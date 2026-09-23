#!/usr/bin/env python
"""Supervised training for the Recent+Predicted-Future-Contribution eviction
method's future predictor.

Two independent per-layer modules, no longer a shared trunk (this is a
deliberate architecture change from the original hidden-state-projection
design -- see the diagnostic history below):

- ImpactPredictor: conditional impact, from the original candidate/current-
  state projection-interaction features (src.method.future_contrib.features).
- FutureQueryPredictor: reuse, from a per-kv-head recent-query window dotted
  against candidate keys, trained via KL divergence against the REAL future
  softmax-attention distribution (see future_attention_distribution below).

Why the reuse branch changed: this project's own diagnostics found the
original hidden-state-similarity feature (frozen random projection of
residual-stream states) carried almost no signal about true future reuse
(Pearson ~0.03-0.09, even in-sample on the model's own training data --
ruling out a chat-template mismatch or insufficient training as the cause).
Switching to real Q/K-space features raised this to ~0.23-0.45 (raw Q.K
logit), and switching further to the REAL softmax-normalized future
attention distribution raised it to ~0.87-0.98 -- softmax(E[q]K^T) !=
E[softmax(qK^T)], so a query predictor must be trained against the
distribution itself, not a mean-future-query regression target. A small
query predictor trained this way (KL loss) reached ~0.74-0.86 Pearson /
~0.73-0.80 top-30% overlap on HELD-OUT documents in a proof-of-concept run,
closing most of the gap to that oracle ceiling.

Base LLM is fully frozen; training data is the same shared full-KV
inference trace corpus used by KVP/ForesightKV/LookaheadKV in this project
(fair-comparison protocol established earlier), NOT a separately generated
dataset.

eviction-policy wiring (S_total = S_recent + lambda*S_future) and the
LongBench/RULER evaluation are deliberately out of scope for this script --
same staging as the other learned baselines (mechanism + training first,
harness wiring next).
"""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from transformers import AutoModelForCausalLM

from src.method.future_contrib.contribution import compute_output_contribution
from src.method.future_contrib.extract import (
    CandidateConfig,
    build_candidate_indices,
    capture_sequence,
    iter_eviction_steps,
)
from src.method.future_contrib.features import build_candidate_feature
from src.method.future_contrib.labels import build_future_labels, calibrate_reuse_threshold, log1p_transform
from src.method.future_contrib.model import FutureQueryPredictor, ImpactPredictor
from src.method.future_contrib.projection import FrozenRandomProjection
from src.method.future_contrib.query_state import recent_query_features


def future_attention_distribution(
    q: torch.Tensor, k: torch.Tensor, cand: torch.Tensor, horizon_start: int, horizon_end: int,
    num_kv_heads: int, kv_repeat: int, scaling: float,
) -> torch.Tensor:
    """REAL causal softmax attention each candidate receives from queries in
    [horizon_start, horizon_end), averaged over query-group heads and over
    tau, then L1-renormalized over the candidate set (a valid probability
    distribution per kv-head, summing to 1 over `cand`). This is the
    FutureQueryPredictor's training target -- deliberately the full softmax
    distribution, not a raw Q.K logit or a mean-future-query proxy (see
    FutureQueryPredictor's docstring for why: softmax(E[q]K^T) !=
    E[softmax(qK^T)], confirmed to matter enormously in this project's own
    diagnostics). Returns [num_kv_heads, num_candidates]."""
    device = q.device
    cand = cand.to(device)
    abs_q_pos = torch.arange(horizon_start, horizon_end, device=device).unsqueeze(1)
    key_pos_full = torch.arange(horizon_end, device=device).unsqueeze(0)
    causal_mask = key_pos_full > abs_q_pos
    q_h_all = q[:, horizon_start:horizon_end, :].float()
    gt = torch.zeros(num_kv_heads, cand.numel(), device=device)
    for g in range(num_kv_heads):
        k_g = k[g, :horizon_end, :].float()
        acc = torch.zeros(horizon_end - horizon_start, cand.numel(), device=device)
        for h_local in range(kv_repeat):
            h = g * kv_repeat + h_local
            scores = torch.matmul(q_h_all[h], k_g.transpose(0, 1)) * scaling
            scores = scores.masked_fill(causal_mask, float("-inf"))
            attn = torch.softmax(scores, dim=-1)
            acc += attn.index_select(1, cand)
        gt[g] = (acc / kv_repeat).mean(dim=0)
    return gt / gt.sum(dim=-1, keepdim=True).clamp_min(1e-12)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_trace_samples(trace_root: str, model_id: str, max_length: int, pattern: str = "by_source_*", limit_runs: int | None = None) -> list[list[int]]:
    """Same shared corpus as scripts/train_foresightkv_paper.py's
    load_trace_samples (prompt_token_ids + generated_token_ids straight from
    generations.jsonl) -- kept as an independent copy per this package's
    isolation-from-baselines convention (spec section 38)."""
    samples: list[list[int]] = []
    run_dirs = sorted(Path(trace_root, model_id).glob(pattern))
    if limit_runs:
        run_dirs = run_dirs[:limit_runs]
    for run_dir in run_dirs:
        gen_path = run_dir / "generations.jsonl"
        if not gen_path.exists():
            continue
        with open(gen_path, "r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                ids = list(row["prompt_token_ids"]) + list(row["generated_token_ids"])
                if len(ids) < 64:
                    continue
                samples.append(ids[:max_length])
    return samples


def load_trace_samples_with_boundary(trace_root: str, model_id: str, max_length: int, pattern: str = "by_source_*", limit_runs: int | None = None) -> list[tuple[list[int], int]]:
    """Same corpus/filtering as load_trace_samples, but ALSO returns each
    sample's real prompt/generation boundary (len(prompt_token_ids), clamped
    to max_length) -- lets a training loop anchor eviction-step sampling at
    the position that actually matters for deployment (this session's
    diagnosis: qasper evicts ONCE right after a long prompt ends, just
    before a short generation begins, but the generic
    iter_eviction_steps/build_candidate_indices ladder sampled document
    depths uniformly from t=budget upward with no notion of "prompt just
    ended" at all -- most of trace_chat's own short multi-turn documents
    don't even resemble that shape). New function so existing callers of
    load_trace_samples (several other training scripts) are unaffected."""
    samples: list[tuple[list[int], int]] = []
    run_dirs = sorted(Path(trace_root, model_id).glob(pattern))
    if limit_runs:
        run_dirs = run_dirs[:limit_runs]
    for run_dir in run_dirs:
        gen_path = run_dir / "generations.jsonl"
        if not gen_path.exists():
            continue
        with open(gen_path, "r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                prompt_ids = list(row["prompt_token_ids"])
                ids = prompt_ids + list(row["generated_token_ids"])
                if len(ids) < 64:
                    continue
                ids = ids[:max_length]
                boundary = min(len(prompt_ids), len(ids))
                samples.append((ids, boundary))
    return samples


def model_arch_from_config(config, model_type: str) -> dict:
    num_kv_heads = config.num_key_value_heads
    num_heads = config.num_attention_heads
    head_dim = getattr(config, "head_dim", None) or (config.hidden_size // num_heads)
    return {
        "hidden_size": config.hidden_size,
        "num_layers": config.num_hidden_layers,
        "num_attention_heads": num_heads,
        "num_key_value_heads": num_kv_heads,
        "kv_repeat": num_heads // num_kv_heads,
        "head_dim": head_dim,
        "model_type": model_type,
    }


@torch.no_grad()
def calibrate_thresholds(
    model, model_type: str, calib_samples: list[list[int]], cand_cfg: CandidateConfig,
    num_layers: int, kv_repeat: int, scaling: float, device, percentile: float, max_seqs: int,
) -> list[float]:
    # Subsample per (layer, step) at collection time, not after pooling everything:
    # a long document can have ~90 eviction steps * 32 layers * (max_candidates *
    # horizon) values each -- pooling all of that across max_seqs sequences before
    # ever subsampling can reach billions of floats and stalls on the final
    # concat/quantile (CPU-bound, GPU sits idle). A percentile estimate doesn't
    # need every point, so cap each call's contribution up front.
    per_call_cap = 4096
    pooled = [[] for _ in range(num_layers)]
    gen = torch.Generator().manual_seed(1234)
    used = 0
    for sample in calib_samples:
        if used >= max_seqs:
            break
        input_ids = torch.tensor([sample], device=device)
        seq_len = input_ids.shape[1]
        steps = list(iter_eviction_steps(seq_len, cand_cfg))
        if not steps:
            continue
        cap = capture_sequence(model, input_ids, model_type)
        for t in steps:
            cand = build_candidate_indices(t, cand_cfg, generator=gen)
            if cand.numel() == 0:
                continue
            for layer_idx in range(num_layers):
                q, k, v = cap.qkv[layer_idx]
                o_proj_weight = model.model.layers[layer_idx].self_attn.o_proj.weight
                c = compute_output_contribution(q, k, v, o_proj_weight, cand, t + 1, cand_cfg.horizon, kv_repeat, scaling)
                flat = c.flatten().cpu()  # move to CPU before subsampling so the CPU `gen` indices match its device
                if flat.numel() > per_call_cap:
                    idx = torch.randperm(flat.numel(), generator=gen)[:per_call_cap]
                    flat = flat[idx]
                if flat.numel():
                    pooled[layer_idx].append(flat)
        used += 1
    thresholds = []
    for layer_idx in range(num_layers):
        if pooled[layer_idx]:
            flat = torch.cat(pooled[layer_idx])
            thresholds.append(calibrate_reuse_threshold(flat, percentile))
        else:
            thresholds.append(0.0)
    return thresholds


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model_type", choices=["llama", "mistral", "qwen3"], required=True)
    p.add_argument("--model_path", required=True)
    p.add_argument("--trace_root", default="/data1/KV_cache_eviction/trace")
    p.add_argument("--trace_model_id", default=None)
    p.add_argument("--trace_pattern", default="by_source_*")
    p.add_argument("--max_length", type=int, default=4096)
    p.add_argument("--output_dir", required=True)

    p.add_argument("--sink_tokens", type=int, default=4)
    p.add_argument("--recent_tokens", type=int, default=64)
    p.add_argument("--budget", type=int, default=256)
    p.add_argument("--step_stride", type=int, default=64)
    p.add_argument("--max_candidates", type=int, default=256)
    p.add_argument("--horizon", type=int, default=256)

    p.add_argument("--proj_dim", type=int, default=32)
    p.add_argument("--proj_seed", type=int, default=0)
    p.add_argument("--shared_hidden_dim", type=int, default=64)
    p.add_argument("--task_hidden_dim", type=int, default=16)
    p.add_argument("--activation", default="silu", choices=["silu", "gelu"])
    p.add_argument("--recent_window", type=int, default=8, help="R: number of recent queries fed to FutureQueryPredictor.")
    p.add_argument("--query_hidden_dim", type=int, default=128)

    p.add_argument("--reuse_percentile", type=float, default=80.0)
    p.add_argument("--calib_seqs", type=int, default=20)
    p.add_argument("--calib_cache", default=None, help="Optional path to save/load calibration thresholds JSON.")

    p.add_argument("--alpha", type=float, default=1.0, help="log1p(alpha*M) transform scale.")
    p.add_argument("--lambda_r", type=float, default=1.0, help="Weight on the reuse KL loss.")
    p.add_argument("--lambda_m", type=float, default=1.0, help="Weight on the impact Huber loss.")

    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--total_training_steps", type=int, default=200)
    p.add_argument("--max_norm", type=float, default=1.0)
    p.add_argument("--checkpoint_interval", type=int, default=50)
    p.add_argument("--seed", type=int, default=422)
    args = p.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, attn_implementation="eager", device_map="cuda",
    )
    model.eval()
    model.requires_grad_(False)  # spec section 1: base LLM completely frozen
    config = model.config
    arch = model_arch_from_config(config, args.model_type)
    num_layers, kv_repeat = arch["num_layers"], arch["kv_repeat"]
    scaling = arch["head_dim"] ** -0.5
    device = model.device

    cand_cfg = CandidateConfig(
        sink_tokens=args.sink_tokens, recent_tokens=args.recent_tokens, budget=args.budget,
        step_stride=args.step_stride, max_candidates=args.max_candidates, horizon=args.horizon,
    )

    model_id = args.trace_model_id or args.model_type
    all_samples = load_trace_samples(args.trace_root, model_id, args.max_length, pattern=args.trace_pattern)
    print(f"Loaded {len(all_samples)} sequences from trace ({args.trace_root}/{model_id})")
    random.Random(args.seed).shuffle(all_samples)
    calib_samples = all_samples[: max(1, len(all_samples) // 20)]  # small held-out calibration slice
    train_samples = all_samples[max(1, len(all_samples) // 20):]
    if not train_samples:
        train_samples = all_samples

    thresholds = None
    if args.calib_cache and os.path.exists(args.calib_cache):
        with open(args.calib_cache) as f:
            saved = json.load(f)
        thresholds = [saved[f"layer_{i}"] for i in range(num_layers)]
        print(f"Loaded calibration thresholds from {args.calib_cache}")
    if thresholds is None:
        print(f"Calibrating layer-wise reuse thresholds (top-{100 - args.reuse_percentile:.0f}%) over {min(args.calib_seqs, len(calib_samples))} held-out sequences...")
        thresholds = calibrate_thresholds(
            model, args.model_type, calib_samples, cand_cfg, num_layers, kv_repeat, scaling, device,
            args.reuse_percentile, args.calib_seqs,
        )
        calib_path = args.calib_cache or os.path.join(args.output_dir, "calibration_thresholds.json")
        with open(calib_path, "w") as f:
            json.dump({f"layer_{i}": thresholds[i] for i in range(num_layers)}, f, indent=2)
        print(f"Saved calibration thresholds to {calib_path}")
    print("thresholds:", [round(x, 5) for x in thresholds])

    projection = FrozenRandomProjection(arch["hidden_size"], args.proj_dim, seed=args.proj_seed, device=device, dtype=torch.float32)
    num_kv_heads = arch["num_key_value_heads"]

    impact_predictors = [
        ImpactPredictor(
            input_dim=3 * args.proj_dim, shared_hidden_dim=args.shared_hidden_dim,
            task_hidden_dim=args.task_hidden_dim, activation=args.activation,
        ).to(device)
        for _ in range(num_layers)
    ]
    query_predictors = [
        FutureQueryPredictor(args.recent_window, arch["head_dim"], hidden_dim=args.query_hidden_dim).to(device)
        for _ in range(num_layers)
    ]

    to_optim = [pm for pred in impact_predictors for pm in pred.parameters()]
    to_optim += [pm for pred in query_predictors for pm in pred.parameters()]
    optimizer = torch.optim.AdamW(to_optim, lr=args.lr, weight_decay=args.weight_decay)

    gen = torch.Generator().manual_seed(args.seed + 1)

    for step in range(args.total_training_steps):
        sample = train_samples[random.randrange(len(train_samples))]
        input_ids = torch.tensor([sample], device=device)
        seq_len = input_ids.shape[1]
        steps = list(iter_eviction_steps(seq_len, cand_cfg))
        if not steps:
            continue

        with torch.no_grad():
            cap = capture_sequence(model, input_ids, args.model_type)

        optimizer.zero_grad(set_to_none=True)
        total_loss = torch.zeros((), device=device)
        n_examples = 0
        loss_components = {"loss_reuse_kl": 0.0, "loss_impact": 0.0}
        identity_checked = False

        for t in steps:
            cand = build_candidate_indices(t, cand_cfg, generator=gen)
            if cand.numel() == 0:
                continue
            horizon_end = min(t + 1 + cand_cfg.horizon, seq_len)
            if horizon_end <= t + 1:
                continue
            for layer_idx in range(num_layers):
                q, k, v = cap.qkv[layer_idx]
                hidden_in = cap.hidden_in[layer_idx]
                o_proj_weight = model.model.layers[layer_idx].self_attn.o_proj.weight

                with torch.no_grad():
                    c = compute_output_contribution(q, k, v, o_proj_weight, cand, t + 1, cand_cfg.horizon, kv_repeat, scaling)
                    labels = build_future_labels(c, thresholds[layer_idx], assert_identity=not identity_checked)
                    identity_checked = True
                    z_i = projection.project(hidden_in[cand].float())      # [num_candidates, proj_dim]
                    c_t = projection.project(hidden_in[t].float())         # [proj_dim]
                    impact_log_target = log1p_transform(labels.conditional_impact, alpha=args.alpha)

                    # FutureQueryPredictor's target: real future softmax-attention
                    # distribution, per kv-head, over this SAME candidate set.
                    gt_dist = future_attention_distribution(
                        q, k, cand, t + 1, horizon_end, num_kv_heads, kv_repeat, scaling
                    ).clamp_min(1e-12)
                    recent_q_flat = recent_query_features(q, t, args.recent_window, num_kv_heads)
                    k_cand = k[:, cand, :].float()  # [num_kv_heads, num_candidates, head_dim]

                x = build_candidate_feature(z_i, c_t)  # [num_candidates, 3*proj_dim]
                impact_log_hat = impact_predictors[layer_idx](x)

                q_hat = query_predictors[layer_idx](recent_q_flat)  # [num_kv_heads, head_dim]
                logits = torch.bmm(k_cand, q_hat.unsqueeze(-1)).squeeze(-1) * scaling  # [num_kv_heads, num_candidates]
                log_pred = torch.log_softmax(logits, dim=-1)
                loss_reuse_kl = F.kl_div(log_pred, gt_dist, reduction="batchmean")

                mask = labels.impact_mask
                if mask.sum() > 0:
                    loss_impact = F.huber_loss(impact_log_hat[mask], impact_log_target[mask], delta=1.0)
                else:
                    loss_impact = impact_log_hat.sum() * 0.0

                loss_total = args.lambda_r * loss_reuse_kl + args.lambda_m * loss_impact
                total_loss = total_loss + loss_total
                n_examples += 1
                loss_components["loss_reuse_kl"] += float(loss_reuse_kl.item())
                loss_components["loss_impact"] += float(loss_impact.item())

        if n_examples == 0:
            continue

        (total_loss / n_examples).backward()
        grad_norm = clip_grad_norm_(to_optim, max_norm=args.max_norm)
        if not torch.isfinite(grad_norm):
            print(f"step={step + 1}: skipped (non-finite grad_norm={grad_norm.item()})")
            optimizer.zero_grad(set_to_none=True)
            continue
        optimizer.step()

        avg = {k: v / n_examples for k, v in loss_components.items()}
        print(
            f"step={step + 1} seq_len={seq_len} n_layer_steps={n_examples} "
            f"loss={total_loss.item() / n_examples:.4f} "
            f"L_R_kl={avg['loss_reuse_kl']:.4f} L_M={avg['loss_impact']:.4f} "
            f"grad_norm={grad_norm.item():.4f}"
        )

        if args.checkpoint_interval and (step + 1) % args.checkpoint_interval == 0:
            _save_checkpoint(args, query_predictors, impact_predictors, projection, thresholds, arch, step + 1)

    _save_checkpoint(args, query_predictors, impact_predictors, projection, thresholds, arch, args.total_training_steps, final=True)


def _save_checkpoint(args, query_predictors, impact_predictors, projection, thresholds, arch, step, final: bool = False) -> None:
    tag = "final" if final else f"step_{step}"
    ckpt_dir = os.path.join(args.output_dir, tag)
    os.makedirs(ckpt_dir, exist_ok=True)
    torch.save(
        {
            "query_predictors": [pred.state_dict() for pred in query_predictors],
            "impact_predictors": [pred.state_dict() for pred in impact_predictors],
            "projection": projection.state_dict(),
            "thresholds": thresholds,
            "model_arch": arch,
            "recent_window": args.recent_window,
            "query_hidden_dim": args.query_hidden_dim,
            "step": step,
            "args": vars(args),
        },
        os.path.join(ckpt_dir, "future_contrib.pt"),
    )
    print(f"Saved checkpoint to {ckpt_dir}")


if __name__ == "__main__":
    main()
