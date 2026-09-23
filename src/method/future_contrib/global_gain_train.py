#!/usr/bin/env python
"""Layer x kv_head gain calibration for rfc's global_layer allocation (v2).

v1 (see git history / earlier run) trained 400 steps of ONE random document +
ONE random eviction step each, with a "true optimal" target built from two
DIFFERENT formulas (LaProx-style S_recent + Gram-matrix S_future) that likely
don't share a common scale -- loss stayed flat around 0.69 the whole run
(never meaningfully below the random-pairwise baseline) and the resulting
gain (mean drifted 16.0 -> 7.6, uniformly, over 400 steps) scored WORSE than
the plain scalar lambda=16 on qasper (34.19 vs 34.70) -- a real regression,
not noise; that flat loss curve was the correct warning sign.

v2 fixes both root causes:
  1. Same-formula target: real_future is now ALSO computed via LaProx's own
     exact formula (attn L2-norm x static VWO norm), just over the FUTURE
     query window instead of the recent one (compute_training_laprox_future
     below) -- so "true optimal S_total" = real_recent + real_future adds
     two quantities on the SAME natural scale, unlike v1's Gram-matrix
     contribution mixed with a differently-scaled LaProx recent term.
  2. Much denser gradient signal per optimizer step: iterates over EVERY
     eviction step in a document (like legacy/MP's own training loops),
     accumulating the pairwise loss across all of them before ONE
     backward+step, instead of v1's one-document-one-step-one-backward.
"""
from __future__ import annotations

import argparse
import json
import os
import random

import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from transformers import AutoModelForCausalLM

from src.utils.scoring import BaselineConfig, laprox_head_token_scores

_LAPROX_POOLING_CFG = BaselineConfig(policy="laprox")  # default snapkv_kernel_size=7/snapkv_pooling="avgpool"
from src.method.future_contrib.extract import CandidateConfig, build_candidate_indices, capture_sequence, iter_eviction_steps
from src.method.future_contrib.model import MultiHeadFutureQueryPredictor
from src.method.future_contrib.query_state import recent_query_features
from src.method.future_contrib.train import load_trace_samples, model_arch_from_config, set_seed


def _raw_attn_window(q_window: torch.Tensor, k: torch.Tensor, k_end: int, kv_repeat: int, scaling: float, causal_from: int | None) -> torch.Tensor:
    """q_window: [num_heads, win_len, head_dim], the query slice this window
    covers. k: [num_kv_heads, seq_len, head_dim]. k_end: keys visible are
    [0, k_end). causal_from: if set, query at window-relative index i sits at
    absolute position causal_from+i and may only attend to keys <= that
    position (recent-window case); if None, every query in the window may
    attend to the FULL [0, k_end) range already sliced in by the caller
    (future-window case builds k_end per-tau outside instead). Returns raw
    softmax attention [num_heads, win_len, k_end]."""
    num_heads, win_len, _ = q_window.shape
    num_kv_heads = k.shape[0]
    out = torch.empty(num_heads, win_len, k_end, device=q_window.device, dtype=torch.float32)
    if causal_from is not None:
        abs_q_pos = torch.arange(causal_from, causal_from + win_len, device=q_window.device).unsqueeze(1)
        key_pos = torch.arange(k_end, device=q_window.device).unsqueeze(0)
        mask = key_pos > abs_q_pos
    else:
        mask = None
    for g in range(num_kv_heads):
        k_g = k[g, :k_end, :].float()
        for h_local in range(kv_repeat):
            h = g * kv_repeat + h_local
            scores = torch.matmul(q_window[h].float(), k_g.transpose(0, 1)) * scaling
            if mask is not None:
                scores = scores.masked_fill(mask, float("-inf"))
            out[h] = torch.softmax(scores, dim=-1)
    return out


def compute_training_laprox_recent(model, layer_idx: int, q, k, v, cur_t: int, obs_window: int, kv_repeat: int, scaling: float) -> torch.Tensor:
    """Exact LaProx S_recent (src.utils.scoring.laprox_head_token_scores,
    called verbatim) reconstructed from raw captured Q/K. Returns
    [num_kv_heads, cur_t+1]."""
    obs_start = max(0, cur_t - obs_window + 1)
    obs_end = cur_t + 1
    attn = _raw_attn_window(q[:, obs_start:obs_end, :], k, obs_end, kv_repeat, scaling, causal_from=obs_start)
    value_full = v[:, :obs_end, :].unsqueeze(0)
    return laprox_head_token_scores(model, layer_idx, attn.unsqueeze(0), value_full, _LAPROX_POOLING_CFG)


def compute_training_laprox_future(model, layer_idx: int, q, k, v, cur_t: int, horizon: int, kv_repeat: int, scaling: float, cand: torch.Tensor) -> torch.Tensor:
    """Same exact LaProx formula, but over the FUTURE query window
    [cur_t+1, cur_t+1+horizon) instead of the recent one -- the true "if we
    had a perfect S_future" target, on the IDENTICAL scale as
    compute_training_laprox_recent (this is what v1 got wrong: it mixed this
    kind of quantity with a differently-scaled Gram-matrix contribution).
    Returns [num_kv_heads, num_candidates] (only requested candidates, to
    avoid materializing the full [kv_heads, seq_len] row like the recent
    variant does -- the future window's key range already covers the whole
    prefix so there's no smaller "obs_end" to exploit)."""
    seq_len = q.shape[1]
    horizon_end = min(cur_t + 1 + horizon, seq_len)
    if horizon_end <= cur_t + 1:
        return torch.zeros(k.shape[0], int(cand.numel()), device=q.device)
    attn = _raw_attn_window(q[:, cur_t + 1 : horizon_end, :], k, horizon_end, kv_repeat, scaling, causal_from=cur_t + 1)
    value_full = v[:, :horizon_end, :].unsqueeze(0)
    full = laprox_head_token_scores(model, layer_idx, attn.unsqueeze(0), value_full, _LAPROX_POOLING_CFG)  # [kv_heads, horizon_end]
    return full[:, cand]


def pairwise_rank_loss(pred: torch.Tensor, true_survive: torch.Tensor, n_pairs: int, generator: torch.Generator) -> torch.Tensor:
    pos_idx = true_survive.nonzero(as_tuple=True)[0]
    neg_idx = (1 - true_survive).nonzero(as_tuple=True)[0]
    if pos_idx.numel() == 0 or neg_idx.numel() == 0:
        return pred.sum() * 0.0
    pi = pos_idx[torch.randint(0, pos_idx.numel(), (n_pairs,), generator=generator, device="cpu")]
    ni = neg_idx[torch.randint(0, neg_idx.numel(), (n_pairs,), generator=generator, device="cpu")]
    margin_logit = pred[pi] - pred[ni]
    return F.softplus(-margin_logit).mean()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model_type", choices=["llama", "mistral", "qwen3"], required=True)
    p.add_argument("--model_path", required=True)
    p.add_argument("--mp_checkpoint", required=True)
    p.add_argument("--trace_root", default=os.environ.get("RESCUE_TRACE_ROOT", "assets/trace"))
    p.add_argument("--trace_model_id", default=None)
    p.add_argument("--trace_pattern", default="by_source_*")
    p.add_argument("--max_length", type=int, default=4096)
    p.add_argument("--output_dir", required=True)

    p.add_argument("--sink_tokens", type=int, default=4)
    p.add_argument("--recent_tokens", type=int, default=64)
    p.add_argument("--budget", type=int, default=512)
    p.add_argument("--step_stride", type=int, default=64)
    p.add_argument("--max_candidates", type=int, default=256)
    p.add_argument("--horizon", type=int, default=256)
    p.add_argument("--obs_window", type=int, default=32)
    p.add_argument("--per_layer_budget", type=int, default=128)
    p.add_argument("--n_pairs", type=int, default=64)
    p.add_argument("--max_t_per_doc", type=int, default=6, help="Subsample cap on eviction steps per document (all of them accumulate into ONE backward).")

    p.add_argument("--init_gain", type=float, default=16.0)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--total_training_steps", type=int, default=150, help="Now one 'step' = one document, accumulating loss over up to max_t_per_doc eviction steps -- effective sample count is ~total_training_steps * max_t_per_doc.")
    p.add_argument("--max_norm", type=float, default=10.0)
    p.add_argument("--checkpoint_interval", type=int, default=30)
    p.add_argument("--seed", type=int, default=422)
    args = p.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    cpu_gen = torch.Generator().manual_seed(args.seed + 2)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, attn_implementation="eager", device_map="cuda",
    )
    model.eval()
    model.requires_grad_(False)
    config = model.config
    arch = model_arch_from_config(config, args.model_type)
    num_layers, kv_repeat, num_kv_heads = arch["num_layers"], arch["kv_repeat"], arch["num_key_value_heads"]
    scaling = arch["head_dim"] ** -0.5
    device = model.device

    payload = torch.load(args.mp_checkpoint, map_location="cpu")
    assert payload["variant"] == "prototype"
    recent_window = int(payload["recent_window"])
    num_prototypes = int(payload["num_prototypes"])
    query_hidden_dim = int(payload.get("query_hidden_dim", 128))
    mp_predictors = []
    for state_dict in payload["query_predictors"]:
        m = MultiHeadFutureQueryPredictor(recent_window, arch["head_dim"], num_prototypes, hidden_dim=query_hidden_dim).to(device)
        m.load_state_dict(state_dict)
        m.eval()
        for prm in m.parameters():
            prm.requires_grad = False
        mp_predictors.append(m)

    gain = torch.nn.Parameter(torch.full((num_layers, num_kv_heads), float(args.init_gain), device=device))
    optimizer = torch.optim.Adam([gain], lr=args.lr)

    cand_cfg = CandidateConfig(
        sink_tokens=args.sink_tokens, recent_tokens=args.recent_tokens, budget=args.budget,
        step_stride=args.step_stride, max_candidates=args.max_candidates, horizon=args.horizon,
    )
    model_id = args.trace_model_id or args.model_type
    all_samples = load_trace_samples(args.trace_root, model_id, args.max_length, pattern=args.trace_pattern)
    print(f"[gain_v2] Loaded {len(all_samples)} sequences from trace ({args.trace_root}/{model_id})")
    random.Random(args.seed).shuffle(all_samples)
    gen = torch.Generator().manual_seed(args.seed + 1)
    total_budget = args.per_layer_budget * num_layers

    for step in range(args.total_training_steps):
        sample = all_samples[random.randrange(len(all_samples))]
        input_ids = torch.tensor([sample], device=device)
        seq_len = input_ids.shape[1]
        steps = list(iter_eviction_steps(seq_len, cand_cfg))
        if not steps:
            continue
        if len(steps) > args.max_t_per_doc:
            steps = random.sample(steps, args.max_t_per_doc)

        with torch.no_grad():
            cap = capture_sequence(model, input_ids, args.model_type)

        optimizer.zero_grad(set_to_none=True)
        total_loss = torch.zeros((), device=device)
        n_examples = 0
        loss_sum = 0.0

        for t in steps:
            cand = build_candidate_indices(t, cand_cfg, generator=gen)
            if cand.numel() < 8:
                continue
            true_layer_scores, pred_layer_scores = [], []
            for layer_idx in range(num_layers):
                q, k, v = cap.qkv[layer_idx]
                with torch.no_grad():
                    real_recent_full = compute_training_laprox_recent(model, layer_idx, q, k, v, t, args.obs_window, kv_repeat, scaling)
                    real_recent = real_recent_full[:, cand]
                    real_future = compute_training_laprox_future(model, layer_idx, q, k, v, t, cand_cfg.horizon, kv_repeat, scaling, cand)
                    recent_q_flat = recent_query_features(q, t, recent_window, num_kv_heads)

                protos, gate_logits = mp_predictors[layer_idx](recent_q_flat)
                k_cand = k[:, cand, :].float()
                logits = torch.einsum("hnd,hkd->hkn", k_cand, protos) * scaling
                dists = torch.softmax(logits, dim=-1)
                gate = torch.softmax(gate_logits, dim=-1)
                pred_future = torch.einsum("hk,hkn->hn", gate, dists)

                true_total = (real_recent + real_future).mean(dim=0).clamp_min(0.0)
                true_norm = true_total / true_total.sum().clamp_min(1e-12)
                pred_total = (real_recent.detach() + gain[layer_idx].unsqueeze(-1) * pred_future).mean(dim=0).clamp_min(0.0)
                pred_norm = pred_total / pred_total.sum().clamp_min(1e-12)

                true_layer_scores.append(true_norm)
                pred_layer_scores.append(pred_norm)

            true_global = torch.cat(true_layer_scores)
            pred_global = torch.cat(pred_layer_scores)
            budget_k = min(int(total_budget), true_global.numel())
            true_survive = torch.zeros_like(true_global)
            true_survive[torch.topk(true_global, budget_k).indices] = 1.0

            loss = pairwise_rank_loss(pred_global, true_survive, args.n_pairs, cpu_gen)
            total_loss = total_loss + loss
            n_examples += 1
            loss_sum += float(loss.item())

        if n_examples == 0:
            continue

        (total_loss / n_examples).backward()
        grad_norm = clip_grad_norm_([gain], max_norm=args.max_norm)
        if not torch.isfinite(grad_norm):
            print(f"[gain_v2] step={step + 1}: skipped (non-finite grad_norm)")
            optimizer.zero_grad(set_to_none=True)
            continue
        optimizer.step()
        with torch.no_grad():
            gain.clamp_(min=0.0)

        print(f"[gain_v2] step={step + 1} seq_len={seq_len} n_t={n_examples} loss={loss_sum / n_examples:.4f} grad_norm={grad_norm.item():.5f} gain_mean={gain.mean().item():.3f} gain_std={gain.std().item():.3f}")

        if args.checkpoint_interval and (step + 1) % args.checkpoint_interval == 0:
            _save(args, gain, num_layers, num_kv_heads, step + 1)

    _save(args, gain, num_layers, num_kv_heads, args.total_training_steps, final=True)


def _save(args, gain, num_layers, num_kv_heads, step, final: bool = False) -> None:
    tag = "final" if final else f"step_{step}"
    ckpt_dir = os.path.join(args.output_dir, tag)
    os.makedirs(ckpt_dir, exist_ok=True)
    torch.save(
        {"gain": gain.detach().cpu(), "num_layers": num_layers, "num_kv_heads": num_kv_heads, "mp_checkpoint": args.mp_checkpoint, "step": step, "args": vars(args)},
        os.path.join(ckpt_dir, "layer_head_gain.pt"),
    )
    print(f"[gain_v2] Saved checkpoint to {ckpt_dir}")


if __name__ == "__main__":
    main()
