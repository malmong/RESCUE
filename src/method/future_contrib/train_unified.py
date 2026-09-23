#!/usr/bin/env python
"""4-way future-formulation comparison, same shared 6-dim feature basis
(src.method.future_contrib.unified), same trace corpus, same candidate
generation config as the proven Q/K FutureQueryPredictor recipe -- only the
training objective differs:

  A  FutureAttn/KL     : per-kv-head softmax score vs real future attention distribution (KL)
  B  DirectContribution: position-level scalar vs mean future output-contribution (Huber, log1p)
  C  Factorized        : position-level (reuse_rate, impact) vs build_future_labels targets (BCE + Huber)
  D  Ranking           : position-level scalar, SAME raw target as B, pairwise ranking loss instead of regression

Run once per --objective; each run is fully independent (own optimizer, own
checkpoint). See src/method/future_contrib/unified.py for the feature/model
definitions shared across all four.
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

from src.method.future_contrib.active_pattern import ACTIVITY_WINDOW, compute_activity_stats_offline
from src.method.future_contrib.contribution import compute_output_contribution
from src.method.future_contrib.extract import CandidateConfig, build_candidate_indices, capture_sequence, iter_eviction_steps
from src.method.future_contrib.labels import build_future_labels, log1p_transform
from src.method.future_contrib.regret import compute_regret_via_grad
from src.method.future_contrib.query_state import recent_query_last_and_mean, recent_query_last_m
from src.method.future_contrib.train import calibrate_thresholds, load_trace_samples, model_arch_from_config, future_attention_distribution, set_seed
from src.method.future_contrib.unified import (
    RICH_REUSE_M,
    build_predictor,
    compute_activepattern_features_per_head,
    compute_richreuse_features_per_head,
    compute_unified_features_per_head,
    to_position_level,
    vwo_norm_candidates,
)


def pairwise_ranking_loss(s: torch.Tensor, u: torch.Tensor, n_shuffles: int = 3, eps: float = 1e-8) -> torch.Tensor:
    n = s.shape[0]
    if n < 2:
        return s.sum() * 0.0
    parts = []
    for _ in range(n_shuffles):
        perm = torch.randperm(n, device=s.device)
        half = n // 2
        i_idx, j_idx = perm[:half], perm[half : 2 * half]
        diff_u = u[i_idx] - u[j_idx]
        valid = diff_u.abs() > eps
        if valid.sum() == 0:
            continue
        sign = torch.sign(diff_u[valid])
        margin_logit = sign * (s[i_idx][valid] - s[j_idx][valid])
        parts.append(F.softplus(-margin_logit))  # -log(sigmoid(x)) == softplus(-x)
    if not parts:
        return s.sum() * 0.0
    return torch.cat(parts).mean()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--objective", required=True, choices=["A", "B", "C", "D", "G", "H", "S", "R"])
    p.add_argument("--model_type", choices=["llama", "mistral", "qwen3"], required=True)
    p.add_argument("--model_path", required=True)
    p.add_argument("--trace_root", default="/data1/KV_cache_eviction/trace_chat")
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
    p.add_argument("--recent_window", type=int, default=8)

    p.add_argument("--hidden_dim", type=int, default=32)
    p.add_argument("--task_hidden_dim", type=int, default=16)

    p.add_argument("--reuse_percentile", type=float, default=80.0)
    p.add_argument("--calib_seqs", type=int, default=20)
    p.add_argument("--calib_cache", default=None)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--survival_frac", type=float, default=0.25, help="Objective S only: fraction of each step's (post-subsample) candidate pool labeled 'survives'.")

    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--total_training_steps", type=int, default=1500)
    p.add_argument("--max_norm", type=float, default=1.0)
    p.add_argument("--checkpoint_interval", type=int, default=300)
    p.add_argument("--seed", type=int, default=422)
    args = p.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

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

    cand_cfg = CandidateConfig(
        sink_tokens=args.sink_tokens, recent_tokens=args.recent_tokens, budget=args.budget,
        step_stride=args.step_stride, max_candidates=args.max_candidates, horizon=args.horizon,
    )

    model_id = args.trace_model_id or args.model_type
    all_samples = load_trace_samples(args.trace_root, model_id, args.max_length, pattern=args.trace_pattern)
    print(f"[{args.objective}] Loaded {len(all_samples)} sequences from trace ({args.trace_root}/{model_id})")
    random.Random(args.seed).shuffle(all_samples)
    calib_samples = all_samples[: max(1, len(all_samples) // 20)]
    train_samples = all_samples[max(1, len(all_samples) // 20) :]
    if not train_samples:
        train_samples = all_samples

    thresholds = None
    if args.objective == "C":
        if args.calib_cache and os.path.exists(args.calib_cache):
            with open(args.calib_cache) as f:
                saved = json.load(f)
            thresholds = [saved[f"layer_{i}"] for i in range(num_layers)]
        else:
            print(f"[{args.objective}] Calibrating layer-wise reuse thresholds...")
            thresholds = calibrate_thresholds(
                model, args.model_type, calib_samples, cand_cfg, num_layers, kv_repeat, scaling, device,
                args.reuse_percentile, args.calib_seqs,
            )
            calib_path = args.calib_cache or os.path.join(args.output_dir, "calibration_thresholds.json")
            with open(calib_path, "w") as f:
                json.dump({f"layer_{i}": thresholds[i] for i in range(num_layers)}, f, indent=2)
        print(f"[{args.objective}] thresholds:", [round(x, 5) for x in thresholds])

    predictors = [build_predictor(args.objective, args.hidden_dim, args.task_hidden_dim).to(device) for _ in range(num_layers)]
    to_optim = [pm for pred in predictors for pm in pred.parameters()]
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

        regret_per_layer = None
        if args.objective == "R":
            # Document-level, not step-level (unlike compute_output_contribution,
            # which is recomputed per eviction step): one forward+backward gives
            # every layer's regret for the WHOLE document at once, then every
            # step below just slices it by that step's candidate positions.
            try:
                regret_per_layer = compute_regret_via_grad(model, input_ids, num_layers, num_kv_heads, kv_repeat)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(f"[{args.objective}] step={step + 1}: regret backward OOM'd on seq_len={seq_len}, skipping document")
                continue

        optimizer.zero_grad(set_to_none=True)
        total_loss = torch.zeros((), device=device)
        n_examples = 0
        loss_sum = 0.0

        for t in steps:
            cand = build_candidate_indices(t, cand_cfg, generator=gen)
            if cand.numel() == 0:
                continue
            horizon_end = min(t + 1 + cand_cfg.horizon, seq_len)
            if horizon_end <= t + 1:
                continue
            for layer_idx in range(num_layers):
                q, k, v = cap.qkv[layer_idx]
                o_proj_weight = model.model.layers[layer_idx].self_attn.o_proj.weight

                with torch.no_grad():
                    c = compute_output_contribution(q, k, v, o_proj_weight, cand, t + 1, cand_cfg.horizon, kv_repeat, scaling)
                    q_last, q_mean = recent_query_last_and_mean(q, t, args.recent_window, num_kv_heads)
                    k_cand = k[:, cand, :]
                    v_cand = v[:, cand, :]
                    wo_norm = vwo_norm_candidates(v_cand, o_proj_weight, kv_repeat)
                    feat_per_head = compute_unified_features_per_head(q_last, q_mean, k_cand, v_cand, cand, t, wo_norm, scaling)

                pred = predictors[layer_idx]
                if args.objective == "A":
                    with torch.no_grad():
                        gt_dist = future_attention_distribution(q, k, cand, t + 1, horizon_end, num_kv_heads, kv_repeat, scaling).clamp_min(1e-12)
                    logits = pred(feat_per_head).squeeze(-1)  # [H, N]
                    log_pred = torch.log_softmax(logits, dim=-1)
                    loss = F.kl_div(log_pred, gt_dist, reduction="batchmean")
                elif args.objective == "G":
                    with torch.no_grad():
                        gt_dist = future_attention_distribution(q, k, cand, t + 1, horizon_end, num_kv_heads, kv_repeat, scaling).clamp_min(1e-12)
                        q_last_m = recent_query_last_m(q, t, args.recent_window, RICH_REUSE_M, num_kv_heads)
                        feat_rich = compute_richreuse_features_per_head(q_last_m, k_cand, v_cand, cand, t, wo_norm, scaling)
                    logits = pred(feat_rich).squeeze(-1)  # [H, N]
                    log_pred = torch.log_softmax(logits, dim=-1)
                    loss = F.kl_div(log_pred, gt_dist, reduction="batchmean")
                elif args.objective == "H":
                    with torch.no_grad():
                        gt_dist = future_attention_distribution(q, k, cand, t + 1, horizon_end, num_kv_heads, kv_repeat, scaling).clamp_min(1e-12)
                        q_last, q_mean = recent_query_last_and_mean(q, t, args.recent_window, num_kv_heads)
                        window_start = max(0, t - ACTIVITY_WINDOW + 1)
                        recency, freq, streak = compute_activity_stats_offline(q, k, cand, window_start, t + 1, num_kv_heads, kv_repeat, scaling)
                        feat_h = compute_activepattern_features_per_head(q_last, q_mean, k_cand, v_cand, cand, t, wo_norm, scaling, recency, freq, streak)
                    logits = pred(feat_h).squeeze(-1)  # [H, N]
                    log_pred = torch.log_softmax(logits, dim=-1)
                    loss = F.kl_div(log_pred, gt_dist, reduction="batchmean")
                else:
                    feat_pos = to_position_level(feat_per_head)  # [N, 6]
                    if args.objective == "B":
                        with torch.no_grad():
                            target = log1p_transform(c.mean(dim=-1), alpha=args.alpha)
                        y_hat = pred(feat_pos).squeeze(-1)
                        loss = F.huber_loss(y_hat, target, delta=1.0)
                    elif args.objective == "D":
                        with torch.no_grad():
                            target_raw = c.mean(dim=-1)
                        s_hat = pred(feat_pos).squeeze(-1)
                        loss = pairwise_ranking_loss(s_hat, target_raw)
                    elif args.objective == "S":
                        # Priority 3: instead of matching the future
                        # contribution DISTRIBUTION (A/B/D), directly predict
                        # the binary decision that actually determines F1 --
                        # "does this candidate survive inside the true top-
                        # budget set of the real future contribution?" -- as
                        # a classification target instead of a regression/
                        # ranking target on the raw magnitude. Threshold is a
                        # FRACTION of this step's (post-subsample) candidate
                        # pool (--survival_frac), NOT cand_cfg.budget -- that
                        # field is the cache-fill TRIGGER point (e.g. 512),
                        # unrelated to how many candidates actually survive
                        # eviction, and since it exceeds max_candidates=256 it
                        # silently made every candidate "survive" (thresh ==
                        # pool min), collapsing this into "always predict 1"
                        # (confirmed: loss converged to ~0 with near-zero
                        # grad_norm from the very first steps).
                        with torch.no_grad():
                            raw = c.mean(dim=-1)
                            budget_k = max(1, min(int(round(args.survival_frac * raw.numel())), raw.numel()))
                            thresh = torch.topk(raw, budget_k).values.min()
                            survive = (raw >= thresh).float()
                        logit = pred(feat_pos).squeeze(-1)
                        loss = F.binary_cross_entropy_with_logits(logit, survive)
                    elif args.objective == "R":
                        # Priority 6: regress against the gradient-based
                        # eviction-regret target (src.method.future_contrib.regret)
                        # instead of a future-ATTENTION proxy -- directly
                        # targets downstream loss impact.
                        if regret_per_layer is None:
                            continue
                        with torch.no_grad():
                            regret_cand = regret_per_layer[layer_idx][:, cand].mean(dim=0)  # [N], kv-head-averaged
                            target = log1p_transform(regret_cand, alpha=args.alpha)
                        y_hat = pred(feat_pos).squeeze(-1)
                        loss = F.huber_loss(y_hat, target, delta=1.0)
                    else:  # C
                        with torch.no_grad():
                            labels = build_future_labels(c, thresholds[layer_idx])
                            impact_target = log1p_transform(labels.conditional_impact, alpha=args.alpha)
                        reuse_hat, impact_log_hat = pred(feat_pos)
                        loss_reuse = F.binary_cross_entropy(reuse_hat.clamp(1e-6, 1 - 1e-6), labels.reuse_rate)
                        mask = labels.impact_mask
                        loss_impact = F.huber_loss(impact_log_hat[mask], impact_target[mask], delta=1.0) if mask.sum() > 0 else impact_log_hat.sum() * 0.0
                        loss = loss_reuse + loss_impact

                total_loss = total_loss + loss
                n_examples += 1
                loss_sum += float(loss.item())

        if n_examples == 0:
            continue

        (total_loss / n_examples).backward()
        grad_norm = clip_grad_norm_(to_optim, max_norm=args.max_norm)
        if not torch.isfinite(grad_norm):
            print(f"[{args.objective}] step={step + 1}: skipped (non-finite grad_norm)")
            optimizer.zero_grad(set_to_none=True)
            continue
        optimizer.step()

        print(f"[{args.objective}] step={step + 1} seq_len={seq_len} n_layer_steps={n_examples} loss={loss_sum / n_examples:.4f} grad_norm={grad_norm.item():.4f}")

        if args.checkpoint_interval and (step + 1) % args.checkpoint_interval == 0:
            _save_checkpoint(args, predictors, thresholds, arch, step + 1)

    _save_checkpoint(args, predictors, thresholds, arch, args.total_training_steps, final=True)


def _save_checkpoint(args, predictors, thresholds, arch, step, final: bool = False) -> None:
    tag = "final" if final else f"step_{step}"
    ckpt_dir = os.path.join(args.output_dir, tag)
    os.makedirs(ckpt_dir, exist_ok=True)
    torch.save(
        {
            "objective": args.objective,
            "predictors": [pred.state_dict() for pred in predictors],
            "model_arch": arch,
            "recent_window": args.recent_window,
            "hidden_dim": args.hidden_dim,
            "task_hidden_dim": args.task_hidden_dim,
            "alpha": args.alpha,
            "thresholds": thresholds,
            "step": step,
            "args": vars(args),
        },
        os.path.join(ckpt_dir, "unified_future.pt"),
    )
    print(f"[{args.objective}] Saved checkpoint to {ckpt_dir}")


if __name__ == "__main__":
    main()
