#!/usr/bin/env python
"""Multi-head future-query predictor: priorities 4 and 5, sharing one model
(src.method.future_contrib.model.MultiHeadFutureQueryPredictor -- K head_dim
prototypes off one shared trunk, same recent-query-window input as the
proven legacy FutureQueryPredictor, K=1 legacy).

--variant horizon (priority 4, "multi-horizon prediction"): prototype k is
supervised against the real future softmax-attention distribution over ITS
OWN horizon chunk (--horizons, e.g. 16,32,64,128 tokens ahead) via its own
KL loss, summed across k. Tests whether jointly learning multiple horizons
(not just picking one) gives a more robust reuse signal -- especially given
this project's own diagnostic that legacy's original horizon=256 mismatches
qasper's real ~30-token generation length, and a straight horizon=32 retrain
alone didn't move the downstream score.

--variant prototype (priority 5, "multi-prototype future query"): all K
prototypes are supervised against the SAME single (long) horizon target, but
via a GATED MIXTURE (gate also produced by the shared trunk, softmax over K)
-- so unlike --variant horizon's fixed assignment, the model can learn to
route different documents/contexts to different "future query hypotheses"
rather than being forced to average them into one query vector the way
legacy's single-prototype architecture must. Diagnosed this session (qasper,
960-record MP-vs-real-future-attention comparison at the eviction boundary):
this variant's 3 prototypes COLLAPSED to near-identical solutions (gate
~uniform 0.35/0.34/0.31, each prototype's per-horizon KL profile nearly
identical to the other two) -- the gated mixture never learned to specialize,
so --variant prototype's extra capacity bought almost nothing over a single
prototype. Global Spearman correlation against the real oracle was decent
(0.66) but Top-budget-recall (do MP's and oracle's actual keep-sets agree at
the real eviction cutoff) was only 0.359 -- the KL training objective
optimizes the whole distribution's shape, not specifically the boundary
region eviction actually depends on. --variant prototype now also accepts
--boundary_loss (see --variant sml below for the three options) on top of
its existing single-target KL, since this session's follow-up (S/M/L hard
horizon specialization measurably FAILED -- all branches converged toward
matching the longest horizon regardless of their own assigned target,
consistent with qasper's real future attention barely varying by horizon,
corr(T16,T256)=0.978 -- and the resulting downstream score differences
across MP/sml_nokb/sml_retain/sml_rank turned out to be noise: qualitative
per-document inspection showed the gaps were dominated by which policy
happened to hit a repetition-collapse generation on which specific
document, not systematic predictor quality) concluded that objective
redesign on the EXISTING simple architecture is a better next step than
continuing to add architectural complexity.

--variant sml: fixes BOTH diagnosed failure modes at once. Prototypes are
HARD-assigned distinct horizons (--sml_horizons, default "16,64,256" = S/M/L,
short-to-long) via their OWN per-branch KL loss (forces genuine
specialization instead of leaving it to gate-driven emergence, which
collapsed under --variant prototype) -- unlike --variant horizon, which also
hard-assigns horizons but serves a single SELECTED branch at eval time,
--variant sml ALSO trains a gated mixture of all K specialized branches
against the production target (the LAST/longest horizon in --sml_horizons,
matching what rfc_objective=MP/oracle actually compare against downstream),
so specialization and mixing are both learned rather than one substituting
for the other. Additionally adds an --boundary_loss ("retain" or "rank", on
top of the existing whole-distribution KL) that directly targets the
top-budget-recall failure: "retain" maximizes predicted mass on the oracle's
real top-eviction-budget token set; "rank" is a margin ranking loss sampled
ONLY from pairs straddling the real budget cutoff (+/- --boundary_window),
since correctly ordering rank-1 vs rank-1000 is easy and eviction-irrelevant
-- rank-90-vs-rank-100 at the actual cutoff is what a KL loss under-weights
and what actually determines eviction quality. Serves through the EXACT SAME
src.method.rfc_multi.MultiHeadRFCScorer eval-time code as --variant
prototype (checkpoint saved with variant="prototype" regardless of training
variant, since gated-mixture serving is identical) -- no eval-code changes
needed, only --learned-checkpoint needs to point at the new checkpoint.

No impact predictor here (downstream eval always runs --rfc-no-impact per
this project's own finding that the impact multiply doesn't help)."""
from __future__ import annotations

import argparse
import json
import os
import random

import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from transformers import AutoModelForCausalLM

from src.method.future_contrib.extract import CandidateConfig, build_candidate_indices, capture_sequence_chunked, iter_eviction_steps
from src.method.future_contrib.model import MultiHeadFutureQueryPredictor
from src.method.future_contrib.query_state import recent_query_features
from src.method.future_contrib.train import future_attention_distribution, load_trace_samples_with_boundary, model_arch_from_config, set_seed


def boundary_retain_loss(gt: torch.Tensor, pred: torch.Tensor, budget: int) -> torch.Tensor:
    """gt, pred: [H,N] probabilities. Directly maximizes predicted mass on
    each head's oracle top-`budget` (real eviction-keep) token set --
    targets TopBRecall, not overall distribution shape."""
    h, n = gt.shape
    b = max(1, min(int(budget), n))
    losses = []
    for head in range(h):
        topb = gt[head].topk(b).indices
        losses.append(-torch.log(pred[head, topb].clamp_min(1e-12)).mean())
    return torch.stack(losses).mean()


def boundary_critical_loss(gt: torch.Tensor, pred: torch.Tensor, k: int) -> torch.Tensor:
    """gt, pred: [H,N] probabilities. Cost-sensitive critical-miss loss:
    a MASS-WEIGHTED (not uniform, unlike boundary_retain_loss) negative
    log-likelihood restricted to the oracle's own top-k highest-mass
    positions -- concentrates penalty on the FEW genuinely critical tokens
    (per this session's oracle-only=42.4 vs predictor-only=23.59 diagnosis:
    average distribution fitting may matter far less than never losing the
    handful of tokens carrying most of the real future attention mass).
    Restricting to a SMALL k (not the full eviction budget, unlike retain)
    is essential for this to differ from plain KL: mass-weighting over ALL
    N candidates is mathematically identical (same gradient) to KL(gt||pred)
    up to a pred-independent constant, so the "cost-sensitive" effect only
    exists because k << N."""
    h, n = gt.shape
    k = max(1, min(int(k), n))
    losses = []
    for head in range(h):
        topk = gt[head].topk(k).indices
        w = gt[head, topk]
        w = w / w.sum().clamp_min(1e-12)
        losses.append(-(w * torch.log(pred[head, topk].clamp_min(1e-12))).sum())
    return torch.stack(losses).mean()


def boundary_rank_loss(
    gt: torch.Tensor, log_pred: torch.Tensor, budget: int, window: int, margin: float, n_pairs: int,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """gt: [H,N] probabilities, log_pred: [H,N] log-probabilities (same
    scale, monotonic score). Margin ranking loss sampled ONLY from pairs
    whose oracle rank straddles the real budget cutoff (+/- window) -- easy
    rank-1-vs-rank-1000 pairs (already correctly ordered by any reasonable
    predictor, and irrelevant to eviction) are excluded by construction."""
    h, n = gt.shape
    b = max(1, min(int(budget), n))
    losses = []
    for head in range(h):
        order = torch.argsort(gt[head], descending=True)
        lo = max(0, b - window)
        hi = min(n, b + window)
        zone = order[lo:hi]
        rel_budget = b - lo
        pos = zone[:rel_budget]
        neg = zone[rel_budget:]
        if pos.numel() == 0 or neg.numel() == 0:
            continue
        k = min(n_pairs, int(pos.numel()), int(neg.numel()))
        pi = pos[torch.randperm(pos.numel(), generator=generator)[:k]]
        ni = neg[torch.randperm(neg.numel(), generator=generator)[:k]]
        s_pos = log_pred[head, pi]
        s_neg = log_pred[head, ni]
        losses.append(F.relu(margin - (s_pos - s_neg)).mean())
    if not losses:
        return torch.zeros((), device=gt.device)
    return torch.stack(losses).mean()


def _boundary_loss(args, gt: torch.Tensor, pred: torch.Tensor, log_pred: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    """Shared --boundary_loss dispatch for both --variant prototype and
    --variant sml (identical formulas, just computed against each variant's
    own production target/mixed-prediction pair)."""
    if args.boundary_loss == "retain":
        return boundary_retain_loss(gt, pred, args.eviction_budget)
    if args.boundary_loss == "rank":
        return boundary_rank_loss(gt, log_pred, args.eviction_budget, args.boundary_window, args.boundary_margin, args.boundary_n_pairs, generator=gen)
    if args.boundary_loss == "critical":
        return boundary_critical_loss(gt, pred, args.critical_k)
    return torch.zeros((), device=gt.device)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--variant", required=True, choices=["horizon", "prototype", "sml", "wta"])
    p.add_argument("--model_type", choices=["llama", "mistral", "qwen3"], required=True)
    p.add_argument("--model_path", required=True)
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
    p.add_argument("--horizon", type=int, default=256, help="variant=prototype only: the single shared horizon all K prototypes target.")
    p.add_argument("--horizons", default="16,32,64,128", help="variant=horizon only: comma-separated per-prototype horizons, shortest first.")
    p.add_argument("--recent_window", type=int, default=8)
    p.add_argument("--query_hidden_dim", type=int, default=128)
    p.add_argument("--num_prototypes", type=int, default=3, help="variant=prototype only (variant=horizon/sml infer K from --horizons/--sml_horizons).")
    p.add_argument("--sml_horizons", default="16,64,256", help="variant=sml only: comma-separated S/M/L branch horizons, shortest first -- the LAST one is also the gated-mixture's production target (matches rfc_objective=MP/oracle's downstream horizon).")
    p.add_argument("--alpha", type=float, default=1.0, help="variant=sml only: weight on the mean per-branch (S/M/L) KL loss, added to the mixture's own KL loss.")
    p.add_argument("--boundary_loss", choices=["none", "retain", "rank", "critical"], default="none", help="variant=sml/prototype: extra loss term on top of the whole-distribution KL, computed on the GATED MIXTURE against the production target. 'retain' maximizes predicted mass on the oracle's real top-eviction_budget set (uniform weighting); 'rank' is a margin ranking loss sampled only from pairs straddling the real budget cutoff (+/- --boundary_window); 'critical' is a MASS-WEIGHTED negative log-likelihood restricted to the oracle's top --critical_k highest-mass positions only (cost-sensitive: a handful of high-mass tokens matter far more than uniformly fitting the whole top-budget set, per this session's oracle-vs-predictor gap diagnosis).")
    p.add_argument("--gamma", type=float, default=1.0, help="variant=sml/prototype: weight on --boundary_loss.")
    p.add_argument("--eviction_budget", type=int, default=128, help="variant=sml/prototype: the real downstream per-layer eviction budget (matches main.py --budget-tokens) that --boundary_loss's top-B / cutoff is defined against (retain/rank only).")
    p.add_argument("--boundary_window", type=int, default=32, help="--boundary_loss=rank only: sample ranking pairs only from oracle rank in [budget-window, budget+window).")
    p.add_argument("--boundary_margin", type=float, default=0.0, help="--boundary_loss=rank only: margin ranking loss margin.")
    p.add_argument("--boundary_n_pairs", type=int, default=32, help="--boundary_loss=rank only: max sampled pairs per head per (layer,step).")
    p.add_argument("--critical_k", type=int, default=10, help="--boundary_loss=critical only: restrict the mass-weighted loss to the oracle's top-k highest-mass positions (small k is what makes this differ from plain KL).")
    p.add_argument("--boundary_bias", type=float, default=0.0, help="Probability that a training step samples t=(real prompt/generation boundary - 1) INSTEAD OF the generic iter_eviction_steps depth ladder -- anchors training on the scenario that actually matters for deployment (long prompt just ended, short generation about to start), which the generic uniform-document-depth ladder does not privilege at all. 0.0 = old behavior (ladder only). Requires the sample's boundary to be valid (0 < boundary < seq_len-1); falls back to the ladder for that step if not.")
    p.add_argument("--wta_epsilon", type=float, default=0.1, help="variant=wta only: fraction of loss spent on the mean-over-all-K-prototypes KL (keeps non-winning branches minimally alive) vs (1-epsilon) on the winning (lowest-KL) branch alone.")
    p.add_argument("--wta_gate_weight", type=float, default=0.5, help="variant=wta only: weight on the auxiliary gate cross-entropy loss (trains gate_logits to predict which branch would win, so eval-time gated-mixture serving routes to the right specialist instead of relying on the frozen gate values used during hindsight branch selection).")

    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--total_training_steps", type=int, default=1500)
    p.add_argument("--max_norm", type=float, default=1.0)
    p.add_argument("--checkpoint_interval", type=int, default=300)
    p.add_argument("--seed", type=int, default=422)
    args = p.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    if args.variant == "horizon":
        horizons = [int(x) for x in args.horizons.split(",")]
        num_prototypes = len(horizons)
    elif args.variant == "sml":
        horizons = [int(x) for x in args.sml_horizons.split(",")]
        num_prototypes = len(horizons)
    else:  # prototype or wta: single shared horizon target
        horizons = [args.horizon] * args.num_prototypes
        num_prototypes = args.num_prototypes

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
        step_stride=args.step_stride, max_candidates=args.max_candidates, horizon=max(horizons),
    )

    model_id = args.trace_model_id or args.model_type
    all_samples = load_trace_samples_with_boundary(args.trace_root, model_id, args.max_length, pattern=args.trace_pattern)
    n_loaded = len(all_samples)
    # iter_eviction_steps only yields t < seq_len-1 starting from
    # first=max(budget, sink+recent) -- any sample not exceeding that produces
    # ZERO eviction steps, so the training loop's `if not steps: continue`
    # silently wastes that whole draw (no gradient at all). Measured this
    # session: with the default budget=512, this corpus (trace_chat,
    # llama31_8b_instruct) is 1199 loaded sequences but only 284 (23.7%)
    # exceed 512 tokens -- so >76% of every random draw across
    # --total_training_steps was contributing NOTHING, meaning
    # --total_training_steps 2000 was really only ~480 effective gradient
    # steps. Filtering up front makes every draw productive -- same nominal
    # step count, ~4x the effective training. A sample is usable if it can
    # feed EITHER the generic ladder OR (--boundary_bias>0) a valid
    # boundary-anchored point, so raising --boundary_bias can rescue some
    # otherwise-too-short documents too.
    min_len = max(cand_cfg.budget, cand_cfg.sink_tokens + cand_cfg.recent_tokens) + 2
    def _usable(ids: list[int], boundary: int) -> bool:
        if len(ids) > min_len:
            return True
        return args.boundary_bias > 0 and 0 < boundary < len(ids) - 1
    all_samples = [(ids, b) for ids, b in all_samples if _usable(ids, b)]
    print(f"[{args.variant}] Loaded {n_loaded} sequences, {len(all_samples)} usable (ladder len>{min_len} or valid boundary) from trace ({args.trace_root}/{model_id})")
    random.Random(args.seed).shuffle(all_samples)

    query_predictors = [
        MultiHeadFutureQueryPredictor(args.recent_window, arch["head_dim"], num_prototypes, hidden_dim=args.query_hidden_dim).to(device)
        for _ in range(num_layers)
    ]
    to_optim = [pm for pred in query_predictors for pm in pred.parameters()]
    optimizer = torch.optim.AdamW(to_optim, lr=args.lr, weight_decay=args.weight_decay)
    gen = torch.Generator().manual_seed(args.seed + 1)

    for step in range(args.total_training_steps):
        sample, boundary = all_samples[random.randrange(len(all_samples))]
        input_ids = torch.tensor([sample], device=device)
        seq_len = input_ids.shape[1]
        use_boundary = args.boundary_bias > 0 and random.random() < args.boundary_bias and 0 < boundary < seq_len - 1
        if use_boundary:
            steps = [boundary - 1]
        else:
            steps = list(iter_eviction_steps(seq_len, cand_cfg))
        if not steps:
            continue

        with torch.no_grad():
            # capture_sequence_chunked instead of plain capture_sequence:
            # for short docs (chunk_size=4096 default, the common case) this
            # is a single chunk, bit-identical behavior; for longer ones
            # (relevant once --max_length exceeds ~4-6K) it avoids
            # materializing a full O(seq^2) eager attention matrix in one
            # shot, which OOMs (same class of bug this session already fixed
            # in src.method.oracle_future.OracleFutureScorer).
            cap = capture_sequence_chunked(model, input_ids, args.model_type)

        optimizer.zero_grad(set_to_none=True)
        total_loss = torch.zeros((), device=device)
        n_examples = 0
        loss_sum = 0.0

        for t in steps:
            cand = build_candidate_indices(t, cand_cfg, generator=gen)
            if cand.numel() == 0:
                continue
            for layer_idx in range(num_layers):
                q, k, v = cap.qkv[layer_idx]

                with torch.no_grad():
                    recent_q_flat = recent_query_features(q, t, args.recent_window, num_kv_heads)
                    k_cand = k[:, cand, :].float()  # [num_kv_heads, num_candidates, head_dim]
                    gt_dists = []
                    for h in horizons:
                        horizon_end = min(t + 1 + h, seq_len)
                        if horizon_end <= t + 1:
                            gt_dists.append(None)
                            continue
                        gt_dists.append(
                            future_attention_distribution(q, k, cand, t + 1, horizon_end, num_kv_heads, kv_repeat, scaling).clamp_min(1e-12)
                        )
                    if all(g is None for g in gt_dists):
                        continue

                protos, gate_logits = query_predictors[layer_idx](recent_q_flat)  # [H,K,d], [H,K]
                logits = torch.einsum("hnd,hkd->hkn", k_cand, protos) * scaling  # [H, K, N]
                dists = torch.softmax(logits, dim=-1)  # [H, K, N]

                if args.variant == "horizon":
                    loss = torch.zeros((), device=device)
                    n_valid = 0
                    for ki, gt in enumerate(gt_dists):
                        if gt is None:
                            continue
                        log_pred = torch.log(dists[:, ki, :].clamp_min(1e-12))
                        loss = loss + F.kl_div(log_pred, gt, reduction="batchmean")
                        n_valid += 1
                    if n_valid == 0:
                        continue
                    loss = loss / n_valid
                elif args.variant == "sml":
                    # gt_dists[-1] = production target (longest/last horizon,
                    # matches rfc_objective=MP/oracle's downstream horizon=256).
                    gt_final = gt_dists[-1]
                    if gt_final is None:
                        continue
                    gate = torch.softmax(gate_logits, dim=-1)  # [H, K]
                    mixed = torch.einsum("hk,hkn->hn", gate, dists)  # [H, N]
                    log_mixed = torch.log(mixed.clamp_min(1e-12))
                    loss_mix = F.kl_div(log_mixed, gt_final, reduction="batchmean")

                    branch_losses = []
                    for ki, gt in enumerate(gt_dists):
                        if gt is None:
                            continue
                        log_pred = torch.log(dists[:, ki, :].clamp_min(1e-12))
                        branch_losses.append(F.kl_div(log_pred, gt, reduction="batchmean"))
                    loss_branch = torch.stack(branch_losses).mean() if branch_losses else torch.zeros((), device=device)

                    loss_boundary = _boundary_loss(args, gt_final, mixed, log_mixed, gen)
                    loss = loss_mix + args.alpha * loss_branch + args.gamma * loss_boundary
                elif args.variant == "wta":
                    # Winner-take-all (hindsight) training: unlike --variant
                    # sml's horizon-forced specialization (measurably failed
                    # -- qasper's real future barely varies by horizon, so
                    # there was nothing distinct to specialize into), this
                    # lets diversity emerge from the DATA itself. Each
                    # (layer, step) example's gradient goes almost entirely
                    # to whichever prototype ALREADY predicts it best (the
                    # "winner"), which is the standard Multiple-Choice-
                    # Learning mechanism for breaking the collapse plain
                    # gated-mixture KL training exhibited (that objective's
                    # unique minimum has every branch converge to the same
                    # global average -- there is no term rewarding
                    # difference between branches). gate_logits is trained
                    # SEPARATELY via cross-entropy against the (detached)
                    # winner index, so eval-time gated-mixture serving
                    # (src.method.rfc_multi.MultiHeadRFCScorer, unchanged)
                    # learns to ROUTE to the right specialist instead of
                    # needing the hindsight winner-selection mechanism itself
                    # at inference time.
                    gt = gt_dists[0]
                    if gt is None:
                        continue
                    kl_per_proto = []
                    for k in range(dists.shape[1]):
                        log_pred_k = torch.log(dists[:, k, :].clamp_min(1e-12))
                        kl_per_proto.append(F.kl_div(log_pred_k, gt, reduction="batchmean"))
                    kl_stack = torch.stack(kl_per_proto)  # [K]
                    winner = int(torch.argmin(kl_stack).item())
                    loss_winner = kl_stack[winner]
                    loss_mean = kl_stack.mean()
                    gate_target = torch.full((gate_logits.shape[0],), winner, dtype=torch.long, device=device)
                    gate_ce = F.cross_entropy(gate_logits, gate_target)
                    loss = (1.0 - args.wta_epsilon) * loss_winner + args.wta_epsilon * loss_mean + args.wta_gate_weight * gate_ce
                else:  # prototype: single shared target, gated mixture
                    gt = gt_dists[0]
                    if gt is None:
                        continue
                    gate = torch.softmax(gate_logits, dim=-1)  # [H, K]
                    mixed = torch.einsum("hk,hkn->hn", gate, dists)  # [H, N]
                    log_pred = torch.log(mixed.clamp_min(1e-12))
                    loss_kl = F.kl_div(log_pred, gt, reduction="batchmean")
                    loss_boundary = _boundary_loss(args, gt, mixed, log_pred, gen)
                    loss = loss_kl + args.gamma * loss_boundary

                total_loss = total_loss + loss
                n_examples += 1
                loss_sum += float(loss.item())

        if n_examples == 0:
            continue

        (total_loss / n_examples).backward()
        grad_norm = clip_grad_norm_(to_optim, max_norm=args.max_norm)
        if not torch.isfinite(grad_norm):
            print(f"[{args.variant}] step={step + 1}: skipped (non-finite grad_norm)")
            optimizer.zero_grad(set_to_none=True)
            continue
        optimizer.step()

        print(f"[{args.variant}] step={step + 1} seq_len={seq_len} n_layer_steps={n_examples} loss={loss_sum / n_examples:.4f} grad_norm={grad_norm.item():.4f}")

        if args.checkpoint_interval and (step + 1) % args.checkpoint_interval == 0:
            _save_checkpoint(args, query_predictors, arch, horizons, num_prototypes, step + 1)

    _save_checkpoint(args, query_predictors, arch, horizons, num_prototypes, args.total_training_steps, final=True)


def _save_checkpoint(args, query_predictors, arch, horizons, num_prototypes, step, final: bool = False) -> None:
    tag = "final" if final else f"step_{step}"
    ckpt_dir = os.path.join(args.output_dir, tag)
    os.makedirs(ckpt_dir, exist_ok=True)
    # "sml" is served IDENTICALLY to "prototype" at eval time (gated mixture
    # over all K branches, see src.method.rfc_multi.MultiHeadRFCScorer,
    # which only special-cases variant=="prototype" vs else) -- only the
    # TRAINING objective differs, so the checkpoint's serving tag is
    # translated here; args.variant (saved separately below) still records
    # the real training variant for provenance.
    serve_variant = "prototype" if args.variant in ("sml", "wta") else args.variant
    torch.save(
        {
            "variant": serve_variant,
            "train_variant": args.variant,
            "query_predictors": [pred.state_dict() for pred in query_predictors],
            "model_arch": arch,
            "recent_window": args.recent_window,
            "query_hidden_dim": args.query_hidden_dim,
            "num_prototypes": num_prototypes,
            "horizons": horizons,
            "step": step,
            "args": vars(args),
        },
        os.path.join(ckpt_dir, "multi_future.pt"),
    )
    print(f"[{args.variant}] Saved checkpoint to {ckpt_dir}")


if __name__ == "__main__":
    main()
