#!/usr/bin/env python
"""Phase C: combine phase_a (GT + 5 observed + ForesightKV) and phase_b
(LookaheadKV) per-document score dumps into the coverage table:
  Shared / Observed-only / Future-only / Missed-by-both
as a fraction of the GT oracle top-B set, per budget, averaged over all
qasper documents that both phases scored.

Observed_avg = mean of rank-percentile-normalized {H2O, SnapKV, LaProx, LAVa, R-KV}
Future_avg   = mean of rank-percentile-normalized {ForesightKV, LookaheadKV}
(rank-percentile: scale-free, robust to R-KV's signed/unbounded scores and
every method's very different native units -- see coverage_lib.rank_pct)
"""
from __future__ import annotations

import os

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from coverage_lib import rank_pct, BUDGETS, SINK_TOKENS, RECENT_TOKENS  # noqa: E402

SCRATCH = Path(os.environ.get(
    "RESCUE_SCRATCH", Path(__file__).resolve().parents[2] / "runs" / "coverage"))

OBSERVED_KEYS = ["h2o", "snapkv", "laprox", "lava", "rkv"]
FUTURE_KEYS = ["foresightkv", "lookaheadkv"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="qasper")
    args = ap.parse_args()
    phase_a_dir = SCRATCH / "coverage_out" / args.task / "phase_a"
    phase_b_dir = SCRATCH / "coverage_out" / args.task / "phase_b_lookaheadkv"

    rows_per_budget = {b: [] for b in BUDGETS}
    n_docs = 0
    for path in sorted(phase_a_dir.glob("*.pt")):
        b_path = phase_b_dir / path.name
        if not b_path.exists():
            continue
        a = torch.load(path, map_location="cpu")
        b = torch.load(b_path, map_location="cpu")
        n_cand = a["cand_end"] - a["cand_start"]
        if a["gt"].numel() != n_cand or b["lookaheadkv"].numel() != n_cand:
            continue
        n_docs += 1

        observed_scores = [rank_pct(a[k]) for k in OBSERVED_KEYS]
        observed_avg = torch.stack(observed_scores, dim=0).mean(dim=0)
        future_scores = [rank_pct(a["foresightkv"]), rank_pct(b["lookaheadkv"])]
        future_avg = torch.stack(future_scores, dim=0).mean(dim=0)
        gt = a["gt"]

        for budget in BUDGETS:
            b_remaining = max(1, min(budget - SINK_TOKENS - RECENT_TOKENS, n_cand))
            gt_top = set(gt.topk(b_remaining).indices.tolist())
            obs_top = set(observed_avg.topk(b_remaining).indices.tolist())
            fut_top = set(future_avg.topk(b_remaining).indices.tolist())

            shared = len(gt_top & obs_top & fut_top)
            obs_only = len(gt_top & (obs_top - fut_top))
            fut_only = len(gt_top & (fut_top - obs_top))
            missed = len(gt_top - obs_top - fut_top)
            denom = len(gt_top)
            rows_per_budget[budget].append((shared / denom, obs_only / denom, fut_only / denom, missed / denom))

    print(f"task: {args.task}\ndocuments aggregated: {n_docs}\n")
    print(f"{'budget':>8} | {'Shared':>8} | {'Obs-only':>9} | {'Fut-only':>9} | {'Missed':>8}")
    print("-" * 55)
    for budget in BUDGETS:
        rows = rows_per_budget[budget]
        if not rows:
            print(f"{budget:>8} | (no data)")
            continue
        t = torch.tensor(rows)
        mean = t.mean(dim=0) * 100
        print(f"{budget:>8} | {mean[0]:>7.1f}% | {mean[1]:>8.1f}% | {mean[2]:>8.1f}% | {mean[3]:>7.1f}%")


if __name__ == "__main__":
    main()
