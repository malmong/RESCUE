#!/usr/bin/env python
"""Figure 1(b): where the oracle's top-B entries actually come from.

For each budget, the oracle set -- the top-B entries by future attention mass --
is split by which signal retains it: both, the observed signal only, the
predicted-future signal only, or neither. The predicted-future-only share is
what a correction can address and an observed policy cannot, and it is the
largest single slice at the budget the paper works at.

    python figures/plot_overlap.py
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from style import REPO_ROOT, save
import matplotlib.pyplot as plt

DATA = REPO_ROOT / "results" / "measurements" / "coverage_by_budget.csv"
CATEGORIES = ["Shared", "Observed-only", "Predicted future-only", "Missed by both"]
KEYS = ["shared", "observed_only", "future_only", "missed"]
# Dark navy -> medium blue -> amber for the novel "future" slice -> light grey.
COLORS = ["#1c3f66", "#4f86c6", "#e08a1e", "#c9c9c9"]
FONT = ["Nimbus Roman", "Liberation Serif", "DejaVu Serif"]


def load() -> tuple[list[int], dict[str, list[float]]]:
    if not DATA.exists():
        raise SystemExit(f"{DATA} not found; run tools/measurements/coverage.py")
    rows = [r for r in csv.DictReader(
        line for line in open(DATA, encoding="utf-8") if not line.startswith("#"))]
    budgets = [int(r["budget"]) for r in rows]
    return budgets, {k: [float(r[k]) for r in rows] for k in KEYS}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    plt.rcParams["font.family"] = FONT
    plt.rcParams["mathtext.fontset"] = "stix"
    budgets, series = load()

    fig, ax = plt.subplots(figsize=(7.2, 5.6), dpi=300)
    x = np.arange(len(budgets))
    bottoms = np.zeros(len(budgets))
    for key, label, color in zip(KEYS, CATEGORIES, COLORS):
        vals = np.array(series[key])
        ax.bar(x, vals, 0.6, bottom=bottoms, label=label, color=color,
               edgecolor="white", linewidth=0.8)
        for xi, (v, b) in enumerate(zip(vals, bottoms)):
            if v >= 4.0:
                ax.text(xi, b + v / 2, f"{v:.1f}", ha="center", va="center",
                        color="white" if color != "#c9c9c9" else "#333333",
                        fontsize=10, fontweight="bold")
        bottoms += vals

    ax.set_xticks(x)
    ax.set_xticklabels([f"$B={b}$" for b in budgets])
    ax.set_xlabel("Cache budget (entries per layer)", fontweight="bold")
    ax.set_ylabel("Share of oracle future-attention top-$B$ entries (%)",
                  fontweight="bold")
    ax.set_ylim(0, 100)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False)
    save(fig, "figF_budget_overlap", args.out)


if __name__ == "__main__":
    main()
