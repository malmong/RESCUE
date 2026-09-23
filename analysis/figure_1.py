#!/usr/bin/env python
"""Figure 1 (fig:budget_overlap). Where the oracle's top-B entries come from.

For each budget, the oracle set -- the top-B entries by future attention mass --
is split by which signal retains it: both, the observed signal only, the
predicted-future signal only, or neither. The predicted-future-only share is
what a correction can address and an observed policy cannot, and it is the
largest single slice at the budget the paper works at.

Measured values come from results/measurements/coverage_by_budget.csv, which
scripts/measurements/coverage.py produces.

    python analysis/figure_1.py
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.font_manager as fm  # noqa: E402
import numpy as np  # noqa: E402

from style import REPO_ROOT, save  # noqa: E402

DATA = REPO_ROOT / "results" / "measurements" / "coverage_by_budget.csv"

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--out", type=Path)
args = ap.parse_args()

FONT_FAMILY = ["Nimbus Roman", "Liberation Serif", "DejaVu Serif"]
plt.rcParams["font.family"] = FONT_FAMILY
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["axes.linewidth"] = 1.1
plt.rcParams["xtick.major.width"] = 1.1
plt.rcParams["ytick.major.width"] = 1.1

if not DATA.exists():
    raise SystemExit(f"{DATA} not found; run scripts/measurements/coverage.py")
with open(DATA, encoding="utf-8") as fh:
    _rows = list(csv.DictReader(l for l in fh if not l.startswith("#")))
BUDGETS = [int(r["budget"]) for r in _rows]
DATA_BY_BUDGET = {int(r["budget"]): (float(r["shared"]), float(r["observed_only"]),
                                     float(r["future_only"]), float(r["missed"]))
                  for r in _rows}
CATEGORIES = ["Shared", "Observed-only", "Predicted future-only", "Missed by both"]
# Dark navy -> medium blue -> amber for the novel "future" slice -> light grey,
# echoing panel (a)'s palette while keeping four categories print-legible.
COLORS = ["#1c3f66", "#4f86c6", "#e08a1e", "#c9c9c9"]

fig, ax = plt.subplots(figsize=(7.2, 5.6), dpi=300)

x = np.arange(len(BUDGETS))
width = 0.6
bottoms = np.zeros(len(BUDGETS))
values_by_cat = list(zip(*[DATA_BY_BUDGET[b] for b in BUDGETS]))  # per-category series across budgets

bars_by_cat = []
for cat, color, vals in zip(CATEGORIES, COLORS, values_by_cat):
    vals = np.array(vals, dtype=float)
    bars = ax.bar(x, vals, width, bottom=bottoms, color=color, edgecolor="white",
                   linewidth=0.8, label=cat, zorder=3)
    bars_by_cat.append((bars, vals, bottoms.copy()))
    bottoms += vals

# in-segment percentage labels (white on dark segments, dark on light segment)
label_color_for_cat = ["white", "white", "white", "#333333"]
for (bars, vals, base), lc in zip(bars_by_cat, label_color_for_cat):
    for rect, v, b in zip(bars, vals, base):
        if v < 3.0:
            continue  # too thin to label without clutter
        ax.text(rect.get_x() + rect.get_width() / 2, b + v / 2, f"{v:.1f}",
                 ha="center", va="center", fontsize=16, color=lc, fontweight="bold", zorder=4)

ax.set_xticks(x)
ax.set_xticklabels([str(b) for b in BUDGETS], fontsize=13, fontweight="bold")
ax.set_ylim(0, 100)
ax.set_yticks(np.arange(0, 101, 20))
ax.set_yticklabels([f"{v}" for v in np.arange(0, 101, 20)], fontsize=12)

ax.set_xlabel("Budget (tokens)", fontsize=14, fontweight="bold", labelpad=8)
ax.set_ylabel("Share of oracle future-attention top-$B$ entries (%)", fontsize=13, fontweight="bold", labelpad=10)
ax.set_title("Llama-3.1-8B-Instruct  |  Qasper", fontsize=14.5, fontweight="bold", pad=16)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", color="#dddddd", linewidth=0.8, zorder=0)
ax.set_axisbelow(True)

legend = ax.legend(
    loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, frameon=False,
    fontsize=11.5, handlelength=1.4, handleheight=1.1, columnspacing=1.4,
)
for text in legend.get_texts():
    text.set_fontweight("bold")

fig.tight_layout()
save(fig, "figure_1", args.out)
