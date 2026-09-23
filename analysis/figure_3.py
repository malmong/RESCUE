#!/usr/bin/env python
"""Figure 3 (fig:headroom).

Where RESCUE lands between the base policy and an oracle.

Left: each task as a span from the mean base score to the ceiling the same
correction reaches when it is driven by real future attention, taken from an
un-evicted reference generation. A long span is a task where a better predictor
of future importance would still be worth something; RESCUE is marked on it.

Right: the share of that span the learned correction takes. The two panels
together are what makes the exceptions legible -- a large raw gain on a task
with a large span is a smaller achievement than a small gain on a task with
almost none.

Oracle runs exist for 14 of the 16 tasks: NarrativeQA and GovReport exceed
device memory under the oracle objective, so they are absent rather than zero.

    python analysis/figure_3.py
"""
from __future__ import annotations

import argparse
import statistics as st
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from style import save  # noqa: E402
from data import BASES, Scores  # noqa: E402
from rescue.models import LONGBENCH_TASKS  # noqa: E402

# The oracle ceiling was run on one base policy; base and RESCUE are averaged
# over all five, as the table accompanying this figure does.
ORACLE_BASE = "snapkv"

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--out", type=Path)
args = ap.parse_args()
sc = Scores()

plt.rcParams.update({"font.family": "serif", "font.serif": ["Nimbus Roman", "DejaVu Serif"],
                     "font.size": 15, "axes.linewidth": 1.0, "mathtext.fontset": "cm"})

PRETTY = {"qasper": "Qasper", "multifieldqa_en": "MultiFieldQA-en", "hotpotqa": "HotpotQA",
          "2wikimqa": "2WikiMQA", "musique": "MuSiQue", "qmsum": "QMSum",
          "multi_news": "MultiNews", "trec": "TREC", "triviaqa": "TriviaQA",
          "samsum": "SAMSum", "passage_count": "PassageCount",
          "passage_retrieval_en": "PassageRetrieval-en", "lcc": "LCC", "narrativeqa": "NarrativeQA", "gov_report": "GovReport",
          "repobench": "RepoBench-P"}

rows = []
for t in LONGBENCH_TASKS:
    orc = sc.get("llama3_8b", "oracle_future", ORACLE_BASE, 128, t)
    if orc is None:
        continue
    base = [sc.get("llama3_8b", "base", b, 128, t) for b in BASES]
    resc = [sc.get("llama3_8b", "rescue", b, 128, t) for b in BASES]
    if any(x is None for x in base + resc):
        continue
    b, r = st.mean(base), st.mean(resc)
    rows.append({"task": PRETTY.get(t, t), "base": b, "rescue": r, "oracle": orc,
                 "head": orc - b, "gain": r - b})
rows.sort(key=lambda d: d["head"])          # smallest headroom at the bottom
y = range(len(rows))

C_SPAN, C_BASE, C_ORC, C_RES, C_BAD = "#cfd8e3", "#5b6b7c", "#08306b", "#e08214", "#a4442e"

fig, (axL, axR) = plt.subplots(1, 2, figsize=(9.8, 4.82),
                               gridspec_kw={"width_ratios": [1.55, 1]})

# ---- left: absolute score, base -> oracle span with RESCUE on it ----
for i, d in enumerate(rows):
    axL.plot([d["base"], d["oracle"]], [i, i], c=C_SPAN, lw=5.5, solid_capstyle="round", zorder=1)
    axL.scatter(d["base"], i, s=46, facecolor="white", edgecolor=C_BASE, lw=1.6, zorder=3)
    axL.scatter(d["oracle"], i, s=46, color=C_ORC, zorder=3)
    bad = d["gain"] < 0
    axL.scatter(d["rescue"], i, s=78, marker="D", color=C_BAD if bad else C_RES,
                edgecolor="white", lw=0.9, zorder=4)
    axL.annotate(f"{d['head']:.1f}", (max(d["oracle"], d["rescue"]), i),
                 xytext=(7, 0), textcoords="offset points", va="center",
                 fontsize=13, fontweight="bold", color="#33404d")
axL.set_yticks(list(y)); axL.set_yticklabels([d["task"] for d in rows])
# Sixteen rows sit close together; only these labels go below body size.
axL.tick_params(axis="y", labelsize=12.5)
axL.set_xlabel("LongBench score")
axL.set_title("Base $\\rightarrow$ oracle span, with RESCUE on it", fontsize=16.5,
              fontweight="bold", color="#1b2836", pad=9)
axL.set_xlim(5, 108); axL.grid(axis="x", color="#eceff3", zorder=0)
axL.set_axisbelow(True)
for sp in ("top", "right"): axL.spines[sp].set_visible(False)
axL.scatter([], [], s=46, facecolor="white", edgecolor=C_BASE, lw=1.6, label="Base (mean of 5 policies)")
axL.scatter([], [], s=78, marker="D", color=C_RES, label="RESCUE")
axL.scatter([], [], s=78, marker="D", color=C_BAD, label="RESCUE (below base)")
axL.scatter([], [], s=46, color=C_ORC, label="Oracle future signal")


# ---- right: fraction of the span recovered ----
for i, d in enumerate(rows):
    if d["head"] < 1.0:
        # A span this short makes the ratio meaningless, not favourable: draw
        # nothing rather than a bar that runs off the axis. The caption says why.
        continue
    frac = d["gain"] / d["head"]
    c = C_BAD if frac < 0 else C_RES
    axR.barh(i, frac, height=0.6, color=c, zorder=2)
    off = 4 if frac >= 0 else -4
    axR.annotate(f"{100*frac:.0f}%", (frac, i), xytext=(off, 0),
                 textcoords="offset points", va="center",
                 ha="left" if frac >= 0 else "right", fontsize=13, fontweight="bold", color="#1b2836")
axR.axvline(0, c=C_BASE, lw=1.1, zorder=3)
axR.axvline(1, c=C_ORC, lw=1.1, ls="--", zorder=3)
axR.set_yticks(list(y)); axR.set_yticklabels([])
axR.set_xlim(-0.45, 1.12)
axR.set_xticks([0, 0.5, 1.0])
axR.set_xticklabels(["base", "50%", "oracle"])
axR.set_xlabel("Fraction of the span recovered")
axR.set_title("Recovery", fontsize=16.5, fontweight="bold", color="#1b2836", pad=9)
axR.grid(axis="x", color="#eceff3", zorder=0); axR.set_axisbelow(True)
for sp in ("top", "right"): axR.spines[sp].set_visible(False)

h, l = axL.get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=4, frameon=False, fontsize=14,
           handletextpad=0.5, columnspacing=1.8, bbox_to_anchor=(0.5, -0.012))
fig.tight_layout(rect=(0, 0.062, 1, 1))
save(fig, "figure_3", args.out)
for d in rows[::-1]:
    fr = "n/a" if d["head"] < 1 else f"{100*d['gain']/d['head']:.0f}%"
    print(f"{d['task']:22s} {d['base']:7.2f} {d['rescue']:7.2f} {d['oracle']:7.2f} "
          f"{d['head']:6.2f} {d['gain']:+6.2f} {fr:>7s}")
print(f"\n{len(rows)} tasks; RESCUE below base on "
      f"{sum(1 for d in rows if d['gain']<0)}; span>3pt but <10% recovered on "
      f"{sum(1 for d in rows if d['head']>3 and d['gain']/d['head']<0.10)}")
