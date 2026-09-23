#!/usr/bin/env python
"""Figure: where RESCUE lands between the base policy and an oracle future signal.

Left: each task as a span from the mean base score to the ceiling the same
correction reaches when it is driven by real future attention, with RESCUE
marked on that span. A long span is a task where a better predictor of future
importance would still be worth something.

Right: the share of that span the learned correction takes. The two panels
together are the point -- a large raw gain on a task with a large span is a
smaller achievement than a small gain on a task with almost none.

    python figures/plot_headroom.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

from style import BLUE, GREY, ORANGE, save
import matplotlib.pyplot as plt

from common import Scores, TASK_LABEL  # noqa: E402
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tables"))
from make_headroom_table import rows  # noqa: E402

SPAN, BAD = "#cfd8e3", "#a4442e"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="llama3_8b")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    rs = rows(Scores(), args.model)
    if not rs:
        raise SystemExit("oracle run absent from results/scores.csv")
    rs = rs[::-1]                       # smallest headroom at the bottom
    y = range(len(rs))
    labels = [TASK_LABEL[t] for t, *_ in rs]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(10.4, 5.1),
                                   gridspec_kw={"width_ratios": [1.55, 1]})

    for i, (_t, orc, b, _head, gain) in enumerate(rs):
        axL.plot([b, orc], [i, i], color=SPAN, lw=6, solid_capstyle="round", zorder=1)
        axL.plot([b], [i], marker="|", color=GREY, ms=11, mew=2, zorder=3)
        axL.plot([orc], [i], marker="|", color=BLUE, ms=11, mew=2, zorder=3)
        axL.plot([b + gain], [i], marker="o", ms=6, zorder=4,
                 color=ORANGE if gain >= 0 else BAD,
                 markeredgecolor="white", markeredgewidth=0.8)
    axL.set_yticks(list(y))
    axL.set_yticklabels(labels)
    axL.tick_params(axis="y", labelsize=9.5)
    axL.set_xlabel("LongBench score")
    axL.set_title("Base $\\rightarrow$ oracle future signal", fontweight="bold")
    axL.plot([], [], marker="|", color=GREY, ls="none", ms=11, mew=2, label="base")
    axL.plot([], [], marker="o", color=ORANGE, ls="none", ms=6, label="RESCUE")
    axL.plot([], [], marker="|", color=BLUE, ls="none", ms=11, mew=2, label="oracle")
    axL.legend(frameon=False, fontsize=9, ncol=3, loc="upper center",
               bbox_to_anchor=(0.5, -0.13))

    # Recovery is a ratio, so a task whose span is almost zero can report a huge
    # share for a gain of a fraction of a point. Those bars are clipped at 100%
    # and labelled with the real figure, so the panel stays readable without
    # hiding the value.
    rec = [100 * g / h if h > 1e-9 and g > 0 else 0.0 for *_, h, g in rs]
    axR.barh(list(y), [min(v, 100.0) for v in rec],
             color=[ORANGE if v > 0 else BAD for v in rec], alpha=0.9)
    for i, (v, (*_, h, g)) in enumerate(zip(rec, rs)):
        if g <= 0:
            label = "loses"
        elif v > 100:
            label = f"{v:.0f}% (span {h:.2f})"
        else:
            label = f"{v:.0f}%"
        axR.text(min(v, 100.0) + 1.5, i, label, va="center", fontsize=8.5,
                 color="#333333")
    axR.set_yticks(list(y))
    axR.set_yticklabels([])
    axR.set_xlim(0, 118)
    axR.set_xlabel("Share of the span taken (%)")
    axR.set_title("Recovery", fontweight="bold")

    fig.tight_layout()
    save(fig, "figH_headroom", args.out)


if __name__ == "__main__":
    main()
