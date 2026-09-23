#!/usr/bin/env python
"""Figure 3: verification at two levels -- per cell, and per document.

Left: each (base, task) cell with the correction applied to every document
against the same correction gated by the selector. Points above the diagonal
are cells the selector improves. The average and the tail are different claims,
and this panel is about the tail.

Right: the same question one level down, over the documents where the two
caches differ. Only accepted corrections move the score, so what matters is not
just how often the selector is right but how large its right and wrong
decisions are.

    python figures/plot_safety.py
"""
from __future__ import annotations

import argparse
import csv
import statistics as st
from pathlib import Path

from style import BLUE, GREY, ORANGE, REPO_ROOT, save
import matplotlib.pyplot as plt

from common import BASE_LABEL, Scores  # noqa: E402

PER_DOC = REPO_ROOT / "results" / "per_document.csv"
MARKERS = {"snapkv": "o", "laprox": "s", "h2o": "^", "lava": "D", "rkv": "v"}


def left_panel(ax, sc: Scores, model: str) -> None:
    lo, hi = 0.0, 0.0
    for base, marker in MARKERS.items():
        xs, ys = [], []
        for t, b, u in sc.paired(model, "base", "rescue_ungated", base):
            g = sc.get(model, "rescue", base, 128, t)
            if g is None:
                continue
            xs.append(u - b)
            ys.append(g - b)
        if not xs:
            continue
        ax.scatter(xs, ys, s=34, marker=marker, alpha=0.8,
                   edgecolor="white", linewidth=0.5, label=BASE_LABEL[base])
        lo = min(lo, min(xs), min(ys))
        hi = max(hi, max(xs), max(ys))
    pad = 0.6
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=GREY, lw=1, zorder=0)
    ax.axhline(0, color=GREY, lw=0.6, ls=":", zorder=0)
    ax.axvline(0, color=GREY, lw=0.6, ls=":", zorder=0)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel("Correction applied to every document")
    ax.set_ylabel("Correction gated by the selector")
    ax.set_title("Per (base, task) cell", fontweight="bold")
    ax.legend(frameon=False, fontsize=9, loc="lower right")


def right_panel(ax) -> str:
    """Accepted/rejected x right/wrong, sized by how much the decision was worth."""
    if not PER_DOC.exists():
        ax.text(0.5, 0.5, "results/per_document.csv not found\n"
                          "run tools/export_per_document.py",
                ha="center", va="center", color=GREY, fontsize=10)
        ax.set_axis_off()
        return ""
    quad: dict[tuple[bool, bool], list[float]] = {}
    with open(PER_DOC, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            b, c, s = (float(r["score_base"]), float(r["score_corr"]),
                       float(r["score_selector"]))
            if abs(c - b) <= 1e-9:
                continue                      # the two caches agree; nothing to decide
            accepted = abs(s - c) < abs(s - b)
            right = (c > b) == accepted
            quad.setdefault((accepted, right), []).append(abs(c - b))
    if not quad:
        ax.set_axis_off()
        return ""
    labels = [("accepted, right", (True, True), ORANGE),
              ("accepted, wrong", (True, False), BLUE),
              ("rejected, right", (False, True), ORANGE),
              ("rejected, wrong", (False, False), BLUE)]
    names = [n for n, k, _ in labels if k in quad]
    counts = [len(quad[k]) for _n, k, _ in labels if k in quad]
    sizes = [st.mean(quad[k]) for _n, k, _ in labels if k in quad]
    colors = [c for _n, k, c in labels if k in quad]
    ax.barh(range(len(names)), counts, color=colors, alpha=0.85)
    for i, (n, s) in enumerate(zip(counts, sizes)):
        ax.text(n, i, f"  {s:.1f} pts each", va="center", fontsize=9, color="#333333")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("Documents")
    ax.set_title("Per document, where the caches differ", fontweight="bold")
    acc = sum(c for n, c in zip(names, counts) if "right" in n) / sum(counts)
    return f"selector agrees with hindsight on {acc * 100:.1f}% of deciding documents"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="llama3_8b")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    fig, (a, b) = plt.subplots(1, 2, figsize=(11.6, 5.0))
    left_panel(a, Scores(), args.model)
    note = right_panel(b)
    if note:
        fig.text(0.5, -0.02, note, ha="center", fontsize=9, color="#555555")
    fig.tight_layout()
    save(fig, "fig3_safety", args.out)


if __name__ == "__main__":
    main()
