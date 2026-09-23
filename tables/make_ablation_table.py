#!/usr/bin/env python
"""Section 5.4 ablation: what the correction contributes, and what the selector
contributes on top of it.

Six arms over the same (base, task) cells, paired so the deltas are within-cell:

    base              the base policy alone
    fullfuture        a policy-agnostic full-future target, applied to every document
    rescue_ungated    the residual target, applied to every document
    fullfuture_gated  the full-future target under the fidelity selector
    rescue            the residual target under the selector -- the method
    oracle_selector   per document, whichever cache scored better in hindsight

"Harmful" counts cells that lose at least a point: the average and the tail are
different claims, and the selector is what moves the tail.

    python tables/make_ablation_table.py --format text
"""
from __future__ import annotations

import argparse
import statistics as st
from pathlib import Path

import csv
from collections import defaultdict

from common import (LATEX_FOOTER, REPO_ROOT, Scores, bootstrap_ci, latex_header,
                    write)

PER_DOC = REPO_ROOT / "results" / "per_document.csv"

ARMS = [
    ("base", "Base", False),
    ("fullfuture", r"\quad Full-future target", True),
    ("rescue_ungated", r"\quad Residual target", True),
    ("fullfuture_gated", r"\quad Full-future target", True),
    ("rescue", r"\quad Residual target (RESCUE)", True),
    ("oracle_selector", "Oracle per-input selector", True),
]
GROUPS = {
    "fullfuture": r"\emph{Correction applied unconditionally:}",
    "fullfuture_gated": r"\emph{Correction gated by the fidelity selector:}",
    "oracle_selector": None,
}


def oracle_cells(model: str) -> list[tuple[str, str, float, float]]:
    """The ceiling on any per-document rule of this form.

    Not an eviction run: for each document, take whichever of the two caches
    scored better once the answer was known, then average over documents and
    tasks exactly as every other row does. It needs per-document scores, which
    a cell average cannot supply.
    """
    if not PER_DOC.exists():
        return []
    acc: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(
        lambda: {"base": [], "orc": []})
    with open(PER_DOC, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["model"] != model:
                continue
            b, c = float(r["score_base"]), float(r["score_corr"])
            acc[(r["base"], r["task"])]["base"].append(b)
            acc[(r["base"], r["task"])]["orc"].append(max(b, c))
    return [(base, task, st.mean(v["base"]), st.mean(v["orc"]))
            for (base, task), v in acc.items()]


def rows(sc: Scores, model: str) -> list[tuple[str, str, float | None, float | None,
                                               float | None, int, int]]:
    out = []
    for arm, label, is_delta in ARMS:
        cells = (oracle_cells(model) if arm == "oracle_selector"
                 else sc.cells(model, "base", arm))
        if not cells:
            out.append((arm, label, None, None, None, 0, 0))
            continue
        avg = st.mean([v for *_, v in cells])
        if not is_delta:
            out.append((arm, label, avg, None, None, 0, len(cells)))
            continue
        deltas = [v - u for *_, u, v in cells]
        mean, _lo, _hi = bootstrap_ci(cells)
        harmful = sum(1 for d in deltas if d <= -1.0)
        out.append((arm, label, avg, mean, min(deltas), harmful, len(cells)))
    return out


def text(sc: Scores, model: str) -> str:
    lines = [f"{model}  --  ablation over paired (base, task) cells",
             f"  {'arm':30s} {'avg':>7s} {'delta':>8s} {'worst':>8s} {'harmful':>9s}"]
    for arm, label, avg, delta, worst, harmful, n in rows(sc, model):
        if avg is None:
            lines.append(f"  {arm:30s} {'(absent from results/scores.csv)':>35s}")
            continue
        d = f"{delta:+8.2f}" if delta is not None else " " * 8
        w = f"{worst:+8.2f}" if worst is not None else " " * 8
        h = f"{harmful:5d}/{n:<4d}" if delta is not None else " " * 9
        lines.append(f"  {label.replace(chr(92) + 'quad', ' ').strip():30s} {avg:7.2f} {d} {w} {h}")
    return "\n".join(lines)


def latex(sc: Scores, model: str) -> str:
    out = [latex_header(
        "Ablation over the paired (base, task) cells. The two targets share features, "
        "architecture, corpus, budget, combination rule and selector, differing only in "
        "whether the base policy's correct selections are excluded from supervision. "
        "``Harmful'' counts cells losing at least a point.",
        "tab:selector_ablation", "lcccc"),
        r"\textbf{Method} & \textbf{Avg. score} & \textbf{$\Delta$ vs. Base} & "
        r"\textbf{Worst $\Delta$} & \textbf{Harmful cells} \\", r"\midrule"]
    for arm, label, avg, delta, worst, harmful, n in rows(sc, model):
        if arm in GROUPS:
            if GROUPS[arm]:
                out.append(r"\midrule")
                out.append(f"\\multicolumn{{5}}{{l}}{{{GROUPS[arm]}}} \\\\")
            else:
                out.append(r"\midrule")
        if avg is None:
            out.append(f"{label} & \\LBmissing & \\LBmissing & \\LBmissing & \\LBmissing \\\\")
            continue
        if delta is None:
            out.append(f"{label} & {avg:.2f} & -- & -- & -- \\\\")
        else:
            bold = r"\textbf{%s}" if arm == "rescue" else "%s"
            out.append(f"{label} & {bold % f'{avg:.2f}'} & {bold % f'{delta:+.2f}'} & "
                       f"${worst:+.2f}$ & {bold % f'{harmful} / {n}'} \\\\")
    out.append(LATEX_FOOTER)
    return "\n".join(out)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="llama3_8b")
    p.add_argument("--format", choices=["latex", "text"], default="latex")
    p.add_argument("--out", type=Path)
    args = p.parse_args()
    sc = Scores()
    write(args.out, latex(sc, args.model) if args.format == "latex" else text(sc, args.model))


if __name__ == "__main__":
    main()
