#!/usr/bin/env python
"""Table 1 (tab:longbench-mlp). Every base policy with and without RESCUE.

Each base is measured against its own baseline. The rescue set is defined by
that base's own misses, so a shared reference would be meaningless; the
question a row pair answers is whether correcting *this* policy helps, not
which policy is best.

    python analysis/table_1.py                 # LaTeX to stdout
    python analysis/table_1.py --format text   # readable in a terminal
    python analysis/table_1.py --out analysis/out/main.tex
"""
from __future__ import annotations

import argparse
import statistics as st
from pathlib import Path

from data import (BASE_LABEL, BASES, MODEL_LABEL, Scores, TASK_LABEL,
                    LATEX_FOOTER, bootstrap_ci, fmt, latex_header, write)
from rescue.models import LONGBENCH_TASKS

SHORT = {"narrativeqa": "NQA", "qasper": "Qsp", "multifieldqa_en": "MFQA",
         "hotpotqa": "HQA", "2wikimqa": "2Wiki", "musique": "MSQ",
         "gov_report": "GovR", "qmsum": "QMS", "multi_news": "MNews",
         "trec": "TREC", "triviaqa": "TrQA", "samsum": "SAM",
         "passage_count": "PCnt", "passage_retrieval_en": "PRe",
         "lcc": "LCC", "repobench": "RB-P"}


WIN_MARGIN = 0.05

def text(sc: Scores, models: list[str]) -> str:
    lines = []
    for model in models:
        rows = sc.cells(model, "base", "rescue")
        if not rows:
            lines.append(f"{MODEL_LABEL[model]}: no cells in results/scores.csv")
            continue
        mean, lo, hi = bootstrap_ci(rows)
        # The paper counts a cell as a win or a loss only past a 0.05-point
        # margin; anything inside that is a tie, since LongBench scores are
        # reported to two decimals and smaller gaps are not meaningful.
        w = sum(1 for *_, u, v in rows if v - u > WIN_MARGIN)
        l = sum(1 for *_, u, v in rows if u - v > WIN_MARGIN)
        lines.append(f"\n{MODEL_LABEL[model]}   B=128, {len(rows)} cells")
        lines.append(f"  {'base':8s} {'base':>7s} {'+RESCUE':>9s} {'delta':>7s}")
        for b in BASES:
            pair = sc.paired(model, "base", "rescue", b)
            if not pair:
                continue
            bm = st.mean([u for _, u, _ in pair])
            rm = st.mean([v for _, _, v in pair])
            lines.append(f"  {BASE_LABEL[b]:8s} {bm:7.2f} {rm:9.2f} {rm - bm:+7.2f}")
        lines.append(f"  mean delta {mean:+.2f}  95% CI [{lo:+.2f}, {hi:+.2f}]"
                     f"   {w}W / {l}L / {len(rows) - w - l}T")
    return "\n".join(lines)


def latex(sc: Scores, models: list[str]) -> str:
    cols = "l" + "r" * len(LONGBENCH_TASKS) + "r"
    head = latex_header(
        "LongBench at a 128-token budget. Each base policy is paired with the same "
        "policy under RESCUE; bold marks the better of the two rows in each pair.",
        "tab:longbench-mlp", cols)
    out = [head, "\\providecommand{\\LBmissing}{\\textemdash}",
           "Policy & " + " & ".join(SHORT[t] for t in LONGBENCH_TASKS) + " & Avg. \\\\",
           "\\midrule"]
    for model in models:
        out.append(f"\\multicolumn{{{len(LONGBENCH_TASKS) + 2}}}{{l}}"
                   f"{{\\emph{{{MODEL_LABEL[model]}}}}} \\\\")
        for b in BASES:
            base = sc.sweep(model, "base", b)
            resc = sc.sweep(model, "rescue", b)
            for label, row in ((BASE_LABEL[b], base), ("\\quad + RESCUE", resc)):
                other = resc if row is base else base
                cells = []
                for t in LONGBENCH_TASKS:
                    v = row.get(t)
                    o = other.get(t)
                    s = fmt(v)
                    if v is not None and o is not None and v > o:
                        s = f"\\textbf{{{s}}}"
                    cells.append(s)
                avg = fmt(st.mean(list(row.values())) if row else None)
                out.append(f"{label} & " + " & ".join(cells) + f" & {avg} \\\\")
        out.append("\\midrule")
    out[-1] = LATEX_FOOTER
    return "\n".join(out)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--format", choices=["latex", "text"], default="latex")
    p.add_argument("--models", nargs="*", default=list(MODEL_LABEL))
    p.add_argument("--out", type=Path)
    args = p.parse_args()
    sc = Scores()
    write(args.out, latex(sc, args.models) if args.format == "latex"
          else text(sc, args.models))


if __name__ == "__main__":
    main()
