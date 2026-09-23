#!/usr/bin/env python
"""Table 6 (tab:budget_sweep).

Budget sweep: the correction is worth more the more the base policy must drop.

The residual view predicts that a policy given more budget retains more of the
oracle set on its own, leaving less residual to rescue. The sweep tests that
downstream rather than at the level of set overlap: the same eighty cells at
B=128, 256 and 1024.

    python tables/make_table_6.py --format text
"""
from __future__ import annotations

import argparse
import statistics as st
from pathlib import Path

from common import (BASE_LABEL, BASES, LATEX_FOOTER, Scores, bootstrap_ci,
                    latex_header, write)

BUDGETS = (128, 256, 1024)


def text(sc: Scores, model: str) -> str:
    lines = [f"{model}  --  RESCUE gain over its own base, by budget",
             f"  {'base':8s}" + "".join(f"{f'B={b}':>10s}" for b in BUDGETS)]
    for b in BASES:
        row = [f"  {BASE_LABEL[b]:8s}"]
        for budget in BUDGETS:
            pair = sc.paired(model, "base", "rescue", b, budget)
            row.append(f"{st.mean([v - u for _, u, v in pair]):+10.2f}" if pair else f"{'--':>10s}")
        lines.append("".join(row))
    lines.append(f"  {'mean':8s}" + "".join(
        f"{bootstrap_ci(sc.cells(model, 'base', 'rescue', budget))[0]:+10.2f}"
        if sc.cells(model, 'base', 'rescue', budget) else f"{'--':>10s}"
        for budget in BUDGETS))
    for budget in BUDGETS:
        cells = sc.cells(model, "base", "rescue", budget)
        if not cells:
            continue
        m, lo, hi = bootstrap_ci(cells)
        base_avg = st.mean([u for *_, u, _ in cells])
        lines.append(f"  B={budget:<5d} base {base_avg:6.2f}   gain {m:+.2f} [{lo:+.2f}, {hi:+.2f}]"
                     f"   over {len(cells)} cells")
    return "\n".join(lines)


def latex(sc: Scores, model: str) -> str:
    out = [latex_header(
        "RESCUE's gain over each base policy at three budgets. Every entry is a "
        "within-cell delta against that base policy at the same budget.",
        "tab:budget_sweep_summary", "l" + "c" * len(BUDGETS)),
        r"\textbf{Base policy} & " + " & ".join(f"$B={b}$" for b in BUDGETS) + r" \\",
        r"\midrule"]
    for b in BASES:
        cells = []
        for budget in BUDGETS:
            pair = sc.paired(model, "base", "rescue", b, budget)
            cells.append(f"${st.mean([v - u for _, u, v in pair]):+.2f}$" if pair
                         else r"\LBmissing")
        out.append(f"{BASE_LABEL[b]} & " + " & ".join(cells) + r" \\")
    out.append(r"\midrule")
    means = []
    for budget in BUDGETS:
        c = sc.cells(model, "base", "rescue", budget)
        means.append(f"$\\mathbf{{{bootstrap_ci(c)[0]:+.2f}}}$" if c else r"\LBmissing")
    out.append(r"\textbf{Mean} & " + " & ".join(means) + r" \\")
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
