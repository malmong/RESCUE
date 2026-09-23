#!/usr/bin/env python
"""Table 5 (tab:budget_matched). Complementarity at a matched budget.

A union of two selections retains more of the oracle set than either alone, but
it is also larger, so the comparison says nothing on its own. Matching the sizes
is what makes the question answerable: given the same number of entries, how
much does each rule retain?

    python analysis/table_4.py --format text
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from data import LATEX_FOOTER, REPO_ROOT, latex_header, write

DATA = REPO_ROOT / "results" / "measurements" / "budget_matched.csv"


def rows():
    if not DATA.exists():
        raise SystemExit(f"{DATA} not found; run scripts/measurements/coverage.py")
    with open(DATA, encoding="utf-8") as fh:
        for r in csv.DictReader(line for line in fh if not line.startswith("#")):
            yield (r["selection"], r["size"], float(r["oracle_coverage_pct"]),
                   r["bold"] == "1")


def text() -> str:
    lines = ["Oracle-set coverage at a matched budget, Llama-3.1-8B-Instruct, B=128",
             f"  {'selection':42s} {'size':>8s} {'coverage':>9s}"]
    for sel, size, cov, bold in rows():
        plain = sel.replace("$", "").replace("\\alpha", "alpha").replace("{=}", "=")
        plain = plain.replace("^{*}", "*").replace("$|$", "|")
        lines.append(f"  {plain:42s} {size:>8s} {cov:8.1f}%" + ("  <-" if bold else ""))
    return "\n".join(lines)


def latex() -> str:
    out = [latex_header(
        "Oracle-set coverage at a matched budget. A union retains more than either "
        "signal alone but is also larger; matching the sizes is what makes the "
        "comparison answerable.",
        "tab:budget_matched", "lcc"),
        r"\textbf{Selection} & \textbf{$|\mathcal K|$} & \textbf{Oracle coverage} \\",
        r"\midrule"]
    for sel, size, cov, bold in rows():
        c = f"\\textbf{{{cov:.1f}\\%}}" if bold else f"{cov:.1f}\\%"
        out.append(f"{sel} & ${size}$ & {c} \\\\")
    out.append(LATEX_FOOTER)
    return "\n".join(out)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--format", choices=["latex", "text"], default="latex")
    p.add_argument("--out", type=Path)
    args = p.parse_args()
    write(args.out, latex() if args.format == "latex" else text())


if __name__ == "__main__":
    main()
