#!/usr/bin/env python
"""Appendix (no numbered table) (sec:lambda_candidates).

Does a finer candidate grid help the selector?

The selector compares two candidates, lambda in {0, 1} -- the base cache and the
fully corrected one. A finer grid would let it interpolate. Against
{0, 0.5, 1, 2} on SnapKV and R-KV, the gain does not move consistently, and the
cost does: building and scoring candidate caches dominates the selector's
latency and scales with their number.

This is a cost argument. Six cells cannot rule out that intermediate lambdas
help somewhere; they show no evidence that they help here.

    python tables/make_appendix_lambda_candidates.py --format text
"""
from __future__ import annotations

import argparse
import statistics as st
from pathlib import Path

from common import BASE_LABEL, LATEX_FOOTER, Scores, TASK_LABEL, latex_header, write

BASES = ("snapkv", "rkv")
TASKS = ("qasper", "multifieldqa_en", "trec")


def rows(sc: Scores, model: str):
    for b in BASES:
        for t in TASKS:
            base = sc.get(model, "base", b, 128, t)
            two = sc.get(model, "rescue", b, 128, t)
            four = sc.get(model, "lambda4", b, 128, t)
            if None in (base, two, four):
                continue
            yield b, t, base, two - base, four - base


def text(sc: Scores, model: str) -> str:
    rs = list(rows(sc, model))
    lines = [f"{model}  --  two candidates against four, B=128",
             f"  {'base':8s} {'task':18s} {'base':>7s} "
             f"{'{0,1}':>9s} {'{0,.5,1,2}':>12s} {'diff':>8s}"]
    for b, t, base, two, four in rs:
        lines.append(f"  {BASE_LABEL[b]:8s} {TASK_LABEL[t]:18s} {base:7.2f} "
                     f"{two:+9.2f} {four:+12.2f} {four - two:+8.2f}")
    if rs:
        d = [f - t for *_, t, f in rs]
        lines.append(f"\n  mean difference {st.mean(d):+.2f} over {len(rs)} cells"
                     f"  ({sum(1 for x in d if x > 0)} better, {sum(1 for x in d if x < 0)} worse)")
    return "\n".join(lines)


def latex(sc: Scores, model: str) -> str:
    rs = list(rows(sc, model))
    out = [latex_header(
        "A finer candidate grid for the selector. Entries are gains over the "
        "corresponding base policy at a $128$-token budget, with the probe, features "
        "and checkpoint unchanged.",
        "tab:lambda_candidates", "llccc"),
        r"\textbf{Base} & \textbf{Task} & \textbf{Base score} & "
        r"$\lambda \in \{0,1\}$ & $\lambda \in \{0,0.5,1,2\}$ \\", r"\midrule"]
    for b, t, base, two, four in rs:
        bold = r"$\mathbf{%s}$"
        a = bold % f"{two:+.2f}" if two >= four else f"${two:+.2f}$"
        c = bold % f"{four:+.2f}" if four > two else f"${four:+.2f}$"
        out.append(f"{BASE_LABEL[b]} & {TASK_LABEL[t]} & {base:.2f} & {a} & {c} \\\\")
    if rs:
        d = [f - t for *_, t, f in rs]
        out.append(r"\midrule")
        out.append(r"\multicolumn{4}{l}{\textbf{Mean difference}} & "
                   f"${st.mean(d):+.2f}$ \\\\")
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
