#!/usr/bin/env python
"""Table 15 (tab:task_decomp_table).

Where the correction earns, and where the selector earns.

The two components do not help the same tasks. On tasks the correction already
handles, the selector's job is to stay out of the way and it costs a little; on
the tasks the correction damages, the selector is the whole result. Reporting
one average hides both halves, which is why the paper reports this split.

Both columns are averaged over the five base policies, so each row is a task
rather than a cell.

    python analysis/table_15.py --format text
"""
from __future__ import annotations

import argparse
import statistics as st
from pathlib import Path

from data import BASES, LATEX_FOOTER, Scores, TASK_LABEL, latex_header, write
from rescue.models import LONGBENCH_TASKS


def rows(sc: Scores, model: str):
    out = []
    for t in LONGBENCH_TASKS:
        base = [sc.get(model, "base", b, 128, t) for b in BASES]
        corr = [sc.get(model, "rescue_ungated", b, 128, t) for b in BASES]
        sel = [sc.get(model, "rescue", b, 128, t) for b in BASES]
        if any(x is None for x in base + corr + sel):
            continue
        c = st.mean([x - y for x, y in zip(corr, base)])
        s = st.mean([x - y for x, y in zip(sel, base)])
        out.append((t, c, s, s - c))
    out.sort(key=lambda r: -r[2])
    return out


def text(sc: Scores, model: str) -> str:
    rs = rows(sc, model)
    lines = [f"{model}  --  correction and selector, by task (mean over five base policies)",
             f"  {'task':21s} {'correction':>11s} {'selector':>9s} {'selector - corr':>16s}"]
    for t, c, s, d in rs:
        lines.append(f"  {TASK_LABEL[t]:21s} {c:+11.2f} {s:+9.2f} {d:+16.2f}")
    if rs:
        helped = [r for r in rs if r[3] > 0]
        lines.append(f"\n  selector improves {len(helped)}/{len(rs)} tasks; "
                     f"mean correction {st.mean([r[1] for r in rs]):+.2f}, "
                     f"mean selector {st.mean([r[2] for r in rs]):+.2f}")
    return "\n".join(lines)


def latex(sc: Scores, model: str) -> str:
    out = [latex_header(
        "The correction and the selector, decomposed by task. Both columns are gains "
        "over the base policy at a $128$-token budget, averaged over the five base "
        "policies. The two do not overlap: the selector costs a little where the "
        "correction already works and is the entire result where it does not.",
        "tab:task_decomp_table", "lccc"),
        r"\textbf{Task} & \textbf{Correction only} & \textbf{Selector} & "
        r"\textbf{Selector $-$ correction} \\", r"\midrule"]
    for t, c, s, d in rows(sc, model):
        out.append(f"{TASK_LABEL[t]} & ${c:+.2f}$ & ${s:+.2f}$ & ${d:+.2f}$ \\\\")
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
