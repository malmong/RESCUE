#!/usr/bin/env python
"""Table 8 (tab:headroom).

How much of the reachable headroom does the correction actually take?

A raw gain says nothing about whether a task was hard. The ceiling here is the
same correction driven by *real* future attention, read off an un-evicted
reference generation: an oracle over the signal, at the same budget and with
the same combination rule, so the gap to it is what a perfect predictor of
future importance would be worth.

Recovery is `(RESCUE - base) / (oracle - base)`, undefined where the correction
loses ground.

    python analysis/table_7.py --format text
"""
from __future__ import annotations

import argparse
import statistics as st
from pathlib import Path

from data import (BASES, LATEX_FOOTER, Scores, TASK_LABEL, latex_header, write)
from rescue.models import LONGBENCH_TASKS

ORACLE_BASE = "snapkv"     # the oracle run is scored on one base policy


def rows(sc: Scores, model: str):
    out = []
    for t in LONGBENCH_TASKS:
        orc = sc.get(model, "oracle_future", ORACLE_BASE, 128, t)
        if orc is None:
            continue
        base = [sc.get(model, "base", b, 128, t) for b in BASES]
        resc = [sc.get(model, "rescue", b, 128, t) for b in BASES]
        if any(x is None for x in base + resc):
            continue
        b, r = st.mean(base), st.mean(resc)
        out.append((t, orc, b, orc - b, r - b))
    out.sort(key=lambda x: -x[3])
    return out


def text(sc: Scores, model: str) -> str:
    rs = rows(sc, model)
    if not rs:
        return "oracle run absent from results/scores.csv"
    lines = [f"{model}  --  headroom against an oracle future signal, B=128",
             f"  {'task':21s} {'oracle':>7s} {'base':>7s} {'headroom':>9s} {'gain':>7s} {'recovery':>9s}"]
    for t, orc, b, head, gain in rs:
        rec = f"{100 * gain / head:8.1f}%" if head > 1e-9 and gain > 0 else "       --"
        lines.append(f"  {TASK_LABEL[t]:21s} {orc:7.2f} {b:7.2f} {head:9.2f} {gain:+7.2f} {rec}")
    heads = [h for *_, h, _ in rs]
    gains = [g for *_, g in rs]
    lines.append(f"\n  mean headroom {st.mean(heads):.2f}, mean gain {st.mean(gains):+.2f}"
                 f"  -- overall recovery {100 * st.mean(gains) / st.mean(heads):.1f}%")
    return "\n".join(lines)


def latex(sc: Scores, model: str) -> str:
    out = [latex_header(
        "Headroom against an oracle future signal at a 128-token budget. The oracle "
        "runs the same correction on real future attention taken from an un-evicted "
        "reference generation; base and RESCUE are averaged over the five base policies. "
        "Recovery is the share of the gap that the learned correction takes.",
        "tab:headroom", "lrrrr"),
        r"\textbf{Task} & \textbf{Oracle} & \textbf{Base (mean)} & \textbf{Headroom} "
        r"& \textbf{RESCUE $\Delta$ (recovery)} \\", r"\midrule"]
    for t, orc, b, head, gain in rows(sc, model):
        rec = f"({100 * gain / head:.1f}\\%)" if head > 1e-9 and gain > 0 else "(--)"
        out.append(f"{TASK_LABEL[t]} & {orc:.2f} & {b:.2f} & {head:.2f} & "
                   f"${gain:+.2f}$ \; {rec} \\\\")
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
