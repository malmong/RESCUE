#!/usr/bin/env python
"""Table 12 (tab:headroom).

How much of the reachable headroom does the correction actually take?

A raw gain says nothing about whether a task was hard. The ceiling here is the
same correction driven by *real* future attention, read off an un-evicted
reference generation: an oracle over the signal, at the same budget and with
the same combination rule, so the gap to it is what a perfect predictor of
future importance would be worth.

Recovery is `(RESCUE - base) / (oracle - base)`, undefined where the correction
loses ground.

    python analysis/table_12.py --format text
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
    # The reported recovery is the mean of the per-task ratios over tasks with
    # more than a point of headroom. PassageCount is excluded because a 0.14
    # point span makes the ratio uninformative rather than favorable, not
    # because it is unfavorable -- its ratio is 257%.
    recs = [100 * g / h for *_, h, g in rs if h > 1.0]
    lines.append(f"\n  mean headroom {st.mean(heads):.2f}, mean gain {st.mean(gains):+.2f}"
                 f"  -- recovery over the {len(recs)} tasks with headroom > 1:"
                 f" mean {st.mean(recs):.1f}%, median {st.median(recs):.1f}%")
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
    rs = rows(sc, model)
    for t, orc, b, head, gain in rs:
        # A ratio is reported only where it means something: the span has to be
        # worth more than a point and the correction has to have moved towards
        # the oracle. PassageCount's 0.14-point span is the reason for the first
        # test, LCC and RepoBench-P for the second.
        # A negative gain still has a well-defined share of the headroom; "(--)"
        # is reserved for the one task whose gain exceeds its headroom outright
        # (PassageCount, +0.37 against 0.14) and for headrooms under a point.
        if head > 1.0 and gain <= head:
            pct = 100 * gain / head
            rec = f"(${pct:+.1f}\\%$)" if gain < 0 else f"({pct:.1f}\\%)"
        else:
            rec = "(--)"
        out.append(f"{TASK_LABEL[t]} & {orc:.2f} & {b:.2f} & {head:.2f} & "
                   f"${gain:+.2f}$ \; {rec} \\\\")
    recs = [100 * g / h for *_, h, g in rs if h > 1.0]
    out.append(r"\midrule")
    out.append(f"Mean & -- & -- & {st.mean(h for *_, h, _ in rs):.2f} & "
               f"\\multicolumn{{1}}{{c}}{{${st.mean(recs):.1f}\\%$ recovered}} \\\\")
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
