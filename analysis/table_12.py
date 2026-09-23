#!/usr/bin/env python
"""Table 12 (tab:cache_select).

Control: give the selector two existing caches instead of two lambdas.

Same one-token probe, same KL criterion; the candidates are SnapKV's cache and
ForesightKV's rather than lambda 0 and 1. If choosing between two off-the-shelf
policies scored as well as correcting one with a trained residual scorer, the
scorer would not be what the selector needs.

ForesightKV rather than LookaheadKV: LookaheadKV runs its own decoder path and
never reaches the eviction machinery that builds candidate caches, so it cannot
be a candidate here. It is also the more expensive of the two comparators,
which makes this the harder control.

    python analysis/table_11.py --format text
"""
from __future__ import annotations

import argparse
import statistics as st
from pathlib import Path

from data import (LATEX_FOOTER, Scores, TASK_LABEL, latex_header, write)
from rescue.models import LONGBENCH_TASKS

BASE = "snapkv"


def rows(sc: Scores, model: str):
    for t in LONGBENCH_TASKS:
        b = sc.get(model, "base", BASE, 128, t)
        f = sc.get(model, "foresightkv", "", 128, t)
        c = sc.get(model, "cache_select", BASE, 128, t)
        r = sc.get(model, "rescue", BASE, 128, t)
        if None in (b, f, c, r):
            continue
        yield t, b, f, c, r


def summarise(rs):
    dc = [c - b for _t, b, _f, c, _r in rs]
    dr = [r - b for _t, b, _f, _c, r in rs]
    df = [f - b for _t, b, f, _c, _r in rs]
    # Choosing the better of the two policies per TASK, knowing the answer
    # afterwards -- the ceiling for any assignment of policies to tasks.
    do = [max(b, f) - b for _t, b, f, _c, _r in rs]
    return dc, dr, df, do


def text(sc: Scores, model: str) -> str:
    rs = list(rows(sc, model))
    if not rs:
        return "cache-selection control absent from results/scores.csv"
    lines = [f"{model}  --  selecting between two policies' caches, B=128",
             f"  {'task':21s} {'SnapKV':>7s} {'Fore':>7s} {'CacheSel':>9s} {'RESCUE':>7s}"
             f" {'dCS':>7s} {'dRES':>7s}"]
    for t, b, f, c, r in rs:
        lines.append(f"  {TASK_LABEL[t]:21s} {b:7.2f} {f:7.2f} {c:9.2f} {r:7.2f}"
                     f" {c - b:+7.2f} {r - b:+7.2f}")
    dc, dr, df, do = summarise(rs)
    lines.append("")
    lines.append(f"  mean delta vs SnapKV   CacheSel {st.mean(dc):+.2f}   RESCUE {st.mean(dr):+.2f}"
                 f"   ForesightKV alone {st.mean(df):+.2f}")
    lines.append(f"  win/loss               CacheSel {sum(1 for x in dc if x > 0)}/{sum(1 for x in dc if x < 0)}"
                 f"        RESCUE {sum(1 for x in dr if x > 0)}/{sum(1 for x in dr if x < 0)}")
    lines.append(f"  per-task hindsight oracle over the two policies {st.mean(do):+.2f}"
                 f"  -- CacheSel chooses per document and reaches {st.mean(dc):+.2f}")
    return "\n".join(lines)


def latex(sc: Scores, model: str) -> str:
    rs = list(rows(sc, model))
    out = [latex_header(
        "Selecting between two policies' caches, against correcting one of them. All "
        "columns are LongBench scores at a 128-token budget. ``CacheSel'' applies the "
        "RESCUE selector to the SnapKV and ForesightKV caches; ``RESCUE'' corrects the "
        "SnapKV cache with the residual scorer under the same selector.",
        "tab:cache_select", "lrrrrr"),
        r"\textbf{Task} & \textbf{SnapKV} & \textbf{ForesightKV} & \textbf{CacheSel} "
        r"& \textbf{RESCUE} & \textbf{$\Delta$ CacheSel} \\", r"\midrule"]
    for t, b, f, c, r in rs:
        best = max(c, r)
        cs = f"\\textbf{{{c:.2f}}}" if c == best else f"{c:.2f}"
        rr = f"\\textbf{{{r:.2f}}}" if r == best else f"{r:.2f}"
        out.append(f"{TASK_LABEL[t]} & {b:.2f} & {f:.2f} & {cs} & {rr} & ${c - b:+.2f}$ \\\\")
    if rs:
        dc, dr, df, _ = summarise(rs)
        out.append(r"\midrule")
        out.append(r"\textbf{Average $\Delta$ vs.\ SnapKV} & -- & "
                   f"${st.mean(df):+.2f}$ & ${st.mean(dc):+.2f}$ & "
                   f"$\\mathbf{{{st.mean(dr):+.2f}}}$ & \\\\")
        out.append(r"\textbf{Win / loss vs.\ SnapKV} & -- & "
                   f"{sum(1 for x in df if x > 0)} / {sum(1 for x in df if x < 0)} & "
                   f"{sum(1 for x in dc if x > 0)} / {sum(1 for x in dc if x < 0)} & "
                   f"\\textbf{{{sum(1 for x in dr if x > 0)} / {sum(1 for x in dr if x < 0)}}} & \\\\")
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
