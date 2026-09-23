#!/usr/bin/env python
"""Table 6 (tab:future_aware).

Comparison with methods that predict future importance instead of correcting.

LookaheadKV and ForesightKV replace the base policy rather than correct it, so
their scores are absolute, not deltas: a single column each, against every base
policy under RESCUE. The point of the table is that the margin is not uniform --
LookaheadKV matches or exceeds RESCUE on several tasks -- which is why the paper
reports the per-base spread rather than one number.

    python analysis/table_5.py --format text
"""
from __future__ import annotations

import argparse
import statistics as st
from pathlib import Path

from data import (BASE_LABEL, BASES, LATEX_FOOTER, Scores, latex_header, write)


def collect(sc: Scores, model: str):
    look = sc.sweep(model, "lookaheadkv")
    fore = sc.sweep(model, "foresightkv")
    out = []
    for b in BASES:
        r = sc.sweep(model, "rescue", b)
        common_l = [t for t in r if t in look]
        common_f = [t for t in r if t in fore]
        out.append((
            b,
            st.mean(list(r.values())) if r else None,
            st.mean([r[t] - look[t] for t in common_l]) if common_l else None,
            st.mean([r[t] - fore[t] for t in common_f]) if common_f else None,
        ))
    return look, fore, out


def text(sc: Scores, model: str) -> str:
    look, fore, rows = collect(sc, model)
    lines = [f"{model}  --  RESCUE against future-aware methods, B=128",
             f"  LookaheadKV {st.mean(list(look.values())):.2f}   "
             f"ForesightKV {st.mean(list(fore.values())):.2f}" if look and fore else "  (comparators absent)",
             f"  {'base':8s} {'+RESCUE':>9s} {'vs Look':>9s} {'vs Fore':>9s}"]
    for b, r, dl, df in rows:
        if r is None:
            continue
        lines.append(f"  {BASE_LABEL[b]:8s} {r:9.2f} "
                     f"{dl:+9.2f} {df:+9.2f}" if dl is not None and df is not None
                     else f"  {BASE_LABEL[b]:8s} {r:9.2f}")
    got = [(dl, df) for _, r, dl, df in rows if dl is not None and df is not None]
    if got:
        lines.append(f"  range vs LookaheadKV {min(x for x, _ in got):+.2f} .. {max(x for x, _ in got):+.2f}")
        lines.append(f"  range vs ForesightKV {min(y for _, y in got):+.2f} .. {max(y for _, y in got):+.2f}")
    return "\n".join(lines)


def latex(sc: Scores, model: str) -> str:
    look, fore, rows = collect(sc, model)
    out = [latex_header(
        "RESCUE against methods that predict future importance instead of correcting a "
        "base policy, at a 128-token budget. The comparators replace the base policy, so "
        "their scores are absolute; each RESCUE row is that base policy corrected.",
        "tab:future_aware", "lccc"),
        r"\textbf{Base policy} & \textbf{+ RESCUE} & \textbf{$\Delta$ vs. LookaheadKV} "
        r"& \textbf{$\Delta$ vs. ForesightKV} \\", r"\midrule"]
    if look:
        out.append(f"LookaheadKV & {st.mean(list(look.values())):.2f} & -- & -- \\\\")
    if fore:
        out.append(f"ForesightKV & {st.mean(list(fore.values())):.2f} & -- & -- \\\\")
    out.append(r"\midrule")
    for b, r, dl, df in rows:
        if r is None:
            out.append(f"{BASE_LABEL[b]} & \\LBmissing & \\LBmissing & \\LBmissing \\\\")
            continue
        dls = f"${dl:+.2f}$" if dl is not None else r"\LBmissing"
        dfs = f"${df:+.2f}$" if df is not None else r"\LBmissing"
        out.append(f"{BASE_LABEL[b]} + RESCUE & {r:.2f} & {dls} & {dfs} \\\\")
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
