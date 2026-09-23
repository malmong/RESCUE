#!/usr/bin/env python
"""Table 11 (tab:probe_length).

Probe length: how many tokens the selector decodes before it decides.

The probe is the selector's only cost that scales with a setting, so its length
is the one knob with a direct latency consequence. It is swept on LaProx and
SnapKV over all sixteen tasks, and the table reports the gain averaged over
both, the worst single cell, and a two-sided sign test of each length against a
single token over the paired cells.

The latency column is read from ``results/latency.csv`` when that file is
present; it is measurement, not arithmetic over scores, so it is absent from a
checkout that has not run ``scripts/run_latency.sh``.

    python analysis/table_10.py --format text
"""
from __future__ import annotations

import argparse
import csv
import statistics as st
from math import comb
from pathlib import Path

from data import (BASE_LABEL, LATEX_FOOTER, REPO_ROOT, Scores, latex_header,
                    write)

LENGTHS = [(1, "rescue"), (2, "probe2"), (4, "probe4"), (8, "probe8")]
SWEPT = ("laprox", "snapkv")
LATENCY = REPO_ROOT / "results" / "latency.csv"


# A cell whose gain moves by less than this is a tie, not a win. LongBench
# scores are reported to two decimals and several tasks move in fixed steps
# (TREC in units of 0.5, PassageCount in 0.05), so without a dead zone the
# sign test counts quantisation as evidence.
TIE = 0.05


def sign_test(pairs: list[tuple[float, float]]) -> tuple[float | None, int, int, int]:
    """Two-sided sign test: how often does the longer probe beat one token?

    Returns (p, wins, losses, ties).
    """
    wins = sum(1 for a, b in pairs if b > a + TIE)
    losses = sum(1 for a, b in pairs if b < a - TIE)
    ties = len(pairs) - wins - losses
    n = wins + losses
    if n == 0:
        return None, wins, losses, ties
    k = min(wins, losses)
    tail = sum(comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * tail), wins, losses, ties


def deltas(sc: Scores, model: str, arm: str) -> list[tuple[str, str, float]]:
    """Per-cell gain over the same base policy, across the swept bases."""
    out = []
    for b in SWEPT:
        for t, base, v in sc.paired(model, "base", arm, b):
            out.append((b, t, v - base))
    return out


def latency() -> dict[int, float]:
    if not LATENCY.exists():
        return {}
    out = {}
    with open(LATENCY, encoding="utf-8") as fh:
        for r in csv.DictReader(line for line in fh if not line.startswith("#")):
            if r.get("component") == "total" and r.get("probe_len"):
                out[int(r["probe_len"])] = float(r["ms"])
    return out


def rows(sc: Scores, model: str):
    ref = {(b, t): d for b, t, d in deltas(sc, model, "rescue")}
    lat = latency()
    for n, arm in LENGTHS:
        d = deltas(sc, model, arm)
        if not d:
            yield n, None, None, None, None, lat.get(n)
            continue
        vals = [x for *_, x in d]
        pairs = [(ref[(b, t)], x) for b, t, x in d if (b, t) in ref]
        vs = None if n == 1 else st.mean([y - x for x, y in pairs])
        wl = None if n == 1 else sign_test(pairs)
        p = None if wl is None else wl[0]
        yield n, st.mean(vals), min(vals), vs, p, lat.get(n), wl


def text(sc: Scores, model: str) -> str:
    lines = [f"{model}  --  selector probe length",
             f"  {'probe':>6s} {'avg gain':>9s} {'worst':>8s} {'vs p=1':>8s} "
             f"{'W/L/T':>10s} {'sign':>7s} {'ms':>8s}"]
    for n, avg, worst, vs, p, ms, wl in rows(sc, model):
        if avg is None:
            lines.append(f"  {n:6d}   (absent from results/scores.csv)")
            continue
        rec = "--" if wl is None else f"{wl[1]}/{wl[2]}/{wl[3]}"
        lines.append(f"  {n:6d} {avg:+9.2f} {worst:+8.2f} "
                     f"{'--' if vs is None else f'{vs:+.2f}':>8s} {rec:>10s} "
                     f"{'--' if p is None else f'{p:.3f}':>7s} "
                     f"{'--' if ms is None else f'{ms:.1f}':>8s}")
    lines.append("")
    lines.append("  per base policy:")
    lines.append(f"    {'base':8s}" + "".join(f"{f'p={n}':>9s}" for n, _ in LENGTHS))
    for b in SWEPT:
        row = [f"    {BASE_LABEL[b]:8s}"]
        for _n, arm in LENGTHS:
            pair = sc.paired(model, "base", arm, b)
            row.append(f"{st.mean([v - u for _, u, v in pair]):+9.2f}" if pair else f"{'--':>9s}")
        lines.append("".join(row))
    return "\n".join(lines)


def latex(sc: Scores, model: str) -> str:
    out = [latex_header(
        "Selector probe length, on LaProx and SnapKV over all sixteen tasks. The gain "
        "is against the same base policy at a 128-token budget; the sign test is "
        "two-sided over the paired cells. Longer probes help, and cost proportionally.",
        "tab:probe_length", "cccccc"),
        r"\textbf{Probe} & \textbf{Avg.\ gain} & \textbf{Worst drop} & "
        r"\textbf{vs.\ $p{=}1$} & \textbf{Sign test} & \textbf{Added logic} \\",
        r"\midrule"]
    for n, avg, worst, vs, p, ms, _wl in rows(sc, model):
        if avg is None:
            out.append(f"{n} & \\LBmissing & \\LBmissing & \\LBmissing & \\LBmissing & \\LBmissing \\\\")
            continue
        cell = (f"\\textbf{{{ms:.1f}\\,ms}}" if (ms is not None and n == 1)
                else f"{ms:.1f}\\,ms" if ms is not None else r"\LBmissing")
        out.append(f"{n} & ${avg:+.2f}$ & ${worst:+.2f}$ & "
                   f"{'--' if vs is None else f'${vs:+.2f}$'} & "
                   f"{'--' if p is None else f'${p:.3f}$'} & {cell} \\\\")
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
