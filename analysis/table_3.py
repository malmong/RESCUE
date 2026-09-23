#!/usr/bin/env python
"""Table 3 (tab:pareto). Accuracy against cost, next to simply raising the budget.

The comparison a deployment faces is not RESCUE against another eviction rule at
the same budget, but RESCUE against spending the same resources on a larger
cache. Putting both on the same axes is not favourable and the table says so:
a deployment that can afford twice the cache should spend it there.

What it also shows is where that stops. RESCUE's overhead is flat in the budget
-- the cost is building and replaying the candidate caches, which scales with
the prompt rather than with how much of it is kept -- while the gain is not. The
same fixed price buys less and less, which is what fixes the operating point.

Latency comes from results/latency.csv; scores from results/scores.csv.

    python analysis/table_3.py --format text
"""
from __future__ import annotations

import argparse
import csv
import statistics as st
from pathlib import Path

from data import LATEX_FOOTER, REPO_ROOT, Scores, latex_header, write

LATENCY = REPO_ROOT / "results" / "latency.csv"
BASE = "snapkv"
BUDGETS = (128, 256, 1024)


def latency() -> dict[tuple[str, int], float]:
    """Median added logic per document, by (arm, budget)."""
    if not LATENCY.exists():
        raise SystemExit(f"{LATENCY} not found; run scripts/run_latency.sh")
    out: dict[tuple[str, int], float] = {}
    with open(LATENCY, encoding="utf-8") as fh:
        for r in csv.DictReader(l for l in fh if not l.startswith("#")):
            arm = "base" if r["method"] == "snapkv" else (
                "rescue" if r["method"] == "rescue" else None)
            if arm is None:
                continue
            # rescue has a `total` row per probe length; this table is the
            # shipped one-token selector.
            if arm == "rescue" and r["probe_len"] not in ("", "1"):
                continue
            comp = r["component"]
            if comp == "total":
                out[(arm, 128)] = float(r["ms"])
            elif comp.startswith("total_b"):
                out[(arm, int(comp.removeprefix("total_b")))] = float(r["ms"])
    return out


def rows(sc: Scores, model: str):
    lat = latency()
    for arm, label in (("base", "SnapKV"), ("rescue", "RESCUE")):
        for b in BUDGETS:
            vals = sc.sweep(model, arm, BASE, b)
            if not vals:
                continue
            yield label, b, st.mean(list(vals.values())), lat.get((arm, b))


def text(sc: Scores, model: str) -> str:
    rs = list(rows(sc, model))
    lines = [f"{model}  --  accuracy against cost, SnapKV base",
             f"  {'method':8s} {'B':>5s} {'score':>7s} {'added logic':>12s} {'retained':>9s}"]
    for label, b, s_, ms in rs:
        lines.append(f"  {label:8s} {b:5d} {s_:7.2f} "
                     f"{'--' if ms is None else f'{ms:.1f} ms':>12s} {b:9d}")
    ref = [r for r in rs if r[0] == "RESCUE" and r[1] == 128]
    if ref:
        _, _, rs_, rt = ref[0]
        lines.append("\n  against RESCUE at B=128:")
        for label, b, s_, ms in rs:
            if (label, b) == ("RESCUE", 128) or ms is None:
                continue
            lines.append(f"    {label}@{b:<5d} score {s_ - rs_:+6.2f}   "
                         f"latency {ms - rt:+8.1f} ms   cache {b / 128:.0f}x")
    return "\n".join(lines)


def latex(sc: Scores, model: str) -> str:
    rs = list(rows(sc, model))
    best = max(s_ for _l, _b, s_, _m in rs)
    out = [latex_header(
        "Accuracy against cost, with SnapKV as the base. Scores are the 16-task LongBench "
        "mean; added logic is the median eviction- and selection-related latency per "
        "document on Qasper. Retained entries per layer is what the cache costs for the "
        "whole of decoding.",
        "tab:pareto", "lrrr"),
        r"\textbf{Method} & \textbf{Score} & \textbf{Added logic} & "
        r"\textbf{Retained / layer} \\", r"\midrule"]
    for i, (label, b, s_, ms) in enumerate(rs):
        if i and rs[i - 1][0] != label:
            out.append(r"\midrule")
        cell = f"\\textbf{{{s_:.2f}}}" if s_ == best else f"{s_:.2f}"
        lat = "--" if ms is None else f"{ms:.1f}" + r"\,ms"
        out.append(f"{label}, $B{{=}}{b}$ & {cell} & {lat} & {b} \\\\")
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
