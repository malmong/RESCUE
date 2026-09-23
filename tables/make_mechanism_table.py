#!/usr/bin/env python
"""Does the correction recover what the base policy actually missed?

Two set-level quantities per base policy, against the downstream gain:

    delta coverage   how much more of the oracle future-attention top-B set the
                     corrected cache retains than the base cache does
    rescue recall    the share of that base's own miss set the scorer brings back

They order the base policies the same way the downstream gain does at the top
and diverge at the bottom, which is the point: coverage is necessary and not
sufficient, because an entry recovered at the cost of one the base had right is
a wash.

Set-level numbers come from results/measurements/mechanism.csv; the downstream
column is computed from results/scores.csv.

    python tables/make_mechanism_table.py --format text
"""
from __future__ import annotations

import argparse
import csv
import statistics as st
from pathlib import Path

from common import (BASE_LABEL, LATEX_FOOTER, REPO_ROOT, Scores, latex_header,
                    write)

DATA = REPO_ROOT / "results" / "measurements" / "mechanism.csv"


def rows(sc: Scores, model: str):
    if not DATA.exists():
        raise SystemExit(f"{DATA} not found; run tools/measurements/coverage.py")
    with open(DATA, encoding="utf-8") as fh:
        meas = list(csv.DictReader(
            line for line in fh if not line.startswith("#")))
    for r in meas:
        b = r["base"]
        pair = sc.paired(model, "base", "rescue", b)
        down = st.mean([v - u for _t, u, v in pair]) if pair else None
        yield b, float(r["delta_coverage_pt"]), float(r["rescue_recall_pct"]), down


def text(sc: Scores, model: str) -> str:
    lines = [f"{model}  --  set-level coverage against the downstream gain, B=128",
             f"  {'base':8s} {'d coverage':>11s} {'rescue recall':>14s} {'downstream':>11s}"]
    for b, cov, rec, down in rows(sc, model):
        d = f"{down:+11.2f}" if down is not None else f"{'--':>11s}"
        lines.append(f"  {BASE_LABEL[b]:8s} {cov:+10.1f}pt {rec:13.1f}% {d}")
    return "\n".join(lines)


def latex(sc: Scores, model: str) -> str:
    out = [latex_header(
        "Set-level effect of the correction per base policy, against its downstream "
        "gain, at a $128$-token budget. $\\Delta$Coverage is the change in the share of "
        "the oracle future-attention top-$B$ set the cache retains; rescue recall is the "
        "share of that policy's own miss set the scorer brings back.",
        "tab:mechanism", "lccc"),
        r"\textbf{Base} & \textbf{$\Delta$Coverage} & \textbf{Rescue recall} & "
        r"\textbf{Downstream $\Delta$} \\", r"\midrule"]
    for b, cov, rec, down in rows(sc, model):
        d = f"${down:+.2f}$" if down is not None else r"\LBmissing"
        pad = r"\phantom{0}" if rec < 10 else ""
        out.append(f"{BASE_LABEL[b]} & ${cov:+.1f}$ pt & {pad}${rec:.1f}\\%$ & {d} \\\\")
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
