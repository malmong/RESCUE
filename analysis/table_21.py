#!/usr/bin/env python
"""Table 21 (tab:doc_tail). The tail, counted in documents rather than cells.

A cell mean hides what a single document can lose. This counts the documents
the correction moves by at least tau points, with and without the selector, so
the safety claim is made at the level a user meets the method.

    python analysis/table_21.py --format text
"""
from __future__ import annotations

import argparse
import csv
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import LATEX_FOOTER, latex_header, write  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
PER_DOC = REPO_ROOT / "results" / "per_document.csv"
TAUS = (1, 5, 10, 20)


def rows() -> tuple[list[float], list[float], int]:
    if not PER_DOC.exists():
        raise SystemExit(f"{PER_DOC} not found; run scripts/export_per_document.py")
    corr, sel = [], []
    with open(PER_DOC, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            b = float(r["score_base"])
            corr.append(float(r["score_corr"]) - b)
            sel.append(float(r["score_selector"]) - b)
    return corr, sel, len(corr)


def counts(d: list[float], tau: float, n: int) -> tuple[float, float, float]:
    lose = sum(1 for x in d if x <= -tau)
    win = sum(1 for x in d if x >= tau)
    return 100 * lose / n, 100 * win / n, win / max(lose, 1)


def text() -> str:
    corr, sel, n = rows()
    out = [f"documents moved by at least tau points, n={n:,}",
           f"  {'tau':>4} {'corr lose':>10} {'corr win':>9} {'ratio':>6}"
           f" {'RESCUE lose':>12} {'RESCUE win':>11} {'ratio':>6}"]
    for tau in TAUS:
        cl, cw, cr = counts(corr, tau, n)
        sl, sw, sr = counts(sel, tau, n)
        out.append(f"  {tau:>4} {cl:9.1f}% {cw:8.1f}% {cr:6.2f}"
                   f" {sl:11.1f}% {sw:10.1f}% {sr:6.2f}")
    out.append(f"\n  mean delta  correction {st.mean(corr):+.3f}   RESCUE {st.mean(sel):+.3f}")
    return "\n".join(out)


def latex() -> str:
    corr, sel, n = rows()
    n_tex = f"{n:,}".replace(",", "{,}")
    out = [latex_header(
        f"Documents moved by at least $\\tau$ points, over all ${n_tex}$ evaluated "
        "documents, against each document's own base policy. ``Ratio'' is wins over "
        "losses: $1.0$ is a coin flip.",
        "tab:doc_tail", "lrrrrrr"),
        r"& \multicolumn{3}{c}{Correction only ($\lambda{=}1$)} & \multicolumn{3}{c}{RESCUE (gated)} \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
        r"$\tau$ & Losses & Wins & Ratio & Losses & Wins & Ratio \\", r"\midrule"]
    for tau in TAUS:
        cl, cw, cr = counts(corr, tau, n)
        sl, sw, sr = counts(sel, tau, n)
        out.append(f"${tau}$ & ${cl:.1f}\\%$ & ${cw:.1f}\\%$ & ${cr:.2f}$ "
                   f"& ${sl:.1f}\\%$ & ${sw:.1f}\\%$ & ${sr:.2f}$ \\\\")
    out.append(LATEX_FOOTER)
    return "\n".join(out)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--format", choices=["text", "latex"], default="latex")
    p.add_argument("--out", type=Path)
    a = p.parse_args()
    write(a.out, text() if a.format == "text" else latex())


if __name__ == "__main__":
    main()
