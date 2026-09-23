#!/usr/bin/env python
"""Table 13 (tab:selector_margin). When the selector's signal is reliable.

The selector picks the retrospectively better cache on 55.6% of the deciding
documents, and its KL margin correlates with the realized score change at only
rho=+0.11. Both are averages over a quantity that varies by four orders of
magnitude, so they describe the selector where it has no signal as much as where
it has one.

Splitting the deciding documents by the size of the selector's own margin
separates the two regimes: near chance where the probe carries no information,
meaningfully better where it does -- and the second regime is where the large
score differences are.

    python analysis/table_13.py --format text
"""
from __future__ import annotations

import argparse
import csv
import statistics as st
from pathlib import Path

from data import LATEX_FOOTER, REPO_ROOT, latex_header, write

PER_DOC = REPO_ROOT / "results" / "per_document.csv"
TOL = 1e-9
QUINTILES = 5


def load(model: str):
    """(|margin|, accepted, right, realized gain) per deciding document.

    A document counts only where the two caches score differently, the
    selector's score matches one of them, and a margin was logged.
    """
    if not PER_DOC.exists():
        raise SystemExit(f"{PER_DOC} not found; run scripts/export_per_document.py")
    out = []
    with open(PER_DOC, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["model"] != model or not r["kl_margin"]:
                continue
            b, c, s = (float(r["score_base"]), float(r["score_corr"]),
                       float(r["score_selector"]))
            if abs(c - b) <= TOL:
                continue
            near_c, near_b = abs(s - c) <= TOL, abs(s - b) <= TOL
            if not (near_c or near_b):
                continue
            # Rejecting serves the base cache, so only an accept moves the score.
            out.append((abs(float(r["kl_margin"])), near_c, (c > b) == near_c,
                        (c - b) if near_c else 0.0))
    out.sort()
    return out


def rows(model: str):
    d = load(model)
    n = len(d)
    for i in range(QUINTILES):
        g = d[i * n // QUINTILES:(i + 1) * n // QUINTILES]
        yield (i + 1, g[0][0], g[-1][0], len(g),
               100 * sum(1 for *_, ok, _ in g if ok) / len(g),
               100 * sum(1 for _, a, _, _ in g if a) / len(g),
               st.mean([x for *_, x in g]))
    yield (None, d[0][0], d[-1][0], n,
           100 * sum(1 for *_, ok, _ in d if ok) / n,
           100 * sum(1 for _, a, _, _ in d if a) / n,
           st.mean([x for *_, x in d]))


def text(model: str) -> str:
    lines = [f"{model}  --  selector behaviour by the size of its own margin",
             f"  {'quintile':10s} {'|margin| range':>24s} {'n':>5s} {'acc':>7s} "
             f"{'accepts':>8s} {'gain/doc':>9s}"]
    for q, lo, hi, n, acc, accept, gain in rows(model):
        lines.append(f"  {('all' if q is None else f'Q{q}'):10s} "
                     f"{f'{lo:.2e} - {hi:.2e}':>24s} {n:5d} {acc:6.1f}% "
                     f"{accept:7.1f}% {gain:+9.2f}")
    return "\n".join(lines)


def latex(model: str) -> str:
    rs = list(rows(model))
    out = [latex_header(
        "Selector behaviour by the size of its own margin, over the deciding documents "
        "that carry a logged margin, in equal fifths. Accuracy is agreement with "
        "hindsight; the gain is realized points per document, which is zero on a "
        "rejection by construction.",
        "tab:selector_margin", "lrrrr"),
        r"\textbf{Margin quintile} & \textbf{Range} & \textbf{Accuracy} & "
        r"\textbf{Accepts} & \textbf{Gain / doc} \\", r"\midrule"]
    names = {1: "Q1 (smallest)", 5: f"Q{QUINTILES} (largest)"}
    for q, lo, hi, _n, acc, accept, gain in rs:
        if q is None:
            out.append(r"\midrule")
            out.append(f"All & --- & ${acc:.1f}\\%$ & ${accept:.1f}\\%$ & ${gain:+.2f}$ \\\\")
            continue
        rng = (f"$<{hi:.1e}$" if q == 1 else
               f"$>{lo:.1e}$" if q == QUINTILES else f"${lo:.1e}$--${hi:.1e}$")
        bold = (lambda x: f"$\\mathbf{{{x}}}$") if q == QUINTILES else (lambda x: f"${x}$")
        out.append(f"{names.get(q, f'Q{q}')} & {rng} & {bold(f'{acc:.1f}')}\\% & "
                   f"${accept:.1f}\\%$ & {bold(f'{gain:+.2f}')} \\\\")
    out.append(LATEX_FOOTER)
    return "\n".join(out)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="llama3_8b")
    p.add_argument("--format", choices=["latex", "text"], default="latex")
    p.add_argument("--out", type=Path)
    args = p.parse_args()
    write(args.out, latex(args.model) if args.format == "latex" else text(args.model))


if __name__ == "__main__":
    main()
