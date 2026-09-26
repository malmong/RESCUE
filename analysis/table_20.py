#!/usr/bin/env python
"""Table 20 (tab:selector_quadrants). What the selector's decisions are worth.

The one-token probe picks the retrospectively better cache on only a little over
half the documents where the two arms differ, yet it recovers a large share of
the oracle headroom. Accuracy alone cannot explain that: a decision's value is
the size of the score difference it captures or forfeits, not whether it was
right.

Two asymmetries carry the result. Rejecting returns the base cache, so a wrong
rejection forfeits an available gain but never causes a loss. And among the
accepts, the right ones are on larger differences than the wrong ones.

Documents whose selector score matches neither arm are excluded: the selector
serves one of the two caches, so a third score means the run does not line up
with either arm and the decision cannot be read.

    python analysis/table_20.py --format text
"""
from __future__ import annotations

import argparse
import csv
import statistics as st
from pathlib import Path

from data import LATEX_FOOTER, REPO_ROOT, latex_header, write

PER_DOC = REPO_ROOT / "results" / "per_document.csv"
TOL = 1e-9
LABELS = [
    ((True, True), "Accept, correction better", "right"),
    ((True, False), "Accept, base better", "wrong"),
    ((False, True), "Reject, base better", "right"),
    ((False, False), "Reject, correction better", "wrong"),
]


def collect(model: str):
    if not PER_DOC.exists():
        raise SystemExit(f"{PER_DOC} not found; run scripts/export_per_document.py")
    quad: dict[tuple[bool, bool], list[float]] = {k: [] for k, _l, _r in LABELS}
    gains: dict[tuple[bool, bool], list[float]] = {k: [] for k, _l, _r in LABELS}
    unmatched = same = 0
    with open(PER_DOC, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["model"] != model:
                continue
            b, c, s = (float(r["score_base"]), float(r["score_corr"]),
                       float(r["score_selector"]))
            if abs(c - b) <= TOL:
                same += 1
                continue
            near_c, near_b = abs(s - c) <= TOL, abs(s - b) <= TOL
            if not (near_c or near_b):
                unmatched += 1
                continue
            accepted = near_c
            right = (c > b) == accepted
            quad[(accepted, right)].append(abs(c - b))
            # Rejecting serves the base cache, so it moves nothing; only an
            # accept contributes, and its sign is whether it was right.
            gains[(accepted, right)].append((c - b) if accepted else 0.0)
    return quad, gains, unmatched, same


def rows(model: str):
    quad, gains, unmatched, same = collect(model)
    total = sum(len(v) for v in quad.values())
    for key, label, verdict in LABELS:
        n = len(quad[key])
        yield (label, verdict, n,
               100 * n / total if total else 0.0,
               st.mean(quad[key]) if n else 0.0,
               sum(gains[key]) / total if total else 0.0)
    right = sum(len(quad[k]) for k, _l, v in LABELS if v == "right")
    net = sum(sum(g) for g in gains.values()) / total if total else 0.0
    # "%" is a comment character in LaTeX; the text renderer strips the escape.
    yield ("Net", f"{100 * right / total:.1f}\\% right" if total else "--",
           total, 100.0, float("nan"), net)
    yield ("__meta__", "", unmatched, float(same), float("nan"), float("nan"))


def text(model: str) -> str:
    lines = [f"{model}  --  selector decisions where the two caches differ",
             f"  {'decision':34s} {'n':>6s} {'share':>7s} {'mean |d|':>9s} {'contribution':>13s}"]
    for label, verdict, n, share, mag, contrib in rows(model):
        if label == "__meta__":
            lines.append(f"\n  excluded: {n} documents whose selector score matched "
                         f"neither arm, and {int(share)} where the two caches scored "
                         f"the same")
            continue
        if label == "Net":
            lines.append(f"  {'-' * 70}")
            lines.append(f"  {'Net':34s} {n:6d} {verdict.replace(chr(92) + '%', '%'):>12s} "
                         f"{'--':>9s} {contrib:+13.2f}")
            continue
        lines.append(f"  {label + ' (' + verdict + ')':34s} {n:6d} {share:6.1f}% "
                     f"{mag:9.2f} {contrib:+13.2f}")
    return "\n".join(lines)


def latex(model: str) -> str:
    out = [latex_header(
        "Selector decisions on the documents where the base and corrected caches score "
        "differently. Rejecting returns the base cache, so rejections contribute nothing "
        "to the gain over the base policy --- a wrong rejection forfeits the difference "
        "rather than losing it. Mean $|\\Delta|$ is the average absolute score difference "
        "between the two arms in that outcome.",
        "tab:selector_quadrants", "lrrrr"),
        r"\textbf{Decision} & \textbf{$n$} & \textbf{Share} & \textbf{Mean $|\Delta|$} "
        r"& \textbf{Contribution} \\", r"\midrule"]
    for label, verdict, n, share, mag, contrib in rows(model):
        if label == "__meta__":
            continue
        if label == "Net":
            out.append(r"\midrule")
            out.append(f"Net & {n} & {verdict} & --- & ${contrib:+.2f}$ \\\\")
            continue
        c = f"${contrib:+.2f}$" if abs(contrib) > 1e-9 else r"\phantom{$+$}0.00"
        out.append(f"{label} \\emph{{({verdict})}} & {n} & {share:.1f}\\% & "
                   f"{mag:.2f} & {c} \\\\")
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
