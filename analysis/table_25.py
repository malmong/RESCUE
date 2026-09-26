#!/usr/bin/env python
"""Table 25 (tab:seed_variance). How much of a reported margin is the seed?

The scorer is small and trained on a few hundred documents, so retraining it
under a different seed moves the downstream score. The margin between two
learning targets has to be read against that spread, which is why this table
sits beside the target comparison rather than in an appendix of its own.

Three seeds, two base policies, the three tasks the paper reports.

    python analysis/table_25.py --format text
"""
from __future__ import annotations

import argparse
import statistics as st
from pathlib import Path

from data import BASE_LABEL, LATEX_FOOTER, Scores, TASK_LABEL, latex_header, write

# This table is eight columns wide and its label column is a \multirow stub, so
# it uses the short policy names. Table 23 sits directly above it and carries
# the "(evict-once)" qualifier for the same rows.
SHORT = dict(BASE_LABEL, h2o="H2O")

BASES = ("h2o", "rkv")
TASKS = ("qasper", "trec", "lcc")
SEEDS = [(0, "rescue"), (1, "seed1"), (2, "seed2")]


def rows(sc: Scores, model: str):
    for b in BASES:
        gains = {}
        for seed, arm in SEEDS:
            per_task = []
            for t in TASKS:
                base = sc.get(model, "base", b, 128, t)
                v = sc.get(model, arm, b, 128, t)
                per_task.append(None if (base is None or v is None) else v - base)
            gains[seed] = per_task
        avgs = [st.mean([g for g in gains[s] if g is not None])
                for s, _a in SEEDS if any(g is not None for g in gains[s])]
        # Population s.d.: the three seeds are the whole set being described,
        # not a sample drawn from a larger pool of seeds.
        sd = st.pstdev(avgs) if len(avgs) > 1 else None
        rng = (max(avgs) - min(avgs)) if len(avgs) > 1 else None
        yield b, gains, sd, rng


def text(sc: Scores, model: str) -> str:
    lines = [f"{model}  --  gain by training seed, over the same base policy",
             f"  {'base':8s} {'task':10s}" + "".join(f"{f'seed {s}':>9s}" for s, _ in SEEDS)
             + f" {'s.d.':>7s} {'range':>7s}"]
    for b, gains, sd, rng in rows(sc, model):
        for i, t in enumerate(TASKS):
            cells = "".join(
                f"{gains[s][i]:+9.2f}" if gains[s][i] is not None else f"{'--':>9s}"
                for s, _a in SEEDS)
            tail = (f" {sd:7.2f} {rng:7.2f}" if i == 0 and sd is not None
                    else " " * 16)
            lines.append(f"  {SHORT[b] if i == 0 else '':8s} {TASK_LABEL[t]:10s}{cells}{tail}")
    return "\n".join(lines)


def latex(sc: Scores, model: str) -> str:
    out = [latex_header(
        "Gain over the same base policy with the scorer retrained under three seeds, "
        "everything else fixed. The s.d.\\ and range are over the three-task averages, "
        "and they bound how finely a difference between two scorers can be read.",
        "tab:seed_variance", "llcccccc"),
        r"& & \multicolumn{3}{c}{\textbf{Gain by seed}} & & "
        r"\multicolumn{2}{c}{\textbf{3-task average}} \\",
        r"\textbf{Base} & \textbf{Task} & 0 & 1 & 2 & & \textbf{s.d.} & \textbf{range} \\",
        r"\midrule"]
    for b, gains, sd, rng in rows(sc, model):
        for i, t in enumerate(TASKS):
            cells = " & ".join(
                f"${gains[s][i]:+.2f}$" if gains[s][i] is not None else r"\LBmissing"
                for s, _a in SEEDS)
            if i == 0:
                tail = (f" & & \\multirow{{{len(TASKS)}}}{{*}}{{${sd:.2f}$}} & "
                        f"\\multirow{{{len(TASKS)}}}{{*}}{{${rng:.2f}$}}"
                        if sd is not None else " & & &")
                out.append(f"{SHORT[b]} & {TASK_LABEL[t]} & {cells}{tail} \\\\")
            else:
                out.append(f"     & {TASK_LABEL[t]} & {cells} & & & \\\\")
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
