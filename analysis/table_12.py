#!/usr/bin/env python
"""Table 12 (tab:kslot).

Bounding the blast radius: restrict the correction to k contested slots.

The obvious way to make a correction safe is to bound how much of the cache it
may change -- reserve the base policy's top-(B-k) outright and let the corrected
score compete only for the remaining k. A bad correction then costs at most k
entries.

It does not pay, and the reason is worth recording: bounding the damage at k
entries also bounds the benefit at k entries, whereas the selector rejects the
correction entirely on the documents where it would hurt and leaves it
unbounded on the rest. The restriction duplicates the selector's job and does
it worse.

    python analysis/table_12.py --format text
"""
from __future__ import annotations

import argparse
from pathlib import Path

from data import BASE_LABEL, LATEX_FOOTER, Scores, TASK_LABEL, latex_header, write

KS = (16, 32, 64)
CELLS = [("snapkv", "qasper"), ("snapkv", "trec"), ("rkv", "qasper"), ("rkv", "trec")]


def rows(sc: Scores, model: str):
    for b, t in CELLS:
        base = sc.get(model, "base", b, 128, t)
        if base is None:
            continue
        gains = [(k, v - base if (v := sc.get(model, f"kslot{k}", b, 128, t)) is not None
                  else None) for k in KS]
        full = sc.get(model, "rescue", b, 128, t)
        yield b, t, base, gains, (full - base if full is not None else None)


def text(sc: Scores, model: str) -> str:
    lines = [f"{model}  --  correction restricted to k contested slots, B=128",
             f"  {'base':8s} {'task':10s} {'base':>7s}" +
             "".join(f"{f'k={k}':>9s}" for k in KS) + f" {'unrestricted':>13s}"]
    for b, t, base, gains, full in rows(sc, model):
        cells = "".join(f"{g:+9.2f}" if g is not None else f"{'--':>9s}" for _k, g in gains)
        f = f"{full:+13.2f}" if full is not None else f"{'--':>13s}"
        lines.append(f"  {BASE_LABEL[b]:8s} {TASK_LABEL[t]:10s} {base:7.2f}{cells}{f}")
    return "\n".join(lines)


def latex(sc: Scores, model: str) -> str:
    out = [latex_header(
        "Restricting the correction to $k$ contested slots, against the unrestricted "
        "correction, at a $128$-token budget. Entries are gains over the corresponding "
        "base policy; the selector is left on throughout.",
        "tab:kslot", "llc" + "c" * len(KS) + "c"),
        r"\textbf{Base} & \textbf{Task} & \textbf{Base score} & " +
        " & ".join(f"$k{{=}}{k}$" for k in KS) + r" & \textbf{Unrestricted} \\",
        r"\midrule"]
    for b, t, base, gains, full in rows(sc, model):
        vals = [g for _k, g in gains if g is not None] + ([full] if full is not None else [])
        best = max(vals) if vals else None
        cells = []
        for _k, g in gains:
            if g is None:
                cells.append(r"\LBmissing")
            else:
                cells.append(f"$\\mathbf{{{g:+.2f}}}$" if g == best else f"${g:+.2f}$")
        f = (r"\LBmissing" if full is None else
             f"$\\mathbf{{{full:+.2f}}}$" if full == best else f"${full:+.2f}$")
        out.append(f"{BASE_LABEL[b]} & {TASK_LABEL[t]} & {base:.2f} & "
                   + " & ".join(cells) + f" & {f} \\\\")
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
