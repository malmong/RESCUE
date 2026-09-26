#!/usr/bin/env python
"""Table 13 (tab:memscale). How the cost grows with the prompt, not the budget.

Two things separate here. The added logic is linear in the prompt for SnapKV and
quadratic for R-KV, because R-KV's own scoring rule builds an n x n redundancy
block per layer -- a cost of the base policy that the correction inherits, and
which the selector pays once per document rather than once per candidate. The
peak allocation is a prefill-time spike: the cache generation actually runs
against is the base policy's, and does not grow with it.

Read from results/measurements/memscale.csv, which
scripts/export_memscale.py regenerates.

    python analysis/table_13.py --format text
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from data import LATEX_FOOTER, REPO_ROOT, latex_header, write

SRC = REPO_ROOT / "results" / "measurements" / "memscale.csv"
LENGTHS = (16, 32, 64)
LABEL = {"snapkv": "SnapKV", "snapkv+rescue": "{+}RESCUE", "rkv+rescue": "R-KV {+}RESCUE"}


def load() -> dict[tuple[str, int], dict[str, float]]:
    if not SRC.exists():
        raise SystemExit(f"{SRC} not found; run scripts/export_memscale.py --runs ...")
    out = {}
    with open(SRC, encoding="utf-8") as fh:
        for r in csv.DictReader(line for line in fh if not line.startswith("#")):
            out[(r["arm"], int(r["length_k"]))] = {
                "added": float(r["added_ms"]), "peak": float(r["peak_gib"]),
                "kv": float(r["decode_kv_mb"]), "n": int(r["n"])}
    return out


def ms(v: float) -> str:
    return f"{v / 1000:.1f} s" if v >= 1000 else f"{v:.0f} ms"


def text() -> str:
    d = load()
    head = "".join(f"{str(L) + 'K':>12s}" for L in LENGTHS)
    w = ["memscale  --  RULER niah_single_2, B=128, median per document",
         "  " + " " * 22 + head]
    for arm in ("snapkv", "snapkv+rescue", "rkv+rescue"):
        lab = LABEL[arm].replace("{+}", "+")
        cells = []
        for L in LENGTHS:
            c = d.get((arm, L))
            cells.append(f"{ms(c['added']) if c else '--':>12s}")
        w.append(f"  added  {lab:15s}" + "".join(cells))
    for arm in ("snapkv", "snapkv+rescue"):
        lab = LABEL[arm].replace("{+}", "+")
        cells = []
        for L in LENGTHS:
            c = d.get((arm, L))
            cells.append(f"{(format(c['peak'], '.1f') + ' GiB' if c else '--'):>12s}")
        w.append(f"  peak   {lab:15s}" + "".join(cells))
    cells = []
    for L in LENGTHS:
        c = d.get(("snapkv+rescue", L))
        cells.append(f"{(format(c['kv'], '.1f') + ' MB' if c else '--'):>12s}")
    w.append("  " + f"{'decode KV, either arm':22s}" + "".join(cells))
    w.append("\n  documents per cell: " + str(sorted({c["n"] for c in d.values()})))
    return "\n".join(w)


def latex() -> str:
    d = load()
    out = [latex_header(
        "RULER \\texttt{niah\\_single\\_2} at $B=128$, median over $100$ documents. "
        "``Added'' is eviction- and selection-related logic; ``decode KV'' is the cache "
        "generation actually runs against.",
        "tab:memscale", "llccc"),
        " & & " + " & ".join(f"\\textbf{{{L}K}}" for L in LENGTHS) + r" \\", r"\midrule",
        r"\multirow{3}{*}{Added logic}"]
    for arm in ("snapkv", "snapkv+rescue", "rkv+rescue"):
        c = [d.get((arm, L)) for L in LENGTHS]
        vals = " & ".join(
            (f"${x['added'] / 1000:.1f}$\\,s" if x and x["added"] >= 1000
             else (f"${x['added']:.0f}$\\,ms" if x else "--")) for x in c)
        out.append(f" & {LABEL[arm]:17s} & {vals} \\\\")
    out.append(r"\addlinespace[2pt]")
    out.append(r"\multirow{2}{*}{Peak memory}")
    for arm in ("snapkv", "snapkv+rescue"):
        c = [d.get((arm, L)) for L in LENGTHS]
        out.append(f" & {LABEL[arm]:17s} & "
                   + " & ".join(f"${x['peak']:.1f}$\\,GiB" if x else "--" for x in c) + r" \\")
    out.append(r"\addlinespace[2pt]")
    c = [d.get(("snapkv+rescue", L)) for L in LENGTHS]
    out.append(r"\multicolumn{2}{l}{Decode KV, either arm} & "
               + " & ".join(f"${x['kv']:.1f}$\\,MB" if x else "--" for x in c) + r" \\")
    out.append(LATEX_FOOTER)
    return "\n".join(out)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--format", choices=["latex", "text"], default="latex")
    p.add_argument("--out", type=Path)
    args = p.parse_args()
    write(args.out, latex() if args.format == "latex" else text())


if __name__ == "__main__":
    main()
