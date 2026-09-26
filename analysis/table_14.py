#!/usr/bin/env python
"""Table 14 (tab:runtime_overhead). What the selector costs, and where.

RESCUE is about twice the most expensive predicted-future method it is compared
against, not less than it: this is accuracy bought with latency, not latency
saved. Almost none of the cost is the scorer -- it is building and replaying one
pruned cache per candidate lambda, which is why the number of candidates, not
the size of the model, is what the cost scales with.

Read from results/latency.csv, which scripts/run_latency.sh regenerates.

    python analysis/table_14.py --format text
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from data import LATEX_FOOTER, REPO_ROOT, latex_header, write

LATENCY = REPO_ROOT / "results" / "latency.csv"
ROWS = [
    ("snapkv", "", "total", "SnapKV", 0),
    ("lookaheadkv", "", "total", "LookaheadKV$^{\\dagger}$", 0),
    ("foresightkv", "", "total", "ForesightKV", 0),
    ("rescue_corr", "", "total", "RESCUE correction only", 0),
    ("rescue", "1", "total", "RESCUE + 1-token selector", 0),
    ("rescue", "1", "probe", "dense probe", 1),
    ("rescue", "1", "candidates", "two candidate caches", 1),
]


def load() -> dict[tuple[str, str, str], float]:
    if not LATENCY.exists():
        raise SystemExit(f"{LATENCY} not found; run scripts/run_latency.sh")
    out = {}
    with open(LATENCY, encoding="utf-8") as fh:
        for r in csv.DictReader(line for line in fh if not line.startswith("#")):
            out[(r["method"], r["probe_len"], r["component"])] = float(r["ms"])
    return out


def text() -> str:
    d = load()
    lines = ["Added logic per document, median, Llama-3.1-8B-Instruct on Qasper",
             f"  {'method':34s} {'ms':>8s}"]
    for method, probe, comp, label, indent in ROWS:
        v = d.get((method, probe, comp))
        if v is None:
            continue
        lines.append(f"  {'  ' * indent + label:34s} {v:8.1f}")
    return "\n".join(lines)


def latex() -> str:
    d = load()
    out = [latex_header(
        "Eviction- and selection-related latency per document, median over $200$ Qasper "
        "documents at $\\sim$5K tokens on Llama-3.1-8B-Instruct. Prefill and generation "
        "are excluded; see the latency protocol for what the interval covers. "
        "$^{\\dagger}$LookaheadKV runs its own decoder path, so its figure is measured "
        "differently and is not directly comparable.",
        "tab:runtime_overhead", "lr"),
        r"\textbf{Method} & \textbf{Latency} \\", r"\midrule"]
    for method, probe, comp, label, indent in ROWS:
        v = d.get((method, probe, comp))
        if v is None:
            continue
        if indent:
            out.append(f"\\quad \\emph{{{label}}} & \\emph{{{v:.1f} ms}} \\\\")
        else:
            out.append(f"{label} & {v:.1f} ms \\\\")
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
