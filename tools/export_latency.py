#!/usr/bin/env python
"""Collect the runtime breakdown into results/latency.csv.

The generator emits one `[TIMING]` line per document when `RESCUE_TIMING=1` is
set, splitting prefill, selection and eviction, and one `[SELTIME]` line inside
the selector splitting the dense probe from the candidate caches. This turns
those logs into:

    results/latency.csv
        method,probe_len,component,ms,n

Medians, not means: the distribution has a long right tail from the longest
documents, and a mean sits well above the typical document. Components:

    added_logic   eviction- and selection-related work only, excluding prefill
                  and generation -- the wrapper-level interval around scoring
                  and pruning, which is where each policy's machinery runs
    probe         decoding the probe token against the unpruned cache
    candidates    building and replaying one pruned cache per candidate lambda
    total         added_logic, repeated for the table that reads it by name

    python tools/export_latency.py --logs runs/latency
"""
from __future__ import annotations

import argparse
import csv
import re
import statistics as st
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

TIMING = re.compile(
    r"\[TIMING\] len=(\d+) prefill=([\d.]+) select=([\d.]+) evict=([\d.]+)")
SELTIME = re.compile(
    r"\[SELTIME\] probe=([\d.]+) trials=([\d.]+) steps=(\d+) cands=(\d+)")
# runs/latency/<tag>.log, where the tag is what run_latency.sh passed.
TAG = re.compile(r"^(?P<method>[a-z]+)(?:_p(?P<probe>\d+))?$")


def parse(path: Path):
    text = path.read_text(errors="ignore")
    logic = [float(m.group(3)) + float(m.group(4)) for m in TIMING.finditer(text)]
    probe = [float(m.group(1)) for m in SELTIME.finditer(text)]
    trials = [float(m.group(2)) for m in SELTIME.finditer(text)]
    return logic, probe, trials


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--logs", required=True, type=Path,
                   help="directory of per-arm logs written by scripts/run_latency.sh")
    p.add_argument("--out", type=Path, default=REPO_ROOT / "results" / "latency.csv")
    args = p.parse_args()

    rows = []
    for log in sorted(args.logs.glob("*.log")):
        m = TAG.match(log.stem)
        if not m:
            print(f"  skipping {log.name}: name does not look like <method>[_p<n>]")
            continue
        method = m.group("method")
        probe_len = m.group("probe") or ""
        logic, probe, trials = parse(log)
        if not logic:
            print(f"  skipping {log.name}: no [TIMING] lines "
                  "(was RESCUE_TIMING=1 set?)")
            continue
        for component, vals in (("added_logic", logic), ("total", logic),
                                ("probe", probe), ("candidates", trials)):
            if vals:
                rows.append((method, probe_len, component,
                             f"{st.median(vals):.1f}", len(vals)))
        print(f"  {log.stem:16s} added_logic {st.median(logic):7.1f} ms  "
              f"over {len(logic)} documents")

    if not rows:
        raise SystemExit("no usable logs; run scripts/run_latency.sh first")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["method", "probe_len", "component", "ms", "n"])
        w.writerows(rows)
    print(f"{len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
