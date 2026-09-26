#!/usr/bin/env python
"""Build results/measurements/memscale.csv from the RULER scaling runs.

Table 13 reports how the added logic and the peak allocation grow with the
prompt, at 16K, 32K and 64K. The numbers come from runs launched with
RESCUE_TIMING=1 and RESCUE_MEMLOG=1, which print one [E2E] line per document
carrying the eviction- and selection-related time, the peak allocation and the
size of the cache generation actually runs against. Medians over the documents,
as the table states.

    python scripts/export_memscale.py --runs /path/to/opencompass_outputs
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import re
import statistics as st
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# (run suffix, arm label). The base-policy rows come from runs of the policy on
# its own; the RESCUE rows from the same policy under the full pipeline.
ARMS = [
    ("mem_base_snapkv_16k", "snapkv", 16),
    ("mem_base_snapkv_32k", "snapkv", 32),
    ("mem_base_snapkv_64k", "snapkv", 64),
    ("sbopt_snapkv_16k", "snapkv+rescue", 16),
    ("sbopt_snapkv_32k", "snapkv+rescue", 32),
    ("sbopt_snapkv_64k", "snapkv+rescue", 64),
    ("sbopt_rkv_16k", "rkv+rescue", 16),
    ("sbopt_rkv_32k", "rkv+rescue", 32),
    ("sbopt_rkv_64k", "rkv+rescue", 64),
]
FIELDS = ("added", "peak", "kv_bytes")


def read(runs: Path, suffix: str) -> dict[str, float] | None:
    dirs = glob.glob(str(runs / f"*{suffix}-*"))
    if not dirs:
        return None
    outs = sorted(glob.glob(os.path.join(dirs[0], "*/logs/infer/*/*.out")),
                  key=os.path.getmtime)
    if not outs:
        return None
    txt = open(outs[-1], errors="ignore").read().replace("\r", "\n")
    got = {}
    for f in FIELDS:
        v = [float(x) for x in re.findall(rf"\b{f}=([0-9.]+)", txt)]
        if v:
            got[f] = st.median(v)
            got["n"] = len(v)
    return got or None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", required=True, type=Path)
    p.add_argument("--out", type=Path,
                   default=REPO_ROOT / "results" / "measurements" / "memscale.csv")
    args = p.parse_args()

    rows, missing = [], []
    for suffix, arm, length in ARMS:
        d = read(args.runs, suffix)
        if not d:
            missing.append(suffix)
            continue
        rows.append([arm, length, f"{d['added']:.1f}",
                     f"{d['peak'] / 1073741824:.2f}",
                     f"{d['kv_bytes'] / 1048576:.2f}", d["n"]])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="") as fh:
        fh.write("# RULER niah_single_2 at B=128 on Llama-3.1-8B-Instruct, medians per\n"
                 "# document. added_ms is eviction- and selection-related logic only;\n"
                 "# decode_kv_mb is the cache generation runs against, not the peak.\n")
        w = csv.writer(fh)
        w.writerow(["arm", "length_k", "added_ms", "peak_gib", "decode_kv_mb", "n"])
        w.writerows(rows)
    print(f"{len(rows)} rows -> {args.out}")
    if missing:
        print(f"  not found in --runs: {', '.join(missing)}")


if __name__ == "__main__":
    main()
