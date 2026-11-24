#!/usr/bin/env python
"""Scorer-quality ladder on one consistent base (scorer quality).

All rungs that exist for only one base policy are compared on SnapKV alone,
so the ladder is a ladder and not a change of base. 16 LongBench tasks, 128
tokens, Llama-3.1-8B-Instruct.
"""
import csv, statistics as st
from collections import defaultdict
from pathlib import Path

S = Path("/path/to/kv_cache/rescue-release/results/scores.csv")
cells = defaultdict(dict)        # (arm, base) -> task -> score
for r in csv.DictReader(open(S, encoding="utf-8")):
    if r["model"] != "llama3_8b" or r["budget"] not in ("128", "0"):
        continue
    cells[(r["arm"], r["base"])][r["task"]] = float(r["score"])

def mean(arm, base):
    d = cells.get((arm, base), {})
    return (st.mean(d.values()), len(d)) if d else (None, 0)

def delta(arm, base, ref="base"):
    a, b = cells.get((arm, base), {}), cells.get((ref, base), {})
    k = sorted(set(a) & set(b))
    return (st.mean(a[t] - b[t] for t in k), len(k)) if k else (None, 0)

print("=== Ladder on SnapKV (delta vs SnapKV base, 16 tasks) ===")
for label, arm in (("oracle rescue set (upper bound)", "oracle_future"),
                   ("full-future scorer + selector", "fullfuture_gated"),
                   ("full-future scorer, always on", "fullfuture"),
                   ("RESCUE (shipped)", "rescue"),
                   ("RESCUE scorer, always on", "rescue_ungated"),
                   ("SnapKV base", "base")):
    d, n = delta(arm, "snapkv")
    m, _ = mean(arm, "snapkv")
    print(f"{label:34s} {('%+.2f'%d) if d is not None else '  n/a':>7s}  "
          f"(abs {m:.2f}, n={n})" if m else f"{label:34s}  n/a")

print("\n=== Same rungs, mean over all 5 base policies where available ===")
for label, arm in (("RESCUE (shipped)", "rescue"),
                   ("RESCUE scorer, always on", "rescue_ungated"),
                   ("full-future scorer + selector", "fullfuture_gated"),
                   ("full-future scorer, always on", "fullfuture")):
    ds = [delta(arm, b)[0] for b in ("snapkv", "laprox", "lava", "rkv", "h2o")]
    ds = [d for d in ds if d is not None]
    print(f"{label:34s} {st.mean(ds):+.2f}  (bases={len(ds)})")

d, _ = mean("dense", "")
print(f"\ndense (no eviction) = {d:.2f}" if d else "")
for b in ("snapkv", "laprox", "lava", "rkv", "h2o"):
    mb, _ = mean("base", b); mr, _ = mean("rescue", b)
    print(f"  {b:7s} base {mb:5.2f} -> rescue {mr:5.2f}  ({mr-mb:+.2f})")
