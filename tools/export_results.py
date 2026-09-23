#!/usr/bin/env python
"""Collect every LongBench score the paper reports into one tidy file.

The experiments were run as several hundred separate jobs whose directory names
encode their settings. Rather than teach each table script that naming scheme,
this converts all of them once into ``results/scores.csv``:

    model,arm,base,budget,task,score

Every script under ``tables/`` and ``figures/`` reads only that file, so the
paper's numbers can be regenerated with no GPU and no evaluation harness. Runs
not listed in ARMS below are diagnostics that no table or figure uses, and are
skipped rather than exported.

    python tools/export_results.py --runs /path/to/opencompass_outputs
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.models.registry import LONGBENCH_TASKS  # noqa: E402

BASES = ("snapkv", "laprox", "h2o", "lava", "rkv")

# run-name suffix -> (model, arm, base, budget)
#
# "arm" is the role a run plays in the paper:
#   base              the base eviction policy on its own
#   rescue            base + residual correction + fidelity selector (the method)
#   rescue_ungated    the same correction applied to every document
#   fullfuture        policy-agnostic full-future target, applied unconditionally
#   fullfuture_gated  the same target under the selector
#   oracle_future     the correction driven by real future attention (a ceiling)
#   dense             no eviction
#   lookaheadkv / foresightkv   future-aware comparators
#   cache_select      the selector choosing between SnapKV's and ForesightKV's caches
#   probe2/4/8        selector probe length other than one token
ARMS: dict[str, tuple[str, str, str | None, int]] = {}

# Some suffixes were reused across arms that differ only in a flag the suffix
# does not encode -- fidp1_rkv exists both as "fidelity0_1p1" (the selector
# choosing between lambda 0 and 1) and as "fidelity1p1" (lambda 1 applied
# unconditionally), and the two score 33.25 and 34.34 on Qasper. Matching on the
# suffix alone silently mixes them, so each arm also states a substring its run
# name must contain.
REQUIRE: dict[str, str] = {}


def _add(suffix: str, model: str, arm: str, base: str | None, budget: int,
         require: str = "") -> None:
    ARMS[suffix] = (model, arm, base, budget)
    if require:
        REQUIRE[suffix] = require


# ---- Llama-3.1-8B-Instruct -------------------------------------------------
for _s, _b in (("abl_baseline", "laprox"), ("snapkvV_base", "snapkv"),
               ("h2oV_base", "h2o"), ("rkvV_base", "rkv"),
               ("p128_lava_base", "lava")):
    _add(_s, "llama3_8b", "base", _b, 128)
for _s, _b in (("fidp1_laprox", "laprox"), ("fidp1_snapkv", "snapkv"),
               ("fidp1_h2o", "h2o"), ("fidp1_rkv", "rkv"),
               ("p128_lava_fid", "lava")):
    _add(_s, "llama3_8b", "rescue", _b, 128, require="fidelity0_1p1")
for _b in BASES:
    _add(f"p128_corr_{_b}", "llama3_8b", "rescue_ungated", _b, 128)
    _add(f"p128_ff_{_b}", "llama3_8b", "fullfuture", _b, 128)
    _add(f"p128_ffsel_{_b}", "llama3_8b", "fullfuture_gated", _b, 128,
         require="fidelity0_1p1")
    # Probe length was swept on LaProx and SnapKV only (Appendix: probe length).
    if _b in ("laprox", "snapkv"):
        for _n in (2, 4, 8):
            _add(f"p128_p{_n}_{_b}", "llama3_8b", f"probe{_n}", _b, 128,
                 require=f"fidelity0_1p{_n}")
    for _budget in (256, 1024):
        _add(f"b{_budget}_{_b}_base", "llama3_8b", "base", _b, _budget)
        _add(f"b{_budget}_{_b}_fid", "llama3_8b", "rescue", _b, _budget,
             require="fidelity0_1p1")
_add("dense16", "llama3_8b", "dense", None, 0)
_add("p128_lookahead", "llama3_8b", "lookaheadkv", None, 128)
_add("p128_foresight64", "llama3_8b", "foresightkv", None, 128)
# orc50 runs the correction against REAL future attention taken from an
# un-evicted reference generation: a ceiling on the future SIGNAL, not on the
# per-document decision. The selector oracle is a different quantity and comes
# from results/per_document.csv.
_add("orc50", "llama3_8b", "oracle_future", "snapkv", 128)
_add("cachesel", "llama3_8b", "cache_select", "snapkv", 128)

# ---- Mistral-7B-Instruct-v0.3 ---------------------------------------------
# m32_* are the runs at the model's own 32768-token position limit. The earlier
# mi_* runs fed it prompts up to 81k, which left its eviction scores computed
# from positions it was never trained on; only NarrativeQA was affected, but
# the m32_* set is the one the paper reports.
_add("m32_dense", "mistral_7b", "dense", None, 0)
for _b in BASES:
    _add(f"m32_base_{_b}", "mistral_7b", "base", _b, 128)
    _add(f"m32_fid_{_b}", "mistral_7b", "rescue", _b, 128, require="fidelity0_1p1")

# ---- Qwen3-8B --------------------------------------------------------------
# qwfl_* carry the selector's abstention floor. The earlier qw_fid_* runs let
# the order of two rounding errors choose lambda on the documents where the KL
# underflows, which on Qwen is a large share of them.
_add("qw_dense", "qwen3_8b", "dense", None, 0)
for _b in BASES:
    _add(f"qw_base_{_b}", "qwen3_8b", "base", _b, 128)
    _add(f"qwfl_{_b}", "qwen3_8b", "rescue", _b, 128, require="fidelity0_1p1")

# A few cells had to be re-run under their own suffix after a fix; same flags,
# same checkpoint, so they stand in for the original cell.
ALIASES = {
    ("p128_nq_rkv_fid", "narrativeqa"): "fidp1_rkv",
    ("p128_nq_h2o_fid", "narrativeqa"): "fidp1_h2o",
}


def score_of(run_dir: str) -> float | None:
    for f in sorted(glob.glob(os.path.join(run_dir, "*/summary/summary_*.csv")),
                    reverse=True):
        for row in csv.reader(open(f, encoding="utf-8")):
            for cell in reversed(row):
                if re.fullmatch(r"\d+\.\d+", (cell or "").strip()):
                    return float(cell)
    return None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", required=True, type=Path,
                   help="directory holding the per-run OpenCompass output folders")
    p.add_argument("--out", type=Path, default=REPO_ROOT / "results" / "scores.csv")
    args = p.parse_args()

    rows: dict[tuple, float] = {}
    seen_suffixes: set[str] = set()
    for d in sorted(glob.glob(str(args.runs / "*-longbench_*"))):
        stem = os.path.basename(d)
        head, task = stem.rsplit("-longbench_", 1)
        if task not in LONGBENCH_TASKS:
            continue
        suffix = head.rsplit("-", 1)[-1]
        seen_suffixes.add(suffix)
        key = ALIASES.get((suffix, task), suffix)
        if key not in ARMS:
            continue
        need = REQUIRE.get(key)
        if need and need not in head:
            continue
        model, arm, base, budget = ARMS[key]
        s = score_of(d)
        if s is None:
            continue
        # Later runs of the same cell win; directories are walked in name order
        # and re-runs carry a newer timestamp inside, which score_of prefers.
        rows[(model, arm, base or "", budget, task)] = s

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "arm", "base", "budget", "task", "score"])
        for k in sorted(rows):
            w.writerow([*k, f"{rows[k]:.2f}"])
    print(f"{len(rows)} cells -> {args.out}")

    missing = [s for s in ARMS if s not in seen_suffixes]
    if missing:
        print(f"  not found in --runs ({len(missing)}): {', '.join(sorted(missing)[:8])}"
              + (" ..." if len(missing) > 8 else ""))
    incomplete = {}
    for (model, arm, base, budget, _), _ in rows.items():
        incomplete[(model, arm, base, budget)] = incomplete.get((model, arm, base, budget), 0) + 1
    partial = {k: v for k, v in incomplete.items() if v < len(LONGBENCH_TASKS)}
    if partial:
        print(f"  incomplete sweeps ({len(partial)}):")
        for k, v in sorted(partial.items())[:10]:
            print(f"    {'/'.join(str(x) for x in k)}  {v}/{len(LONGBENCH_TASKS)}")


if __name__ == "__main__":
    main()
