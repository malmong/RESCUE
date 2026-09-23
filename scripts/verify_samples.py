#!/usr/bin/env python
"""Recompute the sampled document scores and compare them to the stored ones.

This is the check the samples exist for. ``results/scores.csv`` is a set of
numbers, and numbers in a file can be anything; ``results/samples/*.json``
carries the text those numbers came from. Running the official metric over that
text here, and getting the stored score back, is what ties the aggregate to
something a reader can inspect.

It needs OpenCompass, because the metrics are its implementations of the
official LongBench ones -- the same code path the evaluation used.

    python scripts/verify_samples.py
    python scripts/verify_samples.py --task qasper --verbose
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.export_per_document import evaluators  # noqa: E402

SAMPLES = REPO_ROOT / "results" / "samples"
TOL = 1e-6


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", help="check one task instead of all of them")
    p.add_argument("--verbose", action="store_true",
                   help="print each document that disagrees")
    args = p.parse_args()

    files = sorted(SAMPLES.glob("*.json"))
    if args.task:
        files = [f for f in files if f.stem == args.task]
    if not files:
        raise SystemExit(f"no samples under {SAMPLES}; run scripts/export_samples.py")

    EVAL = evaluators()
    total = bad = 0
    for f in files:
        payload = json.loads(f.read_text(encoding="utf-8"))
        ev = EVAL[payload["task"]]()
        mismatched = []
        for d in payload["documents"]:
            got = float(ev.score([d["prediction"]], [d["references"]])["score"])
            total += 1
            if abs(got - d["score"]) > TOL:
                bad += 1
                mismatched.append((d["doc"], d["score"], got))
        mark = "OK" if not mismatched else f"{len(mismatched)} differ"
        print(f"  {payload['task']:21s} {len(payload['documents']):3d} documents  "
              f"{payload['evaluator']:34s} {mark}")
        if mismatched and args.verbose:
            for doc, want, got in mismatched[:5]:
                print(f"      doc {doc}: stored {want:.6f}, recomputed {got:.6f}")

    print(f"\n  {total - bad}/{total} documents reproduce the stored score")
    if bad:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
