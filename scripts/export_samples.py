#!/usr/bin/env python
"""Export raw model predictions for a few cells, so the metrics can be checked.

``results/scores.csv`` is a set of numbers, and numbers in a file can be
anything. These samples are the layer underneath: the actual text the model
generated, the references it was scored against, and the score each document
received. With them, a reader can run the official metric themselves and see
that it reproduces the stored score -- which is what makes the aggregate
believable.

One sample per LongBench metric family, since verifying the metric is the
point and the families are what differ:

    F1              qasper
    ROUGE           gov_report
    classification  trec
    count           passage_count
    retrieval       passage_retrieval_en
    code similarity lcc

Documents per sample are capped: the full runs are 36 GB and a few dozen
documents establish that a metric agrees.

    python scripts/export_samples.py --runs /path/to/opencompass_outputs
    python scripts/verify_samples.py            # recompute and compare
"""
from __future__ import annotations

import argparse
import ast
import glob
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.export_per_document import details, evaluators  # noqa: E402

# (task, run stem) -- the base policy's own run, because a sample should show
# the plainest path through the metric, not the method's.
P = "llama31_8b_instruct"
SAMPLES = {
    "qasper": f"{P}-snapkv-b128tok-snapkvV_base",
    "gov_report": f"{P}-snapkv-b128tok-snapkvV_base",
    "trec": f"{P}-snapkv-b128tok-snapkvV_base",
    "passage_count": f"{P}-snapkv-b128tok-snapkvV_base",
    "passage_retrieval_en": f"{P}-snapkv-b128tok-snapkvV_base",
    "lcc": f"{P}-snapkv-b128tok-snapkvV_base",
}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", required=True, type=Path)
    p.add_argument("--limit", type=int, default=25,
                   help="documents per sample (default 25)")
    p.add_argument("--out", type=Path, default=REPO_ROOT / "results" / "samples")
    args = p.parse_args()

    EVAL = evaluators()
    args.out.mkdir(parents=True, exist_ok=True)
    for task, stem in SAMPLES.items():
        det = details(args.runs, stem, task)
        if not det:
            print(f"  {task:21s} not found under {args.runs}")
            continue
        ev = EVAL[task]()
        keys = sorted(det, key=lambda x: int(x) if x.isdigit() else x)[:args.limit]
        docs = []
        for k in keys:
            pred, refs = det[k]
            docs.append({
                "doc": k,
                "prediction": pred,
                "references": refs,
                "score": round(float(ev.score([pred], [refs])["score"]), 6),
            })
        payload = {
            "model": "llama3_8b",
            "policy": "snapkv",
            "task": task,
            "budget_tokens": 128,
            "evaluator": type(ev).__name__,
            "note": ("Scores are the official LongBench metric on one document at a "
                     "time, which is exactly that document's contribution to the "
                     "cell mean. Run scripts/verify_samples.py to recompute them."),
            "documents": docs,
        }
        dest = args.out / f"{task}.json"
        dest.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n",
                        encoding="utf-8")
        size = dest.stat().st_size / 1024
        print(f"  {task:21s} {len(docs):3d} documents, {size:6.1f} KB  "
              f"({type(ev).__name__})")


if __name__ == "__main__":
    main()
