import os
#!/usr/bin/env python
"""Phase A: unpatched Llama-3.1-8B-Instruct, computes GT oracle + 5 observed
(H2O/SnapKV/LaProx/LAVa/R-KV) + ForesightKV candidate-token importance scores
for every qasper document, dumps one .pt per document to OUT_DIR for
coverage_phase_b_lookaheadkv.py and coverage_aggregate.py to consume.

Usage: python coverage_phase_a.py [--limit N] [--gpu 7]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import torch

import src.models.registry as _registry
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from coverage_lib import compute_doc_scores  # noqa: E402
from src.utils.foresightkv_official import ForesightKVJudgeScorer  # noqa: E402

MODEL_PATH = str(_registry.load("llama3_8b").path)
# ForesightKV's own judge, trained by its authors; set the path to your copy.
FORESIGHTKV_CKPT = os.environ.get("RESCUE_FORESIGHTKV_CKPT", "")
TASK_DATA_PATHS = {
    "qasper": str(_registry.data_root() / "LongBench" / "data" / "qasper.jsonl"),
    "hotpotqa": str(_registry.data_root() / "LongBench" / "data" / "hotpotqa.jsonl"),
    "gov_report": str(_registry.data_root() / "LongBench" / "data" / "gov_report.jsonl"),
    "multifieldqa_en": str(_registry.data_root() / "LongBench" / "data" / "multifieldqa_en.jsonl"),
}
SCRATCH = Path(os.environ.get(
    "RESCUE_SCRATCH", Path(__file__).resolve().parents[2] / "runs" / "coverage"))
MAX_SEQ_LEN = 131072


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=sorted(TASK_DATA_PATHS), default="qasper")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--start", type=int, default=0)
    args = ap.parse_args()

    out_dir = SCRATCH / "coverage_out" / args.task / "phase_a"
    out_dir.mkdir(parents=True, exist_ok=True)
    device = f"cuda:{args.gpu}"

    print(f"[{time.ctime()}] loading model on {device}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
    ).to(device)
    model.eval()

    print(f"[{time.ctime()}] loading ForesightKV judge scorer", flush=True)
    foresight_scorer = ForesightKVJudgeScorer(FORESIGHTKV_CKPT, device=device)

    docs = []
    with open(TASK_DATA_PATHS[args.task], "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            docs.append(json.loads(line))
    docs = docs[args.start :]
    if args.limit:
        docs = docs[: args.limit]
    print(f"[{time.ctime()}] {len(docs)} {args.task} documents to process", flush=True)

    ok, skipped, failed = 0, 0, 0
    for i, item in enumerate(docs):
        doc_id = item.get("_id", str(args.start + i))
        out_path = out_dir / f"{doc_id}.pt"
        if out_path.exists():
            ok += 1
            continue
        t0 = time.time()
        try:
            result = compute_doc_scores(
                model, tokenizer, device, doc_id, item["context"], item["input"], MAX_SEQ_LEN,
                foresight_scorer, task=args.task,
            )
        except Exception as e:
            failed += 1
            print(f"[{time.ctime()}] doc {doc_id} FAILED: {e}", flush=True)
            traceback.print_exc()
            torch.cuda.empty_cache()
            continue
        if result is None:
            skipped += 1
            print(f"[{time.ctime()}] doc {doc_id} skipped (too short)", flush=True)
            continue
        torch.save(
            {
                "doc_id": result.doc_id, "boundary": result.boundary,
                "cand_start": result.cand_start, "cand_end": result.cand_end,
                "prompt_ids": result.prompt_ids,
                "gt": result.gt, "h2o": result.h2o, "snapkv": result.snapkv,
                "laprox": result.laprox, "lava": result.lava, "rkv": result.rkv,
                "foresightkv": result.foresightkv, "recent": result.recent,
            },
            out_path,
        )
        ok += 1
        print(f"[{time.ctime()}] doc {doc_id} ({i+1}/{len(docs)}) ok, "
              f"boundary={result.boundary} n_cand={result.cand_end - result.cand_start} "
              f"({time.time()-t0:.1f}s)", flush=True)

    print(f"[{time.ctime()}] DONE. ok={ok} skipped={skipped} failed={failed}", flush=True)


if __name__ == "__main__":
    main()
