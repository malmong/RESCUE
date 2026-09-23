"""Builds a few-shot intent-classification training corpus from Banking77
(mteb/banking77 on HF) -- NOT part of LongBench (trec's actual eval labels
are TREC coarse/fine question-type classes, banking intents never appear),
so training on it doesn't leak into any eventual full-LongBench comparison.
Structurally mirrors LongBench's own trec.jsonl format exactly (verified by
reading $RESCUE_DATA_ROOT/LongBench/data/trec.jsonl):
context = many "Question: <query>\nType: <label>\n" demonstration pairs,
input = "Question: <held-out query>\nType:", answers = [<held-out label>].

Motivation: the unified NQ(QA)+arXiv(summarization) corpus only covers
"long single document -> short answer/summary" structures. trec is a
completely different shape (many short labeled demonstrations -> classify
final query) and gets *worse* the more the corpus leans QA/summarization
(unified ensemble trec lam5 = 3.25, worse than NQ-only's 7.0 and even
baseline's 5.25) -- diagnosed as a missing third structural family, not a
lambda/capacity problem. This corpus fills that gap.
"""
import json
import random
import os
from pathlib import Path

from datasets import load_dataset

OUT = Path(os.environ.get("RESCUE_CORPUS_DIR", "corpora")) / "classify_corpus.jsonl"
TARGET_N = 260
NUM_DEMOS = 200
SEED = 0


def main():
    ds = load_dataset("mteb/banking77", split="train")
    rows = [{"text": r["text"], "label": r["label_text"]} for r in ds]
    rng = random.Random(SEED)

    written = 0
    with OUT.open("w", encoding="utf-8") as out_f:
        for doc_idx in range(TARGET_N):
            pool = rows[:]
            rng.shuffle(pool)
            demos = pool[:NUM_DEMOS]
            query = pool[NUM_DEMOS]
            context = "".join(f"Question: {d['text']}\nType: {d['label']}\n" for d in demos)
            record = {
                "context": context,
                "input": f"Question: {query['text']}\nType:",
                "answers": [query["label"]],
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
            if written % 50 == 0:
                print(f"written {written}", flush=True)
    print(f"done: {written} usable examples -> {OUT}")


if __name__ == "__main__":
    main()
