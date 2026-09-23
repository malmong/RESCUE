"""Builds the 3rd RESCUE training-corpus domain: a many-shot long-context
classification task in the LongICLBench style (demonstrations of labeled
20-Newsgroups posts, then one held-out post to classify).

NOTE: the official LongICLBench release (TIGER-Lab/LongICLBench, TMLR2025)
does NOT actually include a 20-Newsgroups task -- it ships BANKING77,
DialogRE, Discovery, FewNERD, GoEmotion, TacRED only (confirmed by cloning
github.com/TIGER-AI-Lab/LongICLBench and inspecting its HF dataset siblings).
This builds the same *style* of task (many-shot classification demo + one
query) directly from the raw 20 Newsgroups dataset (SetFit/20_newsgroups on
HF), which is what "LongICLBench's 20 Newsgroups" almost certainly refers to
in spirit even though it isn't literally packaged there.

Matches the NQ/arXiv corpus scripts' output format: one JSON object per line
with "context" (the long part) and "input" (the final instruction), no gold
label needed -- RESCUE training teacher-forces the MODEL's OWN generated
continuation as the trace, not a ground-truth answer.

Usage: python prepare_newsgroups_corpus.py
"""
from __future__ import annotations

import json
import random
import os
from pathlib import Path

from datasets import load_dataset

SCRATCH = Path(os.environ.get("RESCUE_CORPUS_DIR", "corpora"))
OUT_JSONL = SCRATCH / "newsgroups_corpus.jsonl"
TARGET_N = 260          # match NQ/arXiv corpus size
SHOTS_PER_DOC = 45      # labeled demo posts per doc (posts are short; need many to reach real long-context scale)
MIN_CONTEXT_CHARS = 4000
SEED = 0


def build_context(demo_rows: list[dict]) -> str:
    blocks = []
    for i, row in enumerate(demo_rows):
        blocks.append(f"Post {i + 1}:\n{row['text'].strip()}\nCategory: {row['label_text']}\n")
    return "\n".join(blocks)


def main() -> None:
    ds = load_dataset("SetFit/20_newsgroups", split="train")
    rng = random.Random(SEED)
    indices = list(range(len(ds)))
    rng.shuffle(indices)

    categories = sorted(set(ds["label_text"]))
    cat_list_str = ", ".join(categories)

    docs = []
    cursor = 0
    while len(docs) < TARGET_N and cursor + SHOTS_PER_DOC + 1 <= len(indices):
        chunk_idx = indices[cursor:cursor + SHOTS_PER_DOC + 1]
        cursor += SHOTS_PER_DOC + 1
        rows = [ds[i] for i in chunk_idx]
        demo_rows, query_row = rows[:-1], rows[-1]
        context = build_context(demo_rows)
        if len(context) < MIN_CONTEXT_CHARS:
            continue
        query_text = query_row["text"].strip()
        input_text = (
            f"Based on the {len(demo_rows)} labeled example posts above, classify the following new post "
            f"into exactly one of these categories: {cat_list_str}.\n\n"
            f"Post to classify:\n{query_text}\n\n"
            f"Respond with only the category name, nothing else.\nCategory:"
        )
        docs.append({"context": context, "input": input_text})

    print(f"built {len(docs)} docs (target {TARGET_N}), cursor used {cursor}/{len(indices)} source rows")
    with OUT_JSONL.open("w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"saved -> {OUT_JSONL}")
    lens = [len(d["context"]) for d in docs]
    print(f"context char lengths: min={min(lens)} max={max(lens)} mean={sum(lens)/len(lens):.0f}")


if __name__ == "__main__":
    main()
