"""3rd RESCUE training-corpus domain: the real LongICLBench "Discovery" task
(TIGER-Lab/LongICLBench, TMLR2025 -- a real, official task, unlike the
20-Newsgroups substitute this replaces: that HF dataset only ships BANKING77/
DialogRE/Discovery/FewNERD/GoEmotion/TacRED, no Newsgroups task at all).

Uses the "1 Round Prompt" field directly (already a complete, ready-to-feed
prompt string with instructions + demonstrations + the query embedded --
LongICLBench ships these pre-built per round-count, not raw context+question
separately). "1 Round Prompt" alone is already ~48K chars (comfortably over
the 4000-char floor the other 2 corpora use); "5 Round Prompt" runs to
~235K chars, which would just get half-truncated by the 40000-token
MAX_SEQ_LEN budget anyway, so there's no benefit to a higher round count here.

No gold label needed (RESCUE training teacher-forces the model's OWN
generated continuation, not a ground-truth answer), so the "label" column
is unused.

Usage: python prepare_discovery_corpus.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from datasets import load_dataset

SCRATCH = Path(os.environ.get("RESCUE_CORPUS_DIR", "corpora"))
OUT_JSONL = SCRATCH / "discovery_corpus.jsonl"
TARGET_N = 260
MIN_CONTEXT_CHARS = 4000


def main() -> None:
    ds = load_dataset("TIGER-Lab/LongICLBench", split="Discovery")
    n = min(TARGET_N, len(ds))
    docs = []
    for i in range(len(ds)):
        if len(docs) >= TARGET_N:
            break
        row = ds[i]
        prompt = row["1 Round Prompt"].strip()
        if len(prompt) < MIN_CONTEXT_CHARS:
            continue
        docs.append({"context": "", "input": prompt})

    print(f"built {len(docs)} docs (target {TARGET_N}) from {len(ds)} available rows")
    with OUT_JSONL.open("w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"saved -> {OUT_JSONL}")
    lens = [len(d["input"]) for d in docs]
    print(f"prompt char lengths: min={min(lens)} max={max(lens)} mean={sum(lens)/len(lens):.0f}")


if __name__ == "__main__":
    main()
