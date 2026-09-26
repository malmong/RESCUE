"""Builds a summarization training corpus from arXiv paper->abstract pairs
(ccdv/arxiv-summarization) -- NOT part of LongBench (unlike gov_report/qmsum/
multi_news, which ARE LaProx's evaluated summarization datasets), so training
on it doesn't leak into any eventual full-LongBench comparison. Long-document
structure (full paper body) matches gov_report's "long formal document ->
structured summary" shape, unlike short news articles (CNN/DM), testing
whether summarization-shaped training data fixes the word-overlap deficit
diagnosed on gov_report (RESCUE loses domain-identifying keywords that a
QA-boundary-trained corpus doesn't teach it to keep)."""
import json
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

from datasets import load_dataset

OUT = Path(os.environ.get("RESCUE_CORPUS_DIR",
                Path(os.environ.get("RESCUE_FEATURE_ROOT",
                                    REPO_ROOT / "assets" / "train")) / "corpora")) / "arxiv_corpus.jsonl"
TARGET_N = 260
MIN_CONTEXT_CHARS = 4000


def main():
    ds = load_dataset("ccdv/arxiv-summarization", split="train", streaming=True)
    written = 0
    seen = 0
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as out_f:
        for row in ds:
            seen += 1
            article = row["article"]
            abstract = row["abstract"]
            if not article or not abstract:
                continue
            if len(article) < MIN_CONTEXT_CHARS:
                continue
            record = {
                "context": article,
                "input": "Summarize the above scientific paper in a few sentences, covering its main topic, methods, and findings.",
                "answers": [abstract],
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
            if written % 20 == 0:
                print(f"written {written} (scanned {seen})", flush=True)
            if written >= TARGET_N:
                break
    print(f"done: {written} usable examples from {seen} scanned -> {OUT}")


if __name__ == "__main__":
    main()
