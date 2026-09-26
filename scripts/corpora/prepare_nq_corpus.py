"""Builds a qasper-shaped (long article + short factual answer) training
corpus from Google Natural Questions -- NOT part of LongBench (unlike
TriviaQA, which IS one of LaProx's 16 evaluated datasets), so training on it
doesn't leak into any eventual full-LongBench comparison against LaProx.

Streams the official NQ validation split, keeps only examples where at least
one annotator gave a real short answer, strips HTML tokens from the
Wikipedia page to build a plain-text context, and writes qasper-style
{"context", "input", "answers"} records.
"""
import json
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

from datasets import load_dataset

OUT = Path(os.environ.get("RESCUE_CORPUS_DIR",
                Path(os.environ.get("RESCUE_FEATURE_ROOT",
                                    REPO_ROOT / "assets" / "train")) / "corpora")) / "nq_corpus.jsonl"
TARGET_N = 260
MIN_CONTEXT_CHARS = 4000  # roughly >1000 tokens, matching our other corpora's length filter


def clean_tokens(tokens: dict) -> str:
    words = [t for t, is_html in zip(tokens["token"], tokens["is_html"]) if not is_html]
    return " ".join(words)


def main():
    ds = load_dataset("google-research-datasets/natural_questions", "default", split="validation", streaming=True)
    written = 0
    seen = 0
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as out_f:
        for row in ds:
            seen += 1
            ann = row["annotations"]
            answer_text = None
            for texts in ann["short_answers"]:
                if texts["text"]:
                    answer_text = texts["text"][0]
                    break
            if not answer_text:
                continue
            context = clean_tokens(row["document"]["tokens"])
            if len(context) < MIN_CONTEXT_CHARS:
                continue
            record = {
                "context": context,
                "input": row["question"]["text"],
                "answers": [answer_text],
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
