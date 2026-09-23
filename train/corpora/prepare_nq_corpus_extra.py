"""Extends the NQ training corpus with a second, disjoint batch of 260 docs
(same filter as prepare_nq_corpus.py) by skipping the first 260 valid matches
(already in nq_corpus.jsonl) and collecting the next 260 from the same
deterministic stream order -- tests whether more domain-matched data closes
the remaining 2wikimqa gap (48.18 vs baseline 48.27)."""
import json
import os
from pathlib import Path

from datasets import load_dataset

OUT = Path(os.environ.get("RESCUE_CORPUS_DIR", "corpora")) / "nq_corpus_extra.jsonl"
SKIP_N = 260
TARGET_N = 260
MIN_CONTEXT_CHARS = 4000


def clean_tokens(tokens: dict) -> str:
    words = [t for t, is_html in zip(tokens["token"], tokens["is_html"]) if not is_html]
    return " ".join(words)


def main():
    ds = load_dataset("google-research-datasets/natural_questions", "default", split="validation", streaming=True)
    written = 0
    skipped = 0
    seen = 0
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
            if skipped < SKIP_N:
                skipped += 1
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
