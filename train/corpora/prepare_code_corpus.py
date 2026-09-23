"""Builds a code-completion training corpus, in the same shape as the recovered
prepare_{nq,arxiv,newsgroups,classify}_corpus.py builders.

Motivation: the artifact's RESCUE module was trained on NQ + arXiv only, with no
code anywhere in the mix, and code is exactly where it loses — lcc scores 55.48
against a 59.58 LaProx baseline (−4.10), the single largest regression in the
table. This corpus exists to test whether code-shaped training data closes it.

Shape matches LongBench's lcc/repobench (a long source file, continue the next
line), NOT a short function snippet: bigcode/the-stack-smol is gated, and
code_search_net rows are single functions (median ~1.2k chars, far too short for
a 128-token budget to bind), so documents are assembled by concatenating
consecutive functions FROM THE SAME REPOSITORY until the length filter is met.
Mixing repositories inside one document would make the "next line" continuation
incoherent.

Deliberately avoids LongBench's own lcc/repobench data, matching the other
builders' rule of never training on an evaluated dataset.
"""
import json
import os
from pathlib import Path

from datasets import load_dataset

OUT = Path(os.environ.get("RESCUE_CORPUS_DIR", "corpora")) / "code_corpus.jsonl"
TARGET_N = 260
MIN_CONTEXT_CHARS = 4000   # same filter as the NQ/arXiv/newsgroups builders
MAX_CONTEXT_CHARS = 60_000
TARGET_DOC_CHARS = 30_000


def main():
    ds = load_dataset("code_search_net", "python", split="train", streaming=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    docs, buf, buf_repo, seen = [], [], None, 0

    for row in ds:
        seen += 1
        repo = row.get("repository_name") or ""
        fn = row.get("whole_func_string") or row.get("func_code_string") or ""
        path = row.get("func_path_in_repository") or ""
        if not fn:
            continue
        if buf_repo is not None and repo != buf_repo:
            buf = []                       # never span repositories
        buf_repo = repo
        buf.append(f"# {path}\n{fn}\n")
        if sum(len(x) for x in buf) < TARGET_DOC_CHARS:
            continue

        text = "\n".join(buf)
        buf = []
        if not (MIN_CONTEXT_CHARS <= len(text) <= MAX_CONTEXT_CHARS):
            continue
        # hold out the final line as the target continuation, lcc-style
        lines = text.rstrip("\n").split("\n")
        if len(lines) < 10:
            continue
        context, answer = "\n".join(lines[:-1]) + "\n", lines[-1]
        if len(context) < MIN_CONTEXT_CHARS:
            continue
        docs.append({"context": context, "input": "", "answers": [answer]})
        if len(docs) >= TARGET_N:
            break

    with OUT.open("w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    lens = sorted(len(d["context"]) for d in docs)
    print(f"wrote {len(docs)} docs to {OUT} (scanned {seen} functions)")
    print(f"context chars: min {lens[0]}  median {lens[len(lens)//2]}  max {lens[-1]}")


if __name__ == "__main__":
    main()
