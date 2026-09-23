#!/usr/bin/env bash
# Build the scorer's training corpora from the Hugging Face Hub.
#
# Natural Questions and arXiv, both out of domain for LongBench -- the scorer is
# deliberately not fitted to the evaluation distribution. About 56 MB of JSONL;
# not committed, because it is derived data.
set -euo pipefail
cd "$(dirname "$0")/.."

OUT="${RESCUE_CORPUS_DIR:-${RESCUE_FEATURE_ROOT:-$PWD/assets/train}/corpora}"
mkdir -p "$OUT"
export RESCUE_CORPUS_DIR="$OUT"

for s in prepare_nq_corpus prepare_nq_corpus_extra prepare_arxiv_corpus; do
  echo "== $s"
  python "train/corpora/$s.py"
done
echo "corpora -> $OUT"
