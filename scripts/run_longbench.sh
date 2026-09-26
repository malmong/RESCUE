#!/usr/bin/env bash
# Table 1: every base policy, with and without RESCUE, on all sixteen tasks.
#
#   bash scripts/run_longbench.sh llama3_8b 0
#
# One cell at a time on one GPU. Scores land in results/summaries/; rebuild
# results/scores.csv with scripts/export_results.py when the grid is done.
set -euo pipefail

# Python interpreter: override with PYTHON=... for a venv or a specific
# build. Bare `python` is not present on every system.
PYTHON="${PYTHON:-python3}"
cd "$(dirname "$0")/.."

MODEL="${1:-llama3_8b}"
GPU="${2:-0}"
BASES=(snapkv laprox h2o lava rkv)

"$PYTHON" scripts/evaluate.py --model "$MODEL" --method dense --task all --gpu "$GPU"

for base in "${BASES[@]}"; do
  "$PYTHON" scripts/evaluate.py --model "$MODEL" --method "$base" --task all --gpu "$GPU"

  ckpt="results/checkpoints/rescue_${MODEL}_${base}.pt"
  [ -f "$ckpt" ] || { echo "no scorer for $base at $ckpt, skipping" >&2; continue; }
  "$PYTHON" scripts/evaluate.py --model "$MODEL" --method rescue --base "$base" \
      --checkpoint "$ckpt" --task all --gpu "$GPU"
done
