#!/usr/bin/env bash
# Runtime breakdown: what the selector costs, and where.
#
#   bash scripts/run_latency.sh llama3_8b 0
#
# RESCUE_TIMING=1 makes the generator emit one [TIMING] line per document with
# prefill / select / evict split out; scripts/export_latency.py turns the log into
# results/latency.csv, which the overhead table and the probe-length table read.
set -euo pipefail

# Python interpreter: override with PYTHON=... for a venv or a specific
# build. Bare `python` is not present on every system.
PYTHON="${PYTHON:-python3}"
cd "$(dirname "$0")/.."

MODEL="${1:-llama3_8b}"
GPU="${2:-0}"
TASK="${3:-qasper}"
LOG=runs/latency
mkdir -p "$LOG"

run() {  # run <tag> <extra args...>
  local tag="$1"; shift
  RESCUE_TIMING=1 "$PYTHON" scripts/evaluate.py --model "$MODEL" --task "$TASK" \
      --gpu "$GPU" --tag "tm_$tag" "$@" 2>&1 | tee "$LOG/$tag.log"
}

run snapkv      --method snapkv
run dense       --method dense

# ForesightKV needs its authors' judge model, which is not vendored here, so
# this arm runs only if you have fetched it. Table 14's ForesightKV row comes
# from it; everything else in the table does not.
FORESIGHT_CKPT="${FORESIGHT_CKPT:-results/checkpoints/foresightkv_judge_models.pt}"
if [ -f "$FORESIGHT_CKPT" ]; then
  run foresightkv --method foresightkv --checkpoint "$FORESIGHT_CKPT"
else
  echo "no ForesightKV judge model at $FORESIGHT_CKPT, skipping that arm" >&2
fi
for p in 1 2 4 8; do
  run "rescue_p$p" --method rescue --base snapkv \
      --checkpoint "results/checkpoints/rescue_${MODEL}_snapkv.pt" --probe-len "$p"
done

"$PYTHON" scripts/export_latency.py --logs "$LOG"
