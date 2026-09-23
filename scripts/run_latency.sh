#!/usr/bin/env bash
# Runtime breakdown: what the selector costs, and where.
#
#   bash scripts/run_latency.sh llama3_8b 0
#
# RESCUE_TIMING=1 makes the generator emit one [TIMING] line per document with
# prefill / select / evict split out; tools/export_latency.py turns the log into
# results/latency.csv, which the overhead table and the probe-length table read.
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="${1:-llama3_8b}"
GPU="${2:-0}"
TASK="${3:-qasper}"
LOG=runs/latency
mkdir -p "$LOG"

run() {  # run <tag> <extra args...>
  local tag="$1"; shift
  RESCUE_TIMING=1 python evaluation/eval_longbench.py --model "$MODEL" --task "$TASK" \
      --gpu "$GPU" --tag "tm_$tag" "$@" 2>&1 | tee "$LOG/$tag.log"
}

run snapkv      --method snapkv
run foresightkv --method foresightkv
run dense       --method dense
for p in 1 2 4 8; do
  run "rescue_p$p" --method rescue --base snapkv \
      --checkpoint "results/checkpoints/rescue_${MODEL}_snapkv.pt" --probe-len "$p"
done

python tools/export_latency.py --logs "$LOG"
