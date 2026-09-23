#!/usr/bin/env bash
# Section 5.4: what the correction contributes, and what the selector adds.
#
#   bash scripts/run_ablation.sh llama3_8b 0
#
# Four arms beyond the main grid:
#   correction applied to every document      --lambdas 1
#   full-future target, unconditional         the policy-agnostic scorer, --lambdas 1
#   full-future target, gated                 the same scorer under the selector
#   budget sweep                              --budget-tokens 256 / 1024
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="${1:-llama3_8b}"
GPU="${2:-0}"
BASES=(snapkv laprox h2o lava rkv)
FF="results/checkpoints/fullfuture_${MODEL}.pt"

for base in "${BASES[@]}"; do
  ckpt="results/checkpoints/rescue_${MODEL}_${base}.pt"
  [ -f "$ckpt" ] || continue

  python scripts/evaluate.py --model "$MODEL" --method rescue --base "$base" \
      --checkpoint "$ckpt" --task all --gpu "$GPU" --lambdas 1 --tag corr

  if [ -f "$FF" ]; then
    python scripts/evaluate.py --model "$MODEL" --method rescue --base "$base" \
        --checkpoint "$FF" --task all --gpu "$GPU" --lambdas 1 --tag ff
    python scripts/evaluate.py --model "$MODEL" --method rescue --base "$base" \
        --checkpoint "$FF" --task all --gpu "$GPU" --tag ffsel
  fi

  for budget in 256 1024; do
    b_ckpt="results/checkpoints/rescue_${MODEL}_${base}_b${budget}.pt"
    python scripts/evaluate.py --model "$MODEL" --method "$base" \
        --task all --gpu "$GPU" --budget-tokens "$budget" --tag "b${budget}"
    [ -f "$b_ckpt" ] || continue
    python scripts/evaluate.py --model "$MODEL" --method rescue --base "$base" \
        --checkpoint "$b_ckpt" --task all --gpu "$GPU" --budget-tokens "$budget" --tag "b${budget}"
  done

  # Probe length (LaProx and SnapKV in the paper).
  case "$base" in
    snapkv|laprox)
      for p in 2 4 8; do
        python scripts/evaluate.py --model "$MODEL" --method rescue --base "$base" \
            --checkpoint "$ckpt" --task all --gpu "$GPU" --probe-len "$p" --tag "p${p}"
      done ;;
  esac
done
