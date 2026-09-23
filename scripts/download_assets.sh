#!/usr/bin/env bash
# Fetch model weights and LongBench from the Hugging Face Hub.
#
# Weights are ~50 GB in total. Llama-3.1 and Mistral are gated: accept their
# licences on the Hub and run `huggingface-cli login` first, or this will stop
# with a 401 on those two and still fetch the rest.
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL_ROOT="${RESCUE_MODEL_ROOT:-$PWD/assets/models}"
DATA_ROOT="${RESCUE_DATA_ROOT:-$PWD/assets/data}"
mkdir -p "$MODEL_ROOT" "$DATA_ROOT"

command -v huggingface-cli >/dev/null || {
  echo "huggingface-cli not found; pip install -r requirements.txt" >&2; exit 1; }

for repo in meta-llama/Llama-3.1-8B-Instruct \
            mistralai/Mistral-7B-Instruct-v0.3 \
            Qwen/Qwen3-8B; do
  dest="$MODEL_ROOT/${repo##*/}"
  if [ -d "$dest" ]; then echo "have $dest"; continue; fi
  echo "== $repo"
  huggingface-cli download "$repo" --local-dir "$dest" \
    --exclude "*.pth" "original/*" || echo "  skipped $repo (gated? run huggingface-cli login)"
done

echo "== LongBench"
huggingface-cli download THUDM/LongBench --repo-type dataset \
  --local-dir "$DATA_ROOT/LongBench"

cat <<MSG

Done. Point the code at them:
  export RESCUE_MODEL_ROOT=$MODEL_ROOT
  export RESCUE_DATA_ROOT=$DATA_ROOT
MSG
