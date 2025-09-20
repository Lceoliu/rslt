#!/usr/bin/env bash
set -euo pipefail

# Simple inference launcher
# Usage: bash start_infer.sh CHECKPOINT_DIR [CONFIG] [DS_CONFIG] [MODE] [NPY] [OUTPUT]

PYTHONPATH=$(pwd)
export PYTHONPATH
export TOKENIZERS_PARALLELISM=false

CKPT_DIR=${1:?"CHECKPOINT_DIR is required (runs/{ts} or runs/{ts}/checkpoints)"}
CONFIG=${2:-configs/train_default.yaml}
DS_CFG=${3:-configs/ds_config_bf16.json}
MODE=${4:-test}
NPY=${5:-}
OUTPUT=${6:-runs/$(date +%Y%m%d_%H%M%S)/infer.json}

mkdir -p "$(dirname "$OUTPUT")"

echo "Running inference:"
echo "  CKPT_DIR=$CKPT_DIR"
echo "  CONFIG=$CONFIG"
echo "  DS_CFG=$DS_CFG"
echo "  MODE=$MODE"
echo "  NPY=$NPY"
echo "  OUTPUT=$OUTPUT"

CMD=(python3 inference/run_infer.py \
  --checkpoint "$CKPT_DIR" \
  --config "$CONFIG" \
  --ds_config "$DS_CFG" \
  --mode "$MODE" \
  --output "$OUTPUT" \
  --bf16)

if [[ "$MODE" == "single" && -n "$NPY" ]]; then
  CMD+=(--npy "$NPY")
fi

# Optional decoding overrides via env vars
[[ -n "${TEMPERATURE:-}" ]] && CMD+=(--temperature "$TEMPERATURE")
[[ -n "${TOP_K:-}" ]] && CMD+=(--top_k "$TOP_K")
[[ -n "${MAX_NEW_TOKENS:-}" ]] && CMD+=(--max_new_tokens "$MAX_NEW_TOKENS")

echo "Command: ${CMD[*]}"
"${CMD[@]}"

