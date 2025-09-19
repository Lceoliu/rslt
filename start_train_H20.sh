#!/usr/bin/env bash
set -euo pipefail

# H20 multi-GPU training launcher (bf16 enabled)
# Usage: bash start_train_H20.sh [TRAIN_CFG] [DS_CFG] [NGPU]

PYTHONPATH=$(pwd)
export PYTHONPATH
export TOKENIZERS_PARALLELISM=false

TRAIN_CFG=${1:-configs/train_default.yaml}
DS_CFG=${2:-configs/ds_config_bf16.json}
NGPU=${3:-8}

if ! command -v deepspeed >/dev/null 2>&1; then
  echo "[ERROR] deepspeed not found. Install via: pip install deepspeed" >&2
  exit 1
fi

echo "Using TRAIN_CFG=${TRAIN_CFG}"
echo "Using DS_CFG=${DS_CFG}"
echo "Using NGPU=${NGPU}"

deepspeed --num_gpus=${NGPU} training/train_deepspeed.py \
  --config ${TRAIN_CFG} \
  --deepspeed \
  --deepspeed_config ${DS_CFG}

