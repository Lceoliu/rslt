#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:-configs/embedding_default.yaml}
DS_CFG=${2:-configs/ds_config.json}
NGPU=${NGPU:-2}

if ! command -v deepspeed >/dev/null 2>&1; then
  echo "[ERROR] deepspeed not found. Install via: pip install deepspeed" >&2
  exit 1
fi

deepspeed --num_gpus=${NGPU} training/train_deepspeed.py \
  --config ${CONFIG} \
  --deepspeed \
  --deepspeed_config ${DS_CFG}

