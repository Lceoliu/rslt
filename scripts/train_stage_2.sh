export PYTHONPATH=$(pwd)
export TOKENIZERS_PARALLELISM=false

if ! command -v deepspeed >/dev/null 2>&1; then
  echo "[ERROR] deepspeed not found. Install via: pip install deepspeed" >&2
  exit 1
fi

if [ "$(basename "$PWD")" = "scripts" ]; then
    echo "Please run this script from the root directory, not from the scripts directory."
    exit 1
fi

deepspeed --num_gpus=8 --master_port=29501 training/train_deepspeed.py \
  --config configs/train_cslnews_stage2.yaml \
  --deepspeed \
  --deepspeed_config configs/ds_config_stage2.json