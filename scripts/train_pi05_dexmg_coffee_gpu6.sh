#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=/data_all/gzr1/openpi
EXP_NAME="${1:-pi05_dexmg_two_arm_coffee_$(date +%Y%m%d_%H%M%S)}"

cd "$ROOT_DIR"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.85}"

exec ./.venv/bin/python scripts/train.py \
  pi05_dexmg_two_arm_coffee \
  --exp-name "$EXP_NAME" \
  --overwrite
