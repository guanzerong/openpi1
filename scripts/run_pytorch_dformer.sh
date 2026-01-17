#!/usr/bin/env bash
# Start DFormer training with default environment overrides and arguments.
# Launch PyTorch DFormer training with preset GPU and checkpoint defaults.
# Usage: bash scripts/run_pytorch_dformer.sh [config] [exp_name] [pytorch_weight_path] [extra args...]
# Override CUDA_VISIBLE_DEVICES/NUM_GPUS via env vars if needed.
# Defaults target GPUs 3,6 and base weights at /data_all/gzr1/openpi_onlyrgbd/checkpoints/pi05_libero_pytorch.
# Ensure DFormer dependencies (opencv-python, timm, mmengine) are installed before running.
# Set NUM_GPUS=1 to run single-GPU training with the first visible device.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3,6}"
export PYTHONPATH="/data_all/gzr1:${PYTHONPATH:-}"

CONFIG_NAME="${1:-pi05_libero_depth}"
EXP_NAME="${2:-pi05_libero_depth_dformer}"
WEIGHT_PATH="${3:-/data_all/gzr1/openpi_onlyrgbd/checkpoints/pi05_libero_pytorch}"
NUM_GPUS="${NUM_GPUS:-2}"

EXTRA_ARGS=()
if [ "$#" -ge 4 ]; then
  EXTRA_ARGS=("${@:4}")
fi

torchrun --standalone --nnodes=1 --nproc_per_node="${NUM_GPUS}" \
  "${ROOT_DIR}/scripts/train_pytorch.py" "${CONFIG_NAME}" \
  --exp_name "${EXP_NAME}" \
  --pytorch_weight_path "${WEIGHT_PATH}" \
  "${EXTRA_ARGS[@]}"
