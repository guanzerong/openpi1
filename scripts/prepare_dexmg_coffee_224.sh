#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=/data_all/gzr1
OPENPI_DIR="$ROOT_DIR/openpi"
DEXMG_DIR="$ROOT_DIR/code/residual-offpolicy-rl-macrocls-change-xiugai/deps/dexmimicgen"
ROBOSUITE_DIR="$ROOT_DIR/code/residual-offpolicy-rl-macrocls-change-xiugai/deps/robosuite"
RESFIT_DIR="$ROOT_DIR/code/residual-offpolicy-rl-macrocls-change-xiugai"
HF_HOME_DIR="$ROOT_DIR/tmp/huggingface"

OPENPI_PY="$OPENPI_DIR/.venv/bin/python"
RESIDUAL_PY="/home/gzr1/miniconda3/envs/residual1/bin/python"

RAW_ROOT="${1:-$OPENPI_DIR/datasets/dexmimicgen_raw}"
LEROBOT_DIR="${2:-$OPENPI_DIR/datasets/dexmg_two_arm_coffee_224_lerobot}"

SOURCE_HDF5="$RAW_ROOT/generated/two_arm_coffee.hdf5"
RENDERED_HDF5="$RAW_ROOT/generated/two_arm_coffee_224.hdf5"

if [ ! -x "$OPENPI_PY" ]; then
  echo "Missing OpenPI Python interpreter: $OPENPI_PY" >&2
  exit 1
fi

if [ ! -x "$RESIDUAL_PY" ]; then
  echo "Missing residual Python interpreter: $RESIDUAL_PY" >&2
  exit 1
fi

mkdir -p "$RAW_ROOT/generated" "$LEROBOT_DIR"
mkdir -p "$HF_HOME_DIR"
export HF_HOME="$HF_HOME_DIR"

if [ ! -f "$SOURCE_HDF5" ]; then
  RAW_ROOT="$RAW_ROOT" "$OPENPI_PY" - <<'PY'
import os

from huggingface_hub import hf_hub_download, list_repo_files

repo_id = "MimicGen/dexmimicgen_datasets"
local_dir = os.environ["RAW_ROOT"]
target = "generated/two_arm_coffee.hdf5"

files = set(list_repo_files(repo_id, repo_type="dataset"))
if target not in files:
    matches = sorted(f for f in files if f.endswith("two_arm_coffee.hdf5"))
    if not matches:
        raise FileNotFoundError(f"Could not find {target} in {repo_id}")
    target = matches[0]

print(f"Downloading {target} from {repo_id} into {local_dir}")
hf_hub_download(
    repo_id=repo_id,
    filename=target,
    repo_type="dataset",
    local_dir=local_dir,
)
PY
fi

if [ ! -f "$RENDERED_HDF5" ]; then
  CUDA_VISIBLE_DEVICES="${RENDER_CUDA_VISIBLE_DEVICES:-1}" \
  PYTHONPATH="$ROBOSUITE_DIR:$DEXMG_DIR:$OPENPI_DIR/robomimic${PYTHONPATH:+:$PYTHONPATH}" \
    "$RESIDUAL_PY" - <<PY
import runpy
import sys

import dexmimicgen  # Registers TwoArmCoffee with robosuite.

sys.argv = [
    "dataset_states_to_obs.py",
    "--dataset",
    "$SOURCE_HDF5",
    "--output_name",
    "$(basename "$RENDERED_HDF5")",
    "--done_mode",
    "2",
    "--camera_names",
    "agentview",
    "robot0_eye_in_left_hand",
    "robot0_eye_in_right_hand",
    "--camera_height",
    "224",
    "--camera_width",
    "224",
]
runpy.run_path("$OPENPI_DIR/robomimic/robomimic/scripts/dataset_states_to_obs.py", run_name="__main__")
PY
fi

if [ ! -f "$LEROBOT_DIR/meta/info.json" ] && [ ! -f "$LEROBOT_DIR/meta_data/info.json" ]; then
  "$RESIDUAL_PY" "$RESFIT_DIR/resfit/lerobot/dataset/convert_robomimic_to_lerobot.py" \
    --dataset "$RENDERED_HDF5" \
    --output_dir "$LEROBOT_DIR"
fi

(cd "$OPENPI_DIR" && "$OPENPI_PY" scripts/compute_norm_stats.py --config-name pi05_dexmg_two_arm_coffee)

echo "Prepared DexMimicGen coffee dataset at $LEROBOT_DIR"
