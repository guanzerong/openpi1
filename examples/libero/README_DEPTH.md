# OpenPi with Depth Support for LIBERO

This extension adds depth map input capability to OpenPi for the LIBERO benchmark, following the approach from 3D-CAVLA.

## Overview

The depth support implementation includes:

1. **PointNet Depth Projector** (`src/openpi/models/depth_projector.py`)
   - Converts depth maps to point clouds using camera intrinsics
   - Processes point clouds through PointNet for feature extraction
   - Outputs a fixed-dimensional feature vector

2. **Pi0Depth Model** (`src/openpi/models/pi0_depth.py`)
   - Extends Pi0 model to accept depth inputs
   - Concatenates depth features with image tokens in the prefix

3. **Data Transforms** (`src/openpi/policies/libero_depth_policy.py`)
   - Handles depth map parsing and normalization
   - Extends Libero transforms with depth support

4. **Training Configuration** (`src/openpi/training/libero_depth_config.py`)
   - Pre-configured training configs for Libero with depth
   - Supports both full fine-tuning and LoRA

## Files Created

```
openpi/
├── src/openpi/
│   ├── models/
│   │   ├── depth_projector.py      # PointNet encoder for depth
│   │   ├── pi0_depth.py            # Pi0 model with depth support
│   │   └── pi0_depth_config.py     # Configuration for Pi0Depth
│   ├── policies/
│   │   └── libero_depth_policy.py  # Data transforms with depth
│   └── training/
│       └── libero_depth_config.py  # Training configurations
└── examples/libero/
    ├── convert_libero_depth_to_lerobot.py  # Data conversion script
    ├── main_depth.py                        # Evaluation script with depth
    └── README_DEPTH.md                      # This file
```

## Usage

### 1. Data Conversion

Convert the `modified_libero_rlds_cotdep` dataset to LeRobot format:

```bash
python examples/libero/convert_libero_depth_to_lerobot.py \
    --input_dir /path/to/modified_libero_rlds_cotdep \
    --output_dir /path/to/output \
    --task_suite libero_spatial_cotdep
```

### 2. Training

Use the pre-configured training configs:

```bash
# Full fine-tuning
python -m openpi.training.train --config pi05_libero_depth

# LoRA fine-tuning (faster, requires less memory)
python -m openpi.training.train --config pi05_libero_depth_lora
```

### 3. Evaluation

Evaluate with depth input:

```bash
# Start the policy server
python -m openpi.serving.serve --checkpoint /path/to/checkpoint

# Run evaluation
python examples/libero/main_depth.py \
    --host localhost \
    --port 8000 \
    --task_suite_name libero_spatial \
    --use_depth
```

## Architecture

### Depth Processing Pipeline

```
Depth Map (256x256)
    ↓
Point Cloud Conversion (using camera intrinsics)
    ↓
PointNet Encoder
    - Spatial Transformer (3x3 alignment)
    - Shared MLP layers with BatchNorm
    - Max Pooling (permutation invariance)
    ↓
Depth Features (2048-dim)
    ↓
Concatenated with Image Tokens
    ↓
Pi0 Model
```

### Camera Intrinsics (Libero)

- Image size: 256 x 256
- Focal length: fx = fy = 309.019
- Principal point: cx = cy = 128.0

## Training Strategy

Following the 3D-CAVLA approach:

1. **Main Model**: LoRA fine-tuning (rank=32) for efficient adaptation
2. **Depth Projector**: Full training for the new module
3. **This preserves the pre-trained model's capabilities while learning to use depth**

## Notes

- Only third-person camera depth is used (not wrist camera depth)
- Depth values are normalized to [0, 1] range
- The depth projector adds ~10M parameters
- Training with depth requires ~1.5x more memory

## References

- 3D-CAVLA: Depth-assisted VLA approach this implementation is based on
- OpenPi: Base VLA framework
- PointNet: Classic point cloud processing network
