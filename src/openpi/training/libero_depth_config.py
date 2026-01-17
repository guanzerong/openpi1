"""Training configuration for Libero with depth support.

This module provides training configurations for the Pi0 model with depth input
on the Libero dataset with depth maps (modified_libero_rlds_cotdep).
"""
import dataclasses
import pathlib
from typing import TypeAlias

import flax.nnx as nnx
import numpy as np
from typing_extensions import override

import openpi.models.model as _model
from openpi.models import pi0_depth_config
from openpi.policies import libero_depth_policy
import openpi.transforms as _transforms
from openpi.shared import array_typing as at
from openpi.shared import download
from openpi.training.config import (
    DataConfig,
    DataConfigFactory,
    ModelTransformFactory,
    TrainConfig,
    AssetsConfig,
)
from openpi.training import optimizer as _optimizer
from openpi.training import weight_loaders


@dataclasses.dataclass(frozen=True)
class DepthCheckpointWeightLoader(weight_loaders.WeightLoader):
    """Loads a checkpoint and initializes missing depth projector weights."""

    params_path: str

    def load(self, params: at.Params) -> at.Params:
        loaded_params = _model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)
        # Merge in any missing params (e.g., depth_projector, LoRA) from current init.
        return weight_loaders._merge_params(loaded_params, params, missing_regex=".*(lora|depth_projector).*")


ModelType: TypeAlias = _model.ModelType


@dataclasses.dataclass(frozen=True)
class LeRobotLiberoDepthDataConfig(DataConfigFactory):
    """Configuration for Libero dataset with depth support.
    
    This config is designed for the modified_libero_rlds_cotdep dataset
    which includes depth maps alongside RGB images.
    """
    
    # Whether to normalize depth values (disabled by default to match 3dcavla)
    normalize_depth: bool = False

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # Repack transform maps dataset keys to expected keys
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "image",
                        "observation/wrist_image": "wrist_image",
                        "observation/depth": "depth",  # Add depth mapping
                        "observation/state": "state",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        # Data transforms with depth support
        data_transforms = _transforms.Group(
            inputs=[libero_depth_policy.LiberoDepthInputs(
                model_type=model_config.model_type,
                normalize_depth=self.normalize_depth,
            )],
            outputs=[libero_depth_policy.LiberoDepthOutputs()],
        )

        # Model transforms (tokenization, etc.)
        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


# Pre-configured training configs for Libero with depth
def get_pi05_libero_depth_config() -> TrainConfig:
    """Get training configuration for Pi0.5 on Libero with depth."""
    return TrainConfig(
        name="pi05_libero_depth",
        model=pi0_depth_config.Pi0DepthConfig(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            use_depth=True,
            depth_encoder="dformer",
            dformer_config="local_configs.NYUDepthv2.DFormerv2_B",
            dformer_checkpoint="/data_all/gzr1/openpi_onlyrgbd/checkpoints/RGBDProjector/DFormerv2_Base_NYU.pth",
            dformer_train_backbone=True,
        ),
        data=LeRobotLiberoDepthDataConfig(
            repo_id="local/libero_spatial_depth",  # Your depth dataset repo
            base_config=DataConfig(prompt_from_task=True),
            normalize_depth=False,
        ),
        batch_size=64,  # Reduced batch size to avoid OOM with depth processing
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=DepthCheckpointWeightLoader("/data_all/gzr1/openpi_onlyrgbd/checkpoints/pi05_libero/params"),
        num_train_steps=30_000,
    )


def get_pi05_libero_depth_lora_config() -> TrainConfig:
    """Get LoRA training configuration for Pi0.5 on Libero with depth.
    
    Uses LoRA for the main model while fully training the depth projector.
    """
    return TrainConfig(
        name="pi05_libero_depth_lora",
        model=pi0_depth_config.Pi0DepthConfig(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            use_depth=True,
            paligemma_variant="gemma_2b_lora",  # Use LoRA for main model
            depth_encoder="dformer",
            dformer_config="local_configs.NYUDepthv2.DFormerv2_B",
            dformer_checkpoint="/data_all/gzr1/openpi_onlyrgbd/checkpoints/RGBDProjector/DFormerv2_Base_NYU.pth",
            dformer_train_backbone=False,
        ),
        data=LeRobotLiberoDepthDataConfig(
            repo_id="local/libero_spatial_depth",
            base_config=DataConfig(prompt_from_task=True),
            normalize_depth=False,
        ),
        batch_size=4,  # LoRA batch size
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=5_000,
            peak_lr=1e-4,  # Higher LR for LoRA
            decay_steps=500_000,
            decay_lr=1e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=None,  # No EMA for LoRA
        freeze_filter=pi0_depth_config.Pi0DepthConfig(
            pi05=True,
            action_horizon=10,
            paligemma_variant="gemma_2b_lora",
        ).get_freeze_filter(),
        weight_loader=DepthCheckpointWeightLoader("/data_all/gzr1/openpi_onlyrgbd/checkpoints/pi05_libero/params"),
        num_train_steps=30_000,
    )


# Export all depth configs
LIBERO_DEPTH_CONFIGS = [
    get_pi05_libero_depth_config(),
    get_pi05_libero_depth_lora_config(),
]
