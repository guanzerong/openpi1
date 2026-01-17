"""Configuration for Pi0 model with depth support."""
import dataclasses
from typing import TYPE_CHECKING

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0_config
from openpi.models.depth_projector import DepthProjectorConfig
import openpi.models.gemma as _gemma
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils

if TYPE_CHECKING:
    from openpi.models.pi0_depth import Pi0Depth


# Default camera intrinsics for Libero (scaled to 224x224)
DEFAULT_DEPTH_CONFIG = DepthProjectorConfig(
    height=224,
    width=224,
    fx=270.39,
    fy=270.39,
    cx=112.0,
    cy=112.0,
    output_dim=2048,  # Match gemma_2b embedding dimension
)


@dataclasses.dataclass(frozen=True)
class Pi0DepthConfig(pi0_config.Pi0Config):
    """Configuration for Pi0 model with depth support.
    
    Extends Pi0Config with depth-specific parameters.
    """
    # Whether to use depth input
    use_depth: bool = True
    # Depth projector configuration
    depth_config: DepthProjectorConfig = dataclasses.field(default_factory=lambda: DEFAULT_DEPTH_CONFIG)
    # Depth image resolution (H, W) - for Libero this is typically 256x256
    depth_resolution: tuple[int, int] = (224, 224)  # Match IMAGE_RESOLUTION
    # Depth feature extractor type ("pointnet" or "dformer")
    depth_encoder: str = "pointnet"
    # DFormer configuration (PyTorch only)
    dformer_config: str = "local_configs.NYUDepthv2.DFormerv2_B"
    dformer_checkpoint: str | None = "/data_all/gzr1/openpi_onlyrgbd/checkpoints/RGBDProjector/DFormerv2_Base_NYU.pth"
    dformer_train_backbone: bool = False
    dformer_projector_hidden_dim: int = 0
    dformer_force_val_size: tuple[int, int] | None = None

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0Depth":
        from openpi.models.pi0_depth import Pi0Depth

        return Pi0Depth(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        """Returns the input specification including depth."""
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        # Depth specification
        depth_spec = jax.ShapeDtypeStruct([batch_size, *self.depth_resolution], jnp.float32)
        depth_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        with at.disable_typechecking():
            observation_spec = _model.Observation(
                images={
                    "base_0_rgb": image_spec,
                    "left_wrist_0_rgb": image_spec,
                    "right_wrist_0_rgb": image_spec,
                },
                image_masks={
                    "base_0_rgb": image_mask_spec,
                    "left_wrist_0_rgb": image_mask_spec,
                    "right_wrist_0_rgb": image_mask_spec,
                },
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
                depth=depth_spec if self.use_depth else None,
                depth_mask=depth_mask_spec if self.use_depth else None,
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)

        return observation_spec, action_spec

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Returns the freeze filter including depth projector handling.
        
        By default, the depth projector is trained (not frozen).
        """
        base_filter = super().get_freeze_filter()
        
        # Optionally freeze or unfreeze depth projector
        # For now, we keep depth projector trainable by default
        return base_filter
