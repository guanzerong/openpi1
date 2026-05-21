import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class PiperInputs(transforms.DataTransformFn):
    """Repack Piper three-camera observations for pi0 / pi05 training."""

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        top_image = _parse_image(data["observation/top_image"])
        wrist_image = _parse_image(data["observation/wrist_image"])
        down_image = _parse_image(data["observation/down_image"])

        match self.model_type:
            case _model.ModelType.PI0 | _model.ModelType.PI05:
                image_names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
            case _model.ModelType.PI0_FAST:
                image_names = ("base_0_rgb", "base_1_rgb", "wrist_0_rgb")
            case _:
                raise ValueError(f"Unsupported model type: {self.model_type}")

        inputs = {
            "state": np.asarray(data["observation/state"]),
            "image": dict(
                zip(
                    image_names,
                    (top_image, wrist_image, down_image),
                    strict=True,
                )
            ),
            "image_mask": {name: np.True_ for name in image_names},
        }

        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"])

        if "prompt" in data:
            prompt = data["prompt"]
            if isinstance(prompt, bytes):
                prompt = prompt.decode("utf-8")
            inputs["prompt"] = prompt

        return inputs


@dataclasses.dataclass(frozen=True)
class PiperOutputs(transforms.DataTransformFn):
    action_dim: int = 7

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][..., : self.action_dim])}
