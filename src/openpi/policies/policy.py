from collections.abc import Sequence
import logging
import pathlib
import time
from typing import Any, TypeAlias

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
import torch
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

BasePolicy: TypeAlias = _base_policy.BasePolicy


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        pytorch_device: str = "cpu",
        is_pytorch: bool = False,
    ):
        """Initialize the Policy.

        Args:
            model: The model to use for action sampling.
            rng: Random number generator key for JAX models. Ignored for PyTorch models.
            transforms: Input data transformations to apply before inference.
            output_transforms: Output data transformations to apply after inference.
            sample_kwargs: Additional keyword arguments to pass to model.sample_actions.
            metadata: Additional metadata to store with the policy.
            pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda:0").
                          Only relevant when is_pytorch=True.
            is_pytorch: Whether the model is a PyTorch model. If False, assumes JAX model.
        """
        self._model = model
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._is_pytorch_model = is_pytorch
        self._pytorch_device = pytorch_device

        if self._is_pytorch_model:
            self._model = self._model.to(pytorch_device)
            self._model.eval()
            self._sample_actions = model.sample_actions
            self._encode_observation = getattr(model, "encode_observation_features", None)
        else:
            # JAX model setup
            self._sample_actions = nnx_utils.module_jit(model.sample_actions)
            self._encode_observation = (
                nnx_utils.module_jit(model.encode_observation_features)
                if hasattr(model, "encode_observation_features")
                else None
            )
            self._rng = rng or jax.random.key(0)

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        inputs = self._prepare_single_inputs(obs)
        sample_rng_or_pytorch_device = self._next_sample_device()
        sample_kwargs = self._prepare_sample_kwargs(noise, batched=False)
        outputs, model_time = self._run_action_forward(inputs, sample_rng_or_pytorch_device, sample_kwargs)
        outputs = jax.tree.map(lambda x: x[0, ...], outputs)
        outputs = self._output_transform(outputs)
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
        return outputs

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata

    def encode_observation_features(self, obs: dict) -> np.ndarray:
        """Encode observations into contextualized prefix tokens."""
        if self._encode_observation is None:
            raise NotImplementedError("This policy does not expose observation feature encoding.")

        inputs = self._prepare_single_inputs(obs)
        encoded = self._run_encode_forward(inputs)
        return np.asarray(encoded[0, ...], dtype=np.float32)

    def infer_and_encode(self, obs: dict, *, noise: np.ndarray | None = None) -> dict[str, np.ndarray]:
        """Run policy inference and observation encoding on the same observation."""
        if self._encode_observation is None:
            raise NotImplementedError("This policy does not expose observation feature encoding.")

        inputs = self._prepare_single_inputs(obs)
        sample_rng_or_pytorch_device = self._next_sample_device()
        sample_kwargs = self._prepare_sample_kwargs(noise, batched=False)
        outputs, observation_features, model_time = self._run_action_and_encode_forward(
            inputs, sample_rng_or_pytorch_device, sample_kwargs
        )
        outputs = jax.tree.map(lambda x: x[0, ...], outputs)
        outputs = self._output_transform(outputs)
        outputs["observation_features"] = np.asarray(observation_features[0, ...], dtype=np.float32)
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
        return outputs

    def infer_and_encode_batch(self, obs: dict, *, noise: np.ndarray | None = None) -> dict[str, np.ndarray]:
        """Run policy inference and observation encoding on a batched observation dict."""
        if self._encode_observation is None:
            raise NotImplementedError("This policy does not expose observation feature encoding.")

        inputs = self._prepare_batched_inputs(obs)
        sample_rng_or_pytorch_device = self._next_sample_device()
        sample_kwargs = self._prepare_sample_kwargs(noise, batched=True)
        outputs, observation_features, model_time = self._run_action_and_encode_forward(
            inputs, sample_rng_or_pytorch_device, sample_kwargs
        )
        outputs = self._output_transform(outputs)
        batched = {
            "actions": outputs["actions"],
            "observation_features": np.asarray(observation_features, dtype=np.float32),
            "policy_timing": {
                "infer_ms": model_time * 1000,
            },
        }
        return batched

    def _prepare_single_inputs(self, obs: dict) -> dict[str, Any]:
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        return self._convert_inputs_to_model_arrays(inputs, batched=False)

    def _prepare_batched_inputs(self, obs: dict) -> dict[str, Any]:
        obs_arrays = jax.tree.map(np.asarray, obs)
        batch_size = int(np.asarray(obs_arrays["observation/state"]).shape[0])
        transformed_samples = []
        for batch_idx in range(batch_size):
            sample_obs = {}
            for key, value in obs_arrays.items():
                if isinstance(value, np.ndarray) and value.ndim > 0 and value.shape[0] == batch_size:
                    sample_obs[key] = value[batch_idx]
                else:
                    sample_obs[key] = value
            transformed_samples.append(self._input_transform(sample_obs))

        batched_inputs = jax.tree.map(lambda *xs: np.stack(xs, axis=0), *transformed_samples)
        return self._convert_inputs_to_model_arrays(batched_inputs, batched=True)

    def _convert_inputs_to_model_arrays(self, inputs: dict[str, Any], *, batched: bool) -> dict[str, Any]:
        if not self._is_pytorch_model:
            if batched:
                return jax.tree.map(lambda x: jnp.asarray(x), inputs)
            return jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
        if batched:
            return jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device), inputs)
        return jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...], inputs)

    def _next_sample_device(self):
        if self._is_pytorch_model:
            return self._pytorch_device
        self._rng, sample_rng = jax.random.split(self._rng)
        return sample_rng

    def _prepare_sample_kwargs(self, noise: np.ndarray | None, *, batched: bool) -> dict[str, Any]:
        sample_kwargs = dict(self._sample_kwargs)
        if noise is None:
            return sample_kwargs

        converted_noise = (
            torch.from_numpy(np.array(noise)).to(self._pytorch_device) if self._is_pytorch_model else jnp.asarray(noise)
        )
        if not batched and converted_noise.ndim == 2:
            converted_noise = converted_noise[None, ...]
        sample_kwargs["noise"] = converted_noise
        return sample_kwargs

    def _run_action_forward(self, inputs: dict[str, Any], sample_rng_or_pytorch_device, sample_kwargs: dict[str, Any]):
        observation = _model.Observation.from_dict(inputs)
        start_time = time.monotonic()
        actions = self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs)
        model_time = time.monotonic() - start_time
        outputs = self._to_numpy_tree({"state": inputs["state"], "actions": actions})
        return outputs, model_time

    def _run_encode_forward(self, inputs: dict[str, Any]) -> np.ndarray:
        observation = _model.Observation.from_dict(inputs)
        encoded = self._encode_observation(observation)
        return self._to_numpy_array(encoded)

    def _run_action_and_encode_forward(
        self,
        inputs: dict[str, Any],
        sample_rng_or_pytorch_device,
        sample_kwargs: dict[str, Any],
    ):
        observation = _model.Observation.from_dict(inputs)
        start_time = time.monotonic()
        actions = self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs)
        observation_features = self._encode_observation(observation)
        model_time = time.monotonic() - start_time
        outputs = self._to_numpy_tree({"state": inputs["state"], "actions": actions})
        return outputs, self._to_numpy_array(observation_features), model_time

    def _to_numpy_tree(self, tree: dict[str, Any]) -> dict[str, np.ndarray]:
        return jax.tree.map(self._to_numpy_array, tree)

    def _to_numpy_array(self, value: Any) -> np.ndarray:
        if self._is_pytorch_model:
            return np.asarray(value.detach().cpu())
        return np.asarray(value)


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results
