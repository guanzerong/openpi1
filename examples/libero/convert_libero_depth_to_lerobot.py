"""
Convert Libero dataset with depth maps to LeRobot format for OpenPi.

This script converts the modified_libero_rlds_cotdep dataset (which includes depth maps)
to LeRobot format for training OpenPi with depth support.

Usage:
    cd /data_all/gzr1/openpi_onlyrgbd
    uv run examples/libero/convert_libero_depth_to_lerobot.py \
        --data-dir /data_all/gzr1/datasets/modified_libero_rlds_cotdep \
        --task-suite libero_spatial_cotdep \
        --repo-id local/libero_spatial_depth

The dataset will be saved to: $HF_LEROBOT_HOME/{repo_id}
(Default HF_LEROBOT_HOME is ~/.cache/huggingface/lerobot)

For training, update libero_depth_config.py to use matching repo_id:
    data=LeRobotLiberoDepthDataConfig(
        repo_id="local/libero_spatial_depth",  # Must match --repo-id
        base_config=DataConfig(prompt_from_task=True),
    )

Output features (compatible with LeRobotLiberoDepthDataConfig's RepackTransform):
- image: RGB image from third-person camera (256x256x3)
- wrist_image: RGB image from wrist camera (256x256x3)
- depth: Depth map from third-person camera (256x256, float32)
- state: Robot proprioceptive state (8,)
- actions: Robot actions (7,)
- task: Task description -> becomes task_index (used with prompt_from_task=True)

Note: to run the script, you need tensorflow_datasets:
    uv pip install tensorflow tensorflow_datasets
"""

import os
import shutil
from pathlib import Path

from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset
import numpy as np
import tensorflow_datasets as tfds
import tyro


# Task suites available in modified_libero_rlds_cotdep
TASK_SUITES = [
    "libero_spatial_cotdep",
    "libero_object_cotdep", 
    "libero_goal_cotdep",
    "libero_10_cotdep",
]

def main(
    data_dir: str,
    *,
    task_suite: str = "libero_spatial_cotdep",
    repo_id: str = "local/libero_spatial_depth",
    push_to_hub: bool = False,
    max_episodes: int | None = None,
):
    """
    Convert Libero depth dataset to LeRobot format for OpenPi training.
    
    Args:
        data_dir: Path to the modified_libero_rlds_cotdep directory containing RLDS data
        task_suite: Which task suite to convert (e.g., "libero_spatial_cotdep")
        repo_id: Repository ID for output dataset (also determines save location)
        push_to_hub: Whether to push to Hugging Face Hub after conversion
        max_episodes: Maximum number of episodes to convert (for testing; None = all)
    """
    # Validate task suite
    if task_suite not in TASK_SUITES:
        raise ValueError(f"task_suite must be one of {TASK_SUITES}, got '{task_suite}'")
    
    # Output path will be HF_LEROBOT_HOME / repo_id
    output_path = HF_LEROBOT_HOME / repo_id
    print(f"Output path: {output_path}")
    
    # Clean up any existing dataset
    if output_path.exists():
        print(f"Removing existing dataset at {output_path}")
        shutil.rmtree(output_path)
    
    # Create LeRobot dataset with features matching OpenPi's expectations
    # Field names must match official convert_libero_data_to_lerobot.py
    # The RepackTransform in libero_depth_config.py will map these to observation/* keys
    print("Creating LeRobot dataset...")
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        robot_type="panda",
        fps=10,
        features={
            # RGB image from third-person camera
            "image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            # RGB image from wrist camera
            "wrist_image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            # Depth map from third-person camera (float32, 2D)
            "depth": {
                "dtype": "float32",
                "shape": (256, 256),
                "names": ["height", "width"],
            },
            # Robot proprioceptive state
            "state": {
                "dtype": "float32",
                "shape": (8,),
                "names": ["state"],
            },
            # Robot actions
            "actions": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["actions"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )
    
    # Load RLDS dataset
    # The dataset is stored in: data_dir/task_suite/1.0.0/
    dataset_path = os.path.join(data_dir, task_suite, "1.0.0")
    if not os.path.exists(dataset_path):
        # Try without version subdirectory
        dataset_path = os.path.join(data_dir, task_suite)
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(
                f"Dataset not found at {data_dir}/{task_suite}/1.0.0 or {data_dir}/{task_suite}"
            )
    
    print(f"Loading RLDS dataset from: {dataset_path}")
    builder = tfds.builder_from_directory(dataset_path)
    rlds_dataset = builder.as_dataset(split='train')
    
    # Convert episodes
    print(f"Converting {task_suite}...")
    episode_count = 0
    total_frames = 0
    
    for episode in rlds_dataset:
        if max_episodes is not None and episode_count >= max_episodes:
            break
            
        # Process each step in episode
        for step in episode["steps"].as_numpy_iterator():
            obs = step["observation"]
            
            # Extract RGB images (uint8, HWC format)
            image = obs["image"]  # (256, 256, 3)
            wrist_image = obs["wrist_image"]  # (256, 256, 3)
            
            # Extract depth map and ensure it's 2D float32
            depth = obs["depth"]  # (256, 256, 1) or (256, 256)
            if depth.ndim == 3 and depth.shape[-1] == 1:
                depth = np.squeeze(depth, axis=-1)  # -> (256, 256)
            depth = depth.astype(np.float32)
            
            # Extract state and action
            state = obs["state"].astype(np.float32)  # (8,)
            action = step["action"].astype(np.float32)  # (7,)
            
            # Extract language instruction
            language = step["language_instruction"]
            if isinstance(language, bytes):
                language = language.decode("utf-8")
            
            # Add frame to dataset
            # NOTE: "task" field is automatically converted to task_index by LeRobot
            # When training with prompt_from_task=True, this becomes the prompt
            dataset.add_frame({
                "image": image,
                "wrist_image": wrist_image,
                "depth": depth,
                "state": state,
                "actions": action,
                "task": language,
            })
            total_frames += 1
        
        # Save episode
        dataset.save_episode()
        episode_count += 1
        
        if episode_count % 10 == 0:
            print(f"  Processed {episode_count} episodes, {total_frames} frames")
    
    # CRITICAL: Consolidate dataset to finalize parquet files and metadata
    print("Consolidating dataset...")
    dataset.consolidate()
    
    print(f"\n{'='*60}")
    print(f"Conversion complete!")
    print(f"  Output: {output_path}")
    print(f"  Episodes: {episode_count}")
    print(f"  Frames: {total_frames}")
    print(f"\nTo use this dataset for training, set in your config:")
    print(f'  repo_id="{repo_id}"')
    print(f"{'='*60}")
    
    # Optionally push to Hugging Face Hub
    if push_to_hub:
        print("\nPushing to Hugging Face Hub...")
        dataset.push_to_hub(
            tags=["libero", "panda", "rlds", "depth"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )


if __name__ == "__main__":
    tyro.cli(main)
