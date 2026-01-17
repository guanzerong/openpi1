"""
Convert Libero depth dataset to LeRobot format.

Usage:
cd /data_all/gzr1/openpi_onlyrgbd
uv run examples/libero/convert_depth_to_lerobot.py --data_dir /data_all/gzr1/datasets/modified_libero_rlds_cotdep

Note: to run the script, you need to install tensorflow_datasets:
uv pip install tensorflow tensorflow_datasets
"""

import os
import shutil

from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import tensorflow_datasets as tfds
import tyro

REPO_NAME = "local/libero_depth"
RAW_DATASET_NAMES = [
    "libero_spatial_cotdep",
    "libero_object_cotdep", 
    "libero_goal_cotdep",
    "libero_10_cotdep",
]


def main(data_dir: str, *, push_to_hub: bool = False, single_dataset: str | None = None):
    """
    Convert Libero depth dataset to LeRobot format.
    
    Args:
        data_dir: Path to the modified_libero_rlds_cotdep directory
        push_to_hub: Whether to push to Hugging Face Hub
        single_dataset: If specified, only convert this single dataset
    """
    output_path = HF_LEROBOT_HOME / REPO_NAME
    if output_path.exists():
        shutil.rmtree(output_path)

    # Create LeRobot dataset with depth support
    dataset = LeRobotDataset.create(
        repo_id=REPO_NAME,
        robot_type="panda",
        fps=10,
        features={
            "image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "depth": {
                "dtype": "float32",
                "shape": (256, 256),
                "names": ["height", "width"],
            },
            "state": {
                "dtype": "float32",
                "shape": (8,),
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["actions"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    datasets_to_convert = [single_dataset] if single_dataset else RAW_DATASET_NAMES
    
    for raw_dataset_name in datasets_to_convert:
        print(f"Converting {raw_dataset_name}...")
        
        # Find version directory
        dataset_path = os.path.join(data_dir, raw_dataset_name)
        if not os.path.isdir(dataset_path):
            print(f"  Directory not found: {dataset_path}")
            continue
            
        subdirs = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
        version_dirs = [d for d in subdirs if d[0].isdigit()]
        
        if version_dirs:
            version_path = os.path.join(dataset_path, version_dirs[0])
            print(f"  Using version: {version_path}")
            builder = tfds.builder_from_directory(version_path)
            raw_dataset = builder.as_dataset(split='train')
        else:
            try:
                raw_dataset = tfds.load(raw_dataset_name, data_dir=data_dir, split="train")
            except Exception as e:
                print(f"  Failed: {e}")
                continue
        
        episode_count = 0
        for episode in raw_dataset:
            for step in episode["steps"].as_numpy_iterator():
                obs = step["observation"]
                
                # Process depth (squeeze if needed)
                depth = obs["depth"]
                if len(depth.shape) == 3 and depth.shape[-1] == 1:
                    depth = np.squeeze(depth, axis=-1)
                
                dataset.add_frame({
                    "image": obs["image"],
                    "wrist_image": obs["wrist_image"],
                    "depth": depth.astype(np.float32),
                    "state": obs["state"],
                    "actions": step["action"],
                    "task": step["language_instruction"].decode(),
                })
            dataset.save_episode()
            episode_count += 1
            if episode_count % 10 == 0:
                print(f"    {episode_count} episodes...")
        
        print(f"  Done: {episode_count} episodes")

    print(f"\nDataset saved to {output_path}")

    if push_to_hub:
        dataset.push_to_hub(
            tags=["libero", "panda", "rlds", "depth"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )


if __name__ == "__main__":
    tyro.cli(main)
