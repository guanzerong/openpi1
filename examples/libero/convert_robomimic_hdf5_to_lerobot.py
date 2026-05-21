"""
Convert robomimic RGB / RGBD HDF5 datasets into a LeRobot dataset that openpi can read.

Example usage:

uv run examples/libero/convert_robomimic_hdf5_to_lerobot.py \
  --input-dir /data_all/gzr1/openpi/datasets/robomimic_ph_can_lift_square_hdf5 \
  --repo-id robomimic_ph__lift_can_square \
  --root /data_all/gzr1/openpi/datasets
"""

from __future__ import annotations

import math
from pathlib import Path
import shutil

import h5py
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import tqdm
import tyro


TASK_PROMPTS = {
    "lift": "pick up the object on the table and hold it",
    "can": "pick up the coke can and place it on the correct place",
    "square": "pick a square nut and place it on a rod",
    "tool_hang": "assemble a frame consisting of a base piece and hook piece by inserting the hook into the base, and hang a wrench on the hook",
}

TASK_ORDER = ["lift", "can", "square", "tool_hang"]

PRIMARY_IMAGE_KEYS = ("agentview_image", "sideview_image")
PRIMARY_DEPTH_KEYS = ("agentview_depth", "sideview_depth")


def quat_to_axisangle(quat: np.ndarray) -> np.ndarray:
    """Convert an xyzw quaternion to axis-angle / rotvec."""
    quat = np.asarray(quat, dtype=np.float64).copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)

    den = math.sqrt(max(1.0 - quat[3] * quat[3], 0.0))
    if math.isclose(den, 0.0):
        return np.zeros(3, dtype=np.float32)

    return ((quat[:3] * 2.0 * math.acos(quat[3])) / den).astype(np.float32)


def sorted_demo_names(data_group: h5py.Group) -> list[str]:
    return sorted(data_group.keys(), key=lambda name: int(name.split("_")[-1]))


def task_from_path(path: Path) -> str:
    stem = path.stem
    for task_name in TASK_ORDER:
        if stem.startswith(task_name):
            return task_name
    return stem.split("_")[0]


def gather_hdf5_files(input_dir: Path) -> list[Path]:
    task_to_path: dict[str, Path] = {}
    for path in sorted(input_dir.glob("*.hdf5")):
        task_to_path[task_from_path(path)] = path

    ordered = [task_to_path[task] for task in TASK_ORDER if task in task_to_path]
    leftovers = sorted(path for task, path in task_to_path.items() if task not in TASK_ORDER)
    return ordered + leftovers


def get_first_available(obs: h5py.Group, candidates: tuple[str, ...]) -> str:
    for key in candidates:
        if key in obs:
            return key
    raise KeyError(f"None of the keys exist in obs: {candidates}")


def infer_features(first_file: Path, include_depth: bool) -> dict[str, dict]:
    with h5py.File(first_file, "r") as f:
        demo_name = sorted_demo_names(f["data"])[0]
        obs = f["data"][demo_name]["obs"]
        image_key = get_first_available(obs, PRIMARY_IMAGE_KEYS)
        image_shape = tuple(int(x) for x in obs[image_key].shape[1:])
        wrist_image_shape = tuple(int(x) for x in obs["robot0_eye_in_hand_image"].shape[1:])
        features = {
            "image": {
                "dtype": "image",
                "shape": image_shape,
                "names": ["height", "width", "channel"],
            },
            "wrist_image": {
                "dtype": "image",
                "shape": wrist_image_shape,
                "names": ["height", "width", "channel"],
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
        }
        if include_depth:
            depth_key = get_first_available(obs, PRIMARY_DEPTH_KEYS)
            features["depth"] = {
                "dtype": "float32",
                "shape": tuple(int(x) for x in obs[depth_key].shape[1:]),
                "names": ["height", "width", "channel"],
            }
            features["wrist_depth"] = {
                "dtype": "float32",
                "shape": tuple(int(x) for x in obs["robot0_eye_in_hand_depth"].shape[1:]),
                "names": ["height", "width", "channel"],
            }
        return features


def create_dataset(
    *,
    repo_id: str,
    root: Path,
    first_file: Path,
    fps: int,
    include_depth: bool,
    overwrite: bool,
    image_writer_processes: int,
    image_writer_threads: int,
) -> LeRobotDataset:
    output_path = root / repo_id
    if output_path.exists():
        if not overwrite:
            raise FileExistsError(f"Output dataset already exists: {output_path}")
        shutil.rmtree(output_path)

    return LeRobotDataset.create(
        repo_id=repo_id,
        root=output_path,
        robot_type="panda",
        fps=fps,
        features=infer_features(first_file, include_depth),
        image_writer_processes=image_writer_processes,
        image_writer_threads=image_writer_threads,
    )


def build_frame(obs: h5py.Group, actions: np.ndarray, idx: int, *, task_prompt: str, include_depth: bool) -> dict:
    image_key = get_first_available(obs, PRIMARY_IMAGE_KEYS)
    state = np.concatenate(
        [
            np.asarray(obs["robot0_eef_pos"][idx], dtype=np.float32),
            quat_to_axisangle(obs["robot0_eef_quat"][idx]),
            np.asarray(obs["robot0_gripper_qpos"][idx], dtype=np.float32),
        ]
    )

    frame = {
        "image": np.asarray(obs[image_key][idx], dtype=np.uint8),
        "wrist_image": np.asarray(obs["robot0_eye_in_hand_image"][idx], dtype=np.uint8),
        "state": state,
        "actions": np.asarray(actions[idx], dtype=np.float32),
        "task": task_prompt,
    }
    if include_depth:
        depth_key = get_first_available(obs, PRIMARY_DEPTH_KEYS)
        frame["depth"] = np.asarray(obs[depth_key][idx], dtype=np.float32)
        frame["wrist_depth"] = np.asarray(obs["robot0_eye_in_hand_depth"][idx], dtype=np.float32)
    return frame


def convert_hdf5_file(
    dataset: LeRobotDataset,
    hdf5_path: Path,
    *,
    include_depth: bool,
    max_episodes: int | None,
) -> None:
    task_name = task_from_path(hdf5_path)
    task_prompt = TASK_PROMPTS.get(task_name, task_name)

    with h5py.File(hdf5_path, "r") as f:
        demos = sorted_demo_names(f["data"])
        if max_episodes is not None:
            demos = demos[:max_episodes]
        for demo_name in tqdm.tqdm(demos, desc=f"Converting {task_name}", leave=False):
            demo = f["data"][demo_name]
            obs = demo["obs"]
            actions = demo["actions"]
            num_frames = int(actions.shape[0])

            for idx in range(num_frames):
                dataset.add_frame(
                    build_frame(obs, actions, idx, task_prompt=task_prompt, include_depth=include_depth)
                )
            dataset.save_episode()


def main(
    input_dir: Path,
    *,
    repo_id: str = "robomimic_ph__lift_can_square",
    root: Path = Path("/data_all/gzr1/openpi/datasets"),
    include_depth: bool = False,
    max_episodes_per_file: int | None = None,
    overwrite: bool = True,
    fps: int = 10,
    image_writer_processes: int = 5,
    image_writer_threads: int = 10,
) -> None:
    hdf5_files = gather_hdf5_files(input_dir)
    if not hdf5_files:
        raise FileNotFoundError(f"No HDF5 files found in {input_dir}")

    print("Input files:")
    for path in hdf5_files:
        print(f"  - {path}")

    dataset = create_dataset(
        repo_id=repo_id,
        root=root,
        first_file=hdf5_files[0],
        fps=fps,
        include_depth=include_depth,
        overwrite=overwrite,
        image_writer_processes=image_writer_processes,
        image_writer_threads=image_writer_threads,
    )

    for hdf5_path in hdf5_files:
        convert_hdf5_file(
            dataset,
            hdf5_path,
            include_depth=include_depth,
            max_episodes=max_episodes_per_file,
        )

    print(f"Wrote LeRobot dataset to: {root / repo_id}")


if __name__ == "__main__":
    tyro.cli(main)
