from pathlib import Path
import sys

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from examples.libero.convert_robomimic_hdf5_to_lerobot import convert_hdf5_file


def main() -> None:
    output_path = Path("/data_all/gzr1/openpi/datasets/robomimic_ph__lift_can_square_toolhang_lerobot")
    hdf5_path = Path("/data_all/gzr1/robomimic/converted_rgbd/robomimic_ph_224/tool_hang_rgbd_224_v15.hdf5")

    dataset = LeRobotDataset(repo_id="robomimic_ph__lift_can_square_toolhang_lerobot", root=output_path)
    dataset.start_image_writer(num_processes=5, num_threads=10)
    try:
        convert_hdf5_file(dataset, hdf5_path, include_depth=False, max_episodes=None)
    finally:
        dataset.stop_image_writer()


if __name__ == "__main__":
    main()
