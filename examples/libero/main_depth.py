"""LIBERO evaluation script with depth support.

This script extends the standard LIBERO evaluation to include depth map inputs
for models trained with depth support (Pi0Depth).
"""
import collections
import dataclasses
import logging
import math
import pathlib

import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tqdm
import tyro

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data


@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = (
        "libero_spatial"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    )
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 50  # Number of rollouts per task

    #################################################################################################################
    # Depth settings
    #################################################################################################################
    use_depth: bool = True  # Whether to include depth in observations
    depth_resolution: int = 256  # Depth image resolution (should match training)
    normalize_depth: bool = True  # Whether to normalize depth values

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "data/libero/videos"  # Path to save videos
    save_depth_videos: bool = False  # Whether to save depth visualization videos

    seed: int = 7  # Random Seed (for reproducibility)


def normalize_depth_image(depth: np.ndarray, min_depth: float = 0.0, max_depth: float = 10.0) -> np.ndarray:
    """Normalize depth values to [0, 1] range."""
    depth = np.clip(depth, min_depth, max_depth)
    depth = (depth - min_depth) / (max_depth - min_depth)
    return depth.astype(np.float32)


def get_depth_from_env(env, camera_name: str = "agentview") -> np.ndarray:
    """Extract depth map from LIBERO environment.
    
    Args:
        env: LIBERO environment instance
        camera_name: Name of the camera to get depth from
        
    Returns:
        Depth map as float32 array of shape (H, W)
    """
    # Get the simulation object
    sim = env.env.sim
    
    # Get camera ID
    camera_id = sim.model.camera_name2id(camera_name)
    
    # Render depth
    # The depth buffer gives distances from the near clipping plane
    depth = sim.render(
        height=LIBERO_ENV_RESOLUTION,
        width=LIBERO_ENV_RESOLUTION,
        camera_name=camera_name,
        depth=True
    )
    
    # If render returns both rgb and depth, extract depth
    if isinstance(depth, tuple):
        depth = depth[1]
    
    # Convert to meters (if necessary) and flip to match image orientation
    depth = np.ascontiguousarray(depth[::-1, ::-1])
    
    return depth.astype(np.float32)


def eval_libero_depth(args: Args) -> None:
    """Evaluate LIBERO with depth input support."""
    # Set random seed
    np.random.seed(args.seed)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}")
    logging.info(f"Using depth input: {args.use_depth}")

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    # Set max steps based on task suite
    if args.task_suite_name == "libero_spatial":
        max_steps = 220
    elif args.task_suite_name == "libero_object":
        max_steps = 280
    elif args.task_suite_name == "libero_goal":
        max_steps = 300
    elif args.task_suite_name == "libero_10":
        max_steps = 520
    elif args.task_suite_name == "libero_90":
        max_steps = 400
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    # Start evaluation
    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        # Start episodes
        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
            logging.info(f"\nTask: {task_description}")

            # Reset environment
            env.reset()
            action_plan = collections.deque()

            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])

            # Setup
            t = 0
            replay_images = []
            replay_depths = []

            logging.info(f"Starting episode {task_episodes+1}...")
            while t < max_steps + args.num_steps_wait:
                try:
                    # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                    if t < args.num_steps_wait:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue

                    # Get preprocessed RGB images
                    # IMPORTANT: rotate 180 degrees to match train preprocessing
                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
                    )
                    wrist_img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                    )

                    # Save preprocessed image for replay video
                    replay_images.append(img)

                    # Get depth map if enabled
                    depth = None
                    if args.use_depth:
                        try:
                            depth = get_depth_from_env(env, camera_name="agentview")
                            if args.normalize_depth:
                                depth = normalize_depth_image(depth)
                            if args.save_depth_videos:
                                # Convert to visualization format
                                depth_vis = (depth * 255).astype(np.uint8)
                                replay_depths.append(depth_vis)
                        except Exception as e:
                            logging.warning(f"Failed to get depth: {e}")
                            depth = None

                    if not action_plan:
                        # Finished executing previous action chunk -- compute new chunk
                        # Prepare observations dict
                        element = {
                            "observation/image": img,
                            "observation/wrist_image": wrist_img,
                            "observation/state": np.concatenate(
                                (
                                    obs["robot0_eef_pos"],
                                    _quat2axisangle(obs["robot0_eef_quat"]),
                                    obs["robot0_gripper_qpos"],
                                )
                            ),
                            "prompt": str(task_description),
                        }

                        # Add depth if available
                        if args.use_depth and depth is not None:
                            element["observation/depth"] = depth

                        # Query model to get action
                        action_chunk = client.infer(element)["actions"]
                        assert (
                            len(action_chunk) >= args.replan_steps
                        ), f"We want to replan every {args.replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
                        action_plan.extend(action_chunk[: args.replan_steps])

                    action = action_plan.popleft()

                    # Execute action in environment
                    obs, reward, done, info = env.step(action.tolist())
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1

                except Exception as e:
                    logging.error(f"Caught exception: {e}")
                    break

            task_episodes += 1
            total_episodes += 1

            # Save replay videos
            suffix = "success" if done else "failure"
            task_segment = task_description.replace(" ", "_")
            
            # Save RGB video
            imageio.mimwrite(
                pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_{suffix}.mp4",
                [np.asarray(x) for x in replay_images],
                fps=10,
            )
            
            # Save depth video if enabled
            if args.save_depth_videos and replay_depths:
                imageio.mimwrite(
                    pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_{suffix}_depth.mp4",
                    [np.asarray(x) for x in replay_depths],
                    fps=10,
                )

            # Log current results
            logging.info(f"Success: {done}")
            logging.info(f"# episodes completed so far: {total_episodes}")
            logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")

        # Log final results
        logging.info(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        logging.info(f"Current total success rate: {float(total_successes) / float(total_episodes)}")

    logging.info(f"Total success rate: {float(total_successes) / float(total_episodes)}")
    logging.info(f"Total episodes: {total_episodes}")


def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task_description


def _quat2axisangle(quat):
    """Convert quaternion to axis-angle representation."""
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_libero_depth)
