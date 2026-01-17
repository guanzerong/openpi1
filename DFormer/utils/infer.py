import argparse
import importlib
import os
import sys
from typing import Dict, Iterable, Tuple

import numpy as np
import torch

try:
    import cv2
except ImportError as exc:
    raise ImportError("OpenCV (cv2) is required to run infer.py. Please install it, e.g. `pip install opencv-python`.") from exc


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


from models.encoders import DFormerv2  # noqa: E402
from utils.dataloader.dataloader import ValPre  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract multi-stage features from a DFormerV2 backbone given RGB and depth inputs."
    )
    parser.add_argument("--config",default='local_configs.NYUDepthv2.DFormerv2_B' ,required=True, help="Config module path, e.g. local_configs.NYUDepthv2.DFormerv2_S")
    parser.add_argument("--checkpoint", default='/data_all/gzr1/openpi_onlyrgbd/checkpoints/RGBDProjector/DFormerv2_Base_NYU.pth', help="Checkpoint path. Uses config.pretrained_model if omitted.")
    parser.add_argument("--rgb", required=True, help="Path to the RGB image.")
    parser.add_argument("--depth", required=True, help="Path to the depth map (single or three channel).")
    parser.add_argument(
        "--rgb-color",
        default="BGR",
        choices=["RGB", "BGR"],
        help="Color space used when reading the RGB image (default: %(default)s).",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Torch device to run inference on."
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(REPO_ROOT, "outputs", "features"),
        help="Directory to store feature files.",
    )
    parser.add_argument(
        "--output-name",
        default=None,
        help="Base name for the saved feature file (defaults to RGB filename stem).",
    )
    parser.add_argument(
        "--save-format",
        default="npz",
        choices=["npz", "pt"],
        help="Serialization format for features: .npz (NumPy) or .pt (PyTorch).",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing feature file if present.")
    return parser.parse_args()


def load_config(config_str: str):
    cfg_module = importlib.import_module(config_str)
    cfg = getattr(cfg_module, "C")

    # Ensure required attributes exist for preprocessing.
    if not hasattr(cfg, "pad"):
        cfg.pad = False
    if not hasattr(cfg, "x_is_single_channel"):
        cfg.x_is_single_channel = True
    if not hasattr(cfg, "norm_mean") or not hasattr(cfg, "norm_std"):
        raise AttributeError("Config must define norm_mean and norm_std for preprocessing.")
    if not hasattr(cfg, "backbone"):
        raise AttributeError("Config must define backbone (e.g., DFormerv2_S).")
    return cfg


def resolve_checkpoint(args: argparse.Namespace, cfg) -> str:
    candidate = args.checkpoint or getattr(cfg, "pretrained_model", None)
    if candidate is None:
        raise ValueError("No checkpoint specified. Provide --checkpoint or set cfg.pretrained_model.")

    if not os.path.isabs(candidate):
        candidate = os.path.join(REPO_ROOT, candidate)
    if not os.path.exists(candidate):
        raise FileNotFoundError(f"Checkpoint not found at {candidate}")
    return candidate


def build_backbone(cfg) -> torch.nn.Module:
    backbone_name = cfg.backbone
    if not backbone_name.startswith("DFormerv2"):
        raise ValueError(f"Expected a DFormerV2 backbone, got {backbone_name}")

    constructor = getattr(DFormerv2, backbone_name, None)
    if constructor is None:
        raise RuntimeError(f"Backbone {backbone_name} is not defined in models.encoders.DFormerv2")

    model = constructor()
    return model


def load_checkpoint(model: torch.nn.Module, path: str) -> None:
    if hasattr(model, "init_weights"):
        try:
            model.init_weights(pretrained=path)
            return
        except Exception:
            print(f"[Info] Falling back to torch.load for checkpoint {path}")

    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict):
        if "state_dict" in state:
            state = state["state_dict"]
        elif "model" in state:
            state = state["model"]

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[Warning] Missing keys while loading checkpoint: {missing}")
    if unexpected:
        print(f"[Warning] Unexpected keys while loading checkpoint: {unexpected}")


def read_rgb(path: str, mode: str) -> np.ndarray:
    flag = cv2.IMREAD_UNCHANGED
    img = cv2.imread(path, flag)
    if img is None:
        raise FileNotFoundError(f"Failed to read RGB image at {path}")

    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    if mode.upper() == "RGB":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def read_depth(path: str) -> np.ndarray:
    depth = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise FileNotFoundError(f"Failed to read depth map at {path}")

    if depth.ndim == 2:
        depth = cv2.merge([depth, depth, depth])
    elif depth.shape[2] == 1:
        depth = np.repeat(depth, 3, axis=2)
    return depth


def preprocess(rgb: np.ndarray, depth: np.ndarray, cfg) -> Tuple[torch.Tensor, torch.Tensor]:
    gt_dummy = np.zeros(rgb.shape[:2], dtype=np.uint8)
    preprocessor = ValPre(cfg.norm_mean, cfg.norm_std, cfg.x_is_single_channel, cfg)
    rgb_norm, _, depth_norm = preprocessor(rgb, gt_dummy, depth)
    rgb_tensor = torch.from_numpy(np.ascontiguousarray(rgb_norm)).unsqueeze(0).float()
    depth_tensor = torch.from_numpy(np.ascontiguousarray(depth_norm)).unsqueeze(0).float()
    return rgb_tensor, depth_tensor


def save_features(
    features: Iterable[torch.Tensor], out_dir: str, base_name: str, fmt: str, overwrite: bool
) -> str:
    os.makedirs(out_dir, exist_ok=True)
    suffix = ".npz" if fmt == "npz" else ".pt"
    output_path = os.path.join(out_dir, base_name + suffix)

    if os.path.exists(output_path) and not overwrite:
        raise FileExistsError(f"{output_path} already exists. Use --overwrite to replace it.")

    cpu_feats = [feat.detach().cpu() for feat in features]
    if fmt == "npz":
        arrays: Dict[str, np.ndarray] = {f"stage{i}": f.numpy() for i, f in enumerate(cpu_feats)}
        np.savez_compressed(output_path, **arrays)
    else:
        torch.save({f"stage{i}": f for i, f in enumerate(cpu_feats)}, output_path)
    return output_path


def main():
    args = parse_args()
    cfg = load_config(args.config)
    checkpoint_path = resolve_checkpoint(args, cfg)

    model = build_backbone(cfg)
    load_checkpoint(model, checkpoint_path)
    device = torch.device(args.device)
    model.to(device)
    model.eval()

    rgb_img = read_rgb(args.rgb, args.rgb_color)
    depth_img = read_depth(args.depth)
    rgb_tensor, depth_tensor = preprocess(rgb_img, depth_img, cfg)

    rgb_tensor = rgb_tensor.to(device)
    depth_tensor = depth_tensor.to(device)

    with torch.no_grad():
        features = model(rgb_tensor, depth_tensor)
        if not isinstance(features, (tuple, list)):
            features = (features,)

    base_name = args.output_name or os.path.splitext(os.path.basename(args.rgb))[0]
    output_path = save_features(features, args.output_dir, base_name, args.save_format, args.overwrite)

    print(f"Saved {len(features)} stage features to {output_path}")
    for idx, feat in enumerate(features):
        shape = tuple(feat.shape)
        print(f"  stage{idx}: {shape}")


if __name__ == "__main__":
    main()
