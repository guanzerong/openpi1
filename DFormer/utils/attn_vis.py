import argparse
import importlib
import importlib.util
import math
import os
import sys
from typing import Dict, Tuple

import numpy as np
import torch

try:
    import cv2
except ImportError as exc:
    raise ImportError("OpenCV (cv2) is required to run attn_vis.py. Please install it, e.g. `pip install opencv-python`.") from exc


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


from models.encoders import DFormerv2  # noqa: E402
from utils.dataloader.dataloader import ValPre  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize DFormer attention for a single query token and overlay the heatmap on the input image."
    )
    parser.add_argument("--config", default='/data_all/gzr1/openpi_onlyrgbd/DFormer/local_configs/NYUDepthv2/DFormerv2_B.py', help="Config module path, e.g. local_configs.NYUDepthv2.DFormerv2_S")
    parser.add_argument("--checkpoint", default='/data_all/gzr1/openpi_onlyrgbd/checkpoints/RGBDProjector/DFormerv2_Base_NYU.pth', help="Checkpoint path. Uses config.pretrained_model if omitted.")
    parser.add_argument("--rgb", default='/data_all/gzr1/rgb_256.png', help="Path to the RGB image.")
    parser.add_argument("--depth", default='/data_all/gzr1/depth_256.png', help="Path to the depth map (single or three channel).")
    parser.add_argument(
        "--rgb-color",
        default="BGR",
        choices=["RGB", "BGR"],
        help="Color space used when reading the RGB image (default: %(default)s).",
    )
    parser.add_argument("--stage", type=int, required=True, help="Stage index to visualize (0-based).")
    parser.add_argument("--block", type=int, default=-1, help="Block index within the stage (default: last block).")
    parser.add_argument(
        "--aggregate-blocks",
        action="store_true",
        help="Average attention maps across all blocks in the stage (ignores --block).",
    )
    parser.add_argument(
        "--query",
        required=True,
        help="Query location as 'x,y'. Interpreted in stage coordinates unless --query-space=image.",
    )
    parser.add_argument(
        "--query-space",
        choices=["stage", "image"],
        default="stage",
        help="Whether --query is in stage or image coordinates (default: stage).",
    )
    parser.add_argument("--head", type=int, default=None, help="Attention head index to visualize (default: average).")
    parser.add_argument(
        "--reduce",
        choices=["mean", "max"],
        default="mean",
        help="Reduction across heads when --head is not set (default: %(default)s).",
    )
    parser.add_argument(
        "--attn-mode",
        choices=["full", "no_geo", "geo_only", "triplet"],
        default="full",
        help="Attention type to visualize (default: %(default)s).",
    )
    parser.add_argument("--geo-bias", action="store_true", help="Blend geometry bias into the attention heatmap.")
    parser.add_argument("--geo-weight", type=float, default=0.35, help="Blend weight for geometry bias (0-1).")
    parser.add_argument(
        "--geo-mode",
        choices=["pos", "depth", "fuse"],
        default="fuse",
        help="Geometry bias source (default: %(default)s).",
    )
    parser.add_argument("--geo-pos-weight", type=float, default=0.5, help="Position weight when --geo-mode fuse.")
    parser.add_argument("--geo-depth-weight", type=float, default=0.5, help="Depth weight when --geo-mode fuse.")
    parser.add_argument("--geo-only-vis", action="store_true", help="Render only geometry bias heatmap (no attention).")
    parser.add_argument("--alpha", type=float, default=0.6, help="Blend factor for heatmap overlay (default: %(default)s).")
    parser.add_argument(
        "--colormap",
        default="jet",
        help="OpenCV colormap name, e.g. jet, hot, turbo (default: %(default)s).",
    )
    parser.add_argument(
        "--norm",
        choices=["minmax", "percentile"],
        default="minmax",
        help="Normalization for attention map (default: %(default)s).",
    )
    parser.add_argument("--pmin", type=float, default=2.0, help="Percentile min for --norm percentile (default: %(default)s).")
    parser.add_argument("--pmax", type=float, default=98.0, help="Percentile max for --norm percentile (default: %(default)s).")
    parser.add_argument("--gamma", type=float, default=1.0, help="Gamma correction applied after normalization (default: %(default)s).")
    parser.add_argument(
        "--interp",
        choices=["nearest", "linear"],
        default="nearest",
        help="Upsampling interpolation for attention map (default: %(default)s).",
    )
    parser.add_argument("--no-grid", action="store_true", help="Disable grid overlay.")
    parser.add_argument("--no-marker", action="store_true", help="Disable query marker overlay.")
    parser.add_argument(
        "--grid-color",
        default="255,255,255",
        help="Grid line color as 'b,g,r' (default: %(default)s).",
    )
    parser.add_argument(
        "--marker-color",
        default="255,255,255",
        help="Query marker color as 'b,g,r' (default: %(default)s).",
    )
    parser.add_argument("--output", default=None, help="Output image path. Defaults to outputs/attn_vis/....png")
    parser.add_argument("--save-attn", action="store_true", help="Save raw attention map as .npy next to output.")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Torch device to run inference on."
    )
    return parser.parse_args()


def load_config(config_str: str):
    cfg_module = None
    config_module = config_str
    if os.path.isfile(config_module):
        base_dir = os.path.abspath(REPO_ROOT)
        rel = os.path.relpath(os.path.abspath(config_module), base_dir)
        if rel.endswith('.py'):
            rel = rel[:-3]
        config_module = rel.replace(os.sep, ".")
    else:
        if config_module.endswith('.py'):
            config_module = config_module[:-3]
        config_module = config_module.replace(os.sep, ".")

    cfg_module = importlib.import_module(config_module)
    cfg = getattr(cfg_module, "C")

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


def read_rgb_bgr(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Failed to read RGB image at {path}")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
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


def preprocess(rgb_bgr: np.ndarray, depth: np.ndarray, cfg, rgb_color: str) -> Tuple[torch.Tensor, torch.Tensor]:
    target_size = (256, 256)
    rgb_bgr = cv2.resize(rgb_bgr, target_size, interpolation=cv2.INTER_LINEAR)
    depth = cv2.resize(depth, target_size, interpolation=cv2.INTER_LINEAR)
    gt_dummy = np.zeros(rgb_bgr.shape[:2], dtype=np.uint8)
    rgb_model = rgb_bgr
    if rgb_color.upper() == "RGB":
        rgb_model = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)

    preprocessor = ValPre(cfg.norm_mean, cfg.norm_std, cfg.x_is_single_channel, cfg)
    rgb_norm, _, depth_norm = preprocessor(rgb_model, gt_dummy, depth)
    rgb_tensor = torch.tensor(rgb_norm.tolist(), dtype=torch.float32).unsqueeze(0)
    depth_tensor = torch.tensor(depth_norm.tolist(), dtype=torch.float32).unsqueeze(0)
    return rgb_tensor, depth_tensor


def prepare_vis_image(rgb_bgr: np.ndarray, cfg) -> np.ndarray:
    vis = rgb_bgr
    if cfg.pad:
        pad_h = max(0, 531 - vis.shape[0])
        pad_w = max(0, 730 - vis.shape[1])
        if pad_h > 0 or pad_w > 0:
            vis = cv2.copyMakeBorder(vis, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(0.0, 0.0, 0.0))

    target_size = getattr(cfg, "force_val_size", None)
    if target_size is not None:
        tgt_h, tgt_w = target_size
        vis = cv2.resize(vis, (tgt_w, tgt_h), interpolation=cv2.INTER_LINEAR)
    return vis


def parse_pair(value: str) -> Tuple[int, int]:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 2:
        raise ValueError(f"Expected two comma-separated values, got '{value}'")
    return int(parts[0]), int(parts[1])


def parse_color(value: str) -> Tuple[int, int, int]:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 3:
        raise ValueError(f"Expected color as 'b,g,r', got '{value}'")
    return int(parts[0]), int(parts[1]), int(parts[2])


def resolve_colormap(name: str) -> int:
    name = name.lower()
    mapping: Dict[str, int] = {
        "jet": cv2.COLORMAP_JET,
        "hot": cv2.COLORMAP_HOT,
        "ocean": cv2.COLORMAP_OCEAN,
        "turbo": getattr(cv2, "COLORMAP_TURBO", cv2.COLORMAP_JET),
        "viridis": getattr(cv2, "COLORMAP_VIRIDIS", cv2.COLORMAP_JET),
        "plasma": getattr(cv2, "COLORMAP_PLASMA", cv2.COLORMAP_JET),
        "magma": getattr(cv2, "COLORMAP_MAGMA", cv2.COLORMAP_JET),
    }
    if name not in mapping:
        raise ValueError(f"Unsupported colormap '{name}'. Available: {', '.join(mapping)}")
    return mapping[name]


def enable_attention(model: torch.nn.Module, stage: int, block: int, aggregate: bool):
    if stage < 0 or stage >= len(model.layers):
        raise ValueError(f"Stage index {stage} out of range (0..{len(model.layers) - 1}).")

    layer = model.layers[stage]
    if aggregate:
        for layer_idx, lyr in enumerate(model.layers):
            for blk in lyr.blocks:
                blk.Attention.save_attn = layer_idx == stage
                blk.Attention.saved_attn = None
        return [blk.Attention for blk in layer.blocks]

    if block < 0:
        block = len(layer.blocks) + block
    if block < 0 or block >= len(layer.blocks):
        raise ValueError(f"Block index {block} out of range for stage {stage} (0..{len(layer.blocks) - 1}).")

    for layer_idx, lyr in enumerate(model.layers):
        for block_idx, blk in enumerate(lyr.blocks):
            blk.Attention.save_attn = layer_idx == stage and block_idx == block
            blk.Attention.saved_attn = None

    return [layer.blocks[block].Attention]


def build_attention_map(
    saved_attn,
    stage_h: int,
    stage_w: int,
    qx: int,
    qy: int,
    head: int | None,
    reduce: str,
) -> np.ndarray:
    if isinstance(saved_attn, dict):
        attn_w = saved_attn["w"].detach().cpu()
        attn_h = saved_attn["h"].detach().cpu()
        attn_w_q = attn_w[0, qy, :, qx, :]
        attn_h_q = attn_h[0, qx, :, qy, :]
        attn = attn_h_q[:, :, None] * attn_w_q[:, None, :]
    else:
        attn = saved_attn.detach().cpu()
        num_heads = attn.shape[1]
        q_idx = qy * stage_w + qx
        attn = attn[0, :, q_idx, :].reshape(num_heads, stage_h, stage_w)

    if head is not None:
        if head < 0 or head >= attn.shape[0]:
            raise ValueError(f"Head index {head} out of range (0..{attn.shape[0] - 1}).")
        attn_map = attn[head]
    else:
        attn_map = attn.mean(0) if reduce == "mean" else attn.max(0).values

    attn_map = np.array(attn_map.cpu().tolist(), dtype=np.float32)
    return attn_map


def normalize_attn_map(attn_map: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    if args.norm == "percentile":
        lo = np.percentile(attn_map, args.pmin)
        hi = np.percentile(attn_map, args.pmax)
    else:
        lo = attn_map.min()
        hi = attn_map.max()

    if hi > lo:
        attn_map = (attn_map - lo) / (hi - lo)
    else:
        attn_map = attn_map * 0.0

    if args.gamma != 1.0:
        attn_map = np.clip(attn_map, 0.0, 1.0) ** args.gamma

    return attn_map



def compute_geo_bias_map(
    depth_gray: np.ndarray,
    stage_h: int,
    stage_w: int,
    qx: int,
    qy: int,
    mode: str,
    pos_weight: float,
    depth_weight: float,
) -> np.ndarray:
    depth_small = cv2.resize(depth_gray, (stage_w, stage_h), interpolation=cv2.INTER_LINEAR)
    yy, xx = np.mgrid[0:stage_h, 0:stage_w]
    pos = np.abs(xx - qx) + np.abs(yy - qy)

    if mode == "pos":
        bias = pos
    else:
        depth_q = float(depth_small[qy, qx])
        depth = np.abs(depth_small - depth_q)
        if mode == "depth":
            bias = depth
        else:
            bias = pos_weight * pos + depth_weight * depth

    max_val = float(np.max(bias))
    if max_val > 0:
        bias = max_val - bias
        bias = bias / max_val
    else:
        bias = bias * 0
    return bias

def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    checkpoint_path = resolve_checkpoint(args, cfg)

    model = build_backbone(cfg)
    load_checkpoint(model, checkpoint_path)
    device = torch.device(args.device)
    model.to(device)
    model.eval()

    rgb_bgr = read_rgb_bgr(args.rgb)
    depth = read_depth(args.depth)
    depth_gray = depth
    if depth_gray.ndim == 3:
        depth_gray = cv2.cvtColor(depth_gray, cv2.COLOR_BGR2GRAY)
    rgb_tensor, depth_tensor = preprocess(rgb_bgr, depth, cfg, args.rgb_color)

    rgb_tensor = rgb_tensor.to(device)
    depth_tensor = depth_tensor.to(device)

    attention_modules = enable_attention(model, args.stage, args.block, args.aggregate_blocks)

    def run_with_mode(mode: str):
        for mod in attention_modules:
            mod.attn_mode = mode
            mod.saved_attn = None

        with torch.no_grad():
            feats = model(rgb_tensor, depth_tensor)
            if not isinstance(feats, (tuple, list)):
                feats = (feats,)

        saved = [mod.saved_attn for mod in attention_modules if mod.saved_attn is not None]
        if not saved:
            raise RuntimeError("Attention was not captured. Ensure the stage/block selection is correct.")

        return saved, feats

    if args.attn_mode == "triplet":
        saved_full, features = run_with_mode("full")
    else:
        saved_full, features = run_with_mode(args.attn_mode)

    saved_attn = saved_full[0]

    vis_img = prepare_vis_image(rgb_bgr, cfg)
    vis_h, vis_w = vis_img.shape[:2]

    if args.geo_only_vis:
        stride = 4 * (2 ** args.stage)
        stage_h = math.ceil(vis_h / stride)
        stage_w = math.ceil(vis_w / stride)
        qx_raw, qy_raw = parse_pair(args.query)
        scale_x = vis_w / stage_w
        scale_y = vis_h / stage_h
        if args.query_space == "image":
            qx = int(qx_raw / scale_x)
            qy = int(qy_raw / scale_y)
        else:
            qx, qy = qx_raw, qy_raw
        qx = max(0, min(stage_w - 1, qx))
        qy = max(0, min(stage_h - 1, qy))
        geo_map = compute_geo_bias_map(
            depth_gray,
            stage_h,
            stage_w,
            qx,
            qy,
            args.geo_mode,
            args.geo_pos_weight,
            args.geo_depth_weight,
        )
        geo_map = normalize_attn_map(geo_map, args)
        interp = cv2.INTER_NEAREST if args.interp == "nearest" else cv2.INTER_LINEAR
        attn_up = cv2.resize(geo_map, (vis_w, vis_h), interpolation=interp)
        heat = cv2.applyColorMap((attn_up * 255).astype(np.uint8), resolve_colormap(args.colormap))
        overlay = cv2.addWeighted(heat, args.alpha, vis_img, 1.0 - args.alpha, 0)
        if not args.no_grid:
            grid_color = parse_color(args.grid_color)
            for i in range(1, stage_w):
                x = int(round(i * scale_x))
                cv2.line(overlay, (x, 0), (x, vis_h - 1), grid_color, 1)
            for i in range(1, stage_h):
                y = int(round(i * scale_y))
                cv2.line(overlay, (0, y), (vis_w - 1, y), grid_color, 1)
        if not args.no_marker:
            center_x = int(round((qx + 0.5) * scale_x))
            center_y = int(round((qy + 0.5) * scale_y))
            marker_color = parse_color(args.marker_color)
            marker_size = max(8, int(round(min(scale_x, scale_y))))
            cv2.drawMarker(
                overlay,
                (center_x, center_y),
                marker_color,
                markerType=cv2.MARKER_STAR,
                markerSize=marker_size,
                thickness=2,
            )
        if args.output:
            output_path = args.output
        else:
            base_name = os.path.splitext(os.path.basename(args.rgb))[0]
            output_dir = os.path.join(REPO_ROOT, "outputs", "attn_vis")
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"{base_name}_geo_stage{args.stage}_qx{qx}_qy{qy}.png")
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        cv2.imwrite(output_path, overlay)
        if args.save_attn:
            np.save(os.path.splitext(output_path)[0] + "_attn.npy", geo_map)
        print(f"Saved geometry-only visualization to {output_path}")
        print(f"Stage size: {stage_h}x{stage_w}, image size: {vis_h}x{vis_w}, query: ({qx},{qy})")
        return

    out_indices = list(getattr(model, "out_indices", []))
    if args.stage in out_indices:
        stage_feat = features[out_indices.index(args.stage)]
        stage_h, stage_w = stage_feat.shape[2:]
    elif isinstance(saved_attn, dict):
        stage_h = saved_attn["w"].shape[1]
        stage_w = saved_attn["w"].shape[3]
    else:
        stride = 4 * (2 ** args.stage)
        stage_h = math.ceil(vis_h / stride)
        stage_w = math.ceil(vis_w / stride)
        if stage_h * stage_w != saved_attn.shape[-1]:
            raise ValueError(
                f"Unable to infer stage size for stage {args.stage} (got L={saved_attn.shape[-1]}, expected {stage_h * stage_w})."
            )

    qx_raw, qy_raw = parse_pair(args.query)
    scale_x = vis_w / stage_w
    scale_y = vis_h / stage_h

    if args.query_space == "image":
        qx = int(qx_raw / scale_x)
        qy = int(qy_raw / scale_y)
    else:
        qx, qy = qx_raw, qy_raw

    qx = max(0, min(stage_w - 1, qx))
    qy = max(0, min(stage_h - 1, qy))

    def compute_map(saved_list):
        attn_maps = [
            build_attention_map(attn, stage_h, stage_w, qx, qy, args.head, args.reduce) for attn in saved_list
        ]
        attn_map = np.mean(attn_maps, axis=0)
        attn_map = normalize_attn_map(attn_map, args)
        if args.geo_bias:
            geo_map = compute_geo_bias_map(
                depth_gray,
                stage_h,
                stage_w,
                qx,
                qy,
                args.geo_mode,
                args.geo_pos_weight,
                args.geo_depth_weight,
            )
            attn_map = (1.0 - args.geo_weight) * attn_map + args.geo_weight * geo_map
        return attn_map

    def render_overlay(attn_map: np.ndarray) -> np.ndarray:
        interp = cv2.INTER_NEAREST if args.interp == "nearest" else cv2.INTER_LINEAR
        attn_up = cv2.resize(attn_map, (vis_w, vis_h), interpolation=interp)
        heat = cv2.applyColorMap((attn_up * 255).astype(np.uint8), resolve_colormap(args.colormap))
        overlay = cv2.addWeighted(heat, args.alpha, vis_img, 1.0 - args.alpha, 0)

        if not args.no_grid:
            grid_color = parse_color(args.grid_color)
            for i in range(1, stage_w):
                x = int(round(i * scale_x))
                cv2.line(overlay, (x, 0), (x, vis_h - 1), grid_color, 1)
            for i in range(1, stage_h):
                y = int(round(i * scale_y))
                cv2.line(overlay, (0, y), (vis_w - 1, y), grid_color, 1)

        if not args.no_marker:
            center_x = int(round((qx + 0.5) * scale_x))
            center_y = int(round((qy + 0.5) * scale_y))
            marker_color = parse_color(args.marker_color)
            marker_size = max(8, int(round(min(scale_x, scale_y))))
            cv2.drawMarker(
                overlay,
                (center_x, center_y),
                marker_color,
                markerType=cv2.MARKER_STAR,
                markerSize=marker_size,
                thickness=2,
            )

        return overlay

    if args.attn_mode == "triplet":
        attn_full = compute_map(saved_full)
        saved_no, _ = run_with_mode("no_geo")
        attn_no = compute_map(saved_no)
        saved_geo, _ = run_with_mode("geo_only")
        attn_geo = compute_map(saved_geo)

        if args.output:
            base_dir = os.path.dirname(args.output)
            base_name = os.path.splitext(os.path.basename(args.output))[0]
            base_ext = os.path.splitext(args.output)[1] or ".png"
        else:
            base_name = os.path.splitext(os.path.basename(args.rgb))[0]
            base_dir = os.path.join(REPO_ROOT, "outputs", "attn_vis")
            base_ext = ".png"

        os.makedirs(base_dir, exist_ok=True)
        outputs = {
            "attn_plain": attn_no,
            "geo_prior": attn_geo,
            "attn_full": attn_full,
        }

        for suffix, attn_map in outputs.items():
            out_path = os.path.join(base_dir, f"{base_name}_{suffix}{base_ext}")
            overlay = render_overlay(attn_map)
            cv2.imwrite(out_path, overlay)
            if args.save_attn:
                np.save(os.path.splitext(out_path)[0] + "_attn.npy", attn_map)
            print(f"Saved {suffix} visualization to {out_path}")

        print(f"Stage size: {stage_h}x{stage_w}, image size: {vis_h}x{vis_w}, query: ({qx},{qy})")
        return

    attn_map = compute_map(saved_full)
    overlay = render_overlay(attn_map)

    if args.output:
        output_path = args.output
    else:
        base_name = os.path.splitext(os.path.basename(args.rgb))[0]
        output_dir = os.path.join(REPO_ROOT, "outputs", "attn_vis")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{base_name}_stage{args.stage}_qx{qx}_qy{qy}.png")

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    cv2.imwrite(output_path, overlay)

    if args.save_attn:
        np.save(os.path.splitext(output_path)[0] + "_attn.npy", attn_map)

    print(f"Saved attention visualization to {output_path}")
    print(f"Stage size: {stage_h}x{stage_w}, image size: {vis_h}x{vis_w}, query: ({qx},{qy})")


if __name__ == "__main__":
    main()
