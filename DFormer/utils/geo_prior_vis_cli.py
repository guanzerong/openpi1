import argparse
import os
from typing import Dict, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GeoPrior(nn.Module):
    def __init__(self, embed_dim=128, num_heads=4, initial_value=2, heads_range=6):
        super().__init__()
        angle = 1.0 / (10000 ** torch.linspace(0, 1, embed_dim // num_heads // 2))
        angle = angle.unsqueeze(-1).repeat(1, 2).flatten()
        self.initial_value = initial_value
        self.heads_range = heads_range
        self.num_heads = num_heads
        decay = torch.log(1 - 2 ** (-initial_value - heads_range * torch.arange(num_heads, dtype=torch.float) / num_heads))
        self.register_buffer("angle", angle)
        self.register_buffer("decay", decay)

    def generate_pos_decay(self, h: int, w: int) -> torch.Tensor:
        index_h = torch.arange(h).to(self.decay)
        index_w = torch.arange(w).to(self.decay)
        grid = torch.meshgrid([index_h, index_w], indexing="ij")
        grid = torch.stack(grid, dim=-1).reshape(h * w, 2)
        mask = grid[:, None, :] - grid[None, :, :]
        mask = (mask.abs()).sum(dim=-1)
        return mask

    def generate_2d_depth_decay(self, h: int, w: int, depth_grid: torch.Tensor) -> torch.Tensor:
        b, _, h, w = depth_grid.shape
        grid_d = depth_grid.reshape(b, h * w, 1)
        mask_d = grid_d[:, :, None, :] - grid_d[:, None, :, :]
        mask_d = (mask_d.abs()).sum(dim=-1)
        mask_d = mask_d.unsqueeze(1)
        return mask_d

    def forward(self, slen: Tuple[int, int], depth_map: torch.Tensor):
        depth_map = F.interpolate(depth_map, size=slen, mode="bilinear", align_corners=False)
        depth_map = depth_map.float()

        index = torch.arange(slen[0] * slen[1]).to(self.decay)
        sin = torch.sin(index[:, None] * self.angle[None, :])
        sin = sin.reshape(slen[0], slen[1], -1)
        cos = torch.cos(index[:, None] * self.angle[None, :])
        cos = cos.reshape(slen[0], slen[1], -1)

        mask_1 = self.generate_pos_decay(slen[0], slen[1])
        mask_d = self.generate_2d_depth_decay(slen[0], slen[1], depth_map)
        mask_sum = (0.85 * mask_1.cpu() + 0.15 * mask_d.cpu()) * self.decay[:, None, None].cpu()
        retention_rel_pos = ((sin, cos), mask_d, mask_1, mask_sum)
        return retention_rel_pos


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Geometry prior visualization without Gradio.")
    parser.add_argument("--rgb", required=True, help="RGB image path.")
    parser.add_argument("--depth", required=True, help="Depth map path (.png/.npy).")
    parser.add_argument("--x", type=float, default=160, help="Query x (pixel).")
    parser.add_argument("--y", type=float, default=270, help="Query y (pixel).")
    parser.add_argument("--stride", type=int, default=20, help="Patch size for grid downsample.")
    parser.add_argument(
        "--mode",
        choices=["pos", "depth", "fuse", "sum", "all"],
        default="fuse",
        help="Which map to visualize.",
    )
    parser.add_argument("--gama", type=float, default=0.55, help="Fuse weight for position vs depth.")
    parser.add_argument("--head", type=int, default=None, help="Head index for sum (default: mean).")
    parser.add_argument("--legacy-index", action="store_true", help="Use the original (y//stride+1) index formula.")
    parser.add_argument("--output", required=True, help="Output image path (or base path for --mode all).")
    return parser.parse_args()


def load_depth(path: str) -> np.ndarray:
    if path.lower().endswith(".npy"):
        depth = np.load(path)
        if depth.ndim == 3:
            depth = depth[..., 0]
        return depth
    depth = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise FileNotFoundError(f"Failed to read depth map at {path}")
    if depth.ndim == 3:
        depth = cv2.cvtColor(depth, cv2.COLOR_BGR2GRAY)
    return depth



def put_mask(image: np.ndarray, mask: torch.Tensor) -> np.ndarray:
    mask = mask.numpy()
    h, w = image.shape[:2]
    mask = cv2.resize(mask, dsize=(w, h), interpolation=cv2.INTER_LINEAR)
    max_val = np.max(mask)
    if max_val > 0:
        mask = (1 - mask / max_val)
    else:
        mask = mask * 0
    heatmap = cv2.applyColorMap((mask * 255).astype(np.uint8), cv2.COLORMAP_JET)
    result = cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)
    return result


def visualize_geometry_prior(
    rgb_path: str,
    depth_path: str,
    x: float,
    y: float,
    stride: int,
    mode: str,
    gama: float,
    head: int | None,
    legacy_index: bool,
) -> Dict[str, np.ndarray]:
    img = cv2.imread(rgb_path)
    if img is None:
        raise FileNotFoundError(f"Failed to read RGB image at {rgb_path}")
    h_img, w_img = img.shape[:2]

    grid_h = h_img // stride
    grid_w = w_img // stride

    if legacy_index:
        index_num = int(x // stride) + int((y // stride + 1) * grid_w)
    else:
        index_num = int(x // stride) + int((y // stride) * grid_w)

    depth_raw = load_depth(depth_path)
    depth_small = cv2.resize(depth_raw, dsize=(grid_w, grid_h), interpolation=cv2.INTER_LINEAR)
    depth_tensor = torch.tensor(depth_small).reshape(1, 1, grid_h, grid_w)

    respos = GeoPrior()
    ((sin, cos), depth_map, mask_1, mask_sum) = respos((grid_h, grid_w), depth_tensor)

    i = index_num
    temp_mask_d = depth_map[0, 0, i, :].reshape(grid_h, grid_w).cpu()
    temp_mask = mask_1[i, :].reshape(grid_h, grid_w).cpu()

    temp_mask_d = torch.nn.functional.normalize(temp_mask_d, p=2.0, dim=1, eps=1e-12)
    d_min, d_max = torch.min(temp_mask_d), torch.max(temp_mask_d)
    if d_max > d_min:
        temp_mask_d = 255 * (temp_mask_d - d_min) / (d_max - d_min)
    else:
        temp_mask_d = temp_mask_d * 0

    p_min, p_max = torch.min(temp_mask), torch.max(temp_mask)
    if p_max > p_min:
        temp_mask = 255 * ((temp_mask - p_min) / (p_max - p_min))
    else:
        temp_mask = temp_mask * 0

    mask_sum_sel = mask_sum
    if head is not None:
        if mask_sum_sel.ndim >= 3:
            mask_sum_sel = mask_sum_sel[head]
        else:
            raise ValueError(f"mask_sum has shape {mask_sum.shape}, cannot select head {head}.")

    while mask_sum_sel.ndim > 2:
        mask_sum_sel = mask_sum_sel.mean(0)

    temp_mask_sum = mask_sum_sel[i, :].reshape(grid_h, grid_w).cpu()
    s_min, s_max = torch.min(temp_mask_sum), torch.max(temp_mask_sum)
    if s_max > s_min:
        temp_mask_sum = 255 * ((temp_mask_sum - s_min) / (s_max - s_min))
    else:
        temp_mask_sum = temp_mask_sum * 0

    outputs: Dict[str, np.ndarray] = {}
    if mode in ("pos", "all"):
        outputs["pos"] = put_mask(img, temp_mask)
    if mode in ("depth", "all"):
        outputs["depth"] = put_mask(img, temp_mask_d)
    if mode in ("sum", "all"):
        outputs["sum"] = put_mask(img, temp_mask_sum)
    if mode in ("fuse", "all"):
        fused = gama * temp_mask + (1 - gama) * temp_mask_d
        outputs["fuse"] = put_mask(img, fused)

    return outputs


def main() -> None:
    args = parse_args()
    outputs = visualize_geometry_prior(
        args.rgb,
        args.depth,
        args.x,
        args.y,
        args.stride,
        args.mode,
        args.gama,
        args.head,
        args.legacy_index,
    )

    base, ext = os.path.splitext(args.output)
    if not ext:
        ext = ".png"

    if args.mode == "all":
        for key, image in outputs.items():
            out_path = f"{base}_{key}{ext}"
            cv2.imwrite(out_path, image)
            print(f"Saved {key} to {out_path}")
    else:
        image = outputs[args.mode]
        out_path = args.output if args.output.endswith(ext) else f"{args.output}{ext}"
        cv2.imwrite(out_path, image)
        print(f"Saved {args.mode} to {out_path}")


if __name__ == "__main__":
    main()
