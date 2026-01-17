from __future__ import annotations

import importlib
import os
from pathlib import Path
import sys

import torch
from torch import nn
import torch.nn.functional as F  # noqa: N812


def _import_dformer_infer():
    try:
        return importlib.import_module("DFormer.utils.infer")
    except Exception:
        repo_root = Path(__file__).resolve().parents[3]
        candidate = repo_root.parent / "DFormer"
        if candidate.exists():
            sys.path.insert(0, str(candidate.parent))
        return importlib.import_module("DFormer.utils.infer")


def _get_stage_dim(backbone_name: str) -> int:
    if "DFormerv2_L" in backbone_name:
        return 640
    if "DFormerv2_S" in backbone_name:
        return 512
    return 512


class DFormerDepthProjector(nn.Module):
    def __init__(self, config, output_dim: int) -> None:
        super().__init__()
        infer = _import_dformer_infer()

        dformer_cfg = infer.load_config(config.dformer_config)
        force_size = getattr(config, "dformer_force_val_size", None)
        if force_size is not None:
            dformer_cfg.force_val_size = force_size
        self.dformer_cfg = dformer_cfg

        ckpt = getattr(config, "dformer_checkpoint", None) or getattr(dformer_cfg, "pretrained_model", None)
        if ckpt is None:
            raise ValueError("DFormer checkpoint is not set. Provide config.dformer_checkpoint.")
        ckpt_path = self._resolve_checkpoint(ckpt, infer)

        self.backbone = infer.build_backbone(dformer_cfg)
        infer.load_checkpoint(self.backbone, ckpt_path)

        self.train_backbone = bool(getattr(config, "dformer_train_backbone", False))
        if not self.train_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        stage_dim = _get_stage_dim(getattr(dformer_cfg, "backbone", ""))
        hidden_dim = int(getattr(config, "dformer_projector_hidden_dim", 0) or 0)
        if hidden_dim > 0:
            self.projector = nn.Sequential(
                nn.Linear(stage_dim, hidden_dim, bias=True),
                nn.GELU(),
                nn.Linear(hidden_dim, output_dim, bias=True),
            )
        else:
            self.projector = nn.Linear(stage_dim, output_dim, bias=True) if stage_dim != output_dim else nn.Identity()

        self.register_buffer(
            "rgb_mean",
            torch.tensor(dformer_cfg.norm_mean, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "rgb_std",
            torch.tensor(dformer_cfg.norm_std, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "depth_mean",
            torch.tensor([0.48, 0.48, 0.48], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "depth_std",
            torch.tensor([0.28, 0.28, 0.28], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )

    def train(self, mode: bool = True):
        super().train(mode)
        if not self.train_backbone:
            self.backbone.eval()
        return self

    def _resolve_checkpoint(self, ckpt: str, infer) -> str:
        if os.path.isabs(ckpt):
            path = ckpt
        else:
            repo_root = getattr(infer, "REPO_ROOT", None)
            if repo_root is None:
                repo_root = str(Path(__file__).resolve().parents[3].parent)
            path = os.path.join(repo_root, ckpt)
        if not os.path.exists(path):
            raise FileNotFoundError(f"DFormer checkpoint not found at {path}")
        return path

    def _prepare_rgbd(self, rgb: torch.Tensor, depth: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if rgb.ndim != 4 or rgb.shape[1] != 3:
            raise ValueError(f"Expected RGB shape [B, 3, H, W], got {tuple(rgb.shape)}")

        rgb = rgb.to(torch.float32)
        rgb = (rgb + 1.0) / 2.0

        force_size = getattr(self.dformer_cfg, "force_val_size", None)
        if force_size is not None and rgb.shape[-2:] != tuple(force_size):
            rgb = F.interpolate(rgb, size=force_size, mode="bilinear", align_corners=False)

        bgr = rgb[:, [2, 1, 0], :, :]
        bgr = (bgr - self.rgb_mean) / self.rgb_std

        depth = depth.to(torch.float32)
        if depth.ndim == 3:
            depth = depth.unsqueeze(1)
        if depth.shape[1] == 1:
            depth = depth.repeat(1, 3, 1, 1)
        elif depth.shape[1] != 3:
            raise ValueError(f"Unsupported depth channel layout: {tuple(depth.shape)}")

        if force_size is not None and depth.shape[-2:] != tuple(force_size):
            depth = F.interpolate(depth, size=force_size, mode="bilinear", align_corners=False)

        depth_max = depth.amax(dim=(1, 2, 3), keepdim=True)
        depth = torch.where(depth_max <= (1.0 + 1e-3), depth * 255.0, depth)
        depth = depth.clamp(0.0, 255.0) / 255.0
        depth = (depth - self.depth_mean) / self.depth_std

        return bgr, depth

    def forward(self, rgb: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        rgb, depth = self._prepare_rgbd(rgb, depth)

        if self.train_backbone and self.training:
            feats = self.backbone(rgb, depth)
        else:
            self.backbone.eval()
            with torch.no_grad():
                feats = self.backbone(rgb, depth)

        if not isinstance(feats, (tuple, list)):
            feats = (feats,)
        feat = feats[-1]

        if feat.dim() == 4:
            pooled = feat.mean(dim=(2, 3))
        elif feat.dim() == 3:
            pooled = feat.mean(dim=2) if feat.shape[1] <= feat.shape[2] else feat.mean(dim=1)
        else:
            pooled = feat

        return self.projector(pooled)
