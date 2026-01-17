"""PointNet-based depth projector for extracting features from depth maps (PyTorch).

This module mirrors the JAX depth projector in openpi.models.depth_projector, using
PointNet to encode depth-derived point clouds into a fixed-dimensional feature vector.
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F  # noqa: N812


class STN3d(nn.Module):
    """Spatial transformer network for 3D point clouds."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, num_points, 3)
        batch_size = x.shape[0]
        x = x.transpose(1, 2)  # (batch, 3, num_points)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2).values  # (batch, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        identity = torch.eye(3, device=x.device, dtype=x.dtype).reshape(1, 9).repeat(batch_size, 1)
        x = x + identity
        return x.view(-1, 3, 3)


class PointNetEncoder(nn.Module):
    """PointNet encoder for extracting features from point clouds."""

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config

        self.stn = STN3d()

        hidden_dims = config.hidden_dims
        self.conv1 = nn.Conv1d(3, hidden_dims[0], 1)
        self.bn1 = nn.BatchNorm1d(hidden_dims[0])
        self.conv2 = nn.Conv1d(hidden_dims[0], hidden_dims[1], 1)
        self.bn2 = nn.BatchNorm1d(hidden_dims[1])
        self.conv3 = nn.Conv1d(hidden_dims[1], hidden_dims[2], 1)
        self.bn3 = nn.BatchNorm1d(hidden_dims[2])
        self.conv4 = nn.Conv1d(hidden_dims[2], hidden_dims[3], 1)
        self.bn4 = nn.BatchNorm1d(hidden_dims[3])

        self.fc = nn.Linear(hidden_dims[-1], config.output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, num_points, 3)
        trans = self.stn(x)
        x = torch.bmm(x, trans)
        x = x.transpose(1, 2)  # (batch, 3, num_points)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))

        x = torch.max(x, 2).values  # (batch, hidden_dims[-1])
        return self.fc(x)


class DepthProjectorPytorch(nn.Module):
    """Depth map -> point cloud -> PointNet features."""

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.encoder = PointNetEncoder(config)

        u = torch.arange(config.width, dtype=torch.float32)
        v = torch.arange(config.height, dtype=torch.float32)
        v_grid, u_grid = torch.meshgrid(v, u, indexing="ij")
        self.register_buffer("u_grid", u_grid.reshape(-1), persistent=False)
        self.register_buffer("v_grid", v_grid.reshape(-1), persistent=False)

    def depth_to_pointcloud(self, depth: torch.Tensor) -> torch.Tensor:
        cfg = self.config
        if depth.shape[-2:] != (cfg.height, cfg.width):
            raise ValueError(
                f"Depth resolution {tuple(depth.shape[-2:])} does not match config {(cfg.height, cfg.width)}"
            )

        batch_size = depth.shape[0]
        z = depth.reshape(batch_size, -1)

        u = self.u_grid.to(device=depth.device, dtype=depth.dtype)
        v = self.v_grid.to(device=depth.device, dtype=depth.dtype)
        x = (u - cfg.cx) * z / cfg.fx
        y = (v - cfg.cy) * z / cfg.fy

        return torch.stack([x, y, z], dim=-1)

    def forward(self, depth: torch.Tensor) -> torch.Tensor:
        original_shape = depth.shape[:-2]
        depth = depth.reshape(-1, depth.shape[-2], depth.shape[-1])

        points = self.depth_to_pointcloud(depth)

        centroid = points.mean(dim=1, keepdim=True)
        points = points - centroid

        max_dist = torch.sqrt((points**2).sum(dim=-1)).max(dim=1, keepdim=True).values
        max_dist = torch.clamp(max_dist, min=1e-6)
        points = points / max_dist.unsqueeze(-1)

        features = self.encoder(points)
        return features.reshape(*original_shape, -1)
