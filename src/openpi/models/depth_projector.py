"""PointNet-based depth projector for extracting features from depth maps.

This module implements a PointNet encoder in JAX/Flax for processing depth maps.
The depth map is first converted to a point cloud using camera intrinsics,
then processed through PointNet to extract a fixed-dimensional feature vector.

Based on the approach from 3D-CAVLA:
- Depth map (H, W) -> Point cloud (H*W, 3) -> PointNet -> Features
"""
import dataclasses
from typing import Sequence

import flax.nnx as nnx
import jax
import jax.numpy as jnp

from openpi.shared import array_typing as at


@dataclasses.dataclass(frozen=True)
class DepthProjectorConfig:
    """Configuration for the depth projector.
    
    Attributes:
        height: Height of the depth image.
        width: Width of the depth image.
        fx: Camera focal length in x direction.
        fy: Camera focal length in y direction.
        cx: Camera principal point x coordinate.
        cy: Camera principal point y coordinate.
        output_dim: Output feature dimension.
        hidden_dims: Hidden layer dimensions for PointNet MLP.
    """
    height: int = 224  # Match OpenPi IMAGE_RESOLUTION
    width: int = 224
    fx: float = 270.39  # Scaled for 224x224  # Libero camera intrinsics
    fy: float = 270.39
    cx: float = 112.0  # Scaled for 224x224
    cy: float = 112.0
    output_dim: int = 2048
    hidden_dims: Sequence[int] = (64, 128, 256, 512)


class STN3d(nnx.Module):
    """Spatial Transformer Network for 3D point clouds.
    
    Learns a 3x3 transformation matrix to align the input point cloud.
    """
    
    def __init__(self, rngs: nnx.Rngs):
        # Shared MLPs
        self.conv1 = nnx.Linear(3, 64, rngs=rngs)
        self.conv2 = nnx.Linear(64, 128, rngs=rngs)
        self.conv3 = nnx.Linear(128, 1024, rngs=rngs)
        
        # FC layers
        self.fc1 = nnx.Linear(1024, 512, rngs=rngs)
        self.fc2 = nnx.Linear(512, 256, rngs=rngs)
        self.fc3 = nnx.Linear(256, 9, rngs=rngs)
        
        # Batch normalization
        self.bn1 = nnx.BatchNorm(64, rngs=rngs)
        self.bn2 = nnx.BatchNorm(128, rngs=rngs)
        self.bn3 = nnx.BatchNorm(1024, rngs=rngs)
        self.bn4 = nnx.BatchNorm(512, rngs=rngs)
        self.bn5 = nnx.BatchNorm(256, rngs=rngs)
    
    def __call__(self, x: jax.Array, *, train: bool = False) -> jax.Array:
        """Apply spatial transformer to input points.
        
        Args:
            x: Input points of shape (batch, num_points, 3)
            train: Whether in training mode (for batch norm)
            
        Returns:
            Transformation matrix of shape (batch, 3, 3)
        """
        batch_size = x.shape[0]
        
        # Shared MLP layers
        x = nnx.relu(self.bn1(self.conv1(x), use_running_average=not train))
        x = nnx.relu(self.bn2(self.conv2(x), use_running_average=not train))
        x = nnx.relu(self.bn3(self.conv3(x), use_running_average=not train))
        
        # Max pooling over points
        x = jnp.max(x, axis=1)  # (batch, 1024)
        
        # FC layers
        x = nnx.relu(self.bn4(self.fc1(x), use_running_average=not train))
        x = nnx.relu(self.bn5(self.fc2(x), use_running_average=not train))
        x = self.fc3(x)  # (batch, 9)
        
        # Add identity matrix
        identity = jnp.eye(3).flatten()
        x = x + identity
        x = x.reshape(batch_size, 3, 3)
        
        return x


class PointNetEncoder(nnx.Module):
    """PointNet encoder for extracting features from point clouds.
    
    Architecture:
    1. Spatial transformer alignment
    2. Shared MLP layers with batch normalization
    3. Max pooling for permutation invariance
    4. Final projection to output dimension
    """
    
    def __init__(self, config: DepthProjectorConfig, rngs: nnx.Rngs):
        self.config = config
        
        # Spatial transformer network
        self.stn = STN3d(rngs)
        
        # Shared MLP layers
        hidden_dims = config.hidden_dims
        
        self.conv1 = nnx.Linear(3, hidden_dims[0], rngs=rngs)
        self.bn1 = nnx.BatchNorm(hidden_dims[0], rngs=rngs)
        
        self.conv2 = nnx.Linear(hidden_dims[0], hidden_dims[1], rngs=rngs)
        self.bn2 = nnx.BatchNorm(hidden_dims[1], rngs=rngs)
        
        self.conv3 = nnx.Linear(hidden_dims[1], hidden_dims[2], rngs=rngs)
        self.bn3 = nnx.BatchNorm(hidden_dims[2], rngs=rngs)
        
        self.conv4 = nnx.Linear(hidden_dims[2], hidden_dims[3], rngs=rngs)
        self.bn4 = nnx.BatchNorm(hidden_dims[3], rngs=rngs)
        
        # Output projection
        self.fc = nnx.Linear(hidden_dims[-1], config.output_dim, rngs=rngs)
    
    def __call__(self, x: jax.Array, *, train: bool = False) -> jax.Array:
        """Encode point cloud to fixed-dimensional feature vector.
        
        Args:
            x: Input points of shape (batch, num_points, 3)
            train: Whether in training mode
            
        Returns:
            Feature vector of shape (batch, output_dim)
        """
        # Apply spatial transformer
        trans = self.stn(x, train=train)  # (batch, 3, 3)
        x = jnp.einsum('bnd,bdk->bnk', x, trans)  # (batch, num_points, 3)
        
        # Shared MLP layers
        x = nnx.relu(self.bn1(self.conv1(x), use_running_average=not train))
        x = nnx.relu(self.bn2(self.conv2(x), use_running_average=not train))
        x = nnx.relu(self.bn3(self.conv3(x), use_running_average=not train))
        x = nnx.relu(self.bn4(self.conv4(x), use_running_average=not train))
        
        # Max pooling over points (symmetric function for permutation invariance)
        x = jnp.max(x, axis=1)  # (batch, hidden_dims[-1])
        
        # Project to output dimension
        x = self.fc(x)
        
        return x


class DepthProjector(nnx.Module):
    """Full depth projector: depth map -> features.
    
    Converts a depth map to a point cloud using camera intrinsics,
    then encodes it using PointNet.
    """
    
    def __init__(self, config: DepthProjectorConfig, rngs: nnx.Rngs):
        self.config = config
        self.encoder = PointNetEncoder(config, rngs)
        
        # Pixel coordinates are derived on-the-fly to avoid storing raw arrays
        # as module leaves (nnx does not support array leaves in state).
    
    @at.typecheck
    def depth_to_pointcloud(
        self, depth: at.Float[at.Array, "b h w"]
    ) -> at.Float[at.Array, "b n 3"]:
        """Convert depth map to 3D point cloud.
        
        Uses pinhole camera model:
        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy
        Z = depth
        
        Args:
            depth: Depth map of shape (batch, height, width)
            
        Returns:
            Point cloud of shape (batch, height*width, 3)
        """
        cfg = self.config
        batch_size = depth.shape[0]
        
        # Flatten depth
        z = depth.reshape(batch_size, -1)  # (batch, H*W)
        
        # Build pixel coordinate grid (H*W,) as float to match depth dtype.
        v, u = jnp.meshgrid(
            jnp.arange(cfg.height),
            jnp.arange(cfg.width),
            indexing='ij'
        )
        u = u.reshape(-1).astype(z.dtype)
        v = v.reshape(-1).astype(z.dtype)

        # Compute X and Y coordinates
        x = (u - cfg.cx) * z / cfg.fx
        y = (v - cfg.cy) * z / cfg.fy
        
        # Stack to form point cloud
        points = jnp.stack([x, y, z], axis=-1)  # (batch, H*W, 3)
        
        return points
    
    @at.typecheck
    def __call__(
        self, depth: at.Float[at.Array, "*b h w"], *, train: bool = False
    ) -> at.Float[at.Array, "*b emb"]:
        """Extract features from depth map.
        
        Args:
            depth: Depth map of shape (*batch, height, width)
            train: Whether in training mode
            
        Returns:
            Feature vector of shape (*batch, output_dim)
        """
        # Handle arbitrary batch dimensions
        original_shape = depth.shape[:-2]
        depth = depth.reshape(-1, depth.shape[-2], depth.shape[-1])
        
        # Convert to point cloud
        points = self.depth_to_pointcloud(depth)
        
        # Normalize point cloud to unit sphere (optional but helps training)
        # Center the point cloud
        centroid = jnp.mean(points, axis=1, keepdims=True)
        points = points - centroid
        
        # Scale to unit sphere
        max_dist = jnp.max(jnp.sqrt(jnp.sum(points ** 2, axis=-1)), axis=1, keepdims=True)
        max_dist = jnp.maximum(max_dist, 1e-6)  # Avoid division by zero
        points = points / max_dist[..., None]
        
        # Encode with PointNet
        features = self.encoder(points, train=train)
        
        # Reshape back to original batch dimensions
        features = features.reshape(*original_shape, -1)
        
        return features
