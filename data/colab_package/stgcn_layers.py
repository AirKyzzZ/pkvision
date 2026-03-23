"""Graph convolution layers for ST-GCN."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def build_coco_adjacency(num_joints: int = 17) -> np.ndarray:
    """Build the adjacency matrix for COCO 17-keypoint skeleton.

    Connections:
    nose(0) - left_eye(1), right_eye(2)
    left_eye(1) - left_ear(3)
    right_eye(2) - right_ear(4)
    left_shoulder(5) - right_shoulder(6), left_elbow(7), left_hip(11)
    right_shoulder(6) - right_elbow(8), right_hip(12)
    left_elbow(7) - left_wrist(9)
    right_elbow(8) - right_wrist(10)
    left_hip(11) - right_hip(12), left_knee(13)
    right_hip(12) - right_knee(14)
    left_knee(13) - left_ankle(15)
    right_knee(14) - right_ankle(16)
    nose(0) - left_shoulder(5), right_shoulder(6) [neck connection]
    """
    edges = [
        (0, 1), (0, 2), (1, 3), (2, 4),   # head
        (0, 5), (0, 6),                     # neck → shoulders
        (5, 6),                             # shoulder span
        (5, 7), (7, 9),                     # left arm
        (6, 8), (8, 10),                    # right arm
        (5, 11), (6, 12),                   # torso
        (11, 12),                           # hip span
        (11, 13), (13, 15),                 # left leg
        (12, 14), (14, 16),                 # right leg
    ]

    adj = np.eye(num_joints, dtype=np.float32)  # self-loops
    for i, j in edges:
        adj[i, j] = 1.0
        adj[j, i] = 1.0

    return adj


def normalize_adjacency(adj: np.ndarray) -> np.ndarray:
    """Symmetric normalization: D^(-1/2) A D^(-1/2)."""
    d = np.sum(adj, axis=1)
    d_inv_sqrt = np.where(d > 0, np.power(d, -0.5), 0.0)
    d_mat = np.diag(d_inv_sqrt)
    return d_mat @ adj @ d_mat


class SpatialGraphConv(nn.Module):
    """Spatial graph convolution on skeleton joints."""

    def __init__(self, in_channels: int, out_channels: int, adj: torch.Tensor):
        super().__init__()
        self.register_buffer("adj", adj)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, T, V) where V = num_joints."""
        B, C, T, V = x.shape

        # Graph convolution: multiply features by adjacency
        # Reshape for matmul: (B*C*T, V) @ (V, V) → (B*C*T, V)
        x_flat = x.permute(0, 2, 1, 3).reshape(B * T, C, V)
        x_graph = torch.matmul(x_flat, self.adj)  # (B*T, C, V)
        x_graph = x_graph.reshape(B, T, C, V).permute(0, 2, 1, 3)  # (B, C, T, V)

        out = self.conv(x_graph)
        out = self.bn(out)
        out = self.relu(out)
        return out


class TemporalConv(nn.Module):
    """Temporal convolution across frames."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 9):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=(kernel_size, 1),
            padding=(padding, 0),
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, T, V)."""
        return self.relu(self.bn(self.conv(x)))


class STGCNBlock(nn.Module):
    """A single ST-GCN block: spatial graph conv + temporal conv + residual."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        adj: torch.Tensor,
        temporal_kernel: int = 9,
        stride: int = 1,
    ):
        super().__init__()
        self.spatial = SpatialGraphConv(in_channels, out_channels, adj)
        self.temporal = TemporalConv(out_channels, out_channels, temporal_kernel)

        # Residual connection
        if in_channels != out_channels or stride != 1:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.residual = nn.Identity()

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.residual(x)
        out = self.spatial(x)
        out = self.temporal(out)
        return self.relu(out + res)
