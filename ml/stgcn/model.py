"""ST-GCN model for skeleton-based parkour trick classification."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from ml.stgcn.layers import STGCNBlock, build_coco_adjacency, normalize_adjacency


class STGCN(nn.Module):
    """Spatial-Temporal Graph Convolutional Network.

    Based on Yan et al. 2018, adapted for parkour trick classification.

    Input:  (B, C, T, V) — batch, channels (x/y/conf), frames, joints
    Output: (B, num_classes) — class logits
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: int = 3,
        num_joints: int = 17,
        hidden_channels: list[int] | None = None,
    ):
        super().__init__()

        if hidden_channels is None:
            hidden_channels = [64, 64, 128, 128, 256]

        # Build normalized adjacency matrix
        adj_np = normalize_adjacency(build_coco_adjacency(num_joints))
        adj = torch.FloatTensor(adj_np)

        # Input batch normalization
        self.input_bn = nn.BatchNorm1d(in_channels * num_joints)

        # Build ST-GCN blocks
        layers: list[nn.Module] = []
        prev_channels = in_channels

        for out_ch in hidden_channels:
            layers.append(STGCNBlock(prev_channels, out_ch, adj))
            prev_channels = out_ch

        self.blocks = nn.ModuleList(layers)

        # Global average pooling + classifier
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_channels[-1], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (B, C, T, V) tensor where C=3 (x, y, confidence), T=frames, V=17 joints

        Returns:
            (B, num_classes) logits
        """
        B, C, T, V = x.shape

        # Input normalization
        x_flat = x.permute(0, 2, 1, 3).reshape(B * T, C * V)
        x_flat = self.input_bn(x_flat)
        x = x_flat.reshape(B, T, C, V).permute(0, 2, 1, 3)

        # ST-GCN blocks
        for block in self.blocks:
            x = block(x)

        # Global average pooling: (B, C', T, V) → (B, C', 1, 1)
        x = self.gap(x)
        x = x.squeeze(-1).squeeze(-1)  # (B, C')

        x = self.dropout(x)
        return self.fc(x)
