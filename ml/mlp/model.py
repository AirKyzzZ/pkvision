"""MLP model for trick classification from camera-invariant features.

Architecture: Input(3840) → [Linear + BatchNorm + ReLU + Dropout] × 4 → Output(num_classes)
Input is a flattened normalized feature sequence (64 frames × 60 features).
Fast enough for real-time CPU inference.

Usage:
    model = TrickMLP(num_classes=11)  # 10 tricks + "no_trick"
    logits = model(flat_features_batch)  # (B, 3840) → (B, 11)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TrickMLP(nn.Module):
    """MLP classifier for parkour trick detection.

    Takes flattened, normalized feature sequences and outputs class logits.
    Architecture: 4 hidden layers with BatchNorm, ReLU, and Dropout.
    """

    def __init__(
        self,
        num_classes: int,
        input_size: int = 3840,  # 64 frames × 60 features
        hidden_sizes: tuple[int, ...] = (512, 256, 128, 64),
        dropout: float = 0.3,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.input_size = input_size

        layers: list[nn.Module] = []
        in_size = input_size

        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(in_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_size = hidden_size

        layers.append(nn.Linear(in_size, num_classes))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (batch_size, input_size) flattened feature sequences.

        Returns:
            (batch_size, num_classes) logits (not softmaxed).
        """
        return self.network(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get class probabilities via softmax.

        Args:
            x: (batch_size, input_size) flattened feature sequences.

        Returns:
            (batch_size, num_classes) probability distributions.
        """
        logits = self.forward(x)
        return torch.softmax(logits, dim=-1)
