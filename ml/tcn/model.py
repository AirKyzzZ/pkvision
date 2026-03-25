"""TCN (Temporal Convolutional Network) for trick classification.

Replaces the flat MLP with a temporal architecture that processes feature
sequences through dilated causal convolutions with residual connections.

Input:  (B, 75, 64) — features × time (channels-first for Conv1d)
Output: (B, num_classes) — class logits

Architecture:
    Stack of TemporalBlocks with exponentially increasing dilation (1, 2, 4, 8),
    followed by global average pooling and a linear classifier.

Usage:
    model = TrickTCN(num_classes=11)
    logits = model(x)       # (B, 75, 64) → (B, 11)
    probs = model.predict_proba(x)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm


class Chomp1d(nn.Module):
    """Remove trailing padding to enforce causal convolutions.

    After a padded Conv1d, the output has extra timesteps at the end.
    Chomping removes them so the output length matches the input length,
    and each output timestep only depends on current and past inputs.
    """

    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """Single TCN residual block with dilated causal convolutions.

    Structure:
        x → Conv1d → Chomp → BN → ReLU → Dropout →
            Conv1d → Chomp → BN → ReLU → Dropout → (+residual) → ReLU
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ):
        super().__init__()
        padding = (kernel_size - 1) * dilation

        self.conv1 = weight_norm(nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation,
        ))
        self.chomp1 = Chomp1d(padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(
            out_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation,
        ))
        self.chomp2 = Chomp1d(padding)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.bn1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.bn2, self.relu2, self.dropout2,
        )

        # Residual: 1x1 conv if channel dimensions differ, otherwise identity
        self.residual = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )
        self.relu_out = nn.ReLU()

        self._init_weights()

    def _init_weights(self) -> None:
        for module in [self.conv1, self.conv2]:
            # parametrizations.weight_norm exposes .weight as a property;
            # init the direction tensor (original1) which holds the shape.
            if hasattr(module, "parametrizations"):
                nn.init.kaiming_normal_(module.parametrizations.weight.original1)
            else:
                nn.init.kaiming_normal_(module.weight)
        for module in [self.bn1, self.bn2]:
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        if isinstance(self.residual, nn.Conv1d):
            nn.init.kaiming_normal_(self.residual.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        res = self.residual(x)
        return self.relu_out(out + res)


class TrickTCN(nn.Module):
    """Temporal Convolutional Network for trick classification.

    Input:  (B, 75, 64) — features x time
    Output: (B, num_classes) — logits
    """

    def __init__(
        self,
        num_classes: int,
        n_features: int = 75,
        hidden_channels: tuple[int, ...] = (128, 128, 128, 128),
        kernel_size: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.n_features = n_features

        blocks: list[nn.Module] = []
        in_ch = n_features

        for i, out_ch in enumerate(hidden_channels):
            dilation = 2 ** i  # 1, 2, 4, 8
            blocks.append(TemporalBlock(
                in_channels=in_ch,
                out_channels=out_ch,
                kernel_size=kernel_size,
                dilation=dilation,
                dropout=dropout,
            ))
            in_ch = out_ch

        self.tcn = nn.Sequential(*blocks)
        self.fc = nn.Linear(hidden_channels[-1], num_classes)

        nn.init.kaiming_normal_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (B, n_features, T) temporal feature tensor.

        Returns:
            (B, num_classes) logits.
        """
        # x: (B, n_features, T)
        out = self.tcn(x)  # (B, C, T)
        # Global average pooling over time
        out = out.mean(dim=2)  # (B, C)
        return self.fc(out)  # (B, num_classes)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get class probabilities via softmax.

        Args:
            x: (B, n_features, T) temporal feature tensor.

        Returns:
            (B, num_classes) probability distributions.
        """
        logits = self.forward(x)
        return torch.softmax(logits, dim=-1)
