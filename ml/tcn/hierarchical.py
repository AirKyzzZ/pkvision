"""Hierarchical TCN for compositional trick classification.

Instead of flat N-class classification, predicts trick *properties* independently:
  - Rotation axis (lateral, longitudinal, off-axis, sagittal)
  - Direction (forward, backward, left, right)
  - Rotation count (0, 0.5, 1, 1.5, 2, 3)
  - Body shape (tuck, pike, layout, open)
  - Entry type (standing, running, one_leg, wall, edge)
  - Twist count (0, 0.5, 1, 2, 3)

Each head is a simple 4-6 class problem — much easier to learn.
The specific trick is identified by matching predicted properties to TrickDefinitions.

Architecture:
    Shared TCN backbone -> Global Avg Pool -> 6 independent classification heads

Benefits:
    - Each head has 4-6 classes (vs 28+ flat classes)
    - Adding a new trick = add a TrickDefinition, no retraining
    - Compositional: can detect tricks never seen if they combine known properties
    - Interpretable: tells you WHY it thinks it is a certain trick

Usage:
    model = HierarchicalTrickTCN()
    predictions = model(x)  # dict of 6 logit tensors
    trick_id = model.resolve_trick(x)  # best matching TrickDefinition
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from ml.tcn.model import TemporalBlock
from ml.trick_physics import (
    TRICK_DEFINITIONS,
    TrickDefinition,
    BodyShape,
    Direction,
    EntryType,
    RotationAxis,
)


# -- Label Encodings ----------------------------------------------------------

AXIS_CLASSES = [e.value for e in RotationAxis]
DIRECTION_CLASSES = [e.value for e in Direction]
ROTATION_BINS = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
SHAPE_CLASSES = [e.value for e in BodyShape]
ENTRY_CLASSES = [e.value for e in EntryType]
TWIST_BINS = [0.0, 0.5, 1.0, 2.0, 3.0]

HEAD_SIZES = {
    "axis": len(AXIS_CLASSES),
    "direction": len(DIRECTION_CLASSES),
    "rotation": len(ROTATION_BINS),
    "shape": len(SHAPE_CLASSES),
    "entry": len(ENTRY_CLASSES),
    "twist": len(TWIST_BINS),
}


def _bin_value(value: float, bins: list[float]) -> int:
    dists = [abs(value - b) for b in bins]
    return int(np.argmin(dists))


@dataclass
class TrickLabels:
    axis: int
    direction: int
    rotation: int
    shape: int
    entry: int
    twist: int

    def to_dict(self) -> dict[str, int]:
        return {
            "axis": self.axis,
            "direction": self.direction,
            "rotation": self.rotation,
            "shape": self.shape,
            "entry": self.entry,
            "twist": self.twist,
        }


def encode_trick(trick_def: TrickDefinition) -> TrickLabels:
    return TrickLabels(
        axis=AXIS_CLASSES.index(trick_def.rotation_axis.value),
        direction=DIRECTION_CLASSES.index(trick_def.direction.value),
        rotation=_bin_value(trick_def.rotation_count, ROTATION_BINS),
        shape=SHAPE_CLASSES.index(trick_def.body_shape.value),
        entry=ENTRY_CLASSES.index(trick_def.entry.value),
        twist=_bin_value(trick_def.twist_count, TWIST_BINS),
    )


NO_TRICK_LABELS = TrickLabels(
    axis=0, direction=0, rotation=0, shape=0, entry=0, twist=0,
)


def build_label_table() -> dict[str, TrickLabels]:
    table = {"no_trick": NO_TRICK_LABELS}
    for trick_id, trick_def in TRICK_DEFINITIONS.items():
        table[trick_id] = encode_trick(trick_def)
    return table


# -- Model --------------------------------------------------------------------


class HierarchicalTrickTCN(nn.Module):
    """TCN with multiple classification heads for compositional trick detection.

    Input:  (B, 75, 64)
    Output: dict of head_name -> (B, n_classes) logits
    """

    def __init__(
        self,
        n_features: int = 75,
        hidden_channels: tuple[int, ...] = (128, 128, 128, 128),
        kernel_size: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.n_features = n_features

        blocks: list[nn.Module] = []
        in_ch = n_features
        for i, out_ch in enumerate(hidden_channels):
            blocks.append(TemporalBlock(
                in_channels=in_ch,
                out_channels=out_ch,
                kernel_size=kernel_size,
                dilation=2 ** i,
                dropout=dropout,
            ))
            in_ch = out_ch
        self.backbone = nn.Sequential(*blocks)
        self._backbone_dim = hidden_channels[-1]

        self.heads = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(self._backbone_dim, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, n_classes),
            )
            for name, n_classes in HEAD_SIZES.items()
        })

        # Binary: is this a trick at all?
        self.is_trick_head = nn.Sequential(
            nn.Linear(self._backbone_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )

        self._init_heads()

    def _init_heads(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.backbone(x)
        pooled = features.mean(dim=2)

        outputs = {}
        for name, head in self.heads.items():
            outputs[name] = head(pooled)
        outputs["is_trick"] = self.is_trick_head(pooled)
        return outputs

    def predict(self, x: torch.Tensor) -> dict[str, np.ndarray]:
        self.train(False)
        with torch.no_grad():
            logits = self.forward(x)
            return {
                name: tensor.argmax(dim=1).cpu().numpy()
                for name, tensor in logits.items()
            }

    def predict_proba(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        self.train(False)
        with torch.no_grad():
            logits = self.forward(x)
            return {
                name: torch.softmax(tensor, dim=-1)
                for name, tensor in logits.items()
            }

    def resolve_trick(
        self,
        x: torch.Tensor,
        confidence_threshold: float = 0.3,
    ) -> list[dict]:
        """Match predicted properties to the best TrickDefinition.

        Returns list of dicts with trick_id, confidence, properties, head_confidences.
        """
        probs = self.predict_proba(x)
        batch_size = x.shape[0]
        is_trick_probs = probs["is_trick"].cpu().numpy()

        label_table = build_label_table()
        results = []

        for b in range(batch_size):
            if is_trick_probs[b, 0] > is_trick_probs[b, 1]:
                results.append({
                    "trick_id": "no_trick",
                    "confidence": float(is_trick_probs[b, 0]),
                    "properties": {},
                    "head_confidences": {"is_trick": float(is_trick_probs[b, 0])},
                })
                continue

            pred_labels = {}
            head_confs = {"is_trick": float(is_trick_probs[b, 1])}

            for name in HEAD_SIZES:
                head_probs = probs[name][b].cpu().numpy()
                pred_idx = int(head_probs.argmax())
                pred_labels[name] = pred_idx
                head_confs[name] = float(head_probs[pred_idx])

            properties = {
                "axis": AXIS_CLASSES[pred_labels["axis"]],
                "direction": DIRECTION_CLASSES[pred_labels["direction"]],
                "rotation_count": ROTATION_BINS[pred_labels["rotation"]],
                "body_shape": SHAPE_CLASSES[pred_labels["shape"]],
                "entry": ENTRY_CLASSES[pred_labels["entry"]],
                "twist_count": TWIST_BINS[pred_labels["twist"]],
            }

            # Find best matching trick by weighted property matching
            best_trick = "unknown"
            best_score = -1.0

            for trick_id, labels in label_table.items():
                if trick_id == "no_trick":
                    continue
                label_dict = labels.to_dict()
                score = sum(
                    head_confs[name] for name in HEAD_SIZES
                    if pred_labels[name] == label_dict[name]
                )
                if score > best_score:
                    best_score = score
                    best_trick = trick_id

            # Geometric mean of confidences
            all_confs = [head_confs[name] for name in HEAD_SIZES]
            overall_conf = float(np.prod(all_confs) ** (1.0 / len(all_confs)))

            if overall_conf < confidence_threshold:
                best_trick = "uncertain"

            results.append({
                "trick_id": best_trick,
                "confidence": overall_conf,
                "properties": properties,
                "head_confidences": head_confs,
            })

        return results


# -- Loss Function ------------------------------------------------------------


class HierarchicalLoss(nn.Module):
    """Combined cross-entropy loss across all heads."""

    def __init__(self, is_trick_weight: float = 2.0):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.is_trick_weight = is_trick_weight

    def forward(
        self,
        predictions: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        losses = {}
        total = torch.tensor(0.0, device=next(iter(predictions.values())).device)

        for name in HEAD_SIZES:
            loss = self.ce(predictions[name], targets[name])
            losses[name] = loss.item()
            total = total + loss

        is_trick_loss = self.ce(predictions["is_trick"], targets["is_trick"])
        losses["is_trick"] = is_trick_loss.item()
        total = total + self.is_trick_weight * is_trick_loss

        losses["total"] = total.item()
        return total, losses
