"""Model assessment — per-trick accuracy and confusion matrix."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ml.dataset import TrickDataset
from ml.stgcn.model import STGCN
from ml.train import get_device


def load_model(model_path: Path | str) -> tuple[STGCN, list[str], torch.device]:
    """Load a trained model from checkpoint."""
    device = get_device()
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    trick_classes = checkpoint["trick_classes"]
    model = STGCN(
        num_classes=len(trick_classes),
        in_channels=3,
        num_joints=17,
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, trick_classes, device


def compute_confusion_matrix(
    model_path: Path | str,
    keypoints_dir: Path | str,
    labels_path: Path | str,
    target_frames: int = 64,
    batch_size: int = 16,
) -> dict:
    """Compute confusion matrix and per-class accuracy."""
    model, trick_classes, device = load_model(model_path)

    dataset = TrickDataset(
        keypoints_dir=keypoints_dir,
        labels_path=labels_path,
        trick_classes=trick_classes,
        target_frames=target_frames,
        augment=False,
    )

    if len(dataset) == 0:
        return {
            "confusion_matrix": [],
            "per_class_accuracy": {},
            "overall_accuracy": 0.0,
            "trick_classes": trick_classes,
        }

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    num_classes = len(trick_classes)
    matrix = np.zeros((num_classes, num_classes), dtype=int)

    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            logits = model(batch_x)
            preds = logits.argmax(dim=1).cpu().numpy()
            targets = batch_y.numpy()

            for true_cls, pred_cls in zip(targets, preds):
                matrix[true_cls, pred_cls] += 1

    per_class: dict[str, float] = {}
    for i, trick_id in enumerate(trick_classes):
        total = matrix[i].sum()
        correct = matrix[i, i]
        per_class[trick_id] = float(correct / total) if total > 0 else 0.0

    total = matrix.sum()
    overall = float(matrix.diagonal().sum() / total) if total > 0 else 0.0

    return {
        "confusion_matrix": matrix.tolist(),
        "per_class_accuracy": per_class,
        "overall_accuracy": overall,
        "trick_classes": trick_classes,
    }


def print_report(results: dict) -> None:
    """Print a formatted assessment report."""
    trick_classes = results["trick_classes"]
    per_class = results["per_class_accuracy"]

    print("\n" + "=" * 60)
    print("ST-GCN Model Assessment Report")
    print("=" * 60)

    print(f"\nOverall Accuracy: {results['overall_accuracy']:.1%}")

    print("\nPer-Trick Accuracy:")
    print("-" * 40)
    for trick_id in trick_classes:
        acc = per_class.get(trick_id, 0.0)
        bar = "█" * int(acc * 20) + "░" * (20 - int(acc * 20))
        print(f"  {trick_id:20s} {bar} {acc:.1%}")

    if results["confusion_matrix"]:
        print("\nConfusion Matrix:")
        print("-" * 40)
        header = "            " + "  ".join(f"{c[:4]:>4s}" for c in trick_classes)
        print(header)
        for i, row in enumerate(results["confusion_matrix"]):
            row_str = "  ".join(f"{v:4d}" for v in row)
            print(f"  {trick_classes[i][:10]:10s} {row_str}")

    print("=" * 60)
