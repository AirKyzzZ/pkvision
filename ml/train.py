"""ST-GCN training loop with MPS/CUDA/CPU auto-detection."""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from ml.dataset import TrickDataset
from ml.stgcn.model import STGCN


def get_device() -> torch.device:
    """Auto-detect best available device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_model(
    keypoints_dir: Path | str,
    labels_path: Path | str,
    output_path: Path | str,
    epochs: int = 100,
    batch_size: int = 16,
    learning_rate: float = 1e-3,
    target_frames: int = 64,
    val_split: float = 0.2,
    device_override: str | None = None,
) -> dict:
    """Train the ST-GCN model on extracted keypoint data.

    Returns training history dict with loss and accuracy per epoch.
    """
    keypoints_dir = Path(keypoints_dir)
    labels_path = Path(labels_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device(device_override) if device_override else get_device()
    print(f"Training on device: {device}")

    # Discover trick classes from labels (supports both manifest and flat formats)
    with open(labels_path) as f:
        raw = json.load(f)

    if isinstance(raw, dict) and "classes" in raw:
        # Manifest format from synthetic generator or prepare_training
        trick_classes = raw["classes"]
        labels = raw.get("samples", [])
        # Remap "class" key to "trick_id" for dataset compatibility
        for entry in labels:
            if "trick_id" not in entry and "class" in entry:
                entry["trick_id"] = entry["class"]
    elif isinstance(raw, dict) and "labels" in raw:
        # Old labeler format
        labels = raw["labels"]
        trick_classes = sorted(set(entry["trick_id"] for entry in labels))
    else:
        # Flat list format
        labels = raw
        trick_classes = sorted(set(entry["trick_id"] for entry in labels))

    print(f"Found {len(trick_classes)} trick classes: {trick_classes}")

    if len(trick_classes) < 2:
        raise ValueError("Need at least 2 trick classes to train.")

    # Build dataset
    full_dataset = TrickDataset(
        keypoints_dir=keypoints_dir,
        labels_path=labels_path,
        trick_classes=trick_classes,
        target_frames=target_frames,
        augment=True,
    )

    if len(full_dataset) == 0:
        raise ValueError(
            f"No samples found. Ensure keypoint .npy files exist in {keypoints_dir}."
        )

    print(f"Total samples: {len(full_dataset)}")

    # Train/val split
    val_size = max(1, int(len(full_dataset) * val_split))
    train_size = len(full_dataset) - val_size

    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Model
    model = STGCN(
        num_classes=len(trick_classes),
        in_channels=3,
        num_joints=17,
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history: dict[str, list] = {
        "train_loss": [], "val_loss": [],
        "train_acc": [], "val_acc": [],
    }
    best_val_acc = 0.0

    for epoch in range(epochs):
        start_time = time.time()

        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch_x.size(0)
            preds = logits.argmax(dim=1)
            train_correct += (preds == batch_y).sum().item()
            train_total += batch_x.size(0)

        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                logits = model(batch_x)
                loss = criterion(logits, batch_y)

                val_loss += loss.item() * batch_x.size(0)
                preds = logits.argmax(dim=1)
                val_correct += (preds == batch_y).sum().item()
                val_total += batch_x.size(0)

        epoch_train_loss = train_loss / max(train_total, 1)
        epoch_val_loss = val_loss / max(val_total, 1)
        epoch_train_acc = train_correct / max(train_total, 1)
        epoch_val_acc = val_correct / max(val_total, 1)

        history["train_loss"].append(epoch_train_loss)
        history["val_loss"].append(epoch_val_loss)
        history["train_acc"].append(epoch_train_acc)
        history["val_acc"].append(epoch_val_acc)

        elapsed = time.time() - start_time

        if device.type == "mps":
            torch.mps.synchronize()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"Epoch {epoch+1:3d}/{epochs} | "
                f"train_loss={epoch_train_loss:.4f} acc={epoch_train_acc:.3f} | "
                f"val_loss={epoch_val_loss:.4f} acc={epoch_val_acc:.3f} | "
                f"{elapsed:.1f}s"
            )

        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "trick_classes": trick_classes,
                    "epoch": epoch,
                    "val_acc": best_val_acc,
                    "target_frames": target_frames,
                },
                output_path,
            )

    print(f"\nTraining complete. Best val accuracy: {best_val_acc:.3f}")
    print(f"Model saved to: {output_path}")

    return history
