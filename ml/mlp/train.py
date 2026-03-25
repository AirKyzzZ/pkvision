"""Training script for the MLP trick classifier.

Trains on synthetic augmented feature sequences generated from a few real reference clips.
Supports CUDA (Windows dev server), MPS (Mac), and CPU.

Usage:
    # From project root:
    python -m ml.mlp.train \
        --references-dir data/references \
        --output data/models/mlp_v1.pt \
        --epochs 50 \
        --device auto

    # On Windows dev server via SSH:
    ssh windows-dev "cd pkvision && python -m ml.mlp.train --device cuda"
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from core.pose.features import FeatureSequence, _array_to_sequence, ANGLE_NAMES, LIMB_RATIO_NAMES, FEATURES_PER_FRAME
from ml.feature_augment import AugmentConfig, FeatureAugmenter
from ml.mlp.model import TrickMLP

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

TARGET_FRAMES = 64
INPUT_SIZE = TARGET_FRAMES * FEATURES_PER_FRAME


def detect_device(requested: str = "auto") -> torch.device:
    """Auto-detect the best available device."""
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_references(references_dir: Path) -> dict[str, list[np.ndarray]]:
    """Load raw feature arrays from the references directory.

    Returns dict mapping trick_id to list of (n_frames, n_features) arrays.
    """
    refs: dict[str, list[np.ndarray]] = {}

    if not references_dir.exists():
        logger.error("References directory %s does not exist", references_dir)
        return refs

    for trick_dir in sorted(references_dir.iterdir()):
        if not trick_dir.is_dir():
            continue
        trick_id = trick_dir.name
        arrays = []
        for npy_file in sorted(trick_dir.glob("*.npy")):
            arr = np.load(npy_file).astype(np.float32)
            if arr.ndim == 2 and arr.shape[1] >= 60:
                arrays.append(arr)
            else:
                logger.warning("Skipping %s: shape %s", npy_file, arr.shape)
        if arrays:
            refs[trick_id] = arrays
            logger.info("  %s: %d reference clips", trick_id, len(arrays))

    return refs


def generate_training_data(
    references: dict[str, list[np.ndarray]],
    samples_per_trick: int = 500,
    no_trick_samples: int = 500,
    augment_config: AugmentConfig | None = None,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Generate training data from references via augmentation.

    Returns:
        X: (n_samples, 3840) flattened normalized feature arrays.
        y: (n_samples,) integer class labels.
        class_names: list of class names (index 0 = "no_trick").
    """
    config = augment_config or AugmentConfig()
    augmenter = FeatureAugmenter(config=config, seed=seed)

    # Class 0 is always "no_trick"
    class_names = ["no_trick"] + sorted(references.keys())
    class_to_idx = {name: i for i, name in enumerate(class_names)}

    all_X: list[np.ndarray] = []
    all_y: list[int] = []

    # Generate trick samples
    for trick_id, ref_arrays in references.items():
        label = class_to_idx[trick_id]
        ref_sequences = [
            _array_to_sequence(arr, list(ANGLE_NAMES), list(LIMB_RATIO_NAMES))
            for arr in ref_arrays
        ]

        variants = augmenter.augment_references(ref_sequences, n_total=samples_per_trick)

        for variant in variants:
            flat = variant.to_flat_array(target_frames=TARGET_FRAMES, normalize=True)
            all_X.append(flat)
            all_y.append(label)

        logger.info("  %s: %d training samples (class %d)", trick_id, len(variants), label)

    # Generate "no_trick" samples (random noise / static poses)
    rng = np.random.default_rng(seed + 999)
    for _ in range(no_trick_samples):
        # Low-amplitude random features (simulates walking/standing)
        noise = rng.normal(0, 0.1, size=INPUT_SIZE).astype(np.float32)
        all_X.append(noise)
        all_y.append(0)

    logger.info("  no_trick: %d training samples (class 0)", no_trick_samples)

    X = np.array(all_X, dtype=np.float32)
    y = np.array(all_y, dtype=np.int64)

    # Replace NaN with 0 (low-confidence keypoints become neutral)
    nan_count = np.isnan(X).sum()
    if nan_count > 0:
        logger.info("  Replacing %d NaN values with 0 (%.1f%% of features)", nan_count, 100 * nan_count / X.size)
        np.nan_to_num(X, copy=False, nan=0.0)

    # Shuffle
    perm = rng.permutation(len(X))
    return X[perm], y[perm], class_names


def train(
    X: np.ndarray,
    y: np.ndarray,
    class_names: list[str],
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    val_split: float = 0.2,
    device: torch.device | None = None,
) -> tuple[TrickMLP, dict]:
    """Train the MLP classifier.

    Returns:
        model: Trained TrickMLP model.
        metadata: Training metadata (classes, best accuracy, etc.).
    """
    if device is None:
        device = detect_device()

    num_classes = len(class_names)
    n_val = int(len(X) * val_split)
    n_train = len(X) - n_val

    X_train, X_val = X[:n_train], X[n_train:]
    y_train, y_val = y[:n_train], y[n_train:]

    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.long),
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    model = TrickMLP(num_classes=num_classes, input_size=INPUT_SIZE).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_state = None

    logger.info("Training MLP: %d classes, %d train, %d val, device=%s", num_classes, n_train, n_val, device)

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0

        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_X.size(0)
            train_correct += (logits.argmax(dim=1) == batch_y).sum().item()

        scheduler.step()
        train_acc = train_correct / n_train

        # Validate
        model.train(False)
        val_correct = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                logits = model(batch_X)
                val_correct += (logits.argmax(dim=1) == batch_y).sum().item()

        val_acc = val_correct / max(n_val, 1)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == 1:
            logger.info(
                "  Epoch %3d/%d | train_acc=%.3f | val_acc=%.3f (best=%.3f)",
                epoch, epochs, train_acc, val_acc, best_val_acc,
            )

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)

    metadata = {
        "class_names": class_names,
        "num_classes": num_classes,
        "input_size": INPUT_SIZE,
        "target_frames": TARGET_FRAMES,
        "best_val_acc": best_val_acc,
        "epochs": epochs,
    }

    return model, metadata


def save_checkpoint(
    model: TrickMLP,
    metadata: dict,
    path: Path | str,
) -> None:
    """Save model checkpoint with metadata."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "metadata": metadata,
    }, path)
    logger.info("Saved checkpoint to %s", path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PkVision MLP classifier")
    parser.add_argument("--references-dir", type=str, default="data/references",
                        help="Directory with reference .npy files per trick")
    parser.add_argument("--output", type=str, default="data/models/mlp_v1.pt",
                        help="Output checkpoint path")
    parser.add_argument("--samples-per-trick", type=int, default=500,
                        help="Synthetic samples to generate per trick")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = detect_device(args.device)
    logger.info("Device: %s", device)

    # Load references
    logger.info("Loading references from %s", args.references_dir)
    refs = load_references(Path(args.references_dir))
    if not refs:
        logger.error("No references found. Add .npy files to %s/{trick_id}/", args.references_dir)
        sys.exit(1)

    logger.info("Found %d tricks with references", len(refs))

    # Generate training data
    logger.info("Generating training data (%d samples/trick)...", args.samples_per_trick)
    X, y, class_names = generate_training_data(
        refs, samples_per_trick=args.samples_per_trick, seed=args.seed,
    )
    logger.info("Total: %d samples, %d classes", len(X), len(class_names))

    # Train
    model, metadata = train(
        X, y, class_names,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
    )

    # Save
    save_checkpoint(model, metadata, Path(args.output))
    logger.info("Done! Best val accuracy: %.1f%%", metadata["best_val_acc"] * 100)


if __name__ == "__main__":
    main()
