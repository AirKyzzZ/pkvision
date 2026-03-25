"""Training script for the TCN trick classifier.

Trains on pure physics-generated feature sequences — no manually labeled data needed.
Each trick is simulated from its kinematic definition with thousands of variations.

Usage:
    python -m ml.tcn.train --output data/models/tcn_v1.pt --epochs 80 --device auto

    # On Windows dev server via SSH:
    ssh windows-dev "cd pkvision && python -m ml.tcn.train --device cuda"
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from core.pose.features import FEATURES_PER_FRAME

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

TARGET_FRAMES = 64


def detect_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def generate_training_data(
    samples_per_trick: int = 2000,
    no_trick_samples: int = 2000,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Generate training data from physics simulation.

    Returns:
        X: (n_samples, FEATURES_PER_FRAME, TARGET_FRAMES) channels-first for TCN.
        y: (n_samples,) integer class labels.
        class_names: list of class names (index 0 = "no_trick").
    """
    from ml.physics_generator import PhysicsFeatureGenerator

    gen = PhysicsFeatureGenerator(target_frames=TARGET_FRAMES, seed=seed)

    logger.info("Generating physics-based training data (%d samples/trick)...", samples_per_trick)

    all_data = gen.generate_all(samples_per_trick=samples_per_trick)

    # Class 0 = no_trick, rest sorted alphabetically
    trick_names = sorted(k for k in all_data.keys() if k != "no_trick")
    class_names = ["no_trick"] + trick_names
    class_to_idx = {name: i for i, name in enumerate(class_names)}

    all_X: list[np.ndarray] = []
    all_y: list[int] = []

    for class_name in class_names:
        if class_name not in all_data:
            continue
        label = class_to_idx[class_name]
        arrays = all_data[class_name]
        for arr in arrays:
            # arr is (T, 75) — interpolate if needed, then transpose to (75, T) for TCN
            if arr.shape[0] != TARGET_FRAMES:
                x_old = np.linspace(0, 1, arr.shape[0])
                x_new = np.linspace(0, 1, TARGET_FRAMES)
                arr_interp = np.zeros((TARGET_FRAMES, arr.shape[1]), dtype=np.float32)
                for j in range(arr.shape[1]):
                    arr_interp[:, j] = np.interp(x_new, x_old, arr[:, j])
                arr = arr_interp

            all_X.append(arr.T)  # (75, T)
            all_y.append(label)

        logger.info("  %s: %d samples (class %d)", class_name, len(arrays), label)

    X = np.array(all_X, dtype=np.float32)
    y = np.array(all_y, dtype=np.int64)

    # Replace NaN with 0
    nan_count = np.isnan(X).sum()
    if nan_count > 0:
        logger.info("  Replacing %d NaN values with 0", nan_count)
        np.nan_to_num(X, copy=False, nan=0.0)

    # Shuffle
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(X))
    return X[perm], y[perm], class_names


def train(
    X: np.ndarray,
    y: np.ndarray,
    class_names: list[str],
    epochs: int = 80,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    val_split: float = 0.15,
    device: torch.device | None = None,
) -> tuple[nn.Module, dict]:
    """Train the TCN classifier.

    Args:
        X: (n_samples, n_features, n_frames) feature arrays.
        y: (n_samples,) integer labels.
        class_names: list of class names.
        epochs: Training epochs.
        batch_size: Batch size.
        lr: Learning rate.
        weight_decay: L2 regularization.
        val_split: Fraction for validation.
        device: Device to train on.

    Returns:
        model: Trained TrickTCN.
        metadata: Training metadata.
    """
    from ml.tcn.model import TrickTCN

    if device is None:
        device = detect_device()

    num_classes = len(class_names)
    n_val = int(len(X) * val_split)
    n_train = len(X) - n_val

    X_train, X_val = X[:n_train], X[n_train:]
    y_train, y_val = y[:n_train], y[n_train:]

    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    val_ds = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.long),
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    model = TrickTCN(
        num_classes=num_classes,
        n_features=FEATURES_PER_FRAME,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr * 10, epochs=epochs,
        steps_per_epoch=len(train_loader),
    )
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_state = None

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(
        "Training TCN: %d classes, %d train, %d val, %d params, device=%s",
        num_classes, n_train, n_val, n_params, device,
    )

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0

        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            train_loss += loss.item() * batch_X.size(0)
            train_correct += (logits.argmax(dim=1) == batch_y).sum().item()

        train_acc = train_correct / n_train

        # Validate
        model.train(False)
        val_correct = 0
        val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                logits = model(batch_X)
                val_loss += criterion(logits, batch_y).item() * batch_X.size(0)
                val_correct += (logits.argmax(dim=1) == batch_y).sum().item()

        val_acc = val_correct / max(n_val, 1)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == 1 or epoch == epochs:
            logger.info(
                "  Epoch %3d/%d | train_acc=%.3f loss=%.4f | val_acc=%.3f (best=%.3f)",
                epoch, epochs, train_acc, train_loss / n_train, val_acc, best_val_acc,
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    metadata = {
        "class_names": class_names,
        "num_classes": num_classes,
        "n_features": FEATURES_PER_FRAME,
        "target_frames": TARGET_FRAMES,
        "best_val_acc": best_val_acc,
        "epochs": epochs,
        "model_type": "tcn",
    }

    return model, metadata


def save_checkpoint(model: nn.Module, metadata: dict, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "metadata": metadata,
    }, path)
    logger.info("Saved checkpoint to %s", path)


def generate_hierarchical_data(
    samples_per_trick: int = 2000,
    seed: int = 42,
) -> tuple[np.ndarray, dict[str, np.ndarray], list[str]]:
    """Generate training data with hierarchical labels.

    Returns:
        X: (n_samples, FEATURES_PER_FRAME, TARGET_FRAMES) channels-first for TCN.
        labels: dict of head_name -> (n_samples,) integer labels.
        class_names: list of trick names for reference.
    """
    from ml.physics_generator import PhysicsFeatureGenerator
    from ml.tcn.hierarchical import encode_trick, NO_TRICK_LABELS, HEAD_SIZES
    from ml.trick_physics import TRICK_DEFINITIONS

    gen = PhysicsFeatureGenerator(target_frames=TARGET_FRAMES, seed=seed)

    logger.info("Generating hierarchical training data (%d samples/trick)...", samples_per_trick)
    all_data = gen.generate_all(samples_per_trick=samples_per_trick)

    trick_names = sorted(k for k in all_data.keys() if k != "no_trick")
    class_names = ["no_trick"] + trick_names

    all_X: list[np.ndarray] = []
    all_labels: dict[str, list[int]] = {name: [] for name in HEAD_SIZES}
    all_labels["is_trick"] = []

    for class_name in class_names:
        if class_name not in all_data:
            continue
        arrays = all_data[class_name]

        if class_name == "no_trick":
            labels = NO_TRICK_LABELS
            is_trick = 0
        else:
            trick_def = TRICK_DEFINITIONS[class_name]
            labels = encode_trick(trick_def)
            is_trick = 1

        label_dict = labels.to_dict()

        for arr in arrays:
            if arr.shape[0] != TARGET_FRAMES:
                x_old = np.linspace(0, 1, arr.shape[0])
                x_new = np.linspace(0, 1, TARGET_FRAMES)
                arr_interp = np.zeros((TARGET_FRAMES, arr.shape[1]), dtype=np.float32)
                for j in range(arr.shape[1]):
                    arr_interp[:, j] = np.interp(x_new, x_old, arr[:, j])
                arr = arr_interp

            all_X.append(arr.T)
            for name in HEAD_SIZES:
                all_labels[name].append(label_dict[name])
            all_labels["is_trick"].append(is_trick)

        logger.info("  %s: %d samples (is_trick=%d)", class_name, len(arrays), is_trick)

    X = np.array(all_X, dtype=np.float32)
    np.nan_to_num(X, copy=False, nan=0.0)

    label_arrays = {
        name: np.array(vals, dtype=np.int64)
        for name, vals in all_labels.items()
    }

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(X))
    X = X[perm]
    for name in label_arrays:
        label_arrays[name] = label_arrays[name][perm]

    return X, label_arrays, class_names


def train_hierarchical(
    X: np.ndarray,
    labels: dict[str, np.ndarray],
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    val_split: float = 0.15,
    device: torch.device | None = None,
) -> tuple[nn.Module, dict]:
    """Train the hierarchical TCN classifier."""
    from ml.tcn.hierarchical import HierarchicalTrickTCN, HierarchicalLoss, HEAD_SIZES

    if device is None:
        device = detect_device()

    n_val = int(len(X) * val_split)
    n_train = len(X) - n_val

    # Split data
    X_train, X_val = X[:n_train], X[n_train:]
    labels_train = {k: v[:n_train] for k, v in labels.items()}
    labels_val = {k: v[n_train:] for k, v in labels.items()}

    # Build datasets — pack all labels into a single tensor for DataLoader
    all_head_names = list(HEAD_SIZES.keys()) + ["is_trick"]
    train_label_tensor = torch.stack([
        torch.tensor(labels_train[name], dtype=torch.long)
        for name in all_head_names
    ], dim=1)  # (N, 7)
    val_label_tensor = torch.stack([
        torch.tensor(labels_val[name], dtype=torch.long)
        for name in all_head_names
    ], dim=1)

    from torch.utils.data import DataLoader, TensorDataset
    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32), train_label_tensor,
    )
    val_ds = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32), val_label_tensor,
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    model = HierarchicalTrickTCN(n_features=FEATURES_PER_FRAME).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr * 10, epochs=epochs,
        steps_per_epoch=len(train_loader),
    )
    criterion = HierarchicalLoss(is_trick_weight=2.0)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(
        "Training Hierarchical TCN: %d heads, %d train, %d val, %d params, device=%s",
        len(all_head_names), n_train, n_val, n_params, device,
    )

    best_val_acc = 0.0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        train_correct = {name: 0 for name in all_head_names}
        epoch_loss = 0.0

        for batch_X, batch_labels in train_loader:
            batch_X = batch_X.to(device)
            target_dict = {
                name: batch_labels[:, i].to(device)
                for i, name in enumerate(all_head_names)
            }

            optimizer.zero_grad()
            predictions = model(batch_X)
            loss, _ = criterion(predictions, target_dict)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item() * batch_X.size(0)

            for i, name in enumerate(all_head_names):
                preds = predictions[name].argmax(dim=1)
                train_correct[name] += (preds == target_dict[name]).sum().item()

        train_accs = {name: c / n_train for name, c in train_correct.items()}

        # Validate
        model.train(False)
        val_correct = {name: 0 for name in all_head_names}
        with torch.no_grad():
            for batch_X, batch_labels in val_loader:
                batch_X = batch_X.to(device)
                target_dict = {
                    name: batch_labels[:, i].to(device)
                    for i, name in enumerate(all_head_names)
                }
                predictions = model(batch_X)
                for i, name in enumerate(all_head_names):
                    preds = predictions[name].argmax(dim=1)
                    val_correct[name] += (preds == target_dict[name]).sum().item()

        val_accs = {name: c / max(n_val, 1) for name, c in val_correct.items()}
        mean_val_acc = sum(val_accs.values()) / len(val_accs)

        if mean_val_acc > best_val_acc:
            best_val_acc = mean_val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == 1 or epoch == epochs:
            head_str = " | ".join(f"{n}={val_accs[n]:.2f}" for n in ["axis", "rotation", "shape", "is_trick"])
            logger.info(
                "  Epoch %3d/%d | loss=%.4f | val: %s | mean=%.3f (best=%.3f)",
                epoch, epochs, epoch_loss / n_train, head_str, mean_val_acc, best_val_acc,
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    metadata = {
        "n_features": FEATURES_PER_FRAME,
        "target_frames": TARGET_FRAMES,
        "best_val_acc": best_val_acc,
        "per_head_best": {name: 0.0 for name in all_head_names},
        "epochs": epochs,
        "model_type": "hierarchical_tcn",
        "head_sizes": dict(HEAD_SIZES),
    }

    return model, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PkVision TCN classifier (physics-based)")
    parser.add_argument("--output", type=str, default="data/models/tcn_v1.pt",
                        help="Output checkpoint path")
    parser.add_argument("--samples-per-trick", type=int, default=2000,
                        help="Synthetic samples per trick")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hierarchical", action="store_true",
                        help="Use hierarchical multi-head architecture (recommended)")
    args = parser.parse_args()

    device = detect_device(args.device)
    logger.info("Device: %s", device)

    if args.hierarchical:
        X, labels, class_names = generate_hierarchical_data(
            samples_per_trick=args.samples_per_trick,
            seed=args.seed,
        )
        logger.info("Total: %d samples, shape=%s", len(X), X.shape)

        model, metadata = train_hierarchical(
            X, labels,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=device,
        )
    else:
        X, y, class_names = generate_training_data(
            samples_per_trick=args.samples_per_trick,
            seed=args.seed,
        )
        logger.info("Total: %d samples, %d classes, shape=%s", len(X), len(class_names), X.shape)

        model, metadata = train(
            X, y, class_names,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=device,
        )

    save_checkpoint(model, metadata, Path(args.output))
    logger.info("Done! Best val accuracy: %.1f%%", metadata["best_val_acc"] * 100)


if __name__ == "__main__":
    main()
