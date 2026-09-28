#!/usr/bin/env python3
"""Train a skeleton-based parkour trick classifier.

Pipeline:
  1. Extract YOLO keypoints from all parkourtheory clips
  2. Map clip names to FIG trick families
  3. Train a Transformer classifier on keypoint sequences
  4. Evaluate on held-out test set

Based on PoseC3D approach but simplified — no MMAction2 dependency.
Uses raw (T, 17, 2) keypoint sequences as input.

Usage:
    # Full pipeline: extract + train + evaluate
    python scripts/train_skeleton_classifier.py

    # Skip extraction if keypoints already saved
    python scripts/train_skeleton_classifier.py --skip-extract

    # Train with specific params
    python scripts/train_skeleton_classifier.py --epochs 100 --batch-size 32 --lr 1e-4

Run on GPU with nohup:
    nohup python -u scripts/train_skeleton_classifier.py > train_skeleton.log 2>&1 &
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ═══════════════════════════════════════════════════════════════════════
# STEP 1: KEYPOINT EXTRACTION
# ═══════════════════════════════════════════════════════════════════════

def extract_all_keypoints(
    clips_dir: Path,
    output_path: Path,
    max_frames: int = 64,
):
    """Extract YOLO keypoints from all clips and save as pickle."""
    from ultralytics import YOLO

    clips = sorted(clips_dir.glob("*.mp4")) + sorted(clips_dir.glob("*.webm")) + sorted(clips_dir.glob("*.mov"))
    print(f"Extracting keypoints from {len(clips)} clips...")

    yolo = YOLO("yolo11n-pose.pt")
    dataset = []

    for i, clip_path in enumerate(clips):
        if (i + 1) % 50 == 0 or i == 0:
            print(f"  [{i+1}/{len(clips)}] {clip_path.name}")

        try:
            import cv2
            cap = cv2.VideoCapture(str(clip_path))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Sample frames uniformly
            if total_frames <= 0:
                cap.release()
                continue

            n_sample = min(total_frames, max_frames)
            indices = np.linspace(0, total_frames - 1, n_sample, dtype=int)

            keypoints = []
            confidences = []

            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                ret, frame = cap.read()
                if not ret:
                    keypoints.append(np.zeros((17, 2)))
                    confidences.append(np.zeros(17))
                    continue

                results = yolo(frame, conf=0.25, verbose=False)

                kp = np.zeros((17, 2))
                conf = np.zeros(17)

                if results and results[0].keypoints is not None and len(results[0].keypoints) > 0:
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                    best = int(np.argmax(areas))

                    kp_data = results[0].keypoints.xy.cpu().numpy()
                    conf_data = results[0].keypoints.conf.cpu().numpy() if results[0].keypoints.conf is not None else None

                    if best < len(kp_data) and kp_data[best].shape[0] >= 17:
                        kp = kp_data[best][:17]
                        if conf_data is not None and best < len(conf_data):
                            conf = conf_data[best][:17]
                        else:
                            conf = np.where(kp.sum(axis=1) > 0, 1.0, 0.0)

                keypoints.append(kp)
                confidences.append(conf)

            cap.release()

            # Normalize keypoints: center on hip midpoint, scale by torso length
            kp_array = np.array(keypoints)  # (T, 17, 2)
            conf_array = np.array(confidences)  # (T, 17)

            # Center on hip midpoint
            hip_center = (kp_array[:, 11] + kp_array[:, 12]) / 2  # (T, 2)
            hip_valid = (conf_array[:, 11] > 0.3) & (conf_array[:, 12] > 0.3)
            if np.any(hip_valid):
                mean_hip = hip_center[hip_valid].mean(axis=0)
                kp_array = kp_array - mean_hip[None, None, :]

            # Scale by torso length
            shoulder_center = (kp_array[:, 5] + kp_array[:, 6]) / 2
            torso_lengths = np.linalg.norm(shoulder_center - (kp_array[:, 11] + kp_array[:, 12]) / 2, axis=1)
            valid_torso = torso_lengths > 1
            if np.any(valid_torso):
                scale = np.median(torso_lengths[valid_torso])
                if scale > 0:
                    kp_array = kp_array / scale

            dataset.append({
                "clip_name": clip_path.stem,
                "keypoints": kp_array.astype(np.float32),
                "confidences": conf_array.astype(np.float32),
                "fps": fps,
                "n_frames": len(keypoints),
            })

        except Exception as e:
            print(f"  ERROR on {clip_path.name}: {e}")
            continue

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(dataset, f)

    print(f"Saved {len(dataset)} clips to {output_path}")
    return dataset


# ═══════════════════════════════════════════════════════════════════════
# STEP 2: LABEL MAPPING
# ═══════════════════════════════════════════════════════════════════════

def build_label_mapping(dataset: list[dict], fig_path: Path) -> tuple[dict, list[str]]:
    """Map parkourtheory clip names to FIG trick families.

    Returns (clip_name → class_index, class_names list).
    """
    with open(fig_path) as f:
        fig_data = json.load(f)

    # Build reverse mapping: parkourtheory_name → FIG trick name
    pt_to_fig = {}
    for cat_key, cat_data in fig_data.get("categories", {}).items():
        for trick in cat_data.get("tricks", []):
            fig_name = trick["name"]
            # Create a simplified key from the FIG name
            fig_key = fig_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
            pt_to_fig[fig_key] = fig_name
            # Also index aliases
            for alias in trick.get("aliases", []):
                alias_key = alias.lower().replace(" ", "_").replace("(", "").replace(")", "")
                pt_to_fig[alias_key] = fig_name

    # Map clips to FIG families based on physics properties
    # Group by: (flip_count, twist_bucket, direction)
    # This creates coarser classes that are learnable with limited data
    families = {}
    clip_labels = {}

    for item in dataset:
        clip_name = item["clip_name"].lower()

        # Try direct name matching
        matched = False
        for pt_key, fig_name in pt_to_fig.items():
            # Fuzzy match: check if clip name contains key words
            clip_words = set(clip_name.replace("_", " ").split())
            key_words = set(pt_key.replace("_", " ").split())
            overlap = len(clip_words & key_words) / max(len(key_words), 1)
            if overlap >= 0.6:
                family = _get_family(fig_name, fig_data)
                if family not in families:
                    families[family] = len(families)
                clip_labels[clip_name] = families[family]
                matched = True
                break

        if not matched:
            # Assign to "unknown" family — still useful for training
            family = "other"
            if family not in families:
                families[family] = len(families)
            clip_labels[clip_name] = families[family]

    class_names = [""] * len(families)
    for name, idx in families.items():
        class_names[idx] = name

    return clip_labels, class_names


def _get_family(fig_name: str, fig_data: dict) -> str:
    """Get the physics family for a FIG trick (for grouping similar tricks)."""
    for cat_key, cat_data in fig_data.get("categories", {}).items():
        for trick in cat_data.get("tricks", []):
            if trick["name"] == fig_name:
                flip = trick.get("flip", 0)
                twist = trick.get("twist", 0)
                direction = trick.get("direction", "unknown") or "unknown"
                # Create family string
                twist_str = f"_{twist}t" if twist > 0 else ""
                return f"{cat_key}_{direction}_{flip}f{twist_str}"
    return "other"


# ═══════════════════════════════════════════════════════════════════════
# STEP 3: DATASET & MODEL
# ═══════════════════════════════════════════════════════════════════════

class SkeletonDataset(Dataset):
    """Keypoint sequence dataset for trick classification."""

    def __init__(self, data: list[dict], labels: dict, seq_len: int = 64):
        self.samples = []
        self.labels_list = []
        self.seq_len = seq_len

        for item in data:
            clip_name = item["clip_name"].lower()
            if clip_name not in labels:
                continue

            kp = item["keypoints"]  # (T, 17, 2)
            conf = item["confidences"]  # (T, 17)

            # Pad or truncate to seq_len
            T = len(kp)
            if T >= seq_len:
                indices = np.linspace(0, T - 1, seq_len, dtype=int)
                kp = kp[indices]
                conf = conf[indices]
            else:
                pad = seq_len - T
                kp = np.pad(kp, ((0, pad), (0, 0), (0, 0)))
                conf = np.pad(conf, ((0, pad), (0, 0)))

            # Flatten: (seq_len, 17, 2) → (seq_len, 34)
            # Also add confidence as feature: (seq_len, 17*3)
            features = np.concatenate([
                kp.reshape(seq_len, -1),      # (seq_len, 34) — xy coords
                conf.reshape(seq_len, -1),     # (seq_len, 17) — confidences
            ], axis=1)  # (seq_len, 51)

            self.samples.append(torch.FloatTensor(features))
            self.labels_list.append(labels[clip_name])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx], self.labels_list[idx]


class SkeletonTransformer(nn.Module):
    """Simple Transformer classifier on keypoint sequences.

    Input: (batch, seq_len, 51) — 17 keypoints × 3 (x, y, conf)
    Output: (batch, num_classes) — trick family logits
    """

    def __init__(self, num_classes: int, d_model: int = 128, nhead: int = 4,
                 num_layers: int = 4, seq_len: int = 64, input_dim: int = 51):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=0.1, batch_first=True, activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, x):
        # x: (batch, seq_len, 51)
        x = self.input_proj(x) + self.pos_embed
        x = self.transformer(x)
        # Global average pooling over sequence
        x = x.mean(dim=1)
        return self.classifier(x)


# ═══════════════════════════════════════════════════════════════════════
# STEP 4: TRAINING
# ═══════════════════════════════════════════════════════════════════════

def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 100,
    lr: float = 3e-4,
    device: str = "cuda",
    save_path: Path = ROOT / "data" / "models" / "pkvision_skeleton.pt",
):
    """Train the skeleton classifier."""
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = torch.tensor(batch_y, dtype=torch.long, device=device)

            logits = model(batch_x)
            loss = F.cross_entropy(logits, batch_y)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item() * len(batch_y)
            train_correct += (logits.argmax(1) == batch_y).sum().item()
            train_total += len(batch_y)

        scheduler.step()

        # Validate
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = torch.tensor(batch_y, dtype=torch.long, device=device)
                logits = model(batch_x)
                val_correct += (logits.argmax(1) == batch_y).sum().item()
                val_total += len(batch_y)

        train_acc = train_correct / max(train_total, 1)
        val_acc = val_correct / max(val_total, 1)
        avg_loss = train_loss / max(train_total, 1)

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{epochs} | Loss: {avg_loss:.4f} | "
                  f"Train: {train_acc:.1%} | Val: {val_acc:.1%} | "
                  f"LR: {scheduler.get_last_lr()[0]:.6f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model_state": model.state_dict(),
                "val_acc": val_acc,
                "epoch": epoch,
            }, save_path)

    print(f"\nBest val accuracy: {best_val_acc:.1%}")
    return best_val_acc


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Train skeleton trick classifier")
    parser.add_argument("--skip-extract", action="store_true",
                        help="Skip keypoint extraction (use saved .pkl)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seq-len", type=int, default=64,
                        help="Number of frames to sample per clip")
    parser.add_argument("--device", default=None,
                        help="Device (auto-detect if not set)")
    args = parser.parse_args()

    # Auto-detect device
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Device: {device}")

    clips_dir = ROOT / "data" / "parkourtheory_clips"
    keypoints_path = ROOT / "data" / "keypoints" / "parkourtheory_keypoints.pkl"
    fig_path = ROOT / "data" / "fig_tricks_2025.json"

    # Step 1: Extract keypoints
    if args.skip_extract and keypoints_path.exists():
        print(f"Loading keypoints from {keypoints_path}")
        with open(keypoints_path, "rb") as f:
            dataset = pickle.load(f)
        print(f"Loaded {len(dataset)} clips")
    else:
        dataset = extract_all_keypoints(clips_dir, keypoints_path, max_frames=args.seq_len)

    # Step 2: Build labels
    print("\nBuilding label mapping...")
    clip_labels, class_names = build_label_mapping(dataset, fig_path)
    num_classes = len(class_names)
    labeled_count = sum(1 for item in dataset if item["clip_name"].lower() in clip_labels)
    print(f"Classes: {num_classes} | Labeled clips: {labeled_count}/{len(dataset)}")

    # Show class distribution
    label_counts = Counter(clip_labels.values())
    print("\nTop 15 classes:")
    for class_idx, count in label_counts.most_common(15):
        print(f"  {class_names[class_idx]:<40} {count:>4} clips")

    # Step 3: Create datasets
    # 80/20 train/val split
    full_dataset = SkeletonDataset(dataset, clip_labels, seq_len=args.seq_len)
    print(f"\nTotal samples: {len(full_dataset)}")

    if len(full_dataset) < 10:
        print("ERROR: Too few labeled samples. Check label mapping.")
        return

    n_val = max(int(len(full_dataset) * 0.2), 1)
    n_train = len(full_dataset) - n_val
    train_set, val_set = torch.utils.data.random_split(
        full_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)

    print(f"Train: {n_train} | Val: {n_val}")

    # Step 4: Train
    model = SkeletonTransformer(
        num_classes=num_classes,
        d_model=128,
        nhead=4,
        num_layers=4,
        seq_len=args.seq_len,
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: SkeletonTransformer ({total_params:,} params)")
    print(f"Training for {args.epochs} epochs...")
    print(f"{'='*60}\n")

    best_acc = train(
        model, train_loader, val_loader,
        epochs=args.epochs, lr=args.lr, device=device,
    )

    print(f"\nDone! Best validation accuracy: {best_acc:.1%}")
    print(f"Model saved to: data/models/pkvision_skeleton.pt")


if __name__ == "__main__":
    main()
