#!/usr/bin/env python3
"""Fine-tune VideoMAE on Modal T4 GPU for parkour trick classification.

Uses modal.Mount to send training frames directly — no volume upload needed.
Model weights saved to Modal volume for retrieval.

Usage:
    python scripts/train_videomae_modal.py
    python scripts/train_videomae_modal.py --epochs 30 --batch-size 4
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import modal

LOCAL_FRAMES_DIR = Path("data/v5_training/frames")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1-mesa-glx", "libglib2.0-0", "ffmpeg")
    .pip_install(
        "torch", "torchvision", "transformers",
        "numpy", "accelerate",
    )
    .add_local_dir(str(LOCAL_FRAMES_DIR), remote_path="/frames")
)

app = modal.App("pkvision-train", image=image)
volume = modal.Volume.from_name("pkvision-data", create_if_missing=True)

VOLUME_MOUNT = "/vol"
FRAMES_MOUNT = "/frames"
MODEL_DIR = f"{VOLUME_MOUNT}/v5_models"


@app.function(
    gpu="T4",
    timeout=3600,
    volumes={VOLUME_MOUNT: volume},
)
def train_videomae(
    manifest: dict,
    epochs: int = 20,
    batch_size: int = 4,
    lr: float = 5e-5,
    freeze_backbone: bool = True,
) -> dict:
    """Train VideoMAE on the v5 dataset. Runs on Modal T4 GPU."""
    import time

    import numpy as np
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
    from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor

    # ── Dataset ──────────────────────────────────────────────────────────
    class PkVisionV5Dataset(Dataset):
        def __init__(self, frames_dir, samples, classes, processor, num_frames=16):
            self.frames_dir = Path(frames_dir)
            self.processor = processor
            self.num_frames = num_frames
            self.classes = classes
            self.class_to_idx = {c: i for i, c in enumerate(classes)}
            self.samples = [
                (s["file"], self.class_to_idx[s["class"]])
                for s in samples
                if s["class"] in self.class_to_idx
            ]

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            filename, label = self.samples[idx]
            frames = np.load(self.frames_dir / filename)  # (T, H, W, 3)
            T = frames.shape[0]

            if T >= self.num_frames:
                indices = np.linspace(0, T - 1, self.num_frames, dtype=int)
            else:
                indices = list(range(T))
                while len(indices) < self.num_frames:
                    indices.append(indices[-1])
                indices = indices[: self.num_frames]

            sampled = [frames[i] for i in indices]
            inputs = self.processor(sampled, return_tensors="pt")
            pixel_values = inputs["pixel_values"].squeeze(0)
            return pixel_values, label

    # ── Locate frames ────────────────────────────────────────────────────
    device = torch.device("cuda")
    classes = manifest["classes"]
    num_classes = len(classes)

    # Frames are in the mount at /frames/{category}/{file}.npy
    frames_dir = Path(FRAMES_MOUNT)
    all_npy = list(frames_dir.rglob("*.npy"))
    print(f"  Frames found: {len(all_npy)} .npy files in {FRAMES_MOUNT}")

    # Build sample lookup: file -> sample dict
    all_samples = manifest.get("samples", [])
    file_to_sample = {s["file"]: s for s in all_samples}

    # Use explicit splits if available, otherwise random 85/15
    splits = manifest.get("splits", {})
    if splits.get("train") and splits.get("val"):
        train_files = set(splits["train"])
        val_files = set(splits["val"])
        train_samples = [file_to_sample[f] for f in train_files if f in file_to_sample]
        val_samples = [file_to_sample[f] for f in val_files if f in file_to_sample]
    else:
        import random
        random.seed(42)
        shuffled = list(all_samples)
        random.shuffle(shuffled)
        split_idx = max(1, int(len(shuffled) * 0.85))
        train_samples = shuffled[:split_idx]
        val_samples = shuffled[split_idx:]

    print()
    print("  PkVision VideoMAE Training (Modal T4)")
    print("  " + "=" * 45)
    print(f"  Device:       {device} ({torch.cuda.get_device_name()})")
    print(f"  Classes:      {num_classes}")
    print(f"  Train:        {len(train_samples)} samples")
    print(f"  Val:          {len(val_samples)} samples")
    print(f"  Epochs:       {epochs}")
    print(f"  Batch size:   {batch_size}")
    print(f"  LR:           {lr}")
    print(f"  Strategy:     {'head only' if freeze_backbone else 'full model'}")
    print()

    # ── Model ────────────────────────────────────────────────────────────
    print("  Loading VideoMAE...", end=" ", flush=True)
    model_name = "MCG-NJU/videomae-base-finetuned-kinetics"
    processor = VideoMAEImageProcessor.from_pretrained(model_name)
    model = VideoMAEForVideoClassification.from_pretrained(model_name)

    hidden_size = model.classifier.in_features
    model.classifier = nn.Linear(hidden_size, num_classes)
    model.config.num_labels = num_classes
    model.config.id2label = {i: c for i, c in enumerate(classes)}
    model.config.label2id = {c: i for i, c in enumerate(classes)}

    if freeze_backbone:
        for param in model.videomae.parameters():
            param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"OK ({trainable:,}/{total_params:,} trainable)")

    model = model.to(device)

    # ── Datasets & DataLoaders ───────────────────────────────────────────
    print("  Loading datasets...", end=" ", flush=True)
    train_ds = PkVisionV5Dataset(FRAMES_MOUNT, train_samples, classes, processor)
    val_ds = PkVisionV5Dataset(FRAMES_MOUNT, val_samples, classes, processor)
    print(f"OK ({len(train_ds)} train, {len(val_ds)} val)")

    if len(train_ds) < 3:
        raise RuntimeError(f"Too few training samples ({len(train_ds)}). Need at least 3.")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=2)

    # ── Optimizer & Scheduler ────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=0.01,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # ── Training Loop ────────────────────────────────────────────────────
    best_val_acc = 0.0
    best_epoch = 0
    output_path = Path(MODEL_DIR) / "pkvision_videomae_v5.pt"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print()
    for epoch in range(epochs):
        t0 = time.time()

        # Train
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = model(pixel_values=bx)
            loss = criterion(out.logits, by)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * bx.size(0)
            train_correct += (out.logits.argmax(1) == by).sum().item()
            train_total += bx.size(0)
        scheduler.step()

        # Validate
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                out = model(pixel_values=bx)
                loss = criterion(out.logits, by)
                val_loss += loss.item() * bx.size(0)
                val_correct += (out.logits.argmax(1) == by).sum().item()
                val_total += bx.size(0)

        train_acc = train_correct / max(train_total, 1)
        val_acc = val_correct / max(val_total, 1)
        elapsed = time.time() - t0

        saved_marker = ""
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "classes": classes,
                    "epoch": epoch,
                    "val_acc": best_val_acc,
                    "config": {
                        "model_name": model_name,
                        "num_classes": num_classes,
                        "hidden_size": hidden_size,
                    },
                },
                output_path,
            )
            volume.commit()
            saved_marker = " *"

        print(
            f"  Epoch {epoch + 1:3d}/{epochs} | "
            f"train {train_loss / max(train_total, 1):.4f} {train_acc:.1%} | "
            f"val {val_loss / max(val_total, 1):.4f} {val_acc:.1%} | "
            f"{elapsed:.1f}s{saved_marker}"
        )

    print()
    print(f"  Done! Best val accuracy: {best_val_acc:.1%} (epoch {best_epoch + 1})")
    print(f"  Model saved to volume: {output_path}")

    return {
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
        "classes": classes,
        "num_classes": num_classes,
        "model_path": str(output_path),
    }


@app.local_entrypoint()
def main(
    data_dir: str = "data/v5_training",
    output: str = "data/models/pkvision_videomae_v5.pt",
    epochs: int = 20,
    batch_size: int = 4,
    lr: float = 5e-5,
    no_freeze: bool = False,
):
    data_path = Path(data_dir)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    manifest_path = data_path / "manifest.json"
    if not manifest_path.exists():
        print(f"Manifest not found: {manifest_path}")
        print("Run scripts/build_v5_dataset.py first.")
        sys.exit(1)

    with open(manifest_path) as f:
        manifest = json.load(f)

    classes = manifest["classes"]
    all_samples = manifest.get("samples", [])
    frames_dir = data_path / "frames"

    print()
    print("  PkVision Modal Trainer")
    print("  " + "=" * 45)
    print(f"  Local data:   {data_path}")
    print(f"  Classes:      {len(classes)}")
    print(f"  Samples:      {len(all_samples)}")
    print(f"  Epochs:       {epochs}")
    print(f"  Batch size:   {batch_size}")
    print(f"  Strategy:     {'head only' if not no_freeze else 'full model'}")
    print()

    # ── Launch training on Modal GPU ────────────────────────────────────
    # Frames are baked into the image via add_local_dir
    data_size = sum(f.stat().st_size for f in frames_dir.rglob("*.npy")) / 1024 / 1024
    print(f"  Launching training on Modal T4 GPU...")
    print(f"  ({data_size:.0f} MB of frame data baked into image)")
    print()

    result = train_videomae.remote(
        manifest=manifest,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        freeze_backbone=not no_freeze,
    )

    print()
    print(f"  Training complete!")
    print(f"  Best val accuracy: {result['best_val_acc']:.1%}")
    print(f"  Best epoch: {result['best_epoch'] + 1}")
    print()

    # ── Download model weights from volume ───────────────────────────────
    print(f"  Downloading model weights...")
    remote_model_path = "v5_models/pkvision_videomae_v5.pt"

    try:
        model_bytes = b""
        for chunk in volume.read_file(remote_model_path):
            model_bytes += chunk
        with open(output_path, "wb") as f:
            f.write(model_bytes)
        print(f"  Saved to {output_path} ({len(model_bytes) / 1024 / 1024:.1f} MB)")
    except Exception as e:
        print(f"  Failed to download model: {e}")
        print(f"  Model is still on the volume at: {remote_model_path}")
        sys.exit(1)

    print()
    print(f"  Test: python scripts/inference_v5.py --model {output_path} --input video.mp4")
    print()
