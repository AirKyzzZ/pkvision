#!/usr/bin/env python3
"""Fine-tune VideoMAE on PkVision parkour trick dataset.

Takes the pre-trained Kinetics-400 VideoMAE model and fine-tunes it
on your labeled parkour clips to distinguish specific trick types.

Supports MPS (Apple Silicon), CUDA, and CPU.

Usage:
    python scripts/finetune.py
    python scripts/finetune.py --epochs 30 --batch-size 4 --device mps
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor

sys.path.insert(0, str(Path(__file__).parent.parent))


class PkVisionDataset(Dataset):
    def __init__(self, frames_dir, manifest, processor, num_frames=16):
        self.frames_dir = Path(frames_dir)
        self.processor = processor
        self.num_frames = num_frames
        self.classes = manifest["classes"]
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.samples = [
            (s["file"], self.class_to_idx[s["class"]])
            for s in manifest["samples"]
            if s["class"] in self.class_to_idx
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filename, label = self.samples[idx]
        frames = np.load(self.frames_dir / filename)
        T = frames.shape[0]
        if T >= self.num_frames:
            indices = np.linspace(0, T - 1, self.num_frames, dtype=int)
        else:
            indices = list(range(T))
            while len(indices) < self.num_frames:
                indices.append(indices[-1])
            indices = indices[:self.num_frames]
        sampled = [frames[i] for i in indices]
        inputs = self.processor(sampled, return_tensors="pt")
        pixel_values = inputs["pixel_values"].squeeze(0)
        return pixel_values, label


def get_device(override=None):
    if override:
        return torch.device(override)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune VideoMAE for parkour tricks")
    parser.add_argument("--data-dir", default="data/training")
    parser.add_argument("--output", default="data/models/pkvision_videomae.pt")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--device")
    parser.add_argument("--no-freeze", action="store_true", help="Train full model instead of just the head")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    device = get_device(args.device)
    freeze_backbone = not args.no_freeze

    manifest_path = data_dir / "manifest.json"
    if not manifest_path.exists():
        print("Run prepare_training.py first.")
        sys.exit(1)

    with open(manifest_path) as f:
        manifest = json.load(f)

    classes = manifest["classes"]
    num_classes = len(classes)

    print()
    print("  PkVision Fine-Tuning")
    print("  " + "=" * 40)
    print(f"  Device:     {device}")
    print(f"  Classes:    {num_classes} ({', '.join(classes)})")
    print(f"  Samples:    {len(manifest['samples'])}")
    print(f"  Epochs:     {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Strategy:   {'head only' if freeze_backbone else 'full model'}")
    print()

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

    print("  Loading dataset...", end=" ", flush=True)
    dataset = PkVisionDataset(data_dir / "frames", manifest, processor)
    print(f"OK ({len(dataset)} samples)")

    if len(dataset) < 5:
        print("  Too few samples.")
        sys.exit(1)

    val_size = max(1, int(len(dataset) * args.val_split))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    print(f"  Split: {train_size} train, {val_size} val\n")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.01,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_acc = 0.0

    for epoch in range(args.epochs):
        t0 = time.time()

        model.train()
        tl, tc, tt = 0.0, 0, 0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = model(pixel_values=bx)
            loss = criterion(out.logits, by)
            loss.backward()
            optimizer.step()
            tl += loss.item() * bx.size(0)
            tc += (out.logits.argmax(1) == by).sum().item()
            tt += bx.size(0)
        scheduler.step()

        model.eval()
        vl, vc, vt = 0.0, 0, 0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                out = model(pixel_values=bx)
                loss = criterion(out.logits, by)
                vl += loss.item() * bx.size(0)
                vc += (out.logits.argmax(1) == by).sum().item()
                vt += bx.size(0)

        ta = tc / max(tt, 1)
        va = vc / max(vt, 1)
        elapsed = time.time() - t0
        if device.type == "mps":
            torch.mps.synchronize()

        print(
            f"  Epoch {epoch+1:3d}/{args.epochs} | "
            f"train {tl/max(tt,1):.4f} {ta:.0%} | "
            f"val {vl/max(vt,1):.4f} {va:.0%} | "
            f"{elapsed:.1f}s"
        )

        if va >= best_val_acc:
            best_val_acc = va
            torch.save({
                "model_state_dict": model.state_dict(),
                "classes": classes,
                "epoch": epoch,
                "val_acc": best_val_acc,
                "config": {"model_name": model_name, "num_classes": num_classes, "hidden_size": hidden_size},
            }, output_path)

    print(f"\n  Done! Best val accuracy: {best_val_acc:.0%}")
    print(f"  Model: {output_path}")
    print(f"\n  Test: python scripts/test_model.py --input video.mp4\n")


if __name__ == "__main__":
    main()
