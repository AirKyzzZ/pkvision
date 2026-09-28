#!/usr/bin/env python3
"""Train VideoMAE on Modal T4 GPU — category and trick classifiers.

Frames are baked into the image via add_local_dir (~4.8GB).
Augmentation happens on-GPU during training (no pre-augmentation needed).

Usage:
    python scripts/train_v5_modal.py
    python scripts/train_v5_modal.py --mode category --epochs 30
    python scripts/train_v5_modal.py --mode trick --epochs 40
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1-mesa-glx", "libglib2.0-0", "ffmpeg")
    .pip_install(
        "torch", "torchvision", "transformers",
        "numpy", "accelerate", "opencv-python-headless",
    )
)

app = modal.App("pkvision-v5", image=image)
volume = modal.Volume.from_name("pkvision-data", create_if_missing=True)

VOL = "/vol"
FRAMES = f"{VOL}/v5_frames"


@app.function(gpu="T4", timeout=14400, volumes={VOL: volume})
def train(
    manifest: dict,
    model_suffix: str = "category",
    epochs: int = 30,
    batch_size: int = 8,
    lr: float = 5e-5,
    freeze_backbone: bool = True,
    warmup_epochs: int = 5,
) -> dict:
    import os
    import random
    import subprocess
    import time

    volume.reload()

    # Extract frames from tar chunks if not already extracted
    frames_path = Path(FRAMES)
    if not frames_path.exists() or not any(frames_path.rglob("*.npy")):
        print("  Extracting frames from tar chunks...")
        data_dir = Path(VOL) / "v5_data"
        chunks = sorted(data_dir.glob("v5_frames_part_*"))
        if chunks:
            # Reconstruct and extract tar
            cat_cmd = " ".join(str(c) for c in chunks)
            subprocess.run(f"cat {cat_cmd} | tar xzf - -C {VOL}/", shell=True, check=True)
            # Frames now at /vol/frames/{category}/{slug}.npy — move to expected path
            extracted = Path(VOL) / "frames"
            if extracted.exists() and not frames_path.exists():
                extracted.rename(frames_path)
            npy_count = len(list(frames_path.rglob("*.npy")))
            print(f"  Extracted {npy_count} frame files")
        else:
            print(f"  WARNING: No tar chunks found at {data_dir}")

    import numpy as np
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
    from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor

    # ── Dataset with on-GPU augmentation ─────────────────────────────────
    class AugDataset(Dataset):
        def __init__(self, frames_dir, samples, classes, processor, num_frames=16, augment=True):
            self.dir = Path(frames_dir)
            self.processor = processor
            self.nf = num_frames
            self.c2i = {c: i for i, c in enumerate(classes)}
            self.samples = [(s["file"], self.c2i[s["class"]]) for s in samples if s["class"] in self.c2i]
            self.augment = augment

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            fname, label = self.samples[idx]
            frames = np.load(self.dir / fname)  # (T, H, W, 3)
            T, H, W, C = frames.shape

            # Temporal sampling with jitter
            if T >= self.nf:
                if self.augment:
                    # Random temporal offset
                    max_start = T - self.nf
                    start = random.randint(0, max(0, max_start))
                    indices = np.linspace(start, min(start + self.nf - 1, T - 1), self.nf, dtype=int)
                else:
                    indices = np.linspace(0, T - 1, self.nf, dtype=int)
            else:
                indices = list(range(T))
                while len(indices) < self.nf:
                    indices.append(indices[-1])
                indices = indices[:self.nf]

            sampled = frames[indices]  # (nf, H, W, C)

            if self.augment:
                # Horizontal flip (50%)
                if random.random() < 0.5:
                    sampled = sampled[:, :, ::-1, :].copy()

                # Random crop (85-100%)
                crop_frac = random.uniform(0.85, 1.0)
                cs = int(H * crop_frac)
                if cs < H:
                    yo = random.randint(0, H - cs)
                    xo = random.randint(0, W - cs)
                    cropped = sampled[:, yo:yo+cs, xo:xo+cs, :]
                    import cv2
                    sampled = np.stack([cv2.resize(f, (W, H)) for f in cropped])

                # Brightness/contrast
                b = random.uniform(-15, 15)
                c = random.uniform(0.9, 1.1)
                sampled = np.clip(sampled.astype(np.float32) * c + b, 0, 255).astype(np.uint8)

            frames_list = [sampled[i] for i in range(sampled.shape[0])]
            inputs = self.processor(frames_list, return_tensors="pt")
            return inputs["pixel_values"].squeeze(0), label

    # ── Setup ────────────────────────────────────────────────────────────
    device = torch.device("cuda")
    classes = manifest["classes"]
    nc = len(classes)

    all_samples = manifest.get("samples", [])
    f2s = {s["file"]: s for s in all_samples}
    splits = manifest.get("splits", {})

    if splits.get("train") and splits.get("val"):
        train_s = [f2s[f] for f in splits["train"] if f in f2s]
        val_s = [f2s[f] for f in splits["val"] if f in f2s]
    else:
        random.seed(42)
        sh = list(all_samples)
        random.shuffle(sh)
        si = max(1, int(len(sh) * 0.85))
        train_s, val_s = sh[:si], sh[si:]

    # Verify data
    npy_count = len(list(Path(FRAMES).rglob("*.npy")))
    print(f"\n  PkVision v5 — {model_suffix.upper()} training")
    print("  " + "=" * 50)
    print(f"  GPU:      {torch.cuda.get_device_name()}")
    print(f"  Frames:   {npy_count} files on disk")
    print(f"  Classes:  {nc}")
    print(f"  Train:    {len(train_s)}")
    print(f"  Val:      {len(val_s)}")
    print(f"  Epochs:   {epochs} (warmup {warmup_epochs})")
    print(f"  Batch:    {batch_size}")
    print(f"  Augment:  on-GPU (flip + crop + color + temporal jitter)")

    # ── Model ────────────────────────────────────────────────────────────
    print("\n  Loading VideoMAE...", end=" ", flush=True)
    hf = "MCG-NJU/videomae-base-finetuned-kinetics"
    proc = VideoMAEImageProcessor.from_pretrained(hf)
    model = VideoMAEForVideoClassification.from_pretrained(hf)
    hs = model.classifier.in_features
    model.classifier = nn.Linear(hs, nc)
    model.config.num_labels = nc

    if freeze_backbone:
        for p in model.videomae.parameters():
            p.requires_grad = False

    tp = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"OK ({tp:,} trainable)")
    model = model.to(device)

    # ── Loaders ──────────────────────────────────────────────────────────
    train_ds = AugDataset(FRAMES, train_s, classes, proc, augment=True)
    val_ds = AugDataset(FRAMES, val_s, classes, proc, augment=False)

    # Weighted sampler for imbalanced classes
    cc = [0] * nc
    for _, l in train_ds.samples:
        cc[l] += 1
    w = [1.0 / max(cc[l], 1) for _, l in train_ds.samples]
    sampler = WeightedRandomSampler(w, len(w))

    tl = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=2)
    vl = DataLoader(val_ds, batch_size=batch_size, num_workers=2)

    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    # ── Train ────────────────────────────────────────────────────────────
    best_va, best_ep = 0.0, 0
    out_path = Path(VOL) / "v5_models" / f"pkvision_{model_suffix}.pt"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print()
    for ep in range(epochs):
        t0 = time.time()

        # Unfreeze backbone after warmup
        if freeze_backbone and ep == warmup_epochs:
            print(f"  *** Unfreezing backbone (epoch {ep+1}) ***")
            for p in model.videomae.parameters():
                p.requires_grad = True
            opt = torch.optim.AdamW([
                {"params": model.videomae.parameters(), "lr": lr * 0.1},
                {"params": model.classifier.parameters(), "lr": lr},
            ], weight_decay=0.01)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs - warmup_epochs)

        model.train()
        tls, tcs, tts = 0.0, 0, 0
        for bx, by in tl:
            bx, by = bx.to(device), by.to(device)
            opt.zero_grad()
            out = model(pixel_values=bx)
            loss = criterion(out.logits, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tls += loss.item() * bx.size(0)
            tcs += (out.logits.argmax(1) == by).sum().item()
            tts += bx.size(0)
        sched.step()

        model.eval()
        vls, vcs, vts = 0.0, 0, 0
        with torch.no_grad():
            for bx, by in vl:
                bx, by = bx.to(device), by.to(device)
                out = model(pixel_values=bx)
                loss = criterion(out.logits, by)
                vls += loss.item() * bx.size(0)
                vcs += (out.logits.argmax(1) == by).sum().item()
                vts += bx.size(0)

        ta = tcs / max(tts, 1)
        va = vcs / max(vts, 1)
        dt = time.time() - t0

        s = ""
        if va > best_va or ep == 0:
            best_va, best_ep = va, ep
            torch.save({
                "model_state_dict": model.state_dict(),
                "classes": classes,
                "epoch": ep, "val_acc": best_va,
                "config": {"model_name": hf, "num_classes": nc, "hidden_size": hs, "type": model_suffix},
            }, out_path)
            volume.commit()
            s = " *"

        print(f"  {ep+1:3d}/{epochs} | train {tls/max(tts,1):.3f} {ta:.1%} | val {vls/max(vts,1):.3f} {va:.1%} | {dt:.0f}s{s}")

    print(f"\n  Best: {best_va:.1%} (epoch {best_ep+1})")
    return {"best_val_acc": best_va, "best_epoch": best_ep, "classes": classes, "model_suffix": model_suffix}


@app.local_entrypoint()
def main(
    data_dir: str = "data/v5_full_training",
    mode: str = "both",
    epochs: int = 30,
    batch_size: int = 8,
    lr: float = 5e-5,
    warmup_epochs: int = 5,
):
    data_path = Path(data_dir)
    out_dir = Path("data/models")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Data is pre-uploaded to volume as tar chunks (v5_data/v5_frames_part_*)
    # Extraction happens inside the GPU function

    modes = ["category", "trick"] if mode == "both" else [mode]

    for m in modes:
        manifest_path = data_path / f"{m}_manifest.json"
        if not manifest_path.exists():
            print(f"Missing: {manifest_path}. Run build_v5_full_dataset.py first.")
            sys.exit(1)

        with open(manifest_path) as f:
            manifest = json.load(f)

        # Strip augmented samples from manifest (augmentation now on-GPU)
        manifest["samples"] = [s for s in manifest["samples"] if "_aug" not in s["file"]]
        if "splits" in manifest:
            for split_name in ("train", "val", "test"):
                if split_name in manifest["splits"]:
                    manifest["splits"][split_name] = [
                        f for f in manifest["splits"][split_name] if "_aug" not in f
                    ]

        nc = len(manifest["classes"])
        ns = len(manifest["samples"])
        bs = min(batch_size, 4) if m == "trick" else batch_size
        ep = epochs if m == "category" else min(epochs + 10, 50)

        print(f"\n  === {m.upper()} model ({nc} classes, {ns} samples) ===")

        result = train.remote(
            manifest=manifest,
            model_suffix=m,
            epochs=ep,
            batch_size=bs,
            lr=lr,
            warmup_epochs=warmup_epochs,
        )

        print(f"\n  {m.upper()}: {result['best_val_acc']:.1%} val acc (epoch {result['best_epoch']+1})")

        # Download model
        rp = f"v5_models/pkvision_{m}.pt"
        lp = out_dir / f"pkvision_{m}.pt"
        try:
            data = b"".join(volume.read_file(rp))
            with open(lp, "wb") as f:
                f.write(data)
            print(f"  Saved: {lp} ({len(data)/1024/1024:.1f} MB)")
        except Exception as e:
            print(f"  Download failed: {e}")

    print(f"\n  Test: python scripts/inference_v5.py --input data/run_testing/IMG_5985.mov")
