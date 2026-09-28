#!/usr/bin/env python3
"""Train metric learning model on Modal T4 GPU.

Uses VideoMAE backbone as feature extractor + projection head trained with
triplet loss. At inference, embed a query clip → find nearest family in
reference DB.

Data already on volume as tar chunks (v5_data/v5_frames_part_*).

Usage:
    python scripts/train_metric_modal.py
    python scripts/train_metric_modal.py --epochs 40 --embed-dim 128
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

app = modal.App("pkvision-metric", image=image)
volume = modal.Volume.from_name("pkvision-data", create_if_missing=True)

VOL = "/vol"
FRAMES = f"{VOL}/v5_frames"


@app.function(gpu="T4", timeout=14400, volumes={VOL: volume})
def train_metric(
    manifest: dict,
    epochs: int = 40,
    batch_size: int = 16,
    lr: float = 1e-4,
    embed_dim: int = 128,
    margin: float = 0.3,
    warmup_epochs: int = 5,
) -> dict:
    import os
    import random
    import subprocess
    import time

    volume.reload()

    # Extract frames if needed
    frames_path = Path(FRAMES)
    if not frames_path.exists() or not any(frames_path.rglob("*.npy")):
        print("  Extracting frames from tar chunks...")
        data_dir = Path(VOL) / "v5_data"
        chunks = sorted(data_dir.glob("v5_frames_part_*"))
        if chunks:
            cat_cmd = " ".join(str(c) for c in chunks)
            subprocess.run(f"cat {cat_cmd} | tar xzf - -C {VOL}/", shell=True, check=True)
            extracted = Path(VOL) / "frames"
            if extracted.exists() and not frames_path.exists():
                extracted.rename(frames_path)
        npy_count = len(list(frames_path.rglob("*.npy")))
        print(f"  Extracted {npy_count} frame files")

    import cv2
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
    from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor

    # ── Model: VideoMAE backbone + projection head ───────────────────────
    class MetricModel(nn.Module):
        def __init__(self, backbone, hidden_size, embed_dim):
            super().__init__()
            self.backbone = backbone
            self.projector = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_size // 2, embed_dim),
            )

        def forward(self, pixel_values):
            # Get CLS token from VideoMAE
            outputs = self.backbone.videomae(pixel_values=pixel_values)
            cls_token = outputs.last_hidden_state[:, 0]  # (B, hidden_size)
            embedding = self.projector(cls_token)  # (B, embed_dim)
            return F.normalize(embedding, p=2, dim=1)  # L2 normalize

    # ── Dataset: triplet sampling ────────────────────────────────────────
    class TripletDataset(Dataset):
        def __init__(self, frames_dir, families, processor, num_frames=16, augment=True):
            self.dir = Path(frames_dir)
            self.processor = processor
            self.nf = num_frames
            self.augment = augment

            # Build family index
            self.families = []  # list of (family_name, [clip_slugs])
            self.slug_to_family = {}
            self.all_slugs = []

            for fname, fdata in families.items():
                slugs = fdata["clips"]
                # Verify clips exist
                valid = [s for s in slugs if self._find_npy(s) is not None]
                if len(valid) >= 2:
                    self.families.append((fname, valid))
                    for s in valid:
                        self.slug_to_family[s] = fname
                        self.all_slugs.append(s)

            self.family_names = [f[0] for f in self.families]
            print(f"    {len(self.families)} families, {len(self.all_slugs)} clips loaded")

        def _find_npy(self, slug):
            """Find .npy file for a slug across category subdirs."""
            for cat in ("acrobatics", "wall", "swing", "pk_basics"):
                p = self.dir / cat / f"{slug}.npy"
                if p.exists():
                    return p
            return None

        def __len__(self):
            return len(self.all_slugs) * 4  # oversample

        def _load_frames(self, slug):
            path = self._find_npy(slug)
            if path is None:
                return None
            frames = np.load(path)
            T = frames.shape[0]

            # Temporal sampling
            if T >= self.nf:
                if self.augment:
                    start = random.randint(0, max(0, T - self.nf))
                    indices = np.linspace(start, min(start + self.nf - 1, T - 1), self.nf, dtype=int)
                else:
                    indices = np.linspace(0, T - 1, self.nf, dtype=int)
            else:
                indices = list(range(T))
                while len(indices) < self.nf:
                    indices.append(indices[-1])
                indices = indices[:self.nf]

            sampled = frames[indices]

            if self.augment:
                if random.random() < 0.5:
                    sampled = sampled[:, :, ::-1, :].copy()
                # Random crop
                H, W = sampled.shape[1], sampled.shape[2]
                crop_frac = random.uniform(0.85, 1.0)
                cs = int(H * crop_frac)
                if cs < H:
                    yo = random.randint(0, H - cs)
                    xo = random.randint(0, W - cs)
                    sampled = np.stack([cv2.resize(f[yo:yo+cs, xo:xo+cs], (W, H)) for f in sampled])
                # Brightness
                b = random.uniform(-15, 15)
                c = random.uniform(0.9, 1.1)
                sampled = np.clip(sampled.astype(np.float32) * c + b, 0, 255).astype(np.uint8)

            frames_list = [sampled[i] for i in range(sampled.shape[0])]
            inputs = self.processor(frames_list, return_tensors="pt")
            return inputs["pixel_values"].squeeze(0)

        def __getitem__(self, idx):
            idx = idx % len(self.all_slugs)
            anchor_slug = self.all_slugs[idx]
            anchor_family = self.slug_to_family[anchor_slug]

            # Positive: same family, different clip
            family_idx = self.family_names.index(anchor_family)
            _, family_slugs = self.families[family_idx]
            pos_candidates = [s for s in family_slugs if s != anchor_slug]
            if not pos_candidates:
                pos_candidates = [anchor_slug]  # self-pair with different augmentation
            pos_slug = random.choice(pos_candidates)

            # Negative: different family
            neg_family_idx = random.choice([i for i in range(len(self.families)) if i != family_idx])
            _, neg_slugs = self.families[neg_family_idx]
            neg_slug = random.choice(neg_slugs)

            anchor = self._load_frames(anchor_slug)
            positive = self._load_frames(pos_slug)
            negative = self._load_frames(neg_slug)

            if anchor is None or positive is None or negative is None:
                # Fallback — return first valid triplet
                anchor = self._load_frames(self.all_slugs[0])
                positive = self._load_frames(self.all_slugs[1])
                negative = self._load_frames(self.all_slugs[-1])

            return anchor, positive, negative

    # ── Setup ────────────────────────────────────────────────────────────
    device = torch.device("cuda")
    families = manifest["families"]

    print(f"\n  PkVision Metric Learning")
    print("  " + "=" * 50)
    print(f"  GPU:        {torch.cuda.get_device_name()}")
    print(f"  Families:   {len(families)}")
    print(f"  Embed dim:  {embed_dim}")
    print(f"  Margin:     {margin}")
    print(f"  Epochs:     {epochs}")
    print(f"  Batch:      {batch_size}")

    # ── Model ────────────────────────────────────────────────────────────
    print("\n  Loading VideoMAE backbone...", end=" ", flush=True)
    hf = "MCG-NJU/videomae-base-finetuned-kinetics"
    processor = VideoMAEImageProcessor.from_pretrained(hf)
    base_model = VideoMAEForVideoClassification.from_pretrained(hf)
    hidden_size = base_model.classifier.in_features

    model = MetricModel(base_model, hidden_size, embed_dim)

    # Freeze backbone initially
    for p in model.backbone.parameters():
        p.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"OK ({trainable:,} trainable)")
    model = model.to(device)

    # ── Data ─────────────────────────────────────────────────────────────
    print("  Loading triplet dataset...")
    dataset = TripletDataset(FRAMES, families, processor, augment=True)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2,
                        drop_last=True)

    # ── Training ─────────────────────────────────────────────────────────
    triplet_loss = nn.TripletMarginLoss(margin=margin)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best_loss = float("inf")
    out_path = Path(VOL) / "v5_models" / "pkvision_metric.pt"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print()
    for ep in range(epochs):
        t0 = time.time()

        # Unfreeze backbone after warmup
        if ep == warmup_epochs:
            print(f"  *** Unfreezing backbone (epoch {ep+1}) ***")
            for p in model.backbone.videomae.parameters():
                p.requires_grad = True
            opt = torch.optim.AdamW([
                {"params": model.backbone.videomae.parameters(), "lr": lr * 0.05},
                {"params": model.projector.parameters(), "lr": lr},
            ], weight_decay=0.01)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs - warmup_epochs)

        model.train()
        total_loss = 0.0
        n_batches = 0
        correct_triplets = 0
        total_triplets = 0

        for anchor, pos, neg in loader:
            anchor = anchor.to(device)
            pos = pos.to(device)
            neg = neg.to(device)

            opt.zero_grad()
            e_a = model(anchor)
            e_p = model(pos)
            e_n = model(neg)

            loss = triplet_loss(e_a, e_p, e_n)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            total_loss += loss.item()
            n_batches += 1

            # Track triplet accuracy (positive closer than negative)
            d_pos = F.pairwise_distance(e_a, e_p)
            d_neg = F.pairwise_distance(e_a, e_n)
            correct_triplets += (d_pos < d_neg).sum().item()
            total_triplets += anchor.size(0)

        sched.step()
        avg_loss = total_loss / max(n_batches, 1)
        triplet_acc = correct_triplets / max(total_triplets, 1)
        dt = time.time() - t0

        s = ""
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "projector_state_dict": model.projector.state_dict(),
                "families": list(families.keys()),
                "embed_dim": embed_dim,
                "hidden_size": hidden_size,
                "config": {"model_name": hf, "embed_dim": embed_dim, "margin": margin},
            }, out_path)
            volume.commit()
            s = " *"

        print(f"  {ep+1:3d}/{epochs} | loss {avg_loss:.4f} | triplet_acc {triplet_acc:.1%} | {dt:.0f}s{s}")

    print(f"\n  Best loss: {best_loss:.4f}")
    print(f"  Model: {out_path}")

    # ── Build reference embeddings ───────────────────────────────────────
    print("\n  Building reference embedding database...")
    model.eval()
    ref_db = {}

    val_dataset = TripletDataset(FRAMES, families, processor, augment=False)

    with torch.no_grad():
        for family_name, family_slugs in val_dataset.families:
            embeddings = []
            for slug in family_slugs[:10]:  # max 10 refs per family
                frames = val_dataset._load_frames(slug)
                if frames is not None:
                    pv = frames.unsqueeze(0).to(device)
                    emb = model(pv).cpu().numpy()[0]
                    embeddings.append(emb.tolist())
            if embeddings:
                # Average embedding as family centroid
                centroid = np.mean(embeddings, axis=0).tolist()
                ref_db[family_name] = {
                    "centroid": centroid,
                    "num_refs": len(embeddings),
                }

    # Save reference DB
    ref_path = Path(VOL) / "v5_models" / "pkvision_metric_refs.json"
    with open(ref_path, "w") as f:
        json.dump(ref_db, f)
    volume.commit()
    print(f"  Reference DB: {len(ref_db)} families, saved to {ref_path}")

    return {
        "best_loss": best_loss,
        "num_families": len(ref_db),
        "embed_dim": embed_dim,
    }


@app.local_entrypoint()
def main(
    manifest_path: str = "data/v5_metric_training/manifest.json",
    epochs: int = 40,
    batch_size: int = 16,
    lr: float = 1e-4,
    embed_dim: int = 128,
):
    mpath = Path(manifest_path)
    if not mpath.exists():
        print(f"Missing: {mpath}. Run mine_trick_families.py first.")
        sys.exit(1)

    with open(mpath) as f:
        manifest = json.load(f)

    print(f"\n  PkVision Metric Learning")
    print("  " + "=" * 50)
    print(f"  Families:   {manifest['num_families']}")
    print(f"  Clips:      {manifest['total_clips']}")
    print(f"  Epochs:     {epochs}")
    print(f"  Embed dim:  {embed_dim}")
    print()

    result = train_metric.remote(
        manifest=manifest,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        embed_dim=embed_dim,
    )

    print(f"\n  Done! Best loss: {result['best_loss']:.4f}")
    print(f"  Families: {result['num_families']}")

    # Download model + reference DB
    out_dir = Path("data/models")
    out_dir.mkdir(parents=True, exist_ok=True)

    for fname in ("pkvision_metric.pt", "pkvision_metric_refs.json"):
        rp = f"v5_models/{fname}"
        lp = out_dir / fname
        try:
            data = b"".join(volume.read_file(rp))
            with open(lp, "wb") as f:
                f.write(data)
            print(f"  Saved: {lp} ({len(data)/1024/1024:.1f} MB)")
        except Exception as e:
            print(f"  Failed: {fname}: {e}")

    print(f"\n  Test: python scripts/inference_v5.py --input video.mp4")
