#!/usr/bin/env python3
"""Train hierarchical attribute classifiers for trick decomposition.

Trains 4 VideoMAE classifiers that decompose any trick into attributes:
  1. context:   acrobatics / wall / swing / pk_basics
  2. direction: backward / forward / side / none
  3. flip_bin:  0 / 1 / 2+  (simplified)
  4. twist_bin: 0 / 1+ (simplified)

The attribute vector uniquely narrows to a small group of FIG tricks.

Each classifier shares the same VideoMAE backbone but has its own head.
Multi-task training: single forward pass, 4 losses summed.

Usage:
    python scripts/train_attributes.py --epochs 20
    python scripts/train_attributes.py --epochs 20 --amp
    python scripts/train_attributes.py --resume data/models/pkvision_attributes.pt
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor

ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = ROOT / "data" / "v5_attribute_training" / "attribute_manifest.json"
FRAMES_DIR = ROOT / "data" / "v5_full_training" / "frames"
MODEL_OUTPUT = ROOT / "data" / "models" / "pkvision_attributes.pt"
HF_MODEL = "MCG-NJU/videomae-base-finetuned-kinetics"

# Simplified attribute classes for training
ATTR_CONFIG = {
    "context": {
        "classes": ["acrobatics", "wall", "swing", "pk_basics"],
        "weight": 1.0,
    },
    "direction": {
        "classes": ["backward", "forward", "side", "none"],
        "weight": 1.0,
    },
    "flip_bin": {
        "classes": ["0", "1", "2+"],
        "remap": {"0": "0", "0.5": "0", "1": "1", "1.5": "1", "2": "2+", "3+": "2+"},
        "weight": 1.5,  # most important for D-score
    },
    "twist_bin": {
        # Finer bins than the original {0, has_twist} split. High-twist
        # FIG tricks (Backflip 720, Double Cork, Kroc) need more than a
        # binary signal to disambiguate from their 0-twist siblings.
        # 1.5 is collapsed into 1 because only 2 clips in the manifest
        # carry that label (too few to train).
        "classes": ["0", "0.5", "1", "2+"],
        "remap": {"0": "0", "0.5": "0.5", "1": "1", "1.5": "1",
                  "2": "2+", "3+": "2+"},
        "weight": 1.5,
    },
}


# ── Model ────────────────────────────────────────────────────────────────


class MultiTaskAttributeModel(nn.Module):
    """VideoMAE backbone + 4 classification heads (one per attribute)."""

    def __init__(self, backbone, hidden_size, attr_config):
        super().__init__()
        self.backbone = backbone
        self.heads = nn.ModuleDict()
        for attr_name, cfg in attr_config.items():
            n_classes = len(cfg["classes"])
            self.heads[attr_name] = nn.Sequential(
                nn.Dropout(0.1),
                nn.Linear(hidden_size, n_classes),
            )

    def forward(self, pixel_values):
        outputs = self.backbone.videomae(pixel_values=pixel_values)
        cls_token = outputs.last_hidden_state[:, 0]
        return {name: head(cls_token) for name, head in self.heads.items()}


# ── Dataset ──────────────────────────────────────────────────────────────


class AttributeDataset(Dataset):
    def __init__(self, clips, attr_config, processor, num_frames=16, augment=True,
                 domain_aug_intensity=0.0):
        self.clips = clips
        self.attr_config = attr_config
        self.processor = processor
        self.nf = num_frames
        self.augment = augment
        self.domain_aug = None
        if domain_aug_intensity > 0:
            from core.augmentation import CompetitionAugmenter
            self.domain_aug = CompetitionAugmenter(intensity=domain_aug_intensity)

    def __len__(self):
        return len(self.clips)

    def _load_frames(self, npy_path):
        frames = np.load(npy_path, allow_pickle=True)
        T = frames.shape[0]

        if T >= self.nf:
            if self.augment:
                speed = random.uniform(0.8, 1.2) if random.random() < 0.3 else 1.0
                eff = max(self.nf, min(int(self.nf * speed), T))
                start = random.randint(0, max(0, T - eff))
                indices = np.linspace(start, min(start + eff - 1, T - 1), self.nf, dtype=int)
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
            H, W = sampled.shape[1], sampled.shape[2]
            crop_frac = random.uniform(0.80, 1.0)
            cs = int(H * crop_frac)
            if cs < H:
                yo = random.randint(0, H - cs)
                xo = random.randint(0, W - cs)
                sampled = np.stack([cv2.resize(f[yo:yo+cs, xo:xo+cs], (W, H)) for f in sampled])
            b = random.uniform(-25, 25)
            c = random.uniform(0.8, 1.2)
            sampled = np.clip(sampled.astype(np.float32) * c + b, 0, 255).astype(np.uint8)
            if random.random() < 0.3:
                k = random.choice([3, 5])
                sampled = np.stack([cv2.GaussianBlur(f, (k, k), 0) for f in sampled])

        # Domain augmentation: simulate competition footage
        if self.domain_aug is not None:
            sampled = self.domain_aug(sampled)

        fl = [sampled[i] for i in range(sampled.shape[0])]
        inputs = self.processor(fl, return_tensors="pt")
        return inputs["pixel_values"].squeeze(0)

    def __getitem__(self, idx):
        clip = self.clips[idx]
        tensor = self._load_frames(clip["npy_path"])

        labels = {}
        for attr_name, cfg in self.attr_config.items():
            raw_val = clip.get(attr_name, "unknown")
            remap = cfg.get("remap", {})
            val = remap.get(raw_val, raw_val)
            if val in cfg["classes"]:
                labels[attr_name] = cfg["classes"].index(val)
            else:
                labels[attr_name] = -1  # ignore

        return tensor, labels


# ── Device ───────────────────────────────────────────────────────────────


def get_device(override=None):
    if override:
        return torch.device(override)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ── Training ─────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Train attribute classifiers")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--accum-steps", type=int, default=2)
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--device", default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=MODEL_OUTPUT)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--domain-aug", type=float, default=0.0,
                        help="Domain augmentation intensity 0-1 (simulate competition footage)")
    parser.add_argument("--head-only", action="store_true",
                        help="Freeze backbone permanently (linear probe) — ~10× faster, "
                             "useful for fast iteration on bin layouts.")
    parser.add_argument("--class-weighted", action="store_true",
                        help="Apply per-class inverse-frequency weights to the "
                             "cross-entropy loss. Prevents collapse onto the "
                             "majority class when labels are imbalanced.")
    parser.add_argument("--unfreeze-last", type=int, default=0,
                        help="Unfreeze the last N VideoMAE encoder blocks even "
                             "when --head-only is set. 0=none, 1=last block, etc. "
                             "Gives capacity to learn minority-class features.")
    args = parser.parse_args()

    device = get_device(args.device)
    amp_enabled = args.amp and device.type == "cuda"

    # Load manifest
    if not MANIFEST_PATH.exists():
        print(f"  Missing: {MANIFEST_PATH}. Run build_attribute_dataset.py first.")
        sys.exit(1)

    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)
    all_clips = manifest["clips"]

    # Filter clips with valid paths
    valid_clips = [c for c in all_clips if Path(c["npy_path"]).exists()]

    # Stratified train/val split (by context to ensure balance)
    ctx_groups = defaultdict(list)
    for clip in valid_clips:
        ctx_groups[clip["context"]].append(clip)

    train_clips, val_clips = [], []
    rng = random.Random(42)
    for ctx, clips in ctx_groups.items():
        n_val = max(1, int(len(clips) * args.val_split))
        shuffled = clips.copy()
        rng.shuffle(shuffled)
        val_clips.extend(shuffled[:n_val])
        train_clips.extend(shuffled[n_val:])

    print()
    print("  PkVision Attribute Classifiers")
    print("  " + "=" * 50)
    print(f"  Device:      {device}")
    if device.type == "cuda":
        print(f"  GPU:         {torch.cuda.get_device_name()}")
    print(f"  Train clips: {len(train_clips)}")
    print(f"  Val clips:   {len(val_clips)}")
    print(f"  Batch:       {args.batch_size} (eff. {args.batch_size * args.accum_steps})")
    print(f"  Epochs:      {args.epochs}")
    print(f"  LR:          {args.lr}")
    print(f"  AMP:         {amp_enabled}")
    if args.domain_aug > 0:
        print(f"  Domain aug:  {args.domain_aug:.0%} intensity")

    # Print class distributions and compute per-class inverse-frequency
    # weights so the loss doesn't collapse onto the majority class. Weights
    # are normalized to mean 1.0 so the overall loss magnitude stays
    # comparable across runs with and without --class-weighted.
    class_weights: dict[str, torch.Tensor] = {}
    for attr_name, cfg in ATTR_CONFIG.items():
        remap = cfg.get("remap", {})
        counts = Counter()
        for c in train_clips:
            raw = c.get(attr_name, "unknown")
            val = remap.get(raw, raw)
            if val in cfg["classes"]:
                counts[val] += 1
        dist = "  ".join(f"{k}={v}" for k, v in sorted(counts.items()))
        print(f"  {attr_name}: {dist}")

        classes = cfg["classes"]
        n_total = sum(counts.values())
        n_classes = len(classes)
        if args.class_weighted and n_total > 0 and n_classes > 0:
            # Inverse frequency, clipped at 8x so one-sample classes don't
            # explode the loss — we still want the model to attend to them
            # but not let a single val clip dominate every batch update.
            raw_w = [n_total / (n_classes * max(counts.get(cls, 0), 1))
                     for cls in classes]
            mean_w = sum(raw_w) / len(raw_w)
            norm_w = [min(w / mean_w, 8.0) for w in raw_w]
            class_weights[attr_name] = torch.tensor(norm_w, dtype=torch.float32)
            print(f"    weights: {[round(w, 2) for w in norm_w]}")

    # ── Model ────────────────────────────────────────────────────────
    print(f"\n  Loading VideoMAE backbone...", end=" ", flush=True)
    processor = VideoMAEImageProcessor.from_pretrained(HF_MODEL)
    base_model = VideoMAEForVideoClassification.from_pretrained(HF_MODEL)
    hidden_size = base_model.classifier.in_features

    model = MultiTaskAttributeModel(base_model, hidden_size, ATTR_CONFIG)

    start_epoch = 0
    best_val_acc = 0.0
    if args.resume and args.resume.exists():
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        ckpt_sd = dict(ckpt["model_state_dict"])
        own_sd = model.state_dict()
        skipped = []
        for k in list(ckpt_sd.keys()):
            if k in own_sd and own_sd[k].shape != ckpt_sd[k].shape:
                skipped.append((k, tuple(ckpt_sd[k].shape), tuple(own_sd[k].shape)))
                del ckpt_sd[k]
        missing, unexpected = model.load_state_dict(ckpt_sd, strict=False)
        # When the head schema changed (e.g. twist_bin 2→4 classes), restart
        # epoch counting and val tracking — the old best_val_acc was computed
        # on a different label space and is no longer comparable.
        if skipped:
            start_epoch = 0
            best_val_acc = 0.0
            print(f"warm-started (skipped mismatched: {[s[0] for s in skipped]})")
            for k, old, new in skipped:
                print(f"    {k}: {old} -> {new} (reinit)")
        else:
            start_epoch = ckpt.get("epoch", 0) + 1
            best_val_acc = ckpt.get("best_val_acc", 0.0)
            print(f"resumed from epoch {start_epoch}")
    else:
        print("OK")

    # Freeze backbone for warmup (or permanently in --head-only mode).
    if args.head_only or start_epoch <= args.warmup_epochs:
        for p in model.backbone.parameters():
            p.requires_grad = False
    else:
        for p in model.backbone.videomae.parameters():
            p.requires_grad = True

    # Partial unfreeze: even in head-only mode, optionally unfreeze the last
    # N encoder blocks so the model can learn features that separate the
    # minority classes (Kinetics features alone collapse onto majority when
    # only a linear head is trained).
    if args.unfreeze_last > 0:
        layers = model.backbone.videomae.encoder.layer
        n_layers = len(layers)
        first_unfrozen = max(0, n_layers - args.unfreeze_last)
        for i, block in enumerate(layers):
            if i >= first_unfrozen:
                for p in block.parameters():
                    p.requires_grad = True
        print(f"  Unfrozen:    encoder blocks [{first_unfrozen}:{n_layers}]")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable:   {trainable:,} parameters")
    model = model.to(device)

    # Move class weights onto the training device if they were computed.
    for k in list(class_weights.keys()):
        class_weights[k] = class_weights[k].to(device)

    # ── Data ─────────────────────────────────────────────────────────
    train_dataset = AttributeDataset(train_clips, ATTR_CONFIG, processor, augment=True,
                                     domain_aug_intensity=args.domain_aug)
    val_dataset = AttributeDataset(val_clips, ATTR_CONFIG, processor, augment=False)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=0, drop_last=True, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=(device.type == "cuda"),
    )

    print(f"  Batches:     {len(train_loader)}/epoch")

    # ── Optimizer ────────────────────────────────────────────────────
    if start_epoch > args.warmup_epochs:
        opt = torch.optim.AdamW([
            {"params": model.backbone.videomae.parameters(), "lr": args.lr * 0.1},
            {"params": model.heads.parameters(), "lr": args.lr},
        ], weight_decay=0.01)
    else:
        opt = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.lr, weight_decay=0.01,
        )

    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs - start_epoch,
    )
    scaler = torch.amp.GradScaler(enabled=amp_enabled)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # ── Training loop ────────────────────────────────────────────────
    print()
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        # Unfreeze backbone after warmup (skipped in --head-only mode).
        if not args.head_only and epoch == args.warmup_epochs:
            print(f"  *** Unfreezing backbone (epoch {epoch + 1}) ***")
            for p in model.backbone.videomae.parameters():
                p.requires_grad = True
            opt = torch.optim.AdamW([
                {"params": model.backbone.videomae.parameters(), "lr": args.lr * 0.1},
                {"params": model.heads.parameters(), "lr": args.lr},
            ], weight_decay=0.01)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=args.epochs - epoch,
            )
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"  *** {trainable:,} parameters trainable ***")

        # ── Train ────────────────────────────────────────────────────
        model.train()
        total_loss = 0.0
        n_batches = 0
        attr_correct = {a: 0 for a in ATTR_CONFIG}
        attr_total = {a: 0 for a in ATTR_CONFIG}

        opt.zero_grad()
        for batch_idx, (frames, labels) in enumerate(train_loader):
            frames = frames.to(device)

            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                logits = model(frames)

                loss = torch.tensor(0.0, device=device)
                for attr_name, cfg in ATTR_CONFIG.items():
                    gt = labels[attr_name].to(device)
                    valid = gt >= 0
                    if valid.sum() == 0:
                        continue
                    attr_loss = F.cross_entropy(
                        logits[attr_name][valid], gt[valid],
                        weight=class_weights.get(attr_name),
                    )
                    loss = loss + attr_loss * cfg["weight"]

                    preds = logits[attr_name][valid].argmax(dim=-1)
                    attr_correct[attr_name] += (preds == gt[valid]).sum().item()
                    attr_total[attr_name] += valid.sum().item()

                loss = loss / args.accum_steps

            scaler.scale(loss).backward()

            if (batch_idx + 1) % args.accum_steps == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()

            total_loss += loss.item() * args.accum_steps
            n_batches += 1

        # Flush remaining gradients
        if n_batches % args.accum_steps != 0:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad()

        sched.step()
        avg_loss = total_loss / max(n_batches, 1)
        dt = time.time() - t0

        train_accs = {a: attr_correct[a] / max(attr_total[a], 1) for a in ATTR_CONFIG}

        # ── Validate ─────────────────────────────────────────────────
        model.eval()
        val_correct = {a: 0 for a in ATTR_CONFIG}
        val_total = {a: 0 for a in ATTR_CONFIG}

        with torch.no_grad():
            for frames, labels in val_loader:
                frames = frames.to(device)
                logits = model(frames)

                for attr_name in ATTR_CONFIG:
                    gt = labels[attr_name].to(device)
                    valid = gt >= 0
                    if valid.sum() == 0:
                        continue
                    preds = logits[attr_name][valid].argmax(dim=-1)
                    val_correct[attr_name] += (preds == gt[valid]).sum().item()
                    val_total[attr_name] += valid.sum().item()

        val_accs = {a: val_correct[a] / max(val_total[a], 1) for a in ATTR_CONFIG}
        mean_val = sum(val_accs.values()) / len(val_accs)

        # Save best
        saved = ""
        if mean_val > best_val_acc:
            best_val_acc = mean_val
            torch.save({
                "model_state_dict": model.state_dict(),
                "attr_config": ATTR_CONFIG,
                "hidden_size": hidden_size,
                "epoch": epoch,
                "best_val_acc": best_val_acc,
                "config": {"model_name": HF_MODEL},
            }, args.output)
            saved = " *"

        # Print
        acc_str = "  ".join(f"{a[:3]}={val_accs[a]:.0%}" for a in ATTR_CONFIG)
        print(
            f"  {epoch + 1:3d}/{args.epochs} | "
            f"loss {avg_loss:.3f} | "
            f"val: {acc_str} | "
            f"mean={mean_val:.0%} | "
            f"{dt:.0f}s{saved}"
        )

    print(f"\n  Best mean val accuracy: {best_val_acc:.0%}")
    print(f"  Model: {args.output}")

    # ── Final per-class report ───────────────────────────────────────
    print(f"\n  Per-class validation accuracy:")
    model.eval()
    for attr_name, cfg in ATTR_CONFIG.items():
        class_correct = Counter()
        class_total = Counter()

        with torch.no_grad():
            for frames, labels in val_loader:
                frames = frames.to(device)
                logits = model(frames)
                gt = labels[attr_name].to(device)
                valid = gt >= 0
                if valid.sum() == 0:
                    continue
                preds = logits[attr_name][valid].argmax(dim=-1)
                for p, g in zip(preds.cpu(), gt[valid].cpu()):
                    class_total[cfg["classes"][g.item()]] += 1
                    if p.item() == g.item():
                        class_correct[cfg["classes"][g.item()]] += 1

        print(f"\n  {attr_name}:")
        for cls in cfg["classes"]:
            t = class_total.get(cls, 0)
            c = class_correct.get(cls, 0)
            acc = c / max(t, 1)
            print(f"    {cls:<12s} {c:>4d}/{t:<4d} ({acc:.0%})")

    print()


if __name__ == "__main__":
    main()
