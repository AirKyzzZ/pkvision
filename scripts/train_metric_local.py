#!/usr/bin/env python3
"""Train metric learning model locally on CUDA (RTX 2060) or MPS.

v2: Online batch-hard triplet mining, gradient accumulation, validation
recall@1/5, PK batch sampling, stronger augmentation.

Usage:
    python scripts/train_metric_local.py
    python scripts/train_metric_local.py --epochs 30 --P 4 --K 3
    python scripts/train_metric_local.py --resume data/models/pkvision_metric.pt
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler
from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

FRAMES_DIR = ROOT / "data" / "v5_full_training" / "frames"
MANIFEST_PATH = ROOT / "data" / "v5_metric_training" / "manifest.json"
FAMILIES_PATH = ROOT / "data" / "trick_families.json"
MODEL_OUTPUT = ROOT / "data" / "models" / "pkvision_metric.pt"
REFS_OUTPUT = ROOT / "data" / "models" / "pkvision_metric_refs.json"
HF_MODEL = "MCG-NJU/videomae-base-finetuned-kinetics"


# ── Model ────────────────────────────────────────────────────────────────

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
        outputs = self.backbone.videomae(pixel_values=pixel_values)
        cls_token = outputs.last_hidden_state[:, 0]
        embedding = self.projector(cls_token)
        return F.normalize(embedding, p=2, dim=1)


# ── Dataset ──────────────────────────────────────────────────────────────

class ClipDataset(Dataset):
    """Returns (clip_tensor, family_index) for batch-hard mining."""

    def __init__(self, clips, processor, num_frames=16, augment=True):
        self.clips = clips  # list of (slug, npy_path, family_idx)
        self.processor = processor
        self.nf = num_frames
        self.augment = augment

    def __len__(self):
        return len(self.clips)

    def _load_frames(self, npy_path):
        frames = np.load(npy_path)
        T = frames.shape[0]

        if T >= self.nf:
            if self.augment:
                # Temporal jitter with optional speed variation
                speed = random.uniform(0.8, 1.2) if random.random() < 0.3 else 1.0
                effective_nf = int(self.nf * speed)
                effective_nf = max(self.nf, min(effective_nf, T))
                start = random.randint(0, max(0, T - effective_nf))
                indices = np.linspace(
                    start, min(start + effective_nf - 1, T - 1), self.nf, dtype=int
                )
            else:
                indices = np.linspace(0, T - 1, self.nf, dtype=int)
        else:
            indices = list(range(T))
            while len(indices) < self.nf:
                indices.append(indices[-1])
            indices = indices[: self.nf]

        sampled = frames[indices]

        if self.augment:
            # Horizontal flip
            if random.random() < 0.5:
                sampled = sampled[:, :, ::-1, :].copy()
            # Random crop (80-100%)
            H, W = sampled.shape[1], sampled.shape[2]
            crop_frac = random.uniform(0.80, 1.0)
            cs = int(H * crop_frac)
            if cs < H:
                yo = random.randint(0, H - cs)
                xo = random.randint(0, W - cs)
                sampled = np.stack(
                    [cv2.resize(f[yo : yo + cs, xo : xo + cs], (W, H)) for f in sampled]
                )
            # Brightness/contrast jitter
            b = random.uniform(-25, 25)
            c = random.uniform(0.8, 1.2)
            sampled = np.clip(sampled.astype(np.float32) * c + b, 0, 255).astype(np.uint8)
            # Gaussian blur (30% chance)
            if random.random() < 0.3:
                k = random.choice([3, 5])
                sampled = np.stack([cv2.GaussianBlur(f, (k, k), 0) for f in sampled])

        frames_list = [sampled[i] for i in range(sampled.shape[0])]
        inputs = self.processor(frames_list, return_tensors="pt")
        return inputs["pixel_values"].squeeze(0)

    def __getitem__(self, idx):
        slug, npy_path, family_idx = self.clips[idx]
        tensor = self._load_frames(npy_path)
        return tensor, family_idx


class PKBatchSampler(Sampler):
    """Samples P families x K clips per batch for metric learning.

    Ensures every batch has exactly P distinct families with K clips each,
    enabling effective in-batch hard negative mining.
    """

    def __init__(self, clips, P=4, K=2, oversample=3):
        self.P = P
        self.K = K
        self.oversample = oversample

        self.family_indices = defaultdict(list)
        for idx, (slug, path, family_idx) in enumerate(clips):
            self.family_indices[family_idx].append(idx)

        self.valid_families = [
            fam for fam, idxs in self.family_indices.items() if len(idxs) >= K
        ]

    def __iter__(self):
        for _ in range(self.oversample):
            families = self.valid_families.copy()
            random.shuffle(families)
            for i in range(0, len(families) - self.P + 1, self.P):
                batch_families = families[i : i + self.P]
                batch = []
                for fam in batch_families:
                    indices = self.family_indices[fam]
                    if len(indices) >= self.K:
                        selected = random.sample(indices, self.K)
                    else:
                        selected = [random.choice(indices) for _ in range(self.K)]
                    batch.extend(selected)
                yield batch

    def __len__(self):
        return (len(self.valid_families) // self.P) * self.oversample


# ── Loss ─────────────────────────────────────────────────────────────────


def batch_hard_triplet_loss(embeddings, labels, margin):
    """Online batch-hard triplet mining with semi-hard fallback.

    For each anchor, selects:
    - Hardest positive (max d(a,p) among same-family clips)
    - Semi-hard negative if available (d(a,p) < d(a,n) < d(a,p) + margin),
      otherwise hardest negative (min d(a,n) among different-family clips)
    """
    dist = torch.cdist(embeddings, embeddings, p=2)
    B = embeddings.size(0)

    losses = []
    stats = {"active": 0, "semi_hard": 0, "hard": 0, "easy": 0}

    for i in range(B):
        pos_mask = labels == labels[i]
        pos_mask[i] = False
        if pos_mask.sum() == 0:
            continue

        neg_mask = labels != labels[i]
        if neg_mask.sum() == 0:
            continue

        d_ap = dist[i][pos_mask].max()
        neg_dists = dist[i][neg_mask]

        # Prefer semi-hard negatives for stable training
        semi_hard = (neg_dists > d_ap) & (neg_dists < d_ap + margin)
        if semi_hard.sum() > 0:
            d_an = neg_dists[semi_hard].min()
            stats["semi_hard"] += 1
        else:
            d_an = neg_dists.min()
            if d_an < d_ap:
                stats["hard"] += 1
            else:
                stats["easy"] += 1

        triplet_loss = F.relu(d_ap - d_an + margin)
        if triplet_loss > 0:
            stats["active"] += 1
        losses.append(triplet_loss)

    if not losses:
        return torch.tensor(0.0, device=embeddings.device, requires_grad=True), stats

    return torch.stack(losses).mean(), stats


# ── Device ───────────────────────────────────────────────────────────────


def get_device(override=None):
    if override:
        return torch.device(override)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ── Validation ───────────────────────────────────────────────────────────


@torch.no_grad()
def evaluate(model, val_clips, train_clips, processor, device):
    """Compute recall@1 and recall@5 on held-out validation clips."""
    model.eval()

    # Build centroids from a sample of training clips
    family_embs = defaultdict(list)
    sampled = random.sample(train_clips, min(200, len(train_clips)))

    for slug, npy_path, family_idx in sampled:
        frames = np.load(npy_path)
        T = frames.shape[0]
        indices = np.linspace(0, T - 1, min(16, T), dtype=int).tolist()
        while len(indices) < 16:
            indices.append(indices[-1])
        fl = [frames[i] for i in indices[:16]]
        pv = processor(fl, return_tensors="pt")["pixel_values"].to(device)
        emb = model(pv).cpu()[0]
        family_embs[family_idx].append(emb)

    centroids = {}
    for fidx, embs in family_embs.items():
        centroids[fidx] = torch.stack(embs).mean(dim=0)

    if not centroids:
        return {"recall@1": 0.0, "recall@5": 0.0}

    c_indices = sorted(centroids.keys())
    c_matrix = torch.stack([centroids[i] for i in c_indices])

    # Evaluate validation clips (cap at 60 for speed)
    eval_clips = random.sample(val_clips, min(60, len(val_clips)))
    correct_1 = correct_5 = total = 0

    for slug, npy_path, family_idx in eval_clips:
        if family_idx not in c_indices:
            continue

        frames = np.load(npy_path)
        T = frames.shape[0]
        indices = np.linspace(0, T - 1, min(16, T), dtype=int).tolist()
        while len(indices) < 16:
            indices.append(indices[-1])
        fl = [frames[i] for i in indices[:16]]
        pv = processor(fl, return_tensors="pt")["pixel_values"].to(device)
        emb = model(pv).cpu()[0]

        sims = F.cosine_similarity(emb.unsqueeze(0), c_matrix, dim=1)
        top_k = sims.topk(min(5, len(c_indices))).indices
        top_fams = [c_indices[j] for j in top_k]

        if top_fams[0] == family_idx:
            correct_1 += 1
        if family_idx in top_fams[:5]:
            correct_5 += 1
        total += 1

    if total == 0:
        return {"recall@1": 0.0, "recall@5": 0.0}
    return {"recall@1": correct_1 / total, "recall@5": correct_5 / total}


# ── Reference DB Builder ─────────────────────────────────────────────────


def build_refs(model, families, frames_dir, processor, device, max_refs=15):
    """Build reference embedding database from trained model."""
    model.eval()
    ref_db = {}
    total = 0

    families_data = {}
    if FAMILIES_PATH.exists():
        with open(FAMILIES_PATH) as f:
            families_data = json.load(f)
    configs = families_data.get("configs", {})

    for fname, fdata in families.items():
        slugs = fdata["clips"]
        embeddings = []

        for slug in slugs[:max_refs]:
            npy_path = None
            for cat in ("acrobatics", "wall", "swing", "pk_basics"):
                p = Path(frames_dir) / cat / f"{slug}.npy"
                if p.exists():
                    npy_path = p
                    break
            if npy_path is None:
                continue

            try:
                frames = np.load(npy_path)
                T = frames.shape[0]
                indices = np.linspace(0, T - 1, min(16, T), dtype=int).tolist()
                while len(indices) < 16:
                    indices.append(indices[-1])
                fl = [frames[i] for i in indices[:16]]
                pv = processor(fl, return_tensors="pt")["pixel_values"].to(device)
                with torch.no_grad():
                    emb = model(pv).cpu().numpy()[0]
                embeddings.append(emb.tolist())
            except Exception:
                continue

        if embeddings:
            centroid = np.mean(embeddings, axis=0).tolist()
            cfg = configs.get(fname, {})
            ref_db[fname] = {
                "centroid": centroid,
                "num_refs": len(embeddings),
                "fig_tricks": cfg.get("fig_tricks", []),
                "category": cfg.get("category", "unknown"),
            }
            total += len(embeddings)

    return ref_db, total


# ── Training ─────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="PkVision metric learning (v2)")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--P", type=int, default=4, help="Families per batch")
    parser.add_argument("--K", type=int, default=2, help="Clips per family per batch")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--margin", type=float, default=0.3)
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--accum-steps", type=int, default=2, help="Gradient accumulation")
    parser.add_argument("--val-split", type=float, default=0.15, help="Validation fraction")
    parser.add_argument("--val-every", type=int, default=5, help="Validate every N epochs")
    parser.add_argument("--device", default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=MODEL_OUTPUT)
    parser.add_argument("--amp", action="store_true", help="Mixed precision (CUDA only)")
    args = parser.parse_args()

    device = get_device(args.device)
    batch_size = args.P * args.K
    amp_enabled = args.amp and device.type == "cuda"

    # Load manifest
    if not MANIFEST_PATH.exists():
        print(f"  Missing: {MANIFEST_PATH}. Run mine_trick_families.py first.")
        sys.exit(1)

    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)
    families = manifest["families"]
    family_names = sorted(families.keys())
    family_to_idx = {name: i for i, name in enumerate(family_names)}

    # Resolve clip paths
    all_clips = []
    for fname, fdata in families.items():
        fidx = family_to_idx[fname]
        for slug in fdata["clips"]:
            for cat in ("acrobatics", "wall", "swing", "pk_basics"):
                p = FRAMES_DIR / cat / f"{slug}.npy"
                if p.exists():
                    all_clips.append((slug, p, fidx))
                    break

    # Stratified train/val split
    fam_clips = defaultdict(list)
    for clip in all_clips:
        fam_clips[clip[2]].append(clip)

    train_clips, val_clips = [], []
    rng = random.Random(42)
    for fidx, clips in fam_clips.items():
        n_val = max(1, int(len(clips) * args.val_split)) if len(clips) >= 4 else 0
        shuffled = clips.copy()
        rng.shuffle(shuffled)
        val_clips.extend(shuffled[:n_val])
        train_clips.extend(shuffled[n_val:])

    print()
    print("  PkVision Metric Learning v2")
    print("  " + "=" * 50)
    print(f"  Device:       {device}")
    if device.type == "cuda":
        print(f"  GPU:          {torch.cuda.get_device_name()}")
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  VRAM:         {vram:.1f} GB")
    print(f"  Families:     {len(family_names)}")
    print(f"  Train clips:  {len(train_clips)}")
    print(f"  Val clips:    {len(val_clips)}")
    print(f"  Batch:        P={args.P} x K={args.K} = {batch_size}")
    print(f"  Effective:    {batch_size * args.accum_steps} (accum={args.accum_steps})")
    print(f"  Embed dim:    {args.embed_dim}")
    print(f"  Margin:       {args.margin}")
    print(f"  Epochs:       {args.epochs}")
    print(f"  AMP:          {amp_enabled}")

    # ── Model ────────────────────────────────────────────────────────
    print(f"\n  Loading VideoMAE backbone...", end=" ", flush=True)
    processor = VideoMAEImageProcessor.from_pretrained(HF_MODEL)
    base_model = VideoMAEForVideoClassification.from_pretrained(HF_MODEL)
    hidden_size = base_model.classifier.in_features

    model = MetricModel(base_model, hidden_size, args.embed_dim)

    start_epoch = 0
    best_loss = float("inf")
    best_recall = 0.0
    if args.resume and args.resume.exists():
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_loss = ckpt.get("best_loss", float("inf"))
        best_recall = ckpt.get("best_recall", 0.0)
        print(f"resumed from epoch {start_epoch} (loss={best_loss:.4f})")
    else:
        print("OK")

    # Freeze backbone initially (unless resuming past warmup)
    if start_epoch <= args.warmup_epochs:
        for p in model.backbone.parameters():
            p.requires_grad = False
    else:
        # Resuming with backbone already unfrozen
        for p in model.backbone.videomae.parameters():
            p.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable:    {trainable:,} parameters")
    model = model.to(device)

    # ── Data ─────────────────────────────────────────────────────────
    print("  Loading dataset...", end=" ", flush=True)
    train_dataset = ClipDataset(train_clips, processor, augment=True)
    pk_sampler = PKBatchSampler(train_clips, P=args.P, K=args.K, oversample=3)
    loader = DataLoader(
        train_dataset,
        batch_sampler=pk_sampler,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    print(f"{len(pk_sampler)} batches/epoch")

    # ── Optimizer ────────────────────────────────────────────────────
    if start_epoch > args.warmup_epochs:
        # Resuming with backbone unfrozen — use differential LR
        opt = torch.optim.AdamW([
            {"params": model.backbone.videomae.parameters(), "lr": args.lr * 0.1},
            {"params": model.projector.parameters(), "lr": args.lr},
        ], weight_decay=0.01)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=args.epochs - start_epoch,
        )
    else:
        opt = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.lr,
            weight_decay=0.01,
        )
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler(enabled=amp_enabled)

    # ── Training loop ────────────────────────────────────────────────
    args.output.parent.mkdir(parents=True, exist_ok=True)

    print()
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        # Unfreeze backbone after warmup
        if epoch == args.warmup_epochs:
            print(f"  *** Unfreezing backbone (epoch {epoch + 1}) ***")
            for p in model.backbone.videomae.parameters():
                p.requires_grad = True
            opt = torch.optim.AdamW(
                [
                    {"params": model.backbone.videomae.parameters(), "lr": args.lr * 0.05},
                    {"params": model.projector.parameters(), "lr": args.lr},
                ],
                weight_decay=0.01,
            )
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=args.epochs - args.warmup_epochs
            )
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"  *** {trainable:,} parameters now trainable ***")

        model.train()
        total_loss = 0.0
        n_batches = 0
        ep_stats = {"active": 0, "semi_hard": 0, "hard": 0, "easy": 0}

        opt.zero_grad()
        for batch_idx, (clips, labels) in enumerate(loader):
            clips = clips.to(device)
            labels = labels.to(device)

            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                embeddings = model(clips)
                loss, stats = batch_hard_triplet_loss(embeddings, labels, args.margin)
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
            for k in ep_stats:
                ep_stats[k] += stats.get(k, 0)

        # Flush remaining accumulated gradients
        if n_batches % args.accum_steps != 0:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad()

        sched.step()
        avg_loss = total_loss / max(n_batches, 1)
        dt = time.time() - t0

        total_tri = sum(ep_stats.values())
        sh_pct = ep_stats["semi_hard"] / max(total_tri, 1) * 100
        hd_pct = ep_stats["hard"] / max(total_tri, 1) * 100

        saved = ""
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "projector_state_dict": model.projector.state_dict(),
                    "families": family_names,
                    "embed_dim": args.embed_dim,
                    "hidden_size": hidden_size,
                    "epoch": epoch,
                    "best_loss": best_loss,
                    "best_recall": best_recall,
                    "config": {
                        "model_name": HF_MODEL,
                        "embed_dim": args.embed_dim,
                        "margin": args.margin,
                    },
                },
                args.output,
            )
            saved = " *"

        line = (
            f"  {epoch + 1:3d}/{args.epochs} | "
            f"loss {avg_loss:.4f} | "
            f"active {ep_stats['active']:>4d} | "
            f"sh {sh_pct:.0f}% hd {hd_pct:.0f}% | "
            f"{dt:.0f}s{saved}"
        )

        # Validation
        if val_clips and (epoch + 1) % args.val_every == 0:
            metrics = evaluate(model, val_clips, train_clips, processor, device)
            r1, r5 = metrics["recall@1"], metrics["recall@5"]
            line += f" | R@1={r1:.0%} R@5={r5:.0%}"
            if r1 > best_recall:
                best_recall = r1
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "projector_state_dict": model.projector.state_dict(),
                        "families": family_names,
                        "embed_dim": args.embed_dim,
                        "hidden_size": hidden_size,
                        "epoch": epoch,
                        "best_loss": best_loss,
                        "best_recall": best_recall,
                        "config": {
                            "model_name": HF_MODEL,
                            "embed_dim": args.embed_dim,
                            "margin": args.margin,
                        },
                    },
                    args.output.with_name("pkvision_metric_best_recall.pt"),
                )
                line += " *R"
            model.train()

        print(line)

    print(f"\n  Best loss:   {best_loss:.4f}")
    print(f"  Best R@1:    {best_recall:.0%}")
    print(f"  Model:       {args.output}")

    # ── Build reference DB ───────────────────────────────────────────
    print("\n  Building reference embedding database...")
    ckpt = torch.load(args.output, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)

    ref_db, total_refs = build_refs(model, families, FRAMES_DIR, processor, device)

    with open(REFS_OUTPUT, "w") as f:
        json.dump(ref_db, f)
    print(f"  Saved: {REFS_OUTPUT} ({len(ref_db)} families, {total_refs} embeddings)")

    print(f"\n  Test: python scripts/test_metric.py --input data/final_clips/backflip.mp4")
    print()


if __name__ == "__main__":
    main()
