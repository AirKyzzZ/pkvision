#!/usr/bin/env python3
"""Build reference embedding database from the metric model.

v2: Stores per-clip embeddings + centroid + optional k-means clusters
for better intra-family variation capture. At inference, query is compared
against all reference embeddings (not just centroid) for more robust matching.

Usage:
    python scripts/build_metric_refs.py
    python scripts/build_metric_refs.py --clusters 3
    python scripts/build_metric_refs.py --model data/models/pkvision_metric_best_recall.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


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


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_frames(path, processor, num_frames=16):
    """Load .npy frames and process for VideoMAE."""
    frames = np.load(path)
    T = frames.shape[0]
    if T >= num_frames:
        indices = np.linspace(0, T - 1, num_frames, dtype=int)
    else:
        indices = list(range(T))
        while len(indices) < num_frames:
            indices.append(indices[-1])
        indices = indices[:num_frames]
    sampled = [frames[i] for i in indices]
    inputs = processor(sampled, return_tensors="pt")
    return inputs["pixel_values"]


def kmeans_simple(embeddings, k, max_iter=20):
    """Simple k-means clustering on L2-normalized embeddings."""
    n = len(embeddings)
    if n <= k:
        return embeddings.copy(), list(range(n))

    # Init: pick k diverse points (farthest-point sampling)
    embs = np.array(embeddings)
    centers = [embs[0]]
    for _ in range(k - 1):
        dists = np.min([np.linalg.norm(embs - c, axis=1) for c in centers], axis=0)
        centers.append(embs[np.argmax(dists)])
    centers = np.array(centers)

    for _ in range(max_iter):
        # Assign
        dists = np.stack([np.linalg.norm(embs - c, axis=1) for c in centers])
        assigns = np.argmin(dists, axis=0)
        # Update
        new_centers = []
        for ci in range(k):
            members = embs[assigns == ci]
            if len(members) > 0:
                c = members.mean(axis=0)
                c = c / (np.linalg.norm(c) + 1e-8)  # re-normalize
                new_centers.append(c)
            else:
                new_centers.append(centers[ci])
        new_centers = np.array(new_centers)
        if np.allclose(centers, new_centers, atol=1e-6):
            break
        centers = new_centers

    return centers.tolist(), assigns.tolist()


def main():
    parser = argparse.ArgumentParser(description="Build metric reference DB (v2)")
    parser.add_argument(
        "--model", type=Path,
        default=ROOT / "data" / "models" / "pkvision_metric.pt",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "data" / "models" / "pkvision_metric_refs.json",
    )
    parser.add_argument("--max-refs", type=int, default=20, help="Max clips per family")
    parser.add_argument("--clusters", type=int, default=0,
                        help="K-means clusters per family (0=centroid only)")
    args = parser.parse_args()

    families_path = ROOT / "data" / "trick_families.json"
    frames_dir = ROOT / "data" / "v5_full_training" / "frames"

    device = get_device()
    print(f"  Building metric reference DB on {device}")

    # Load model
    print("  Loading metric model...", end=" ", flush=True)
    ckpt = torch.load(args.model, map_location=device, weights_only=False)
    config = ckpt["config"]
    embed_dim = config["embed_dim"]
    hidden_size = ckpt["hidden_size"]

    processor = VideoMAEImageProcessor.from_pretrained(config["model_name"])
    base = VideoMAEForVideoClassification.from_pretrained(config["model_name"])

    model = MetricModel(base, hidden_size, embed_dim)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    print("OK")

    # Load families
    with open(families_path) as f:
        data = json.load(f)

    families = data["families"]
    configs = data["configs"]

    # Build reference embeddings
    ref_db = {}
    total = 0

    print(f"\n  Embedding {len(families)} families...")
    for fname, slugs in sorted(families.items()):
        if len(slugs) < 1:
            continue

        embeddings = []
        slug_names = []
        for slug in slugs[: args.max_refs]:
            npy_path = None
            for cat in ("acrobatics", "wall", "swing", "pk_basics"):
                p = frames_dir / cat / f"{slug}.npy"
                if p.exists():
                    npy_path = p
                    break
            if npy_path is None:
                continue

            try:
                pv = load_frames(npy_path, processor).to(device)
                with torch.no_grad():
                    emb = model(pv).cpu().numpy()[0]
                embeddings.append(emb.tolist())
                slug_names.append(slug)
            except Exception:
                continue

        if not embeddings:
            continue

        centroid = np.mean(embeddings, axis=0)
        centroid = (centroid / (np.linalg.norm(centroid) + 1e-8)).tolist()

        fig_tricks = configs.get(fname, {}).get("fig_tricks", [])
        category = configs.get(fname, {}).get("category", "unknown")

        entry = {
            "centroid": centroid,
            "num_refs": len(embeddings),
            "embeddings": embeddings,  # all individual embeddings for kNN
            "slugs": slug_names,
            "fig_tricks": fig_tricks,
            "category": category,
        }

        # Optional clustering
        if args.clusters > 0 and len(embeddings) >= args.clusters * 2:
            k = min(args.clusters, len(embeddings) // 2)
            cluster_centers, assignments = kmeans_simple(embeddings, k)
            entry["clusters"] = cluster_centers
            entry["cluster_assignments"] = assignments

        ref_db[fname] = entry
        total += len(embeddings)

        extra = ""
        if "clusters" in entry:
            extra = f" ({len(entry['clusters'])} clusters)"
        print(
            f"  {fname:<25s} {len(embeddings):>3d} refs{extra}"
            f"  ({', '.join(fig_tricks[:2])})"
        )

    # Save
    with open(args.output, "w") as f:
        json.dump(ref_db, f)

    size_mb = args.output.stat().st_size / 1024 / 1024
    print(f"\n  Saved {len(ref_db)} families ({total} embeddings, {size_mb:.1f} MB)")
    print(f"  Output: {args.output}")


if __name__ == "__main__":
    main()
