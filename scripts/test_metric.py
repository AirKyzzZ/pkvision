#!/usr/bin/env python3
"""Test metric learning model on video clips.

Embeds a query clip and finds the nearest trick family by cosine similarity
to the reference centroids.

Usage:
    python scripts/test_metric.py --input video.mp4
    python scripts/test_metric.py --input data/final_clips/backflip.mp4
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor

ROOT = Path(__file__).resolve().parent.parent


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


def get_device(override=None):
    if override:
        return torch.device(override)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def read_video_crop_frames(video_path, processor, num_frames=16):
    """Read video, crop person, sample frames."""
    import av
    try:
        container = av.open(str(video_path))
        frames = [f.to_ndarray(format="rgb24") for f in container.decode(video=0)]
        container.close()
    except Exception:
        cap = cv2.VideoCapture(str(video_path))
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()

    if not frames:
        return None

    T = len(frames)

    # Resize to 256x256
    resized = [cv2.resize(f, (256, 256)) for f in frames]

    # Sample frames
    if T >= num_frames:
        indices = np.linspace(0, T - 1, num_frames, dtype=int)
    else:
        indices = list(range(T))
        while len(indices) < num_frames:
            indices.append(indices[-1])
        indices = indices[:num_frames]

    sampled = [resized[i] for i in indices]
    inputs = processor(sampled, return_tensors="pt")
    return inputs["pixel_values"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--model", default="data/models/pkvision_metric.pt")
    parser.add_argument("--refs", default="data/models/pkvision_metric_refs.json")
    parser.add_argument("--device", default=None)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    device = get_device(args.device)

    # Load model
    print(f"\n  Loading metric model...", end=" ", flush=True)
    ckpt = torch.load(args.model, map_location=device, weights_only=False)
    config = ckpt["config"]
    processor = VideoMAEImageProcessor.from_pretrained(config["model_name"])
    base = VideoMAEForVideoClassification.from_pretrained(config["model_name"])
    model = MetricModel(base, ckpt["hidden_size"], config["embed_dim"])
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    print("OK")

    # Load reference DB
    with open(args.refs) as f:
        ref_db = json.load(f)

    # Load FIG scores
    fig_scores = {}
    fig_path = ROOT / "data" / "fig_tricks_2025.json"
    if fig_path.exists():
        with open(fig_path) as f:
            fig = json.load(f)
        for cat in fig["categories"].values():
            for t in cat["tricks"]:
                fig_scores[t["name"].lower()] = t["score"]

    # Embed query
    video_path = args.input
    print(f"  Embedding {Path(video_path).name}...", end=" ", flush=True)
    t0 = time.time()
    pv = read_video_crop_frames(video_path, processor)
    if pv is None:
        print("FAILED")
        sys.exit(1)

    with torch.no_grad():
        query_emb = model(pv.to(device)).cpu().numpy()[0]
    print(f"OK ({time.time()-t0:.1f}s)")

    # Find nearest families (use all embeddings if available, fallback to centroid)
    similarities = {}
    for fname, fdata in ref_db.items():
        if "embeddings" in fdata and fdata["embeddings"]:
            # kNN: max similarity across all reference clips
            sims = [float(np.dot(query_emb, np.array(e))) for e in fdata["embeddings"]]
            similarities[fname] = max(sims)  # best match among all refs
        else:
            centroid = np.array(fdata["centroid"])
            similarities[fname] = float(np.dot(query_emb, centroid))

    # Sort by similarity
    ranked = sorted(similarities.items(), key=lambda x: -x[1])

    # Display
    print(f"\n  Top-{args.top_k} matches for {Path(video_path).name}:")
    print(f"  {'Rank':<6s} {'Family':<25s} {'Similarity':>10s}  {'FIG Tricks':<30s}  {'D-score':<8s}")
    print(f"  {'-'*6} {'-'*25} {'-'*10}  {'-'*30}  {'-'*8}")

    for i, (fname, sim) in enumerate(ranked[:args.top_k]):
        fdata = ref_db[fname]
        fig_tricks = fdata.get("fig_tricks", [])
        fig_str = ", ".join(fig_tricks[:2])
        d_scores = [fig_scores.get(t.lower(), 0) for t in fig_tricks]
        d_str = f"D={max(d_scores):.1f}" if d_scores and max(d_scores) > 0 else "D=?"
        print(f"  {i+1:<6d} {fname:<25s} {sim:>10.3f}  {fig_str:<30s}  {d_str}")

    print()


if __name__ == "__main__":
    main()
