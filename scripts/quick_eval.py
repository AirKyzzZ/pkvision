#!/usr/bin/env python3
"""Quick evaluation of metric model on known .npy clips."""

import json
import sys
from pathlib import Path

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

    def forward(self, pv):
        o = self.backbone.videomae(pixel_values=pv)
        return F.normalize(self.projector(o.last_hidden_state[:, 0]), p=2, dim=1)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    frames_dir = ROOT / "data" / "v5_full_training" / "frames"

    # Load model
    print("  Loading model...", end=" ", flush=True)
    ckpt = torch.load(ROOT / "data/models/pkvision_metric.pt", map_location=device, weights_only=False)
    cfg = ckpt["config"]
    proc = VideoMAEImageProcessor.from_pretrained(cfg["model_name"])
    base = VideoMAEForVideoClassification.from_pretrained(cfg["model_name"])
    model = MetricModel(base, ckpt["hidden_size"], cfg["embed_dim"])
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device).eval()
    print("OK")

    # Load refs
    with open(ROOT / "data/models/pkvision_metric_refs.json") as f:
        refs = json.load(f)
    print(f"  Reference DB: {len(refs)} families\n")

    # Test cases: (npy_path_relative_to_frames, expected_family)
    test_cases = [
        ("acrobatics/back_layout.npy", "backflip"),
        ("acrobatics/dive_front_flip.npy", "frontflip"),
        ("acrobatics/double_corkscrew.npy", "corkscrew"),
        ("acrobatics/gainer_full.npy", "gainer"),
        ("wall/wall_flip.npy", "wallflip"),
        ("acrobatics/double_side_flip.npy", "sideflip"),
        ("acrobatics/cartwheel.npy", "cartwheel"),
        ("swing/flyaway_full.npy", "flyaway"),
        ("acrobatics/triple_back_flip.npy", "backflip"),
        ("acrobatics/front_full.npy", "frontflip"),
        ("acrobatics/double_front_flip.npy", "frontflip"),
        ("acrobatics/butterfly_twist.npy", "btwist"),
    ]

    print(f"  {'Clip':<35s} {'Expected':<14s} {'Top-1':<14s} {'Sim':>6s}  {'Top-2':<14s} {'OK?'}")
    print(f"  {'-'*35} {'-'*14} {'-'*14} {'-'*6}  {'-'*14} {'-'*3}")

    correct = 0
    total = 0

    for npy_rel, expected in test_cases:
        p = frames_dir / npy_rel
        if not p.exists():
            print(f"  {npy_rel:<35s} MISSING")
            continue

        frames = np.load(p)
        T = frames.shape[0]
        idx = np.linspace(0, T - 1, min(16, T), dtype=int).tolist()
        while len(idx) < 16:
            idx.append(idx[-1])
        sampled = [frames[i] for i in idx[:16]]
        pv = proc(sampled, return_tensors="pt")["pixel_values"].to(device)

        with torch.no_grad():
            emb = model(pv).cpu().numpy()[0]

        sims = {}
        for fname, fdata in refs.items():
            # Use centroid only — max over all embeddings inflates similarity
            sims[fname] = float(np.dot(emb, np.array(fdata["centroid"])))

        ranked = sorted(sims.items(), key=lambda x: -x[1])
        top1, sim1 = ranked[0]
        top2, sim2 = ranked[1]
        ok = "Y" if top1 == expected else "N"
        if ok == "Y":
            correct += 1
        total += 1
        print(f"  {npy_rel:<35s} {expected:<14s} {top1:<14s} {sim1:>6.3f}  {top2:<14s} {ok}")

    print(f"\n  Accuracy: {correct}/{total} ({correct / max(total, 1):.0%})")


if __name__ == "__main__":
    main()
