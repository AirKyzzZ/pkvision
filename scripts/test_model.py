#!/usr/bin/env python3
"""Test the fine-tuned PkVision model on a video.

Usage:
    python scripts/test_model.py --input video.mp4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import av
import numpy as np
import torch
import torch.nn as nn
from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor


def read_video(path, n=16):
    try:
        c = av.open(str(path))
        af = [f.to_ndarray(format="rgb24") for f in c.decode(video=0)]
        c.close()
        if not af:
            return None
        total = len(af)
        indices = np.linspace(0, total - 1, min(n, total), dtype=int)
        frames = [af[i] for i in indices]
        while len(frames) < n:
            frames.append(frames[-1])
        return frames[:n]
    except Exception as e:
        print(f"Error: {e}")
        return None


def get_device(override=None):
    if override:
        return torch.device(override)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", required=True)
    parser.add_argument("--model", "-m", default="data/models/pkvision_videomae.pt")
    parser.add_argument("--device")
    args = parser.parse_args()

    vp = Path(args.input)
    mp = Path(args.model)
    if not vp.exists():
        print(f"Not found: {vp}"); sys.exit(1)
    if not mp.exists():
        print(f"No model at {mp}. Run finetune.py first."); sys.exit(1)

    device = get_device(args.device)

    print(f"\n  PkVision Trick Classification")
    print(f"  {'='*40}")
    print(f"  Video:  {vp.name}")
    print(f"  Device: {device}\n")

    print("  Loading model...", end=" ", flush=True)
    ckpt = torch.load(mp, map_location="cpu", weights_only=False)
    classes = ckpt["classes"]
    cfg = ckpt["config"]
    proc = VideoMAEImageProcessor.from_pretrained(cfg["model_name"])
    model = VideoMAEForVideoClassification.from_pretrained(cfg["model_name"])
    model.classifier = nn.Linear(cfg["hidden_size"], cfg["num_classes"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    print(f"OK ({len(classes)} classes)")

    print("  Reading video...", end=" ", flush=True)
    frames = read_video(vp)
    if not frames:
        print("FAIL"); sys.exit(1)
    print(f"OK")

    print("  Classifying...", end=" ", flush=True)
    inp = proc(frames, return_tensors="pt")
    inp = {k: v.to(device) for k, v in inp.items()}
    with torch.no_grad():
        probs = torch.softmax(model(pixel_values=inp["pixel_values"]).logits, dim=1)[0]
    print("OK\n")

    si = probs.argsort(descending=True)
    print("  RESULTS:")
    print("  " + "-" * 45)
    for r, idx in enumerate(si):
        cls = classes[idx.item()]
        cf = probs[idx].item()
        bar = "#" * int(cf * 30)
        mk = " << TOP" if r == 0 else ""
        print(f"  {r+1}. {cls:25s} {cf:5.1%} {bar}{mk}")
    print("  " + "-" * 45)

    top = classes[si[0].item()]
    conf = probs[si[0]].item()
    print(f"\n  DETECTED: {top.upper().replace('_',' ')} ({conf:.0%})\n")


if __name__ == "__main__":
    main()
