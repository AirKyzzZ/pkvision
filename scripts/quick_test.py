#!/usr/bin/env python3
"""Quick test: Can the model identify tricks in a video it's never seen?

Usage:
    python scripts/quick_test.py --input path/to/video.mp4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import av
import numpy as np
import torch
from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor

TRICK_CLASSES = {
    "somersaulting", "gymnastics tumbling", "parkour",
    "cartwheeling", "capoeira", "bouncing on trampoline",
    "vault", "breakdancing",
}

PARKOUR_MAP = {
    "somersaulting": ["back_flip", "front_flip", "side_flip"],
    "gymnastics tumbling": ["back_flip", "front_flip", "double_front", "double_back"],
    "parkour": ["kong_vault", "precision_jump", "wall_run"],
    "cartwheeling": ["aerial", "cartwheel", "side_flip"],
    "capoeira": ["b_twist", "raiz", "aerial"],
    "bouncing on trampoline": ["back_flip", "front_flip", "cork"],
    "vault": ["kong_vault", "speed_vault", "dash_vault"],
    "breakdancing": ["b_twist", "windmill"],
}


def read_video(path, n=16):
    try:
        c = av.open(str(path))
        af = [f.to_ndarray(format="rgb24") for f in c.decode(video=0)]
        c.close()
        if not af: return None
        idx = np.linspace(0, len(af)-1, min(n, len(af)), dtype=int)
        fr = [af[i] for i in idx]
        while len(fr) < n: fr.append(fr[-1])
        return fr[:n]
    except: return None


def get_device():
    if torch.cuda.is_available(): return torch.device("cuda")
    if hasattr(torch.backends,"mps") and torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", required=True)
    parser.add_argument("--device")
    args = parser.parse_args()

    vp = Path(args.input)
    if not vp.exists():
        print(f"Not found: {vp}"); sys.exit(1)

    dev = torch.device(args.device) if args.device else get_device()

    print()
    print("  PkVision — Quick Trick Detection")
    print("  " + "=" * 38)
    print(f"  Video:  {vp.name}")
    print(f"  Device: {dev}")
    print()

    mn = "MCG-NJU/videomae-base-finetuned-kinetics"
    print("  Loading model...", end=" ", flush=True)
    proc = VideoMAEImageProcessor.from_pretrained(mn)
    model = VideoMAEForVideoClassification.from_pretrained(mn).to(dev)
    model.eval()
    print("OK")

    print("  Reading video...", end=" ", flush=True)
    frames = read_video(vp)
    if not frames: print("FAIL"); sys.exit(1)
    print(f"OK ({len(frames)} frames)")

    print("  Classifying...", end=" ", flush=True)
    inp = proc(list(frames), return_tensors="pt")
    inp = {k: v.to(dev) for k, v in inp.items()}
    with torch.no_grad():
        probs = torch.softmax(model(**inp).logits, dim=1)[0]
    print("OK\n")

    top = probs.argsort(descending=True)[:10]

    print("  RESULTS:")
    print("  " + "-" * 50)

    best_lbl, best_conf = None, 0.0
    for rank, idx in enumerate(top):
        lbl = model.config.id2label[idx.item()]
        cf = probs[idx].item()
        bar = "#" * int(cf * 30)
        is_t = lbl in TRICK_CLASSES
        mk = " << TRICK" if is_t else ""
        if is_t and cf > best_conf: best_lbl, best_conf = lbl, cf
        print(f"  {rank+1:2d}. {lbl:35s} {cf:5.1%} {bar}{mk}")

    print("  " + "-" * 50)

    if best_lbl:
        pk = PARKOUR_MAP.get(best_lbl, [])
        print(f"\n  TRICK DETECTED: {best_lbl} ({best_conf:.0%})")
        print(f"  Likely parkour trick(s): {', '.join(t.replace('_',' ').title() for t in pk)}")
        print(f"\n  To distinguish specific tricks (e.g. backflip vs gainer),")
        print(f"  fine-tune the model with your labeled training data.")
    else:
        print(f"\n  No parkour trick detected.")

    print()


if __name__ == "__main__":
    main()
