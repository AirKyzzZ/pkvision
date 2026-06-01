# scripts/make_proposals.py
"""Generate + cache {trick,cues,conf} proposals for a clip list.

Usage: python3 scripts/make_proposals.py --limit 50 --proposer local
Caches to data/labeling/proposals/<proposer>/<slug>.json (never recomputed).
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.labeling.clip_ref import ClipRef
from core.labeling.cue_model import CueModel, CUE_CLASSES
from core.labeling.proposer import LocalModelProposer

POSE_GLOB = str(ROOT / "data" / "keypoints" / "**" / "shard_*.npz")
CLIPS = ROOT / "data" / "parkourtheory_clips_cropped"


def load_skeletons() -> dict:
    skel = {}
    for sh in sorted(glob.glob(POSE_GLOB, recursive=True)):
        with np.load(sh) as z:
            for s in z.files:
                skel[s] = z[s].astype(np.float32)
    return skel


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--proposer", choices=["local"], default="local")
    ap.add_argument("--ckpt", default="")  # optional cue_model checkpoint; untrained if empty
    args = ap.parse_args()

    out = ROOT / "data" / "labeling" / "proposals" / args.proposer
    out.mkdir(parents=True, exist_ok=True)
    skel = load_skeletons()

    import torch
    model = CueModel(CUE_CLASSES, t=48)
    if args.ckpt:
        model.load_state_dict(torch.load(args.ckpt, map_location="cpu", weights_only=True))
    model.eval()

    proposer = LocalModelProposer(model=model)
    slugs = sorted(skel)[: args.limit]
    written = 0
    for s in slugs:
        dst = out / f"{s}.json"
        if dst.exists():
            continue
        ref = ClipRef(s, video_path=CLIPS / f"{s}.mp4", frames_path=None, skeleton=skel[s])
        prop = proposer.propose(ref)
        dst.write_text(json.dumps(asdict(prop), indent=2))
        written += 1
    print(f"wrote {written} proposals to {out}")


if __name__ == "__main__":
    main()
