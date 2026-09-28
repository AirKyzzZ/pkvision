#!/usr/bin/env python3
"""Zero-shot FIG trick recognition using CLIP.

Matches video frames against text descriptions of FIG tricks.
No training needed — uses CLIP's pre-trained vision-language alignment.

For each video clip:
1. Extract key frames
2. Encode with CLIP image encoder
3. Compare against text embeddings of all FIG trick descriptions
4. Return top-k matches with D-scores

Usage:
    python scripts/zero_shot_clip.py --input data/parkourtheory_clips/back_layout.mp4
    python scripts/zero_shot_clip.py --eval          # evaluate on all test clips
    python scripts/zero_shot_clip.py --eval-attrs     # evaluate attribute prediction
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
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
FIG_PATH = ROOT / "data" / "fig_tricks_2025.json"
CLIPS_DIR = ROOT / "data" / "parkourtheory_clips"
FRAMES_DIR = ROOT / "data" / "v5_full_training" / "frames"

# CLIP model — ViT-B/32 is fast, ViT-L/14 is more accurate
DEFAULT_MODEL = "ViT-B-32"
DEFAULT_PRETRAINED = "openai"


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ── FIG trick text descriptions ──────────────────────────────────────


def build_fig_descriptions():
    """Generate natural language descriptions for each FIG trick."""
    with open(FIG_PATH) as f:
        fig = json.load(f)

    tricks = []
    for cat_name, cat in fig["categories"].items():
        context = {
            "swing": "on a bar",
            "wall": "on a wall",
            "acrobatics": "on the ground",
            "pk_basics": "over an obstacle",
        }.get(cat_name, "")

        for trick in cat["tricks"]:
            name = trick["name"]
            flip = trick.get("flip", 0)
            twist = trick.get("twist", 0)
            direction = trick.get("direction", "")
            score = trick.get("score", 0)

            # Build natural language description
            parts = ["a person performing"]

            # Direction
            if direction == "backward":
                parts.append("a backward")
            elif direction == "forward":
                parts.append("a forward")
            elif direction == "side":
                parts.append("a sideways")
            else:
                parts.append("a")

            # Rotation description
            if flip >= 3:
                parts.append("triple somersault")
            elif flip >= 2:
                parts.append("double somersault")
            elif flip >= 1:
                parts.append("somersault")
            elif flip >= 0.5:
                parts.append("half rotation")
            else:
                parts.append("parkour movement")

            # Twists
            if twist >= 3:
                parts.append(f"with {twist:.0f} twists")
            elif twist >= 2:
                parts.append("with two twists")
            elif twist >= 1:
                parts.append("with a full twist")
            elif twist >= 0.5:
                parts.append("with a half twist")

            # Context
            if context:
                parts.append(context)

            desc = " ".join(parts)

            # Also create a simple name-based description
            simple_desc = f"a person doing a {name.lower()} in parkour"

            tricks.append({
                "name": name,
                "d_score": score,
                "category": cat_name,
                "flip": flip,
                "twist": twist,
                "direction": direction,
                "descriptions": [desc, simple_desc],
            })

    return tricks


# ── Attribute text descriptions ──────────────────────────────────────


ATTRIBUTE_PROMPTS = {
    "direction": {
        "backward": [
            "a person flipping backward",
            "a backward somersault in parkour",
            "a backflip movement",
        ],
        "forward": [
            "a person flipping forward",
            "a forward somersault in parkour",
            "a frontflip movement",
        ],
        "side": [
            "a person flipping sideways",
            "a sideways rotation in parkour",
            "a cartwheel or sideflip movement",
        ],
        "none": [
            "a person vaulting over an obstacle",
            "a parkour precision jump",
            "a person running and jumping without flipping",
        ],
    },
    "flip_count": {
        "0": [
            "a person jumping without flipping",
            "a vault or jump with no rotation",
            "a parkour movement without somersault",
        ],
        "1": [
            "a person doing a single flip",
            "one complete somersault rotation",
            "a single backflip or frontflip",
        ],
        "2": [
            "a person doing a double flip",
            "two complete somersault rotations",
            "a double backflip or double frontflip",
        ],
        "3+": [
            "a person doing a triple flip",
            "three or more somersault rotations",
            "a triple backflip",
        ],
    },
    "context": {
        "acrobatics": [
            "a person doing acrobatics on flat ground",
            "a ground-based flip or somersault",
            "gymnastics tumbling on the floor",
        ],
        "wall": [
            "a person doing a trick off a wall",
            "a wall flip or wall spin in parkour",
            "a person pushing off a vertical wall to flip",
        ],
        "swing": [
            "a person swinging on a bar and releasing",
            "a gymnastic bar release move",
            "a person doing a flyaway from a horizontal bar",
        ],
        "pk_basics": [
            "a person vaulting over a rail or wall",
            "a parkour vault like kong or speed vault",
            "a person jumping between obstacles",
        ],
    },
}


# ── Video loading ────────────────────────────────────────────────────


def load_video_frames(path, num_frames=16, size=224):
    """Load and resize video frames for CLIP."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    if not frames:
        return None

    T = len(frames)
    if T >= num_frames:
        indices = np.linspace(0, T - 1, num_frames, dtype=int)
    else:
        indices = list(range(T))
        while len(indices) < num_frames:
            indices.append(indices[-1])
        indices = indices[:num_frames]

    images = []
    for i in indices:
        frame = frames[i]
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (size, size))
        images.append(Image.fromarray(frame))

    return images


def load_npy_frames(path, num_frames=16):
    """Load .npy frames as PIL images."""
    frames = np.load(path)
    T = frames.shape[0]
    if T >= num_frames:
        indices = np.linspace(0, T - 1, num_frames, dtype=int)
    else:
        indices = list(range(T))
        while len(indices) < num_frames:
            indices.append(indices[-1])
        indices = indices[:num_frames]

    return [Image.fromarray(frames[i]) for i in indices]


# ── Main ─────────────────────────────────────────────────────────────


def main():
    import open_clip

    parser = argparse.ArgumentParser(description="Zero-shot FIG trick recognition (CLIP)")
    parser.add_argument("--input", help="Single video/npy to classify")
    parser.add_argument("--eval", action="store_true", help="Evaluate on test clips")
    parser.add_argument("--eval-attrs", action="store_true", help="Evaluate attribute prediction")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--pretrained", default=DEFAULT_PRETRAINED)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--num-frames", type=int, default=8, help="Frames to sample per clip")
    args = parser.parse_args()

    device = get_device()
    print(f"\n  Zero-Shot CLIP Trick Recognition")
    print(f"  {'=' * 44}")
    print(f"  Model:  {args.model} ({args.pretrained})")
    print(f"  Device: {device}")

    # Load CLIP
    print(f"  Loading CLIP...", end=" ", flush=True)
    t0 = time.time()
    model, _, preprocess = open_clip.create_model_and_transforms(
        args.model, pretrained=args.pretrained, device=device,
    )
    tokenizer = open_clip.get_tokenizer(args.model)
    model.eval()
    print(f"OK ({time.time() - t0:.1f}s)")

    # Build FIG trick descriptions
    fig_tricks = build_fig_descriptions()
    print(f"  FIG tricks: {len(fig_tricks)}")

    # Encode all trick descriptions
    print(f"  Encoding trick descriptions...", end=" ", flush=True)
    all_texts = []
    text_to_trick_idx = []
    for i, trick in enumerate(fig_tricks):
        for desc in trick["descriptions"]:
            all_texts.append(desc)
            text_to_trick_idx.append(i)

    tokens = tokenizer(all_texts).to(device)
    with torch.no_grad():
        text_features = model.encode_text(tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    print(f"OK ({len(all_texts)} descriptions)")

    def classify_frames(images):
        """Classify a list of PIL images against FIG tricks."""
        # Encode frames
        frame_tensors = torch.stack([preprocess(img) for img in images]).to(device)
        with torch.no_grad():
            image_features = model.encode_image(frame_tensors)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # Average frame embeddings
        avg_emb = image_features.mean(dim=0, keepdim=True)
        avg_emb = avg_emb / avg_emb.norm(dim=-1, keepdim=True)

        # Similarity against all text descriptions
        sims = (avg_emb @ text_features.T).squeeze(0).cpu().numpy()

        # Aggregate per trick (max across descriptions)
        trick_sims = {}
        for j, trick_idx in enumerate(text_to_trick_idx):
            name = fig_tricks[trick_idx]["name"]
            if name not in trick_sims or sims[j] > trick_sims[name]:
                trick_sims[name] = float(sims[j])

        ranked = sorted(trick_sims.items(), key=lambda x: -x[1])
        return ranked

    # ── Single clip mode ─────────────────────────────────────────────

    if args.input:
        path = Path(args.input)
        if path.suffix == ".npy":
            images = load_npy_frames(str(path), args.num_frames)
        else:
            images = load_video_frames(str(path), args.num_frames)

        if images is None:
            print(f"  ERROR: Cannot load {path}")
            sys.exit(1)

        ranked = classify_frames(images)

        print(f"\n  Top-{args.top_k} matches for {path.name}:")
        print(f"  {'Rank':<6s} {'Trick':<30s} {'Sim':>8s}  {'D-score':>8s}  {'Category'}")
        print(f"  {'-' * 6} {'-' * 30} {'-' * 8}  {'-' * 8}  {'-' * 12}")

        for i, (name, sim) in enumerate(ranked[: args.top_k]):
            trick = next(t for t in fig_tricks if t["name"] == name)
            d = trick["d_score"]
            cat = trick["category"]
            print(f"  {i + 1:<6d} {name:<30s} {sim:>8.3f}  D={d:<6.1f}  {cat}")
        print()
        return

    # ── Eval mode: test on known clips ───────────────────────────────

    if args.eval:
        test_cases = [
            ("acrobatics/back_layout.npy", "Backflip"),
            ("acrobatics/dive_front_flip.npy", "Frontflip"),
            ("acrobatics/gainer_full.npy", "Gainer 360"),
            ("acrobatics/double_corkscrew.npy", "Double Cork"),
            ("acrobatics/double_side_flip.npy", "Double Sideflip"),
            ("acrobatics/cartwheel.npy", "Cartwheel"),
            ("acrobatics/double_front_flip.npy", "Double Frontflip"),
            ("swing/flyaway_full.npy", "Swing Gainer 360"),
            ("wall/wall_flip.npy", "Wall Backflip"),
            ("acrobatics/triple_back_flip.npy", "Triple Backflip"),
            ("acrobatics/front_full.npy", "Frontflip 360"),
            ("acrobatics/butterfly_twist.npy", "B-360"),
        ]

        # Map expected names to trick indices
        fig_names = {t["name"].lower(): t["name"] for t in fig_tricks}

        print(f"\n  Evaluating on {len(test_cases)} test clips:")
        print(f"  {'Clip':<35s} {'Expected':<22s} {'Top-1':<22s} {'Sim':>6s} {'OK?'}")
        print(f"  {'-' * 35} {'-' * 22} {'-' * 22} {'-' * 6} {'-' * 3}")

        correct = 0
        top5 = 0
        total = 0

        for npy_rel, expected in test_cases:
            path = FRAMES_DIR / npy_rel
            if not path.exists():
                print(f"  {npy_rel:<35s} MISSING")
                continue

            images = load_npy_frames(str(path), args.num_frames)
            ranked = classify_frames(images)

            top1_name = ranked[0][0]
            top1_sim = ranked[0][1]
            top5_names = [r[0] for r in ranked[:5]]

            ok = "Y" if top1_name == expected else "N"
            in5 = top1_name == expected or expected in top5_names

            if ok == "Y":
                correct += 1
            if in5:
                top5 += 1
            total += 1

            print(f"  {npy_rel:<35s} {expected:<22s} {top1_name:<22s} {top1_sim:>6.3f} {ok}")

        print(f"\n  Accuracy: top-1={correct}/{total} ({correct / max(total, 1):.0%}), "
              f"top-5={top5}/{total} ({top5 / max(total, 1):.0%})")
        print()
        return

    # ── Attribute eval mode ──────────────────────────────────────────

    if args.eval_attrs:
        # Encode attribute prompts
        print(f"\n  Evaluating zero-shot attribute prediction:")

        # Load attribute manifest
        attr_path = ROOT / "data" / "v5_attribute_training" / "attribute_manifest.json"
        with open(attr_path) as f:
            attr_manifest = json.load(f)

        for attr_name, class_prompts in ATTRIBUTE_PROMPTS.items():
            # Encode text prompts for each class
            class_names = list(class_prompts.keys())
            class_embeddings = []

            for cls_name in class_names:
                prompts = class_prompts[cls_name]
                toks = tokenizer(prompts).to(device)
                with torch.no_grad():
                    feats = model.encode_text(toks)
                    feats = feats / feats.norm(dim=-1, keepdim=True)
                class_embeddings.append(feats.mean(dim=0))

            class_matrix = torch.stack(class_embeddings)
            class_matrix = class_matrix / class_matrix.norm(dim=-1, keepdim=True)

            # Test on a sample of clips
            clips = attr_manifest["clips"]
            sample = [c for c in clips if c.get(attr_name) or c.get(attr_name.replace("_count", "_bin"))]
            if len(sample) > 100:
                import random
                random.seed(42)
                sample = random.sample(sample, 100)

            correct = 0
            total = 0
            attr_key = attr_name if attr_name in sample[0] else attr_name.replace("_count", "_bin")

            for clip in sample:
                npy_path = Path(clip["npy_path"])
                if not npy_path.exists():
                    continue

                gt = clip.get(attr_key, clip.get(attr_name))
                if gt not in class_names:
                    # Map flip_bin/twist_bin to simplified classes
                    if attr_name == "flip_count":
                        if gt in ("0", "0.5"):
                            gt = "0"
                        elif gt in ("1", "1.5"):
                            gt = "1"
                        else:
                            gt = "2" if "2" in class_names else "3+"
                    continue

                images = load_npy_frames(str(npy_path), args.num_frames)
                frame_tensors = torch.stack([preprocess(img) for img in images]).to(device)
                with torch.no_grad():
                    img_feats = model.encode_image(frame_tensors)
                    img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
                avg = img_feats.mean(dim=0, keepdim=True)
                avg = avg / avg.norm(dim=-1, keepdim=True)

                sims = (avg @ class_matrix.T).squeeze(0)
                pred_idx = sims.argmax().item()
                pred = class_names[pred_idx]

                if pred == gt:
                    correct += 1
                total += 1

            if total > 0:
                print(f"  {attr_name:<15s}: {correct}/{total} ({correct / total:.0%})")

        print()


if __name__ == "__main__":
    main()
