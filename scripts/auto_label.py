#!/usr/bin/env python3
"""Auto-label video clips using a pre-trained Kinetics-400 VideoMAE model.

Uses HuggingFace VideoMAE (pre-trained on Kinetics-400) to classify clips,
then maps Kinetics classes to PkVision trick categories for review.

Usage:
    python scripts/auto_label.py
    python scripts/auto_label.py --clips-dir data/clips --top-k 5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import av
import numpy as np
import torch
from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor

sys.path.insert(0, str(Path(__file__).parent.parent))

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}

# Kinetics classes that indicate a trick is happening
TRICK_CLASSES = {
    "somersaulting", "gymnastics tumbling", "parkour",
    "cartwheeling", "capoeira", "bouncing on trampoline",
}

# Map Kinetics classes to PkVision categories
KINETICS_TO_CATEGORY = {
    "somersaulting": "flip",
    "gymnastics tumbling": "flip",
    "parkour": "parkour",
    "cartwheeling": "flip",
    "capoeira": "flip",
    "bouncing on trampoline": "flip",
    "breakdancing": "other",
    "vault": "vault",
    "high kick": "precision",
    "side kick": "precision",
    "rock climbing": "precision",
}


def load_model(device: torch.device):
    """Load VideoMAE pre-trained on Kinetics-400."""
    print("Loading VideoMAE model (first run downloads ~350MB)...")
    model_name = "MCG-NJU/videomae-base-finetuned-kinetics"
    processor = VideoMAEImageProcessor.from_pretrained(model_name)
    model = VideoMAEForVideoClassification.from_pretrained(model_name)
    model = model.to(device)
    return processor, model


def read_video_frames(path: Path, num_frames: int = 16) -> list[np.ndarray] | None:
    """Read evenly-spaced frames from a video using PyAV."""
    try:
        container = av.open(str(path))
        stream = container.streams.video[0]

        # Collect all frames first (safe for all formats)
        all_frames = []
        for frame in container.decode(video=0):
            all_frames.append(frame.to_ndarray(format="rgb24"))

        container.close()

        if not all_frames:
            return None

        total = len(all_frames)
        if total <= num_frames:
            indices = list(range(total))
        else:
            indices = np.linspace(0, total - 1, num_frames, dtype=int).tolist()

        frames = [all_frames[i] for i in indices]

        # Pad if needed
        while len(frames) < num_frames:
            frames.append(frames[-1])

        return frames[:num_frames]
    except Exception as e:
        return None


def classify_video(
    frames: list[np.ndarray],
    processor,
    model,
    device: torch.device,
    top_k: int = 5,
) -> list[tuple[str, float]]:
    """Classify a video and return top-k predictions."""
    inputs = processor(list(frames), return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)[0]

    top_indices = probs.argsort(descending=True)[:top_k]
    results = []
    for idx in top_indices:
        label = model.config.id2label[idx.item()]
        conf = probs[idx].item()
        results.append((label, conf))

    return results


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    parser = argparse.ArgumentParser(description="Auto-label clips with pre-trained VideoMAE")
    parser.add_argument("--clips-dir", default="data/clips", help="Directory with video clips")
    parser.add_argument("--output", default="data/clips/auto_labels.json", help="Output file")
    parser.add_argument("--top-k", type=int, default=5, help="Top K predictions per clip")
    parser.add_argument("--device", help="Force device (cpu/mps/cuda)")
    args = parser.parse_args()

    clips_dir = Path(args.clips_dir)
    output_path = Path(args.output)
    device = torch.device(args.device) if args.device else get_device()

    video_files = sorted(
        p for p in clips_dir.iterdir()
        if p.suffix.lower() in VIDEO_EXTENSIONS
    )
    print(f"Found {len(video_files)} video clips")
    print(f"Device: {device}\n")

    processor, model = load_model(device)
    model.eval()

    results = []
    trick_count = 0

    for i, vpath in enumerate(video_files):
        pct = f"[{i+1}/{len(video_files)}]"
        print(f"{pct} {vpath.name}...", end=" ", flush=True)

        frames = read_video_frames(vpath)
        if frames is None:
            print("SKIP (can't read)")
            results.append({
                "file": vpath.name,
                "status": "error",
                "predictions": [],
            })
            continue

        preds = classify_video(frames, processor, model, device, args.top_k)
        top_label, top_conf = preds[0]

        is_trick = any(label in TRICK_CLASSES for label, conf in preds if conf > 0.05)
        category = KINETICS_TO_CATEGORY.get(top_label, "unknown")

        entry = {
            "file": vpath.name,
            "status": "auto",
            "is_trick": is_trick,
            "kinetics_label": top_label,
            "kinetics_confidence": round(top_conf, 3),
            "category": category,
            "top_predictions": [
                {"label": label, "confidence": round(conf, 3)}
                for label, conf in preds
            ],
            "pkvision_trick_id": None,
        }
        results.append(entry)

        if is_trick:
            trick_count += 1

        tag = "TRICK" if is_trick else "----"
        print(f"{top_label} ({top_conf:.0%}) [{tag}]")

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Auto-labeling complete!")
    print(f"{'='*60}")
    print(f"Total clips:   {len(results)}")
    print(f"Likely tricks: {trick_count}")
    print(f"Non-tricks:    {len(results) - trick_count}")
    print(f"\nSaved to: {output_path}")

    label_counts: dict[str, int] = {}
    for r in results:
        lbl = r.get("kinetics_label", "unknown")
        label_counts[lbl] = label_counts.get(lbl, 0) + 1

    print(f"\nTop Kinetics classes detected:")
    for label, count in sorted(label_counts.items(), key=lambda x: -x[1])[:15]:
        bar = "#" * min(count, 40)
        print(f"  {label:30s} {count:3d} {bar}")

    print(f"\nNext: Review in labeler UI and assign specific parkour trick IDs")
    print(f"  python scripts/labeler_ui.py")


if __name__ == "__main__":
    main()
