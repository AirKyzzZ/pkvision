#!/usr/bin/env python3
"""Prepare training data by merging manual labels with auto-labels.

Combines your hand-labeled clips with VideoMAE auto-classifications
to create a training-ready dataset with enough samples per class.

Strategy:
- Use manual labels where available (highest priority)
- Use Kinetics auto-labels for the rest, mapped to parkour categories
- Group rare tricks to ensure minimum samples per class
- Extract frames using PyAV (no YOLO needed for VideoMAE fine-tuning)

Usage:
    python scripts/prepare_training.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import av
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}

# Map Kinetics labels to parkour training classes
KINETICS_TO_TRAIN_CLASS = {
    "somersaulting": "flip",
    "gymnastics tumbling": "tumbling",
    "parkour": "parkour_movement",
    "cartwheeling": "cartwheel",
    "bouncing on trampoline": "trampoline_flip",
    "vault": "vault",
    "capoeira": "flip",
    "breakdancing": "other",
    "high jump": "other",
    "high kick": "other",
    "drop kicking": "other",
    "hurdling": "other",
}

# Group manual trick labels into trainable classes
# (we need at least 5 samples per class for meaningful training)
TRICK_TO_TRAIN_CLASS = {
    # Flips (backward rotation)
    "back_flip": "back_flip",
    "double_back": "back_flip",  # Group with backflip for now
    # Flips (forward rotation)
    "front_flip": "front_flip",
    "double_front": "front_flip",
    "double_pike": "front_flip",
    # Twists & corks
    "double_cork": "twist_cork",
    "cork": "twist_cork",
    "b_twist": "twist_cork",
    "double_full": "twist_cork",
    "tak_full": "twist_cork",
    "twist_180": "twist_cork",
    "twist_360": "twist_cork",
    "twist_540": "twist_cork",
    "twist_720": "twist_cork",
    "raiz": "twist_cork",
    "castaway": "twist_cork",
    "flash_kick": "twist_cork",
    # Side movements
    "side_flip": "side_flip",
    "aerial": "side_flip",
    "cartwheel": "side_flip",
    "palm_flip": "side_flip",
    "krok": "side_flip",
    # Vaults
    "kong_vault": "vault",
    "double_kong": "vault",
    "speed_vault": "vault",
    "dash_vault": "vault",
    # Parkour movement
    "precision_jump": "parkour_movement",
    "wall_run": "parkour_movement",
    "cat_leap": "parkour_movement",
    "climb_up": "parkour_movement",
    "tic_tac": "parkour_movement",
    # Other
    "standing": "other",
    "walking": "other",
    "other_trick": "other",
    "combo": "other",
    "wall_flip": "back_flip",
    "wall_back_flip": "back_flip",
    "webster": "front_flip",
    "gainer": "back_flip",
    "flip_360": "twist_cork",
    "triple_cork": "twist_cork",
}

MIN_FRAMES = 16


def read_video_frames(path: Path, num_frames: int = 16) -> np.ndarray | None:
    """Read evenly-spaced frames as numpy array."""
    try:
        container = av.open(str(path))
        all_frames = [f.to_ndarray(format="rgb24") for f in container.decode(video=0)]
        container.close()
        if not all_frames:
            return None
        total = len(all_frames)
        indices = np.linspace(0, total - 1, min(num_frames, total), dtype=int)
        frames = [all_frames[i] for i in indices]
        while len(frames) < num_frames:
            frames.append(frames[-1])
        return np.stack(frames[:num_frames])  # (T, H, W, 3)
    except Exception:
        return None


def main():
    clips_dir = Path("data/clips")
    manual_labels_path = clips_dir / "labels.json"
    auto_labels_path = clips_dir / "auto_labels.json"
    output_dir = Path("data/training")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load manual labels
    manual_by_file: dict[str, str] = {}
    if manual_labels_path.exists():
        with open(manual_labels_path) as f:
            data = json.load(f)
        labels_list = data.get("labels", []) if isinstance(data, dict) else data
        for entry in labels_list:
            filename = entry["file"]
            trick_id = entry["trick_id"]
            # Map to training class
            train_class = TRICK_TO_TRAIN_CLASS.get(trick_id, trick_id)
            manual_by_file[filename] = train_class

    # Load auto labels
    auto_by_file: dict[str, str] = {}
    if auto_labels_path.exists():
        with open(auto_labels_path) as f:
            auto_data = json.load(f)
        for entry in auto_data:
            if not entry.get("is_trick"):
                continue
            kinetics_label = entry.get("kinetics_label", "")
            train_class = KINETICS_TO_TRAIN_CLASS.get(kinetics_label, "other")
            auto_by_file[entry["file"]] = train_class

    # Merge: manual takes priority
    merged: dict[str, str] = {}
    for filename in auto_by_file:
        if filename in manual_by_file:
            merged[filename] = manual_by_file[filename]
        else:
            merged[filename] = auto_by_file[filename]

    # Add any manual labels not in auto
    for filename, cls in manual_by_file.items():
        if filename not in merged:
            merged[filename] = cls

    # Count per class
    class_counts: dict[str, int] = {}
    for cls in merged.values():
        class_counts[cls] = class_counts.get(cls, 0) + 1

    print("Training class distribution:")
    print("-" * 40)
    total_usable = 0
    classes_to_use = []
    for cls, count in sorted(class_counts.items(), key=lambda x: -x[1]):
        usable = count >= 3
        marker = "  OK" if usable else "  SKIP (< 3 samples)"
        print(f"  {cls:25s} {count:3d} clips{marker}")
        if usable:
            total_usable += count
            classes_to_use.append(cls)

    print(f"\nUsable classes: {len(classes_to_use)}")
    print(f"Usable clips: {total_usable}")
    print(f"Classes: {', '.join(classes_to_use)}")

    # Filter to usable classes
    filtered = {f: c for f, c in merged.items() if c in classes_to_use}

    # Extract frames and save
    print(f"\nExtracting {len(filtered)} video frames...")
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(exist_ok=True)

    training_manifest = []
    processed = 0
    failed = 0

    for filename, train_class in sorted(filtered.items()):
        filepath = clips_dir / filename
        if not filepath.exists():
            failed += 1
            continue

        print(f"  [{processed+1}/{len(filtered)}] {filename}...", end=" ", flush=True)

        frames = read_video_frames(filepath)
        if frames is None:
            print("SKIP")
            failed += 1
            continue

        # Save frames as numpy array
        out_name = filepath.stem + ".npy"
        np.save(frames_dir / out_name, frames)

        training_manifest.append({
            "file": out_name,
            "original": filename,
            "class": train_class,
        })
        processed += 1
        print(f"{train_class}")

    # Save manifest
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump({
            "classes": classes_to_use,
            "samples": training_manifest,
            "class_counts": {c: sum(1 for s in training_manifest if s["class"] == c) for c in classes_to_use},
        }, f, indent=2)

    print(f"\n{'='*50}")
    print(f"Training data prepared!")
    print(f"{'='*50}")
    print(f"Processed: {processed}")
    print(f"Failed: {failed}")
    print(f"Classes: {len(classes_to_use)}")
    print(f"Manifest: {manifest_path}")
    print(f"Frames: {frames_dir}")

    # Final class distribution
    print(f"\nFinal distribution:")
    for cls in classes_to_use:
        count = sum(1 for s in training_manifest if s["class"] == cls)
        bar = "#" * min(count, 40)
        print(f"  {cls:25s} {count:3d} {bar}")

    print(f"\nNext: python scripts/finetune.py")


if __name__ == "__main__":
    main()
