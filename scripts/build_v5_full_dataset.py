#!/usr/bin/env python3
"""Build the FULL v5 training dataset from ALL 1,618 parkourtheory clips.

Maps every clip to a FIG category (acrobatics/wall/swing/pk_basics) using the
parkourtheory type field, then optionally assigns FIG trick labels for clips
that have a direct mapping.

Outputs two datasets:
1. Category dataset (1,618 clips → 4 classes) for the category classifier
2. Trick dataset (96 clips → 96 FIG tricks) for the trick classifier

Both include augmented copies (horizontal flip + temporal jitter).

Usage:
    python scripts/build_v5_full_dataset.py
    python scripts/build_v5_full_dataset.py --augment-factor 5
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import av
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent

CLIPS_DIR_CROPPED = ROOT / "data" / "parkourtheory_clips_cropped"
CLIPS_DIR_ORIGINAL = ROOT / "data" / "parkourtheory_clips"
# Prefer cropped clips if available
CLIPS_DIR = CLIPS_DIR_CROPPED if CLIPS_DIR_CROPPED.exists() and any(CLIPS_DIR_CROPPED.iterdir()) else CLIPS_DIR_ORIGINAL
DETAILED_PATH = ROOT / "data" / "parkourtheory_detailed.json"
FIG_MAP_PATH = ROOT / "data" / "fig_to_parkourtheory_map_v2.json"
FIG_PATH = ROOT / "data" / "fig_tricks_2025.json"
OUTPUT_DIR = ROOT / "data" / "v5_full_training"

# Map parkourtheory type prefixes to FIG categories
TYPE_TO_CATEGORY = {
    "wall": "wall",
    "bar": "swing",
    "beam": "swing",
    "parallel bar": "swing",
    "pole": "swing",
    "ceiling": "swing",
    "vault": "pk_basics",
    "jump": "pk_basics",
    "roll": "acrobatics",
    "flip": "acrobatics",
    "twist": "acrobatics",
    "kick": "acrobatics",
    "freeze": "pk_basics",
    "handstand": "acrobatics",
    "rail": "swing",
    "pommel horse": "swing",
    "spin": "acrobatics",
}


def classify_type(trick_type: str) -> str:
    """Map a parkourtheory type string to a FIG category."""
    if not trick_type:
        return "acrobatics"  # default

    t = trick_type.lower().strip()

    # Priority: wall > bar/beam > vault > flip/twist
    # Check compound types in priority order
    if "wall" in t and ("bar" in t or "beam" in t):
        return "wall"  # wall takes priority
    if "vault" in t and "wall" in t:
        return "pk_basics"  # vault context
    if "vault" in t and ("bar" in t or "beam" in t):
        return "pk_basics"

    # Check first component
    parts = [p.strip() for p in re.split(r"[/,]", t)]
    for part in parts:
        for prefix, cat in TYPE_TO_CATEGORY.items():
            if part.startswith(prefix):
                return cat

    return "acrobatics"  # default for unknown


def read_video_frames(path: Path, num_frames: int = 16) -> np.ndarray | None:
    """Read evenly-spaced frames as numpy array (T, H, W, 3)."""
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
        frames = np.stack(frames[:num_frames])
        # Resize to 256x256
        if frames.shape[1] != 256 or frames.shape[2] != 256:
            frames = np.stack([cv2.resize(f, (256, 256)) for f in frames])
        return frames
    except Exception as e:
        return None


def augment_frames(frames: np.ndarray, rng: np.random.Generator, aug_idx: int) -> np.ndarray:
    """Apply augmentation to a frame sequence.

    Augmentations:
    - Horizontal flip (50% chance)
    - Temporal speed jitter (resample frames with slight variation)
    - Random crop + resize (slight zoom 90-100%)
    - Color jitter (brightness/contrast)
    """
    T, H, W, C = frames.shape
    out = frames.copy()

    # Horizontal flip (50%)
    if rng.random() < 0.5:
        out = out[:, :, ::-1, :].copy()

    # Temporal jitter: shift start point slightly
    if T > 4:
        shift = rng.integers(-2, 3)  # -2 to +2 frames
        indices = np.clip(np.arange(T) + shift, 0, T - 1)
        out = out[indices]

    # Random crop (90-100% of frame, then resize back)
    crop_frac = rng.uniform(0.85, 1.0)
    crop_size = int(H * crop_frac)
    max_offset = H - crop_size
    if max_offset > 0:
        y_off = rng.integers(0, max_offset + 1)
        x_off = rng.integers(0, max_offset + 1)
        out = np.stack([
            cv2.resize(f[y_off:y_off + crop_size, x_off:x_off + crop_size], (W, H))
            for f in out
        ])

    # Brightness/contrast jitter
    brightness = rng.uniform(-20, 20)
    contrast = rng.uniform(0.85, 1.15)
    out = np.clip(out.astype(np.float32) * contrast + brightness, 0, 255).astype(np.uint8)

    return out


def name_to_slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def main():
    parser = argparse.ArgumentParser(description="Build full v5 dataset from all parkourtheory clips")
    parser.add_argument("--augment-factor", type=int, default=5,
                        help="Number of augmented copies per clip (default: 5)")
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--test-clips-dir", type=Path, default=ROOT / "data" / "final_clips",
                        help="Directory with test clips to include in training")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    # ── Load data ────────────────────────────────────────────────────────
    print("Loading parkourtheory metadata...")
    with open(DETAILED_PATH) as f:
        detailed = json.load(f)
    pt_by_slug = {name_to_slug(e["name"]): e for e in detailed}
    print(f"  {len(detailed)} tricks in database")

    print("Loading FIG mapping...")
    with open(FIG_MAP_PATH) as f:
        fig_map = json.load(f)
    with open(FIG_PATH) as f:
        fig_data = json.load(f)

    # Build FIG trick → category + score lookup
    fig_info = {}
    for cat_key, cat_data in fig_data["categories"].items():
        for trick in cat_data["tricks"]:
            fig_info[trick["name"]] = {"category": cat_key, "score": trick["score"]}

    # Build reverse map: pt_slug → fig_name
    slug_to_fig: dict[str, tuple[str, str]] = {}  # slug → (fig_name, category)
    for cat_key, entries in fig_map.items():
        if cat_key.startswith("_"):
            continue
        for fig_name, pt_name in entries.items():
            slug = name_to_slug(pt_name)
            # Keep lowest-score fig trick for each slug (most visually representative)
            if slug in slug_to_fig:
                existing_fig = slug_to_fig[slug][0]
                existing_score = fig_info.get(existing_fig, {}).get("score", 999)
                new_score = fig_info.get(fig_name, {}).get("score", 999)
                if new_score >= existing_score:
                    continue
            slug_to_fig[slug] = (fig_name, cat_key)

    # ── Classify all clips ───────────────────────────────────────────────
    print("\nClassifying all clips...")
    clips = sorted(CLIPS_DIR.glob("*.mp4"))
    print(f"  {len(clips)} clips found")

    clip_entries: list[dict] = []
    cat_counts = defaultdict(int)
    fig_match_count = 0

    for clip_path in clips:
        slug = clip_path.stem

        # Try FIG mapping first
        fig_name = None
        if slug in slug_to_fig:
            fig_name, fig_cat = slug_to_fig[slug]
            category = fig_cat
            fig_match_count += 1
        else:
            # Use parkourtheory type to determine category
            entry = pt_by_slug.get(slug, {})
            trick_type = entry.get("type", "")
            category = classify_type(trick_type)

        cat_counts[category] += 1
        clip_entries.append({
            "path": str(clip_path),
            "slug": slug,
            "category": category,
            "fig_name": fig_name,
        })

    print(f"\n  Category distribution:")
    for cat, count in sorted(cat_counts.items(), key=lambda x: -x[1]):
        bar = "#" * min(count // 5, 50)
        print(f"    {cat:15s} {count:5d}  {bar}")
    print(f"    {'TOTAL':15s} {sum(cat_counts.values()):5d}")
    print(f"  FIG-mapped clips: {fig_match_count}")

    # ── Also include test clips ──────────────────────────────────────────
    test_clip_dir = args.test_clips_dir
    test_clips_added = 0
    if test_clip_dir.exists():
        # Known test clips with their categories
        test_clip_labels = {
            "backflip": "acrobatics",
            "frontflip": "acrobatics",
            "gainer": "acrobatics",
            "back_double_full": "acrobatics",
            "double_cork": "acrobatics",
        }
        for clip_file in test_clip_dir.glob("*.mp4"):
            stem = clip_file.stem.lower()
            category = test_clip_labels.get(stem)
            if category:
                clip_entries.append({
                    "path": str(clip_file),
                    "slug": f"test_{stem}",
                    "category": category,
                    "fig_name": None,
                })
                test_clips_added += 1
        if test_clips_added:
            print(f"  Added {test_clips_added} test clips")

    # ── Extract frames ───────────────────────────────────────────────────
    frames_dir = args.output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    categories = sorted(cat_counts.keys())
    cat_samples: dict[str, list[dict]] = defaultdict(list)  # category → samples
    trick_samples: list[dict] = []  # FIG trick-level samples

    total_clips = len(clip_entries)
    processed = 0
    failed = 0
    total_samples = 0
    t0 = time.time()

    print(f"\nExtracting {args.num_frames} frames × {1 + args.augment_factor} versions from {total_clips} clips...")

    for i, entry in enumerate(clip_entries):
        clip_path = Path(entry["path"])
        slug = entry["slug"]
        category = entry["category"]
        fig_name = entry["fig_name"]

        if (i + 1) % 100 == 0 or (i + 1) == total_clips:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (total_clips - i - 1) / rate if rate > 0 else 0
            print(f"  [{i + 1:5d}/{total_clips}] rate={rate:.1f}/s ETA={eta:.0f}s processed={processed} failed={failed}")

        # Read original frames
        frames = read_video_frames(clip_path, num_frames=args.num_frames)
        if frames is None:
            failed += 1
            continue

        processed += 1

        # Save original
        cat_dir = frames_dir / category
        cat_dir.mkdir(parents=True, exist_ok=True)
        out_name = f"{slug}.npy"
        np.save(cat_dir / out_name, frames)
        rel_path = f"{category}/{out_name}"

        sample = {"file": rel_path, "category": category, "slug": slug}
        if fig_name:
            sample["fig_name"] = fig_name
        cat_samples[category].append(sample)
        if fig_name:
            trick_samples.append(sample)
        total_samples += 1

        # Save augmented copies
        for aug_idx in range(args.augment_factor):
            aug_frames = augment_frames(frames, rng, aug_idx)
            aug_name = f"{slug}_aug{aug_idx}.npy"
            np.save(cat_dir / aug_name, aug_frames)
            aug_rel = f"{category}/{aug_name}"

            aug_sample = {"file": aug_rel, "category": category, "slug": slug}
            if fig_name:
                aug_sample["fig_name"] = fig_name
            cat_samples[category].append(aug_sample)
            if fig_name:
                trick_samples.append(aug_sample)
            total_samples += 1

    elapsed = time.time() - t0

    # ── Build splits ─────────────────────────────────────────────────────
    print(f"\nBuilding train/val/test splits...")

    # Category-level splits (stratified by category)
    cat_splits = {"train": [], "val": [], "test": []}
    for cat in categories:
        samples = cat_samples[cat]
        # Group by base slug (keep augmentations together)
        by_slug = defaultdict(list)
        for s in samples:
            by_slug[s["slug"]].append(s)

        slugs = list(by_slug.keys())
        rng.shuffle(slugs)
        n = len(slugs)
        n_train = max(1, int(n * 0.8))
        n_val = max(1, int(n * 0.1))

        train_slugs = set(slugs[:n_train])
        val_slugs = set(slugs[n_train:n_train + n_val])
        test_slugs = set(slugs[n_train + n_val:])

        for s in samples:
            if s["slug"] in train_slugs:
                cat_splits["train"].append(s["file"])
            elif s["slug"] in val_slugs:
                cat_splits["val"].append(s["file"])
            else:
                cat_splits["test"].append(s["file"])

    # Trick-level splits (stratified by FIG name)
    trick_splits = {"train": [], "val": [], "test": []}
    trick_by_fig = defaultdict(list)
    for s in trick_samples:
        trick_by_fig[s["fig_name"]].append(s)

    for fig_name, samples in trick_by_fig.items():
        by_slug = defaultdict(list)
        for s in samples:
            by_slug[s["slug"]].append(s)

        slugs = list(by_slug.keys())
        # With 1 slug per trick, put all augmentations in train, use original for val
        if len(slugs) == 1:
            for s in samples:
                if "_aug" in s["file"]:
                    trick_splits["train"].append(s["file"])
                else:
                    # Use original for both train and val
                    trick_splits["train"].append(s["file"])
                    trick_splits["val"].append(s["file"])
        else:
            rng.shuffle(slugs)
            train_slugs = set(slugs[:max(1, int(len(slugs) * 0.8))])
            for s in samples:
                if s["slug"] in train_slugs:
                    trick_splits["train"].append(s["file"])
                else:
                    trick_splits["val"].append(s["file"])

    # ── Build manifests ──────────────────────────────────────────────────
    all_cat_samples = [s for samples in cat_samples.values() for s in samples]
    fig_trick_classes = sorted(set(s["fig_name"] for s in trick_samples))

    cat_manifest = {
        "type": "category",
        "classes": categories,
        "samples": [{"file": s["file"], "class": s["category"]} for s in all_cat_samples],
        "splits": cat_splits,
        "config": {
            "num_frames": args.num_frames,
            "augment_factor": args.augment_factor,
            "num_classes": len(categories),
            "num_samples": len(all_cat_samples),
        },
    }

    trick_manifest = {
        "type": "trick",
        "classes": fig_trick_classes,
        "class_info": {name: fig_info.get(name, {}) for name in fig_trick_classes},
        "samples": [{"file": s["file"], "class": s["fig_name"]} for s in trick_samples],
        "splits": trick_splits,
        "config": {
            "num_frames": args.num_frames,
            "augment_factor": args.augment_factor,
            "num_classes": len(fig_trick_classes),
            "num_samples": len(trick_samples),
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cat_manifest_path = args.output_dir / "category_manifest.json"
    trick_manifest_path = args.output_dir / "trick_manifest.json"

    with open(cat_manifest_path, "w") as f:
        json.dump(cat_manifest, f, indent=2)
    with open(trick_manifest_path, "w") as f:
        json.dump(trick_manifest, f, indent=2)

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"V5 Full Dataset Built")
    print(f"{'=' * 60}")

    print(f"\nCATEGORY DATASET (4 classes):")
    print(f"  {'Category':15s} {'Clips':>6s} {'+ Aug':>6s} {'Total':>7s}  {'Train':>6s} {'Val':>5s} {'Test':>5s}")
    for cat in categories:
        n_clips = sum(1 for s in cat_samples[cat] if "_aug" not in s["file"])
        n_aug = len(cat_samples[cat]) - n_clips
        n_total = len(cat_samples[cat])
        n_train = sum(1 for f in cat_splits["train"] if f.startswith(cat + "/"))
        n_val = sum(1 for f in cat_splits["val"] if f.startswith(cat + "/"))
        n_test = sum(1 for f in cat_splits["test"] if f.startswith(cat + "/"))
        print(f"  {cat:15s} {n_clips:6d} {n_aug:6d} {n_total:7d}  {n_train:6d} {n_val:5d} {n_test:5d}")
    print(f"  {'TOTAL':15s} {processed:6d} {processed * args.augment_factor:6d} {len(all_cat_samples):7d}  "
          f"{len(cat_splits['train']):6d} {len(cat_splits['val']):5d} {len(cat_splits['test']):5d}")

    print(f"\nTRICK DATASET ({len(fig_trick_classes)} classes):")
    print(f"  Samples: {len(trick_samples)} ({len(trick_splits['train'])} train, {len(trick_splits['val'])} val)")

    print(f"\nTime: {elapsed:.0f}s")
    print(f"Failed: {failed}")
    disk = sum(f.stat().st_size for f in frames_dir.rglob("*.npy"))
    print(f"Disk: {disk / 1024 / 1024 / 1024:.1f} GB")
    print(f"\nManifests:")
    print(f"  {cat_manifest_path}")
    print(f"  {trick_manifest_path}")
    print(f"\nNext: python scripts/train_v5_modal.py")


if __name__ == "__main__":
    main()
