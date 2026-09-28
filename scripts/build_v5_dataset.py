#!/usr/bin/env python3
"""Build v5 training dataset from FIG-labeled parkourtheory clips for VideoMAE.

Reads the FIG→parkourtheory mapping and FIG trick definitions, extracts 16
evenly-spaced frames per clip, deduplicates clips that map to multiple FIG
rotation variants, and produces a stratified train/val/test split.

Inputs:
    data/fig_to_parkourtheory_map_v2.json  — FIG→parkourtheory name mapping
    data/fig_tricks_2025.json              — FIG trick definitions (categories, scores)
    data/parkourtheory_clips_cropped/      — 256×256 cropped clips (preferred)
    data/parkourtheory_clips/              — original clips (fallback)

Output:
    data/v5_training/frames/{fig_trick_slug}/{clip_slug}.npy
    data/v5_training/manifest.json

Usage:
    python scripts/build_v5_dataset.py
    python scripts/build_v5_dataset.py --num-frames 16 --seed 42 --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import av
import numpy as np

ROOT = Path(__file__).resolve().parent.parent

DEFAULT_MAP_PATH = ROOT / "data" / "fig_to_parkourtheory_map_v2.json"
DEFAULT_FIG_PATH = ROOT / "data" / "fig_tricks_2025.json"
DEFAULT_CROPPED_DIR = ROOT / "data" / "parkourtheory_clips_cropped"
DEFAULT_ORIGINAL_DIR = ROOT / "data" / "parkourtheory_clips"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "v5_training"


# ---------------------------------------------------------------------------
# Frame extraction
# ---------------------------------------------------------------------------

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
        # Pad by repeating last frame if video is too short
        while len(frames) < num_frames:
            frames.append(frames[-1])
        return np.stack(frames[:num_frames])
    except Exception as e:
        print(f"    ERROR reading {path.name}: {e}")
        return None


# ---------------------------------------------------------------------------
# Name → filename slug
# ---------------------------------------------------------------------------

def name_to_slug(name: str) -> str:
    """Convert parkourtheory trick name to filename slug.

    'Back Layout' → 'back_layout'
    'Cork Zero'   → 'cork_zero'
    """
    return name.lower().replace(" ", "_").replace("-", "_")


# ---------------------------------------------------------------------------
# Load FIG trick info indexed by name
# ---------------------------------------------------------------------------

def load_fig_trick_info(fig_path: Path) -> dict[str, dict]:
    """Return {trick_name: {category, score, idx}} from FIG JSON."""
    with open(fig_path) as f:
        fig = json.load(f)

    info: dict[str, dict] = {}
    idx = 0
    for cat_key, cat_data in fig["categories"].items():
        for trick in cat_data["tricks"]:
            info[trick["name"]] = {
                "category": cat_key,
                "score": trick["score"],
                "idx": idx,
            }
            idx += 1
    return info


# ---------------------------------------------------------------------------
# Resolve clip path (cropped > original)
# ---------------------------------------------------------------------------

def find_clip(slug: str, cropped_dir: Path, original_dir: Path) -> Path | None:
    """Return path to clip, preferring cropped version."""
    for ext in (".mp4", ".mov", ".avi", ".mkv", ".webm"):
        cropped = cropped_dir / f"{slug}{ext}"
        if cropped.exists():
            return cropped
        original = original_dir / f"{slug}{ext}"
        if original.exists():
            return original
    return None


# ---------------------------------------------------------------------------
# Stratified split
# ---------------------------------------------------------------------------

def stratified_split(
    samples: list[dict],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> dict[str, list[str]]:
    """Split sample files into train/val/test, stratified by category."""
    rng = np.random.default_rng(seed)

    # Group by category
    by_category: dict[str, list[str]] = defaultdict(list)
    for s in samples:
        by_category[s["category"]].append(s["file"])

    splits: dict[str, list[str]] = {"train": [], "val": [], "test": []}

    for _cat, files in sorted(by_category.items()):
        shuffled = list(files)
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_train = max(1, int(n * train_ratio))
        n_val = max(0, int(n * val_ratio))
        # Ensure at least 1 in train; distribute rest
        if n == 1:
            splits["train"].extend(shuffled)
            continue
        if n == 2:
            splits["train"].append(shuffled[0])
            splits["val"].append(shuffled[1])
            continue

        splits["train"].extend(shuffled[:n_train])
        splits["val"].extend(shuffled[n_train : n_train + n_val])
        splits["test"].extend(shuffled[n_train + n_val :])

    return splits


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build v5 VideoMAE training dataset from FIG-labeled parkourtheory clips."
    )
    parser.add_argument(
        "--map", type=Path, default=DEFAULT_MAP_PATH,
        help="Path to fig_to_parkourtheory_map_v2.json",
    )
    parser.add_argument(
        "--fig", type=Path, default=DEFAULT_FIG_PATH,
        help="Path to fig_tricks_2025.json",
    )
    parser.add_argument(
        "--cropped-dir", type=Path, default=DEFAULT_CROPPED_DIR,
        help="Directory with cropped clips (preferred)",
    )
    parser.add_argument(
        "--original-dir", type=Path, default=DEFAULT_ORIGINAL_DIR,
        help="Directory with original clips (fallback)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help="Output directory for v5_training",
    )
    parser.add_argument(
        "--num-frames", type=int, default=16,
        help="Number of frames to extract per clip (default: 16)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for train/val/test split",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Only print what would be done, don't extract frames",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. Load inputs
    # ------------------------------------------------------------------
    print("Loading FIG trick definitions...")
    fig_info = load_fig_trick_info(args.fig)
    print(f"  {len(fig_info)} FIG tricks loaded")

    print("Loading FIG→parkourtheory mapping...")
    with open(args.map) as f:
        raw_map = json.load(f)

    # raw_map is {category: {fig_name: pt_name, ...}, ...} with _description/_generated metadata
    mapping: dict[str, tuple[str, str]] = {}  # fig_name → (pt_name, category)
    for cat_key, cat_entries in raw_map.items():
        if cat_key.startswith("_"):
            continue
        for fig_name, pt_name in cat_entries.items():
            mapping[fig_name] = (pt_name, cat_key)

    print(f"  {len(mapping)} FIG→parkourtheory mappings")

    # ------------------------------------------------------------------
    # 2. Deduplicate: many FIG tricks → same parkourtheory clip
    #    Keep the LOWEST-rotation variant (most visually accurate label)
    # ------------------------------------------------------------------
    # Group by parkourtheory clip slug
    clip_to_fig_tricks: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for fig_name, (pt_name, cat_key) in mapping.items():
        slug = name_to_slug(pt_name)
        clip_to_fig_tricks[slug].append((fig_name, cat_key))

    # For each clip, pick the lowest-score FIG trick as the label
    # (lowest rotation = most visually representative)
    clip_assignments: dict[str, dict] = {}  # slug → {fig_name, category, variants}
    skipped_no_fig = 0

    for slug, fig_tricks in clip_to_fig_tricks.items():
        # Sort by FIG D-score ascending — lowest first
        scored = []
        for fig_name, cat_key in fig_tricks:
            info = fig_info.get(fig_name)
            if info:
                scored.append((info["score"], fig_name, cat_key))
            else:
                # Trick in mapping but not in FIG definitions — skip
                skipped_no_fig += 1

        if not scored:
            continue

        scored.sort(key=lambda x: x[0])
        _score, best_fig_name, best_cat = scored[0]
        variants = [name for _, name, _ in scored[1:]]

        clip_assignments[slug] = {
            "fig_name": best_fig_name,
            "category": best_cat,
            "variants": variants,
        }

    if skipped_no_fig:
        print(f"  {skipped_no_fig} mappings skipped (trick not in FIG definitions)")
    print(f"  {len(clip_assignments)} unique clips after deduplication")

    # ------------------------------------------------------------------
    # 3. Resolve clip paths
    # ------------------------------------------------------------------
    print("\nResolving clip paths...")
    resolved: dict[str, tuple[Path, dict]] = {}  # slug → (path, assignment)
    missing = []
    cropped_count = 0
    original_count = 0

    for slug, assignment in sorted(clip_assignments.items()):
        clip_path = find_clip(slug, args.cropped_dir, args.original_dir)
        if clip_path is None:
            missing.append((slug, assignment["fig_name"]))
            continue
        resolved[slug] = (clip_path, assignment)
        if str(args.cropped_dir) in str(clip_path):
            cropped_count += 1
        else:
            original_count += 1

    print(f"  Found: {len(resolved)} clips ({cropped_count} cropped, {original_count} original)")
    if missing:
        print(f"  Missing: {len(missing)} clips")
        for slug, fig_name in missing[:10]:
            print(f"    {slug}.mp4 (FIG: {fig_name})")
        if len(missing) > 10:
            print(f"    ... and {len(missing) - 10} more")

    if not resolved:
        print("\nERROR: No clips found. Check your data directories.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 4. Extract frames and build manifest
    # ------------------------------------------------------------------
    frames_dir = args.output_dir / "frames"
    samples: list[dict] = []
    class_info: dict[str, dict] = {}
    failed = 0
    total_bytes = 0
    t0 = time.time()

    if not args.dry_run:
        frames_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'[DRY RUN] ' if args.dry_run else ''}Extracting {args.num_frames} frames from {len(resolved)} clips...")

    for i, (slug, (clip_path, assignment)) in enumerate(sorted(resolved.items())):
        fig_name = assignment["fig_name"]
        category = assignment["category"]
        fig_slug = name_to_slug(fig_name)

        # Build class_info entry
        if fig_name not in class_info:
            info = fig_info.get(fig_name, {})
            class_info[fig_name] = {
                "category": category,
                "score": info.get("score", 0),
                "idx": info.get("idx", -1),
            }
            if assignment["variants"]:
                class_info[fig_name]["rotation_variants"] = assignment["variants"]

        rel_path = f"{category}/{slug}.npy"
        progress = f"[{i + 1}/{len(resolved)}]"

        if args.dry_run:
            print(f"  {progress} {clip_path.name} → {rel_path} (FIG: {fig_name})")
            samples.append({
                "file": rel_path,
                "class": fig_name,
                "category": category,
                "source_clip": clip_path.name,
            })
            continue

        print(f"  {progress} {clip_path.name}...", end=" ", flush=True)

        frames = read_video_frames(clip_path, num_frames=args.num_frames)
        if frames is None:
            print("FAILED")
            failed += 1
            continue

        # Save under category subdir
        cat_dir = frames_dir / category
        cat_dir.mkdir(parents=True, exist_ok=True)
        out_path = cat_dir / f"{slug}.npy"
        np.save(out_path, frames)
        total_bytes += out_path.stat().st_size

        samples.append({
            "file": rel_path,
            "class": fig_name,
            "category": category,
            "source_clip": clip_path.name,
        })
        print(f"OK {frames.shape} → {fig_name}")

    elapsed = time.time() - t0

    if not samples:
        print("\nERROR: No samples extracted.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 5. Build class list and splits
    # ------------------------------------------------------------------
    classes = sorted(set(s["class"] for s in samples))
    splits = stratified_split(samples, seed=args.seed)

    # ------------------------------------------------------------------
    # 6. Build and save manifest
    # ------------------------------------------------------------------
    manifest = {
        "classes": classes,
        "class_info": class_info,
        "samples": samples,
        "splits": splits,
        "config": {
            "num_frames": args.num_frames,
            "seed": args.seed,
            "num_classes": len(classes),
            "num_samples": len(samples),
        },
    }

    manifest_path = args.output_dir / "manifest.json"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    # ------------------------------------------------------------------
    # 7. Print stats
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print(f"{'[DRY RUN] ' if args.dry_run else ''}V5 Dataset Built")
    print(f"{'=' * 60}")

    # Per-category distribution
    cat_counts: dict[str, int] = defaultdict(int)
    for s in samples:
        cat_counts[s["category"]] += 1

    print(f"\nPer-category distribution:")
    print(f"  {'Category':<20s} {'Samples':>8s}")
    print(f"  {'-' * 20} {'-' * 8}")
    for cat, count in sorted(cat_counts.items(), key=lambda x: -x[1]):
        bar = "#" * min(count, 40)
        print(f"  {cat:<20s} {count:>8d}  {bar}")

    # Per-class sample counts
    class_counts: dict[str, int] = defaultdict(int)
    for s in samples:
        class_counts[s["class"]] += 1

    print(f"\nPer-class sample counts ({len(classes)} classes):")
    print(f"  {'Class':<35s} {'Cat':<14s} {'Count':>6s} {'Score':>6s}")
    print(f"  {'-' * 35} {'-' * 14} {'-' * 6} {'-' * 6}")
    for cls_name in sorted(classes):
        count = class_counts[cls_name]
        info = class_info.get(cls_name, {})
        cat = info.get("category", "?")
        score = info.get("score", 0)
        print(f"  {cls_name:<35s} {cat:<14s} {count:>6d} {score:>6.1f}")

    # Split sizes
    print(f"\nSplit sizes:")
    print(f"  Train: {len(splits['train']):>6d}")
    print(f"  Val:   {len(splits['val']):>6d}")
    print(f"  Test:  {len(splits['test']):>6d}")
    print(f"  Total: {len(samples):>6d}")

    # Stats
    if not args.dry_run:
        if total_bytes > 1024 * 1024 * 1024:
            size_str = f"{total_bytes / (1024**3):.2f} GB"
        elif total_bytes > 1024 * 1024:
            size_str = f"{total_bytes / (1024**2):.1f} MB"
        else:
            size_str = f"{total_bytes / 1024:.1f} KB"
        print(f"\nDisk usage: {size_str}")

    print(f"Time: {elapsed:.1f}s")
    if failed:
        print(f"Failed: {failed}")
    print(f"Manifest: {manifest_path}")
    print(f"Frames:   {frames_dir}/")
    print(f"\nNext: python scripts/finetune.py --dataset {args.output_dir}")


if __name__ == "__main__":
    main()
