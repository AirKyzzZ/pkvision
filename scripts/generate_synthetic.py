#!/usr/bin/env python3
"""Generate synthetic skeleton training data from real examples.

Takes your real YOLO keypoint extractions and generates thousands of
plausible variations by transforming the skeleton sequences.

From 3 real backflips → 500 synthetic backflip skeleton sequences.

Usage:
    # First extract real keypoints (if not done)
    python scripts/extract_poses.py --clips-dir data/clips --labels data/clips/labels.json --output data/clips/keypoints

    # Generate synthetic data
    python scripts/generate_synthetic.py
    python scripts/generate_synthetic.py --samples-per-trick 1000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic skeleton data")
    parser.add_argument("--keypoints-dir", default="data/clips/keypoints",
                        help="Directory with real .npy keypoint files")
    parser.add_argument("--labels", default="data/clips/labels.json",
                        help="Labels file")
    parser.add_argument("--output", default="data/synthetic",
                        help="Output directory for synthetic data")
    parser.add_argument("--samples-per-trick", type=int, default=500,
                        help="Synthetic samples to generate per trick")
    parser.add_argument("--frames", type=int, default=64,
                        help="Target frames per sample")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    keypoints_dir = Path(args.keypoints_dir)
    if not keypoints_dir.exists() or not list(keypoints_dir.glob("*.npy")):
        print(f"No keypoint files in {keypoints_dir}")
        print(f"\nExtract poses first:")
        print(f"  python scripts/extract_poses.py --clips-dir data/clips --labels data/clips/labels.json --output {keypoints_dir}")
        sys.exit(1)

    from ml.synthetic import generate_synthetic_dataset

    print()
    print("  PkVision — Synthetic Data Generator")
    print("  " + "=" * 42)
    print(f"  Real keypoints: {keypoints_dir}")
    print(f"  Samples/trick:  {args.samples_per_trick}")
    print(f"  Target frames:  {args.frames}")
    print()

    manifest = generate_synthetic_dataset(
        real_data_dir=str(keypoints_dir),
        labels_path=args.labels,
        output_dir=args.output,
        samples_per_trick=args.samples_per_trick,
        target_frames=args.frames,
        seed=args.seed,
    )

    print(f"\nTo train ST-GCN on synthetic data:")
    print(f"  python scripts/train.py --keypoints-dir {args.output} --labels {args.output}/manifest.json")


if __name__ == "__main__":
    main()
