#!/usr/bin/env python3
"""Train the ST-GCN model on extracted keypoint data.

Usage:
    python scripts/train.py --epochs 100 --batch-size 16
    python scripts/train.py --epochs 50 --device mps
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    parser = argparse.ArgumentParser(description="Train ST-GCN model for trick classification")
    parser.add_argument("--keypoints-dir", default="data/clips/keypoints", help="Directory with .npy keypoint files")
    parser.add_argument("--labels", default="data/clips/labels.json", help="Labels file")
    parser.add_argument("--output", default="data/models/stgcn_best.pt", help="Output model path")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--frames", type=int, default=64, help="Target frames per sample")
    parser.add_argument("--device", help="Force device (cpu/mps/cuda)")
    parser.add_argument("--val-split", type=float, default=0.2, help="Validation split ratio")
    args = parser.parse_args()

    keypoints_dir = Path(args.keypoints_dir)
    labels_path = Path(args.labels)

    if not labels_path.exists():
        print(f"Error: Labels file not found: {labels_path}")
        print("\nTo get started:")
        print("  1. Add video clips to data/clips/")
        print("  2. Label them: python scripts/label.py")
        print("  3. Extract poses: python scripts/extract_poses.py --clips-dir data/clips/ --labels data/clips/labels.json --output data/clips/keypoints/")
        print("  4. Train: python scripts/train.py")
        sys.exit(1)

    if not keypoints_dir.exists() or not list(keypoints_dir.glob("*.npy")):
        print(f"Error: No keypoint files found in {keypoints_dir}")
        print("\nRun pose extraction first:")
        print(f"  python scripts/extract_poses.py --clips-dir data/clips/ --labels {labels_path} --output {keypoints_dir}/")
        sys.exit(1)

    from ml.train import train_model

    history = train_model(
        keypoints_dir=keypoints_dir,
        labels_path=labels_path,
        output_path=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        target_frames=args.frames,
        val_split=args.val_split,
        device_override=args.device,
    )

    print(f"\nFinal train accuracy: {history['train_acc'][-1]:.1%}")
    print(f"Final val accuracy:   {history['val_acc'][-1]:.1%}")


if __name__ == "__main__":
    main()
