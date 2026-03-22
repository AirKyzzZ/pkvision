#!/usr/bin/env python3
"""Bulk extract keypoints from labeled video clips for ST-GCN training.

Usage:
    python scripts/extract_poses.py --clips-dir data/clips/ --labels data/clips/labels.json --output data/clips/keypoints/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    parser = argparse.ArgumentParser(description="Extract poses from labeled clips")
    parser.add_argument("--clips-dir", required=True, help="Directory containing video clips")
    parser.add_argument("--labels", required=True, help="Path to labels.json")
    parser.add_argument("--output", required=True, help="Output directory for .npy keypoint files")
    args = parser.parse_args()

    clips_dir = Path(args.clips_dir)
    labels_path = Path(args.labels)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(labels_path) as f:
        labels = json.load(f)

    if not labels:
        print("No labels found. Add entries to labels.json first.")
        print("Use: python scripts/label.py to annotate clips.")
        sys.exit(0)

    from core.pose.detector import PoseDetector

    print(f"Loading YOLO pose detector...")
    detector = PoseDetector()

    processed = 0
    failed = 0

    for entry in labels:
        clip_file = entry["file"]
        trick_id = entry["trick_id"]
        start_ms = entry.get("start_ms", 0)
        end_ms = entry.get("end_ms", None)

        clip_path = clips_dir / clip_file
        if not clip_path.exists():
            print(f"  SKIP: {clip_file} (not found)")
            failed += 1
            continue

        print(f"  Processing: {clip_file} ({trick_id})...", end=" ")

        try:
            frames = list(detector.process_video(clip_path))

            # Filter to time range
            if end_ms is not None:
                frames = [f for f in frames if start_ms <= f.timestamp_ms <= end_ms]
            elif start_ms > 0:
                frames = [f for f in frames if f.timestamp_ms >= start_ms]

            if not frames:
                print("no frames detected")
                failed += 1
                continue

            # Build (3, T, 17) array: x, y, confidence
            T = len(frames)
            data = np.zeros((3, T, 17), dtype=np.float32)

            for t, frame in enumerate(frames):
                kps = np.array(frame.keypoints, dtype=np.float32)
                confs = np.array(frame.keypoint_confidences, dtype=np.float32)

                # Normalize to [0, 1]
                h, w = frame.frame_shape
                if w > 0:
                    kps[:, 0] /= w
                if h > 0:
                    kps[:, 1] /= h

                data[0, t, :] = kps[:, 0]
                data[1, t, :] = kps[:, 1]
                data[2, t, :] = confs

            # Save
            output_name = Path(clip_file).stem + ".npy"
            np.save(output_dir / output_name, data)
            print(f"OK ({T} frames)")
            processed += 1

        except Exception as e:
            print(f"ERROR: {e}")
            failed += 1

    print(f"\nDone. Processed: {processed}, Failed: {failed}")
    print(f"Keypoint files saved to: {output_dir}")


if __name__ == "__main__":
    main()
