#!/usr/bin/env python3
"""Convert existing (3, T, 17) keypoint files into (n_frames, 60) feature references.

Reads labeled keypoints from data/clips/keypoints/, extracts camera-invariant features,
and saves them as DTW references in data/references/{trick_id}/.

Usage:
    python scripts/build_references.py
    python scripts/build_references.py --min-clips 2  # only tricks with >= 2 clips
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.models import FrameAnalysis
from core.pose.angles import get_joint_angles
from core.pose.features import extract_features, ANGLE_NAMES, LIMB_RATIO_NAMES
from core.recognition.segmentation import RunSegmenter
from core.recognition.strategies.dtw import save_reference

FRAME_SCALE = np.array([640.0, 480.0], dtype=np.float32)


def _skeleton_to_frames(data: np.ndarray) -> list[FrameAnalysis]:
    """Convert a (3, T, 17) array to FrameAnalysis objects for segmentation."""
    C, T, V = data.shape
    frames = []
    for t in range(T):
        kp = np.stack([data[0, t, :], data[1, t, :]], axis=-1) * FRAME_SCALE
        conf = data[2, t, :]
        angles = get_joint_angles(kp.astype(np.float32), conf.astype(np.float32))
        frames.append(FrameAnalysis(
            frame_idx=t, timestamp_ms=t * 33.33,
            keypoints=kp.astype(np.float32),
            keypoint_confidences=conf.astype(np.float32),
            angles=angles,
        ))
    return frames


def keypoints_3tv_to_features(
    data: np.ndarray,
    segment: bool = True,
) -> np.ndarray | None:
    """Convert a (3, T, 17) skeleton array to a (T, 60) feature array.

    If segment=True, extracts only the highest-intensity trick segment
    (trims setup/landing). Falls back to full clip if no segment detected.
    """
    if data.ndim != 3 or data.shape[0] != 3 or data.shape[2] != 17:
        return None

    C, T, V = data.shape
    if T < 4:
        return None

    frames = _skeleton_to_frames(data)

    # Segment: pick the highest-intensity active zone
    if segment and T > 30:
        segmenter = RunSegmenter(min_window_ms=150, max_window_ms=2500)
        segments = segmenter.segment(frames)
        if segments:
            best = max(segments, key=lambda s: s.peak_intensity)
            frames = frames[best.start_frame:best.end_frame + 1]

    # Extract features from (possibly trimmed) frames
    keypoints_seq = [np.asarray(f.keypoints, dtype=np.float32) for f in frames]
    confidences_seq = [np.asarray(f.keypoint_confidences, dtype=np.float32) for f in frames]
    timestamps = [f.timestamp_ms for f in frames]

    features = extract_features(keypoints_seq, confidences_seq, timestamps)
    if features.n_frames == 0:
        return None

    return features.to_array()


def main():
    parser = argparse.ArgumentParser(description="Build DTW feature references from labeled keypoints")
    parser.add_argument("--keypoints-dir", default="data/clips/keypoints",
                        help="Directory with (3, T, 17) .npy keypoint files")
    parser.add_argument("--relabels", default="data/training/relabels.json",
                        help="Path to relabels.json mapping filenames to trick IDs")
    parser.add_argument("--labels", default="data/clips/labels.json",
                        help="Path to labels.json")
    parser.add_argument("--output-dir", default="data/references",
                        help="Output directory for feature references")
    parser.add_argument("--min-clips", type=int, default=1,
                        help="Minimum clips per trick to include")
    parser.add_argument("--no-segment", action="store_true",
                        help="Use full clips instead of segmenting trick zones")
    args = parser.parse_args()

    kp_dir = Path(args.keypoints_dir)
    out_dir = Path(args.output_dir)
    kp_stems = {f.stem: f for f in kp_dir.glob("*.npy")}

    # Build trick -> stems mapping from relabels + labels
    trick_to_stems: dict[str, list[str]] = {}

    if Path(args.relabels).exists():
        relabels = json.load(open(args.relabels))
        for filename, tid in relabels.items():
            stem = Path(filename).stem
            if stem in kp_stems:
                trick_to_stems.setdefault(tid, []).append(stem)

    if Path(args.labels).exists():
        labels = json.load(open(args.labels))
        labels_list = labels.get("labels", labels) if isinstance(labels, dict) else labels
        for entry in labels_list:
            stem = Path(entry["file"]).stem
            tid = entry.get("trick_id")
            if tid and stem in kp_stems:
                trick_to_stems.setdefault(tid, []).append(stem)

    # Deduplicate
    for tid in trick_to_stems:
        trick_to_stems[tid] = sorted(set(trick_to_stems[tid]))

    # Filter by min_clips
    tricks = {tid: stems for tid, stems in trick_to_stems.items() if len(stems) >= args.min_clips}

    print(f"Found {sum(len(s) for s in tricks.values())} clips across {len(tricks)} tricks")
    print(f"(filtered: >= {args.min_clips} clips per trick)\n")

    total_saved = 0
    total_failed = 0

    for tid in sorted(tricks.keys()):
        stems = tricks[tid]
        saved = 0

        for stem in stems:
            data = np.load(kp_stems[stem]).astype(np.float32)
            features = keypoints_3tv_to_features(data, segment=not args.no_segment)

            if features is not None:
                save_reference(features, tid, references_dir=out_dir, name=stem)
                saved += 1
            else:
                print(f"  SKIP {stem} — invalid shape {data.shape}")
                total_failed += 1

        print(f"  {tid}: {saved}/{len(stems)} references saved")
        total_saved += saved

    print(f"\nDone: {total_saved} references saved, {total_failed} skipped")
    print(f"Output: {out_dir}/")

    # Summary
    print(f"\nTo train the MLP classifier:")
    print(f"  python -m ml.mlp.train --references-dir {out_dir} --device auto")
    print(f"\nTo analyze a video with the ensemble:")
    print(f"  python scripts/analyze.py --input video.mp4 --strategy ensemble")


if __name__ == "__main__":
    main()
