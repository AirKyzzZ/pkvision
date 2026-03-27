#!/usr/bin/env python3
"""Analyze a full parkour run — detect and classify every trick.

Pipeline:
  1. YOLO pose detection → 17 keypoints per frame
  2. Feature extraction → 75-dim camera-invariant features
  3. Trick segmentation → find high-intensity trick zones
  4. MLP classification → identify each trick from 1,642 classes

Usage:
    python scripts/test_run.py data/mlp_testing/IMG_4738.MOV
    python scripts/test_run.py data/mlp_testing/IMG_4738.MOV --top-k 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))


def extract_keypoints_from_video(video_path: str) -> tuple[np.ndarray, float]:
    """Extract YOLO 17-keypoint poses and FPS from video.

    Returns: (keypoints (3, T, 17), fps)
    """
    from ultralytics import YOLO

    model = YOLO("yolo11n-pose.pt")

    # Get FPS from video
    import cv2
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    print(f"  Video: {total_frames} frames @ {fps:.1f} FPS ({total_frames/fps:.1f}s)")

    results = model(video_path, stream=True, verbose=False)

    frames = []
    for result in results:
        if result.keypoints is not None and len(result.keypoints) > 0:
            kp = result.keypoints[0]
            xy = kp.xyn[0].cpu().numpy()
            conf = kp.conf[0].cpu().numpy()
            frame = np.zeros((3, 17), dtype=np.float32)
            frame[0] = xy[:, 0]
            frame[1] = xy[:, 1]
            frame[2] = conf
            frames.append(frame)
        else:
            frames.append(np.zeros((3, 17), dtype=np.float32))

    arr = np.stack(frames, axis=0).transpose(1, 0, 2)
    return arr.astype(np.float32), fps


def segment_tricks(kp_array: np.ndarray, fps: float, min_gap_s: float = 0.5) -> list[tuple[int, int]]:
    """Segment a run into individual trick zones using angular velocity spikes.

    Returns list of (start_frame, end_frame) tuples.
    """
    C, T, V = kp_array.shape

    # Compute center-of-mass vertical velocity (trick indicator)
    # kp_array[1, :, [5,6,11,12]] has shape (T, 4) — average across the 4 joints
    com_y = np.mean(kp_array[1, :, :][:, [5, 6, 11, 12]], axis=1)  # (T,)
    com_x = np.mean(kp_array[0, :, :][:, [5, 6, 11, 12]], axis=1)  # (T,)

    # Body angle (nose relative to COM)
    body_angle = np.arctan2(
        kp_array[1, :, 0] - com_y,
        kp_array[0, :, 0] - com_x,
    )

    # Angular velocity (rotation speed)
    angular_vel = np.zeros(T)
    angular_vel[1:] = np.abs(np.diff(body_angle))
    # Handle angle wrapping
    angular_vel = np.minimum(angular_vel, 2 * np.pi - angular_vel)

    # Smooth angular velocity
    kernel_size = max(3, int(fps * 0.1))  # 100ms window
    kernel = np.ones(kernel_size) / kernel_size
    smooth_vel = np.convolve(angular_vel, kernel, mode="same")

    # Vertical movement (jump detection)
    vert_vel = np.zeros(T)
    vert_vel[1:] = np.abs(np.diff(com_y))
    smooth_vert = np.convolve(vert_vel, kernel, mode="same")

    # Combined intensity (rotation + vertical movement)
    intensity = smooth_vel * 0.6 + smooth_vert * 0.4
    intensity = intensity / (intensity.max() + 1e-8)

    # Threshold: frames with intensity > mean + 0.5 * std are "active"
    threshold = np.mean(intensity) + 0.5 * np.std(intensity)
    active = intensity > threshold

    # Find contiguous active regions
    segments = []
    in_segment = False
    start = 0

    min_gap_frames = int(min_gap_s * fps)
    min_trick_frames = int(0.3 * fps)  # Minimum 300ms for a trick
    max_trick_frames = int(4.0 * fps)  # Maximum 4s for a single trick

    for i in range(T):
        if active[i] and not in_segment:
            start = i
            in_segment = True
        elif not active[i] and in_segment:
            # Check gap — if short gap, keep going
            gap_end = min(i + min_gap_frames, T)
            if np.any(active[i:gap_end]):
                continue
            end = i
            in_segment = False

            # Extend boundaries slightly (tricks start before peak intensity)
            pad = int(0.15 * fps)
            seg_start = max(0, start - pad)
            seg_end = min(T - 1, end + pad)

            duration = seg_end - seg_start
            if min_trick_frames <= duration <= max_trick_frames:
                segments.append((seg_start, seg_end))

    # Handle segment at end of video
    if in_segment:
        end = T - 1
        pad = int(0.15 * fps)
        seg_start = max(0, start - pad)
        duration = end - seg_start
        if min_trick_frames <= duration <= max_trick_frames:
            segments.append((seg_start, end))

    # Merge overlapping segments
    if segments:
        merged = [segments[0]]
        for seg in segments[1:]:
            if seg[0] <= merged[-1][1] + min_gap_frames:
                merged[-1] = (merged[-1][0], max(merged[-1][1], seg[1]))
            else:
                merged.append(seg)
        # Split segments that are too long
        final = []
        for start, end in merged:
            duration = end - start
            if duration > max_trick_frames:
                # Split into chunks
                mid = (start + end) // 2
                final.append((start, mid))
                final.append((mid, end))
            else:
                final.append((start, end))
        return final

    return segments


def classify_segment(
    kp_segment: np.ndarray,
    mlp_strategy,
    top_k: int = 5,
) -> list[tuple[str, float]]:
    """Classify a keypoint segment using the MLP.

    Returns list of (trick_name, confidence) tuples.
    """
    from scripts.build_references import keypoints_3tv_to_features
    from core.pose.features import _array_to_sequence, ANGLE_NAMES, LIMB_RATIO_NAMES

    # Convert to features
    features = keypoints_3tv_to_features(kp_segment, segment=False)
    if features is None:
        return []

    # Convert to FeatureSequence and flatten
    seq = _array_to_sequence(features, list(ANGLE_NAMES), list(LIMB_RATIO_NAMES))
    flat = seq.to_flat_array(target_frames=64, normalize=True)
    np.nan_to_num(flat, copy=False, nan=0.0)

    # Run MLP inference
    import torch
    with torch.no_grad():
        x = torch.tensor(flat, dtype=torch.float32).unsqueeze(0)
        probs = mlp_strategy._model.predict_proba(x).squeeze(0).numpy()

    # Get top-K predictions (excluding no_trick)
    class_names = mlp_strategy._class_names
    results = []
    for idx in np.argsort(probs)[::-1][:top_k + 1]:
        name = class_names[idx]
        if name == "no_trick":
            continue
        results.append((name, float(probs[idx])))
        if len(results) >= top_k:
            break

    return results


def main():
    parser = argparse.ArgumentParser(description="Analyze a parkour run")
    parser.add_argument("video", help="Path to video file")
    parser.add_argument("--model", default="data/models/mlp_v3_full.pt",
                        help="MLP model checkpoint")
    parser.add_argument("--top-k", type=int, default=3, help="Top-K predictions per segment")
    parser.add_argument("--min-confidence", type=float, default=0.05)
    args = parser.parse_args()

    video_path = args.video
    if not Path(video_path).exists():
        print(f"Video not found: {video_path}")
        sys.exit(1)

    print(f"\nPkVision — Run Analysis")
    print(f"{'=' * 60}")

    # Step 1: Load MLP model
    print(f"\n[1] Loading MLP model...")
    from ml.mlp.inference import MLPStrategy
    mlp = MLPStrategy(checkpoint_path=args.model, min_confidence=args.min_confidence)
    print(f"  Model: {len(mlp.class_names)} classes loaded")

    # Step 2: Extract keypoints
    print(f"\n[2] Extracting YOLO keypoints from video...")
    kp_array, fps = extract_keypoints_from_video(video_path)
    print(f"  Keypoints: {kp_array.shape} ({kp_array.shape[1]} frames)")

    # Step 3: Segment tricks
    print(f"\n[3] Segmenting tricks...")
    segments = segment_tricks(kp_array, fps)
    print(f"  Found {len(segments)} trick segments")

    # Step 4: Classify each segment
    print(f"\n[4] Classifying each segment...")
    print(f"{'─' * 60}")

    detections = []
    for i, (start, end) in enumerate(segments):
        t_start = start / fps
        t_end = end / fps
        duration = t_end - t_start
        n_frames = end - start

        # Extract segment keypoints
        segment_kp = kp_array[:, start:end + 1, :]

        # Classify
        predictions = classify_segment(segment_kp, mlp, top_k=args.top_k)

        if predictions:
            top_name, top_conf = predictions[0]
            # Format trick name nicely
            display_name = top_name.replace("_", " ").title()

            print(f"\n  Trick #{i+1} @ {t_start:.1f}s - {t_end:.1f}s ({duration:.1f}s, {n_frames} frames)")
            for rank, (name, conf) in enumerate(predictions):
                display = name.replace("_", " ").title()
                bar = "█" * int(conf * 30)
                marker = " ◄── BEST" if rank == 0 else ""
                print(f"    {rank+1}. {display:40s} {conf:5.1%} {bar}{marker}")

            detections.append({
                "trick_num": i + 1,
                "start_s": round(t_start, 2),
                "end_s": round(t_end, 2),
                "duration_s": round(duration, 2),
                "predictions": [{"name": n.replace("_", " ").title(), "trick_id": n, "confidence": round(c, 4)} for n, c in predictions],
            })
        else:
            print(f"\n  Trick #{i+1} @ {t_start:.1f}s - {t_end:.1f}s — no classification")

    # Summary
    print(f"\n{'=' * 60}")
    print(f"RUN SUMMARY")
    print(f"{'=' * 60}")
    if detections:
        print(f"\nDetected {len(detections)} tricks:\n")
        for det in detections:
            top = det["predictions"][0]
            print(f"  {det['trick_num']:2d}. {top['name']:40s} ({top['confidence']:.0%}) @ {det['start_s']:.1f}s")
    else:
        print("\nNo tricks detected.")

    print()


if __name__ == "__main__":
    main()
