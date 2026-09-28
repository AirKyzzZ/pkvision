#!/usr/bin/env python3
"""Batch crop all parkourtheory clips to 256×256 athlete-centered videos.

Uses YOLO person detection to find the athlete bounding box, smooths it,
and crops to a standardized square format for VideoMAE training.

Usage:
    python scripts/batch_crop.py
    python scripts/batch_crop.py --workers 8 --output-size 256
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
YOLO_MODEL_PATH = str(PROJECT_ROOT / "yolo11n-pose.pt")
DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "parkourtheory_clips"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "parkourtheory_clips_cropped"

# Tuned for short single-trick clips (2-5 seconds)
DEFAULT_OUTPUT_SIZE = 256
DEFAULT_PADDING = 1.4
DEFAULT_SMOOTH_WINDOW = 7
NO_DETECTION_THRESHOLD = 0.5  # Copy original if >50% frames have no person


def detect_person_boxes(
    video_path: str,
    model,
    conf_threshold: float = 0.5,
) -> tuple[np.ndarray, int, int, float]:
    """Detect the main athlete in every frame using YOLO.

    Returns:
        boxes: (T, 4) array of [x1, y1, x2, y2] in pixel coords.
               NaN for frames with no detection.
        width: Video width
        height: Video height
        fps: Video FPS
    """
    cap = cv2.VideoCapture(video_path)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if total == 0:
        return np.empty((0, 4)), W, H, fps

    results = model(video_path, stream=True, classes=[0], conf=conf_threshold, verbose=False)

    boxes = np.full((total, 4), np.nan)
    frame_idx = 0

    for result in results:
        if frame_idx >= total:
            break
        if result.boxes is not None and len(result.boxes) > 0:
            box_data = result.boxes.xyxy.cpu().numpy()
            areas = (box_data[:, 2] - box_data[:, 0]) * (box_data[:, 3] - box_data[:, 1])
            best = np.argmax(areas)
            boxes[frame_idx] = box_data[best]
        frame_idx += 1

    return boxes, W, H, fps


def smooth_and_stabilize_boxes(
    boxes: np.ndarray,
    W: int,
    H: int,
    padding_factor: float = DEFAULT_PADDING,
    min_crop_size: int = 200,
    smooth_window: int = DEFAULT_SMOOTH_WINDOW,
) -> np.ndarray:
    """Smooth bounding boxes and compute stable crop regions.

    Produces a square crop region that tracks the athlete smoothly
    with consistent padding. Tuned for short single-trick clips.

    Args:
        boxes: (T, 4) raw detections [x1, y1, x2, y2]. NaN = no detection.
        W, H: Original video dimensions.
        padding_factor: How much padding around the person (1.4 = 40% extra).
        min_crop_size: Minimum crop dimension in pixels.
        smooth_window: Moving average window for smoothing.

    Returns:
        crops: (T, 4) crop regions [x1, y1, x2, y2] in original pixel coords.
    """
    T = len(boxes)

    valid = ~np.isnan(boxes[:, 0])
    if np.sum(valid) < 3:
        return np.tile([0, 0, W, H], (T, 1)).astype(np.float32)

    # Interpolate missing detections
    for col in range(4):
        valid_idx = np.where(valid)[0]
        valid_vals = boxes[valid_idx, col]
        boxes[:, col] = np.interp(np.arange(T), valid_idx, valid_vals)

    cx = (boxes[:, 0] + boxes[:, 2]) / 2
    cy = (boxes[:, 1] + boxes[:, 3]) / 2
    bw = boxes[:, 2] - boxes[:, 0]
    bh = boxes[:, 3] - boxes[:, 1]
    size = np.maximum(bw, bh)

    crop_size = np.maximum(size * padding_factor, min_crop_size)

    # Smooth — use smaller window for short clips, clamp to valid range
    win = min(smooth_window, max(T, 1))
    if win >= 3:
        kernel = np.ones(win) / win
        cx = np.convolve(cx, kernel, mode="same")
        cy = np.convolve(cy, kernel, mode="same")
        crop_size = np.convolve(crop_size, kernel, mode="same")

    # Stable crop size (median) with ±20% variation
    stable_size = float(np.median(crop_size))
    crop_size = np.clip(crop_size, stable_size * 0.8, stable_size * 1.2)

    x1 = cx - crop_size / 2
    y1 = cy - crop_size / 2
    x2 = cx + crop_size / 2
    y2 = cy + crop_size / 2

    # Clamp to frame boundaries (shift, don't squeeze)
    for i in range(T):
        if x1[i] < 0:
            x2[i] -= x1[i]
            x1[i] = 0
        if y1[i] < 0:
            y2[i] -= y1[i]
            y1[i] = 0
        if x2[i] > W:
            x1[i] -= x2[i] - W
            x2[i] = W
        if y2[i] > H:
            y1[i] -= y2[i] - H
            y2[i] = H
        x1[i] = max(0, x1[i])
        y1[i] = max(0, y1[i])

    return np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)


def crop_video(
    video_path: str,
    crops: np.ndarray,
    output_path: str,
    output_size: int = DEFAULT_OUTPUT_SIZE,
    fps: float = 30.0,
) -> str:
    """Crop and resize video frames according to crop regions."""
    cap = cv2.VideoCapture(video_path)
    T = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (output_size, output_size))

    for i in range(T):
        ret, frame = cap.read()
        if not ret:
            break

        x1, y1, x2, y2 = crops[i].astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2 = min(frame.shape[1], x2)
        y2 = min(frame.shape[0], y2)

        cropped = frame[y1:y2, x1:x2]
        if cropped.size == 0:
            cropped = frame

        resized = cv2.resize(cropped, (output_size, output_size))
        writer.write(resized)

    writer.release()
    cap.release()
    return output_path


def process_one_clip(
    clip_path: Path,
    output_path: Path,
    model,
    output_size: int,
    padding_factor: float,
    smooth_window: int,
) -> tuple[str, str]:
    """Process a single clip: detect, smooth, crop.

    Returns:
        (clip_name, status) where status is "cropped", "copied", or "failed: <reason>"
    """
    name = clip_path.name
    try:
        boxes, W, H, fps = detect_person_boxes(str(clip_path), model)

        if len(boxes) == 0:
            shutil.copy2(str(clip_path), str(output_path))
            return name, "copied"

        detected_ratio = np.sum(~np.isnan(boxes[:, 0])) / len(boxes)

        if detected_ratio < NO_DETECTION_THRESHOLD:
            shutil.copy2(str(clip_path), str(output_path))
            return name, "copied"

        crops = smooth_and_stabilize_boxes(
            boxes, W, H,
            padding_factor=padding_factor,
            smooth_window=smooth_window,
        )

        crop_video(str(clip_path), crops, str(output_path), output_size=output_size, fps=fps)
        return name, "cropped"

    except Exception as e:
        return name, f"failed: {e}"


def main():
    parser = argparse.ArgumentParser(
        description="Batch crop parkourtheory clips to athlete-centered 256×256 videos"
    )
    parser.add_argument(
        "--input-dir", type=Path, default=DEFAULT_INPUT_DIR,
        help=f"Input clips directory (default: {DEFAULT_INPUT_DIR})",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument("--output-size", type=int, default=DEFAULT_OUTPUT_SIZE)
    parser.add_argument("--padding", type=float, default=DEFAULT_PADDING)
    parser.add_argument("--smooth-window", type=int, default=DEFAULT_SMOOTH_WINDOW)
    parser.add_argument("--workers", type=int, default=1, help="Parallel workers (default: 1, YOLO not thread-safe)")
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir

    if not input_dir.exists():
        print(f"Error: input directory not found: {input_dir}")
        sys.exit(1)

    clips = sorted(
        p for p in input_dir.iterdir()
        if p.suffix.lower() in (".mp4", ".mov")
    )
    if not clips:
        print(f"No .mp4/.MOV clips found in {input_dir}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Filter out already-cropped clips (resume-friendly)
    todo = []
    skipped = 0
    for clip in clips:
        out = output_dir / clip.with_suffix(".mp4").name
        if out.exists():
            skipped += 1
        else:
            todo.append(clip)

    total = len(clips)
    remaining = len(todo)

    print(f"\nPkVision — Batch Crop")
    print(f"{'=' * 55}")
    print(f"  Input:       {input_dir}")
    print(f"  Output:      {output_dir}")
    print(f"  Output size: {args.output_size}×{args.output_size}")
    print(f"  Padding:     {args.padding}x")
    print(f"  Smoothing:   {args.smooth_window} frames")
    print(f"  Workers:     {args.workers}")
    print(f"  Clips:       {total} total, {skipped} already done, {remaining} to process")
    print(f"{'=' * 55}")

    if remaining == 0:
        print("\n  All clips already cropped. Nothing to do.")
        return

    # Load YOLO model once (shared across sequential processing)
    # Note: YOLO inference is GPU-bound, so we process sequentially through
    # the model but can parallelize the crop+write step via ThreadPoolExecutor
    from ultralytics import YOLO

    print(f"\n  Loading YOLO model: {YOLO_MODEL_PATH}")
    model = YOLO(YOLO_MODEL_PATH)

    cropped_count = 0
    copied_count = 0
    failed_count = 0
    failures = []
    start_time = time.time()

    # YOLO model isn't thread-safe for inference, so we process detection
    # sequentially. The crop_video step is I/O-bound and could be parallelized,
    # but for simplicity and correctness we process clips sequentially.
    # The ThreadPoolExecutor is used for the crop+write phase of each clip.
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {}
        for clip in todo:
            out = output_dir / clip.with_suffix(".mp4").name
            future = executor.submit(
                process_one_clip,
                clip, out, model,
                args.output_size, args.padding, args.smooth_window,
            )
            futures[future] = clip.name

        for i, future in enumerate(as_completed(futures), 1):
            name, status = future.result()

            if status == "cropped":
                cropped_count += 1
            elif status == "copied":
                copied_count += 1
            else:
                failed_count += 1
                failures.append((name, status))

            elapsed = time.time() - start_time
            per_clip = elapsed / i
            eta = per_clip * (remaining - i)
            eta_str = time.strftime("%H:%M:%S", time.gmtime(eta)) if eta > 0 else "done"

            print(
                f"  [{i}/{remaining}] {name[:45]:<45s} {status:<10s} "
                f"ETA: {eta_str}",
                end="\r" if i < remaining else "\n",
            )

    elapsed_total = time.time() - start_time

    print(f"\n{'=' * 55}")
    print(f"  Summary")
    print(f"{'=' * 55}")
    print(f"  Total clips:    {total}")
    print(f"  Cropped:        {cropped_count}")
    print(f"  Copied (no det):{copied_count}")
    print(f"  Skipped (exist):{skipped}")
    print(f"  Failed:         {failed_count}")
    print(f"  Time:           {time.strftime('%H:%M:%S', time.gmtime(elapsed_total))}")

    if failures:
        print(f"\n  Failures:")
        for name, status in failures:
            print(f"    {name}: {status}")

    print()


if __name__ == "__main__":
    main()
