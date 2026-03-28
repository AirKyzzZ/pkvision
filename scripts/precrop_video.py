#!/usr/bin/env python3
"""YOLO-based video pre-cropping for better GVHMR tracking.

Detects the athlete with YOLO, computes a stable bounding box across frames,
and outputs a cropped video where the athlete fills ~70% of the frame.
This dramatically improves GVHMR tracking on wide-angle competition footage.

Pipeline: Full video -> YOLO person detection -> Smoothed bbox -> Crop -> Cropped video

Usage:
    python scripts/precrop_video.py --input video.mp4 --output cropped.mp4
    # Then run GVHMR on cropped.mp4:
    cd C:/Users/pc/GVHMR && python tools/demo/demo.py --video cropped.mp4 -s
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

YOLO_MODEL_PATH = "C:/Users/pc/GVHMR/inputs/checkpoints/yolo/yolov8x.pt"


def detect_person_boxes(
    video_path: str,
    model_path: str = YOLO_MODEL_PATH,
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
    from ultralytics import YOLO

    model = YOLO(model_path)

    cap = cv2.VideoCapture(video_path)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    print(f"  Detecting persons in {total} frames ({W}x{H} @ {fps}fps)...")

    # Run YOLO on entire video (batched)
    results = model(video_path, stream=True, classes=[0], conf=conf_threshold, verbose=False)

    boxes = np.full((total, 4), np.nan)
    frame_idx = 0

    for result in results:
        if result.boxes is not None and len(result.boxes) > 0:
            # Select the largest person (by box area)
            box_data = result.boxes.xyxy.cpu().numpy()  # (N, 4)
            areas = (box_data[:, 2] - box_data[:, 0]) * (box_data[:, 3] - box_data[:, 1])
            best = np.argmax(areas)
            boxes[frame_idx] = box_data[best]

        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"    Frame {frame_idx}/{total}", end="\r")

    detected = np.sum(~np.isnan(boxes[:, 0]))
    print(f"  Detected person in {detected}/{total} frames ({detected/total*100:.0f}%)")

    return boxes, W, H, fps


def smooth_and_stabilize_boxes(
    boxes: np.ndarray,
    W: int,
    H: int,
    padding_factor: float = 1.8,
    min_crop_size: int = 400,
    smooth_window: int = 15,
) -> np.ndarray:
    """Smooth bounding boxes and compute stable crop regions.

    Produces a crop region that:
    - Tracks the athlete smoothly (no jitter)
    - Has consistent aspect ratio
    - Keeps the athlete centered with padding

    Args:
        boxes: (T, 4) raw detections [x1, y1, x2, y2]. NaN = no detection.
        W, H: Original video dimensions.
        padding_factor: How much padding around the person (1.8 = 80% extra).
        min_crop_size: Minimum crop dimension in pixels.
        smooth_window: Moving average window for smoothing.

    Returns:
        crops: (T, 4) crop regions [x1, y1, x2, y2] in original pixel coords.
    """
    T = len(boxes)

    # Interpolate missing detections
    valid = ~np.isnan(boxes[:, 0])
    if np.sum(valid) < 5:
        # Too few detections — return full frame
        return np.tile([0, 0, W, H], (T, 1)).astype(np.float32)

    for col in range(4):
        valid_idx = np.where(valid)[0]
        valid_vals = boxes[valid_idx, col]
        boxes[:, col] = np.interp(np.arange(T), valid_idx, valid_vals)

    # Convert to center + size
    cx = (boxes[:, 0] + boxes[:, 2]) / 2
    cy = (boxes[:, 1] + boxes[:, 3]) / 2
    bw = boxes[:, 2] - boxes[:, 0]
    bh = boxes[:, 3] - boxes[:, 1]
    size = np.maximum(bw, bh)  # Square crop based on larger dimension

    # Apply padding
    crop_size = np.maximum(size * padding_factor, min_crop_size)

    # Smooth everything to prevent jitter
    kernel = np.ones(smooth_window) / smooth_window
    cx = np.convolve(cx, kernel, mode="same")
    cy = np.convolve(cy, kernel, mode="same")
    crop_size = np.convolve(crop_size, kernel, mode="same")

    # Use a single stable crop size (median across frames)
    # This prevents zoom changes that confuse GVHMR
    stable_size = float(np.median(crop_size))
    # But allow 20% variation to handle approach/distance changes
    crop_size = np.clip(crop_size, stable_size * 0.8, stable_size * 1.2)

    # Compute crop regions
    x1 = cx - crop_size / 2
    y1 = cy - crop_size / 2
    x2 = cx + crop_size / 2
    y2 = cy + crop_size / 2

    # Clamp to frame boundaries (shift if needed, don't squeeze)
    for i in range(T):
        if x1[i] < 0:
            x2[i] -= x1[i]
            x1[i] = 0
        if y1[i] < 0:
            y2[i] -= y1[i]
            y1[i] = 0
        if x2[i] > W:
            x1[i] -= (x2[i] - W)
            x2[i] = W
        if y2[i] > H:
            y1[i] -= (y2[i] - H)
            y2[i] = H
        x1[i] = max(0, x1[i])
        y1[i] = max(0, y1[i])

    crops = np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)
    return crops


def crop_video(
    video_path: str,
    crops: np.ndarray,
    output_path: str,
    output_size: int = 720,
    fps: float = 30.0,
) -> str:
    """Crop and resize video frames according to crop regions.

    Args:
        video_path: Input video.
        crops: (T, 4) crop regions [x1, y1, x2, y2].
        output_path: Output video path.
        output_size: Output resolution (square: output_size x output_size).
        fps: Output FPS.
    """
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
            cropped = frame  # Fallback to full frame

        resized = cv2.resize(cropped, (output_size, output_size))
        writer.write(resized)

        if (i + 1) % 100 == 0:
            print(f"    Cropping: {i+1}/{T}", end="\r")

    print(f"    Cropping: {T}/{T}")
    writer.release()
    cap.release()
    return output_path


def main():
    parser = argparse.ArgumentParser(description="YOLO pre-crop for GVHMR")
    parser.add_argument("--input", required=True, help="Input video path")
    parser.add_argument("--output", default=None, help="Output cropped video path")
    parser.add_argument("--output-size", type=int, default=720,
                        help="Output square resolution (default: 720)")
    parser.add_argument("--padding", type=float, default=1.8,
                        help="Padding factor around person (1.8 = 80%% extra)")
    parser.add_argument("--smooth-window", type=int, default=15,
                        help="Smoothing window in frames")
    args = parser.parse_args()

    input_path = args.input
    if args.output is None:
        stem = Path(input_path).stem
        output_path = str(Path(input_path).parent / f"{stem}_cropped.mp4")
    else:
        output_path = args.output

    print(f"\nPkVision — YOLO Pre-Crop for GVHMR")
    print(f"{'=' * 50}")
    print(f"  Input:  {input_path}")
    print(f"  Output: {output_path}")

    # Step 1: Detect person in every frame
    boxes, W, H, fps = detect_person_boxes(input_path)

    # Step 2: Smooth and stabilize crop regions
    print(f"\n  Stabilizing crop regions...")
    crops = smooth_and_stabilize_boxes(
        boxes, W, H,
        padding_factor=args.padding,
        smooth_window=args.smooth_window,
    )

    # Print crop stats
    crop_w = crops[:, 2] - crops[:, 0]
    crop_h = crops[:, 3] - crops[:, 1]
    print(f"  Crop size: {np.median(crop_w):.0f}x{np.median(crop_h):.0f} "
          f"(original: {W}x{H})")
    print(f"  Athlete fills ~{np.median(boxes[~np.isnan(boxes[:,0]), 2] - boxes[~np.isnan(boxes[:,0]), 0]) / np.median(crop_w) * 100:.0f}%% of cropped frame")

    # Step 3: Crop video
    print(f"\n  Writing cropped video ({args.output_size}x{args.output_size})...")
    crop_video(input_path, crops, output_path, output_size=args.output_size, fps=fps)

    print(f"\n  Done! Now run GVHMR on the cropped video:")
    print(f"  cd C:/Users/pc/GVHMR && C:/Users/pc/gvhmr_env/Scripts/python.exe tools/demo/demo.py --video {output_path} -s")
    print()


if __name__ == "__main__":
    main()
