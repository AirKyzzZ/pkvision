"""Shared video utilities — reading, YOLO tracking, box smoothing, clip saving.

Extracted from active_learn.py and inference_v5.py to avoid duplication.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np


@dataclass
class TrackingResult:
    """YOLO detection + keypoint tracking output for a full video."""
    boxes: np.ndarray          # (T, 4) bounding boxes [x1, y1, x2, y2]
    head_y: np.ndarray         # (T,) nose Y position
    hip_y: np.ndarray          # (T,) hip midpoint Y position
    body_angle: np.ndarray     # (T,) nose-hip angle
    ankle_y: np.ndarray        # (T,) ankle Y position
    fps: float
    total_frames: int


def read_video(path: str | Path) -> tuple[list[np.ndarray], float]:
    """Read all frames and FPS from a video file."""
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames, fps


def detect_and_track(frames: list[np.ndarray], yolo_model) -> TrackingResult:
    """Run YOLO-pose on all frames with IoU-based athlete tracking.

    Tracks the main athlete using center proximity + area + IoU continuity.
    Extracts keypoints for inversion detection.
    """
    T = len(frames)
    boxes = np.full((T, 4), np.nan)
    head_y = np.full(T, np.nan)
    hip_y = np.full(T, np.nan)
    body_angle = np.full(T, np.nan)
    ankle_y = np.full(T, np.nan)
    prev_box = None

    for i, frame in enumerate(frames):
        results = yolo_model(frame, conf=0.25, verbose=False)
        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            continue

        box_data = results[0].boxes.xyxy.cpu().numpy()
        areas = (box_data[:, 2] - box_data[:, 0]) * (box_data[:, 3] - box_data[:, 1])

        frame_h, frame_w = frame.shape[:2]
        frame_cx, frame_cy = frame_w / 2, frame_h / 2
        min_area = (frame_h * frame_w) * 0.005

        scores = np.zeros(len(box_data))
        for j, b in enumerate(box_data):
            if areas[j] < min_area:
                scores[j] = -999
                continue
            bcx = (b[0] + b[2]) / 2
            bcy = (b[1] + b[3]) / 2
            dist = np.sqrt((bcx - frame_cx) ** 2 + (bcy - frame_cy) ** 2)
            max_dist = np.sqrt(frame_cx ** 2 + frame_cy ** 2)
            center_score = 1 - (dist / max_dist)
            scores[j] = center_score * 0.4 + (areas[j] / areas.max()) * 0.3

            if prev_box is not None:
                ix1 = max(prev_box[0], b[0])
                iy1 = max(prev_box[1], b[1])
                ix2 = min(prev_box[2], b[2])
                iy2 = min(prev_box[3], b[3])
                inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                a1 = (prev_box[2] - prev_box[0]) * (prev_box[3] - prev_box[1])
                union = a1 + areas[j] - inter
                iou = inter / max(union, 1)
                scores[j] += iou * 0.3

        best = int(np.argmax(scores))
        boxes[i] = box_data[best]
        prev_box = box_data[best]

        # Extract keypoints
        if results[0].keypoints is not None and len(results[0].keypoints) > 0:
            kp = results[0].keypoints.xy.cpu().numpy()
            if best < len(kp):
                kp_person = kp[best]
                if kp_person.shape[0] >= 17:
                    # Head (nose, fallback to eyes)
                    if kp_person[0].sum() > 0:
                        head_y[i] = kp_person[0][1]
                    elif kp_person[1].sum() > 0 and kp_person[2].sum() > 0:
                        head_y[i] = (kp_person[1][1] + kp_person[2][1]) / 2

                    # Hip midpoint
                    if kp_person[11].sum() > 0 and kp_person[12].sum() > 0:
                        hip_y[i] = (kp_person[11][1] + kp_person[12][1]) / 2
                    elif kp_person[11].sum() > 0:
                        hip_y[i] = kp_person[11][1]
                    elif kp_person[12].sum() > 0:
                        hip_y[i] = kp_person[12][1]

                    # Body angle
                    if not np.isnan(head_y[i]) and not np.isnan(hip_y[i]):
                        hx = kp_person[0][0] if kp_person[0].sum() > 0 else (kp_person[1][0] + kp_person[2][0]) / 2
                        hip_x = (kp_person[11][0] + kp_person[12][0]) / 2 if kp_person[11].sum() > 0 else kp_person[12][0]
                        body_angle[i] = np.arctan2(head_y[i] - hip_y[i], hx - hip_x)

                    # Ankle Y
                    if kp_person[15].sum() > 0 and kp_person[16].sum() > 0:
                        ankle_y[i] = (kp_person[15][1] + kp_person[16][1]) / 2

    return TrackingResult(
        boxes=boxes, head_y=head_y, hip_y=hip_y,
        body_angle=body_angle, ankle_y=ankle_y,
        fps=0.0, total_frames=T,  # fps set by caller
    )


def smooth_boxes(boxes: np.ndarray) -> np.ndarray:
    """Interpolate missing detections and smooth bounding boxes."""
    T = len(boxes)
    smoothed = boxes.copy()
    valid = ~np.isnan(smoothed[:, 0])
    if np.sum(valid) < 2:
        return smoothed
    for col in range(4):
        valid_idx = np.where(valid)[0]
        smoothed[:, col] = np.interp(np.arange(T), valid_idx, smoothed[valid_idx, col])
    kernel = np.ones(7) / 7
    for col in range(4):
        smoothed[:, col] = np.convolve(smoothed[:, col], kernel, mode="same")
    return smoothed


def save_clip(
    frames: list[np.ndarray],
    fps: float,
    output_path: str | Path,
    codec: str = "mp4v",
) -> Path:
    """Write frames to an MP4 file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
    for frame in frames:
        writer.write(frame)
    writer.release()
    return output_path
