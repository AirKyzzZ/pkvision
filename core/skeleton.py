"""Skeleton-based physics extraction from YOLO keypoints.

Extracts rotation counts, twist amounts, and direction from 2D keypoint
trajectories. This is the core signal that VLMs can't see but skeletons reveal.

COCO-17 keypoints:
  0=nose, 1-2=eyes, 3-4=ears, 5-6=shoulders, 7-8=elbows,
  9-10=wrists, 11-12=hips, 13-14=knees, 15-16=ankles
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import find_peaks
from scipy.ndimage import uniform_filter1d


@dataclass
class TrickPhysics:
    """Physics properties extracted from keypoint trajectory."""
    flip_count: float       # number of complete rotations (0.5, 1, 2, etc.)
    twist_count: float      # number of twists (0, 0.5, 1, 1.5, etc.)
    direction: str          # forward, backward, side
    duration: float         # seconds
    max_height: float       # peak height relative to start (normalized)
    is_off_axis: bool       # cork/krok style off-axis rotation
    confidence: float       # 0-1, how reliable the extraction is


def extract_physics(
    keypoints: np.ndarray,
    confidences: np.ndarray | None = None,
    fps: float = 30.0,
    smooth_window: int = 5,
) -> TrickPhysics:
    """Extract trick physics from a keypoint sequence.

    Args:
        keypoints: (T, 17, 2) array of x,y coordinates per joint per frame.
        confidences: (T, 17) confidence scores. If None, all treated as valid.
        fps: Frame rate of the video.
        smooth_window: Smoothing kernel size for signals.

    Returns:
        TrickPhysics with estimated flip/twist/direction.
    """
    T = len(keypoints)
    duration = T / fps

    if confidences is None:
        confidences = np.ones((T, 17))

    # Extract key joint trajectories
    nose = keypoints[:, 0]          # (T, 2)
    l_shoulder = keypoints[:, 5]
    r_shoulder = keypoints[:, 6]
    l_hip = keypoints[:, 11]
    r_hip = keypoints[:, 12]
    l_ankle = keypoints[:, 15]
    r_ankle = keypoints[:, 16]

    # Confidence masks
    nose_valid = confidences[:, 0] > 0.3
    shoulder_valid = (confidences[:, 5] > 0.3) & (confidences[:, 6] > 0.3)
    hip_valid = (confidences[:, 11] > 0.3) & (confidences[:, 12] > 0.3)

    # -- MIDPOINTS --
    hip_center = (l_hip + r_hip) / 2
    shoulder_center = (l_shoulder + r_shoulder) / 2

    # -- FLIP COUNTING --
    # Compute body angle: angle of spine (shoulder→hip) relative to vertical
    # In image coords, Y increases downward
    spine_dy = hip_center[:, 1] - shoulder_center[:, 1]  # positive = normal (hips below shoulders)
    spine_dx = hip_center[:, 0] - shoulder_center[:, 0]
    body_angle = np.arctan2(spine_dy, spine_dx)  # radians

    # Smooth body angle
    valid_mask = shoulder_valid & hip_valid
    body_angle_smooth = _interpolate_and_smooth(body_angle, valid_mask, smooth_window)

    # Unwrap angle to track cumulative rotation
    body_angle_unwrapped = np.unwrap(body_angle_smooth)
    total_rotation = abs(body_angle_unwrapped[-1] - body_angle_unwrapped[0])
    flip_count_raw = total_rotation / (2 * np.pi)

    # Snap to nearest 0.5
    flip_count = round(flip_count_raw * 2) / 2

    # -- INVERSION DETECTION (backup flip counting) --
    # head_y > hip_y in image coords = inverted
    head_y = np.where(nose_valid, nose[:, 1], np.nan)
    hip_y_center = np.where(hip_valid, hip_center[:, 1], np.nan)

    inversion_signal = np.zeros(T)
    for i in range(T):
        if not np.isnan(head_y[i]) and not np.isnan(hip_y_center[i]):
            inversion_signal[i] = head_y[i] - hip_y_center[i]

    inv_smooth = uniform_filter1d(inversion_signal, size=max(smooth_window, 3))

    # Count inversion peaks (= individual flips)
    if np.any(inv_smooth > 0):
        threshold = np.percentile(inv_smooth[inv_smooth > 0], 50)
        peaks, _ = find_peaks(inv_smooth, height=threshold, distance=int(fps * 0.15))
        inversion_flips = len(peaks)
    else:
        inversion_flips = 0

    # Use the more reliable estimate
    if flip_count < 0.5 and inversion_flips > 0:
        flip_count = float(inversion_flips)
    elif abs(flip_count - inversion_flips) > 1:
        # Large disagreement — trust inversion peaks more
        flip_count = max(flip_count, float(inversion_flips))

    # -- TWIST COUNTING --
    # Shoulder line angle relative to hip line = twist
    shoulder_dx = r_shoulder[:, 0] - l_shoulder[:, 0]
    shoulder_dy = r_shoulder[:, 1] - l_shoulder[:, 1]
    shoulder_angle = np.arctan2(shoulder_dy, shoulder_dx)

    hip_dx = r_hip[:, 0] - l_hip[:, 0]
    hip_dy = r_hip[:, 1] - l_hip[:, 1]
    hip_angle = np.arctan2(hip_dy, hip_dx)

    # Relative twist = difference between shoulder and hip rotation
    twist_signal = shoulder_angle - hip_angle
    twist_smooth = _interpolate_and_smooth(twist_signal, shoulder_valid & hip_valid, smooth_window)
    twist_unwrapped = np.unwrap(twist_smooth)
    total_twist = abs(twist_unwrapped[-1] - twist_unwrapped[0])
    twist_count_raw = total_twist / (2 * np.pi)

    # Also measure absolute shoulder rotation (for b-twist style moves)
    shoulder_smooth = _interpolate_and_smooth(shoulder_angle, shoulder_valid, smooth_window)
    shoulder_unwrapped = np.unwrap(shoulder_smooth)
    abs_shoulder_rotation = abs(shoulder_unwrapped[-1] - shoulder_unwrapped[0]) / (2 * np.pi)

    twist_count = round(max(twist_count_raw, abs_shoulder_rotation * 0.5) * 2) / 2

    # -- DIRECTION --
    # Compare horizontal movement of hip center
    valid_hip_frames = np.where(hip_valid)[0]
    if len(valid_hip_frames) >= 4:
        start_x = hip_center[valid_hip_frames[:3], 0].mean()
        mid_idx = len(valid_hip_frames) // 2
        mid_x = hip_center[valid_hip_frames[mid_idx-1:mid_idx+2], 0].mean()

        # Check vertical vs horizontal dominance for sideflip detection
        start_y = hip_center[valid_hip_frames[:3], 1].mean()
        mid_y = hip_center[valid_hip_frames[mid_idx-1:mid_idx+2], 1].mean()

        dx = mid_x - start_x
        dy = mid_y - start_y

        # Sideflip: rotation axis is sagittal (front-back), movement is lateral
        # Check if shoulder line stays roughly perpendicular to movement
        shoulder_width_var = np.var(shoulder_dx[shoulder_valid])
        shoulder_width_mean = np.abs(np.mean(shoulder_dx[shoulder_valid]))

        if shoulder_width_var > shoulder_width_mean * 0.5 and flip_count >= 0.5:
            direction = "side"
        elif spine_dy[valid_hip_frames[0]] > 0 and body_angle_unwrapped[-1] < body_angle_unwrapped[0]:
            direction = "backward"
        else:
            direction = "forward"
    else:
        direction = "backward"  # default

    # -- OFF-AXIS DETECTION --
    # Off-axis (cork/krok) = significant twist during flip, body not fully vertical
    is_off_axis = flip_count >= 0.5 and twist_count >= 0.5 and twist_count < flip_count * 1.5

    # -- HEIGHT --
    ankle_y = np.where(
        (confidences[:, 15] > 0.3) & (confidences[:, 16] > 0.3),
        (l_ankle[:, 1] + r_ankle[:, 1]) / 2,
        np.nan,
    )
    valid_ankle = ~np.isnan(ankle_y)
    both_valid = shoulder_valid & hip_valid
    if np.sum(valid_ankle) > 3:
        torso_len = np.nanmean(np.abs(hip_center[both_valid, 1] - shoulder_center[both_valid, 1])) if np.any(both_valid) else 1.0
        max_height = (np.nanmax(ankle_y) - np.nanmin(ankle_y)) / max(torso_len, 1)
    else:
        max_height = 0.0

    # -- CONFIDENCE --
    # Based on how many valid keypoint frames we had
    valid_ratio = np.mean(valid_mask)
    confidence = min(valid_ratio * 1.5, 1.0)

    return TrickPhysics(
        flip_count=max(flip_count, 0),
        twist_count=max(twist_count, 0),
        direction=direction,
        duration=duration,
        max_height=max_height,
        is_off_axis=is_off_axis,
        confidence=confidence,
    )


def extract_keypoints_from_clip(
    video_path: str,
    yolo_model=None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Extract YOLO keypoints from a video clip.

    Returns:
        (keypoints, confidences, fps) where keypoints is (T, 17, 2)
        and confidences is (T, 17).
    """
    import cv2

    if yolo_model is None:
        from ultralytics import YOLO
        yolo_model = YOLO("yolo11n-pose.pt")

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    all_keypoints = []
    all_confidences = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = yolo_model(frame, conf=0.25, verbose=False)

        kp = np.zeros((17, 2))
        conf = np.zeros(17)

        if results and results[0].keypoints is not None and len(results[0].keypoints) > 0:
            # Take the largest detection (most likely the athlete)
            boxes = results[0].boxes.xyxy.cpu().numpy()
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            best = int(np.argmax(areas))

            kp_data = results[0].keypoints.xy.cpu().numpy()
            conf_data = results[0].keypoints.conf.cpu().numpy() if results[0].keypoints.conf is not None else None

            if best < len(kp_data) and kp_data[best].shape[0] >= 17:
                kp = kp_data[best][:17]
                if conf_data is not None and best < len(conf_data):
                    conf = conf_data[best][:17]
                else:
                    conf = np.where(kp.sum(axis=1) > 0, 1.0, 0.0)

        all_keypoints.append(kp)
        all_confidences.append(conf)

    cap.release()

    return np.array(all_keypoints), np.array(all_confidences), fps


def _interpolate_and_smooth(
    signal: np.ndarray,
    valid_mask: np.ndarray,
    window: int,
) -> np.ndarray:
    """Interpolate NaN/invalid values and smooth."""
    result = signal.copy()
    valid_idx = np.where(valid_mask)[0]

    if len(valid_idx) < 2:
        return np.zeros_like(signal)

    # Interpolate invalid frames
    result = np.interp(np.arange(len(signal)), valid_idx, result[valid_idx])

    # Smooth
    result = uniform_filter1d(result, size=window)

    return result
