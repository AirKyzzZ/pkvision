"""Inversion-based trick segmentation.

Detects trick boundaries by finding frames where the athlete is inverted
(head below hips in image coordinates). Combines inversion signal with
angular velocity for robust detection.

Extracted from active_learn.py lines 277-372.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TrickSegment:
    """A detected trick within a video."""
    start_frame: int
    end_frame: int
    duration: float        # seconds
    peak_inversion: float  # max inversion signal in segment
    index: int = 0         # trick number in the run


def segment_tricks(
    head_y: np.ndarray,
    hip_y: np.ndarray,
    body_angle: np.ndarray,
    fps: float,
    inv_threshold: float = 0.15,
    min_duration: float = 0.3,
    max_duration: float = 4.0,
    merge_gap: float = 0.4,
    pad_seconds: float = 0.3,
) -> list[TrickSegment]:
    """Detect trick segments from YOLO keypoint signals.

    Args:
        head_y: (T,) nose Y position per frame (image coords, down = positive).
        hip_y: (T,) hip midpoint Y per frame.
        body_angle: (T,) nose-hip angle per frame.
        fps: Video frame rate.
        inv_threshold: Minimum inversion signal to consider active.
        min_duration: Shortest valid trick in seconds.
        max_duration: Longest valid trick in seconds.
        merge_gap: Merge segments closer than this (seconds).
        pad_seconds: Padding before/after each segment.

    Returns:
        List of TrickSegment with frame ranges.
    """
    T = len(head_y)
    smooth_k = max(int(fps * 0.1), 3)
    kernel = np.ones(smooth_k) / smooth_k

    # Signal 1: INVERSION — head below hips (head_y > hip_y in image coords)
    inversion = np.zeros(T)
    for i in range(T):
        if not np.isnan(head_y[i]) and not np.isnan(hip_y[i]):
            inversion[i] = max(0, head_y[i] - hip_y[i])

    if inversion.max() > 0:
        inv_positive = inversion[inversion > 0]
        inversion_norm = inversion / np.percentile(inv_positive, 90) if len(inv_positive) > 0 else inversion
    else:
        inversion_norm = inversion
    inversion_smooth = np.convolve(np.clip(inversion_norm, 0, 2), kernel, mode="same")

    # Signal 2: Angular velocity (rotation speed)
    angles_clean = np.copy(body_angle)
    valid_ang = ~np.isnan(angles_clean)
    if np.sum(valid_ang) >= 2:
        valid_idx = np.where(valid_ang)[0]
        angles_clean = np.interp(np.arange(T), valid_idx, angles_clean[valid_idx])
    else:
        angles_clean = np.zeros(T)

    ang_vel = np.abs(np.diff(angles_clean, prepend=angles_clean[0]))
    ang_vel = np.where(ang_vel > np.pi, 2 * np.pi - ang_vel, ang_vel)
    ang_vel_smooth = np.convolve(ang_vel, kernel, mode="same")
    if ang_vel_smooth.max() > 0:
        av_positive = ang_vel_smooth[ang_vel_smooth > 0]
        ang_vel_norm = ang_vel_smooth / np.percentile(av_positive, 90) if len(av_positive) > 0 else ang_vel_smooth
    else:
        ang_vel_norm = ang_vel_smooth

    # Combined score: inversion-weighted
    trick_score = inversion_smooth * 0.7 + ang_vel_norm * 0.3
    trick_score = np.convolve(trick_score, kernel, mode="same")

    # Active frames
    active = (inversion_smooth > inv_threshold) | (trick_score > 0.5)

    # Find contiguous regions
    segments = []
    in_segment = False
    start = 0
    for i in range(T):
        if active[i] and not in_segment:
            start = i
            in_segment = True
        elif not active[i] and in_segment:
            segments.append((start, i))
            in_segment = False
    if in_segment:
        segments.append((start, T - 1))

    # Pad
    pad_frames = int(fps * pad_seconds)
    segments = [(max(0, s - pad_frames), min(T - 1, e + pad_frames)) for s, e in segments]

    # Merge nearby
    merged = []
    for seg in segments:
        if merged and (seg[0] - merged[-1][1]) / fps < merge_gap:
            merged[-1] = (merged[-1][0], seg[1])
        else:
            merged.append(seg)
    segments = merged

    # Filter by duration and actual inversion content
    results = []
    for idx, (start, end) in enumerate(segments):
        dur = (end - start) / fps
        if dur < min_duration or dur > max_duration:
            continue
        seg_inv = inversion[start:end + 1]
        seg_ang = ang_vel_smooth[start:end + 1]
        has_inversion = np.any(seg_inv > 0)
        has_rotation = (
            np.max(seg_ang) > np.percentile(ang_vel_smooth[ang_vel_smooth > 0], 70)
            if np.any(ang_vel_smooth > 0) else False
        )
        if has_inversion or has_rotation:
            results.append(TrickSegment(
                start_frame=start,
                end_frame=end,
                duration=dur,
                peak_inversion=float(np.max(seg_inv)),
                index=len(results),
            ))

    return results
