"""Context detection — classify trick environment from 2D keypoints.

Determines if a trick is performed on ground, wall, or bar/rail based on
keypoint motion patterns. This enables Layer 1 filtering in the FIG matching pipeline.

Detection strategies:
- BAR: Wrists stay at a fixed position while body rotates below/above (pendulum).
- WALL: Significant vertical displacement with wrists contacting a vertical surface
        (wrists at consistent horizontal position during approach/contact).
- GROUND: Default — no apparatus interaction detected.
"""

from __future__ import annotations

import numpy as np

from ml.trick_physics import TrickContext


# COCO-17 keypoint indices
L_WRIST, R_WRIST = 9, 10
L_HIP, R_HIP = 11, 12
L_ANKLE, R_ANKLE = 15, 16
L_SHOULDER, R_SHOULDER = 5, 6


def detect_context(
    keypoints: np.ndarray,
    start: int,
    end: int,
    fps: float = 30.0,
) -> TrickContext:
    """Detect trick context from 2D keypoint trajectory.

    Args:
        keypoints: (T, 17, 3) array — x, y, confidence per joint per frame.
        start, end: Frame range for this trick segment.
        fps: Video frame rate.

    Returns:
        TrickContext.GROUND, TrickContext.WALL, or TrickContext.BAR_OR_RAIL.
    """
    seg = keypoints[start:end + 1]
    T = len(seg)
    if T < 5:
        return TrickContext.GROUND

    # Extract wrist and body positions (only use high-confidence frames)
    l_wrist = seg[:, L_WRIST, :2]  # (T, 2) x,y
    r_wrist = seg[:, R_WRIST, :2]
    l_wrist_conf = seg[:, L_WRIST, 2]
    r_wrist_conf = seg[:, R_WRIST, 2]

    hip_center = (seg[:, L_HIP, :2] + seg[:, R_HIP, :2]) / 2
    shoulder_center = (seg[:, L_SHOULDER, :2] + seg[:, R_SHOULDER, :2]) / 2

    # Torso length for normalization
    torso_len = np.median(np.linalg.norm(shoulder_center - hip_center, axis=1))
    if torso_len < 1:
        torso_len = 100.0  # fallback

    # ── BAR DETECTION ──
    # On a bar, wrists are fixed at a point while the body swings.
    # Key signal: wrist displacement is small while hip displacement is large.
    if _detect_bar(l_wrist, r_wrist, l_wrist_conf, r_wrist_conf, hip_center, torso_len):
        return TrickContext.BAR_OR_RAIL

    # ── WALL DETECTION ──
    # On a wall, the athlete runs AT a wall, plants feet/hands on it, then flips off.
    # Key signal: significant upward movement before the trick, with horizontal
    # deceleration (body approaches wall then pushes off).
    if _detect_wall(seg, hip_center, torso_len, fps):
        return TrickContext.WALL

    return TrickContext.GROUND


def _detect_bar(
    l_wrist: np.ndarray,
    r_wrist: np.ndarray,
    l_conf: np.ndarray,
    r_conf: np.ndarray,
    hip_center: np.ndarray,
    torso_len: float,
) -> bool:
    """Detect bar/rail from fixed-wrist + swinging-body pattern."""
    T = len(l_wrist)
    if T < 10:
        return False

    # Use the more confident wrist
    l_good = l_conf > 0.5
    r_good = r_conf > 0.5

    for wrist, good in [(l_wrist, l_good), (r_wrist, r_good)]:
        if np.sum(good) < T * 0.3:
            continue

        wrist_good = wrist[good]
        hip_good = hip_center[good]

        # Wrist should be nearly stationary (< 0.3 torso lengths total range)
        wrist_range = np.max(wrist_good, axis=0) - np.min(wrist_good, axis=0)
        wrist_movement = np.linalg.norm(wrist_range) / torso_len

        # Hip should move significantly (> 1.0 torso lengths)
        hip_range = np.max(hip_good, axis=0) - np.min(hip_good, axis=0)
        hip_movement = np.linalg.norm(hip_range) / torso_len

        if wrist_movement < 0.5 and hip_movement > 1.5:
            return True

    return False


def _detect_wall(
    seg: np.ndarray,
    hip_center: np.ndarray,
    torso_len: float,
    fps: float,
) -> bool:
    """Detect wall from upward approach + contact pattern."""
    T = len(seg)
    if T < 10:
        return False

    # Wall tricks typically show:
    # 1. Rapid upward movement (Y decreases in image coords) in the first half
    # 2. Feet going higher than hips (feet on wall)
    # 3. Large vertical range relative to horizontal

    # Check vertical range vs horizontal range of hip center
    y_range = np.max(hip_center[:, 1]) - np.min(hip_center[:, 1])
    x_range = np.max(hip_center[:, 0]) - np.min(hip_center[:, 0])

    # Wall tricks have dominant vertical movement (portrait/vertical video)
    vert_dominant = y_range > x_range * 1.5

    # Check if ankles go above hips at some point (feet on wall)
    ankle_center = (seg[:, L_ANKLE, :2] + seg[:, R_ANKLE, :2]) / 2
    # In image coords, Y increases downward, so ankle_y < hip_y means ankle is above
    ankle_above_hip = np.sum(ankle_center[:, 1] < hip_center[:, 1] - torso_len * 0.3)
    feet_elevated = ankle_above_hip > T * 0.15

    # Check for significant upward velocity in the approach phase (first 40%)
    approach = hip_center[:int(T * 0.4)]
    if len(approach) > 3:
        y_vel = np.diff(approach[:, 1])  # negative = upward in image coords
        upward_speed = -np.mean(y_vel) / torso_len
        fast_upward = upward_speed > 0.05
    else:
        fast_upward = False

    # Need at least 2 of 3 signals
    signals = sum([vert_dominant, feet_elevated, fast_upward])
    return signals >= 2


def detect_context_from_translation(
    transl: np.ndarray,
    start: int,
    end: int,
) -> TrickContext:
    """Detect context from GVHMR 3D translation data.

    Simpler approach using 3D world coordinates:
    - WALL: Large height gain before trick (athlete runs up wall)
    - BAR: Translation shows pendulum pattern (fixed pivot point)
    - GROUND: Default
    """
    if transl is None:
        return TrickContext.GROUND

    seg = transl[start:end + 1]
    T = len(seg)
    if T < 5:
        return TrickContext.GROUND

    # Pre-trick height analysis
    pre_start = max(0, start - int(T * 0.5))
    pre_transl = transl[pre_start:start]

    if len(pre_transl) > 3:
        # Wall: significant height gain in approach (> 0.5m)
        height_gain = seg[0, 1] - pre_transl[0, 1]
        if height_gain > 0.5:
            return TrickContext.WALL

    # Bar: check if there's a fixed pivot point (translation shows circular motion)
    # In bar tricks, the COM traces an arc around the bar
    y_range = np.max(seg[:, 1]) - np.min(seg[:, 1])
    xz_range = np.max(np.linalg.norm(seg[:, [0, 2]], axis=1)) - np.min(np.linalg.norm(seg[:, [0, 2]], axis=1))

    # Bar swings have very large vertical range relative to horizontal
    if y_range > 1.5 and y_range > xz_range * 2.0:
        return TrickContext.BAR_OR_RAIL

    return TrickContext.GROUND
