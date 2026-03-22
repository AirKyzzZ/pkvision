"""Joint angle and velocity calculations from YOLO keypoints."""

from __future__ import annotations

import math

import numpy as np

from core.models import FrameAnalysis, FrameResult
from core.pose.constants import KEYPOINT_INDEX

# Joint angle definitions: each angle is computed from 3 keypoints (A-B-C)
# The angle is measured at point B (vertex).
ANGLE_DEFINITIONS: dict[str, tuple[str, str, str]] = {
    "left_knee": ("left_hip", "left_knee", "left_ankle"),
    "right_knee": ("right_hip", "right_knee", "right_ankle"),
    "left_hip": ("left_shoulder", "left_hip", "left_knee"),
    "right_hip": ("right_shoulder", "right_hip", "right_knee"),
    "left_elbow": ("left_shoulder", "left_elbow", "left_wrist"),
    "right_elbow": ("right_shoulder", "right_elbow", "right_wrist"),
    "left_shoulder": ("left_elbow", "left_shoulder", "left_hip"),
    "right_shoulder": ("right_elbow", "right_shoulder", "right_hip"),
}

# Composite angles computed from midpoints
COMPOSITE_ANGLES: dict[str, tuple[list[str], list[str], list[str]]] = {
    "spine": (
        ["left_shoulder", "right_shoulder"],  # midpoint A
        ["left_hip", "right_hip"],  # midpoint B (vertex)
        ["left_knee", "right_knee"],  # midpoint C
    ),
    "neck": (
        ["nose"],  # point A
        ["left_shoulder", "right_shoulder"],  # midpoint B (vertex)
        ["left_hip", "right_hip"],  # midpoint C
    ),
}

# Minimum confidence required for a keypoint to be used in angle calculation
MIN_KEYPOINT_CONFIDENCE = 0.3


def _compute_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Compute the angle at point B in the triangle A-B-C, in degrees."""
    ba = a - b
    bc = c - b

    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)

    return float(np.degrees(np.arccos(cos_angle)))


def _midpoint(keypoints: np.ndarray, names: list[str]) -> np.ndarray:
    """Compute the midpoint of multiple keypoints."""
    indices = [KEYPOINT_INDEX[n] for n in names]
    return keypoints[indices].mean(axis=0)


def _keypoints_confident(
    confidences: np.ndarray, keypoint_names: list[str], min_conf: float = MIN_KEYPOINT_CONFIDENCE
) -> bool:
    """Check if all named keypoints meet the minimum confidence threshold."""
    for name in keypoint_names:
        idx = KEYPOINT_INDEX[name]
        if confidences[idx] < min_conf:
            return False
    return True


def get_joint_angles(
    keypoints: np.ndarray,
    confidences: np.ndarray,
    min_confidence: float = MIN_KEYPOINT_CONFIDENCE,
) -> dict[str, float]:
    """Compute all joint angles from keypoints.

    Returns a dict mapping angle name → angle in degrees.
    Angles with low-confidence keypoints are set to NaN.
    For bilateral joints, returns the average of left and right as "knee", "hip", etc.
    """
    angles: dict[str, float] = {}

    # Standard 3-point angles
    for angle_name, (kp_a, kp_b, kp_c) in ANGLE_DEFINITIONS.items():
        if _keypoints_confident(confidences, [kp_a, kp_b, kp_c], min_confidence):
            a = keypoints[KEYPOINT_INDEX[kp_a]]
            b = keypoints[KEYPOINT_INDEX[kp_b]]
            c = keypoints[KEYPOINT_INDEX[kp_c]]
            angles[angle_name] = _compute_angle(a, b, c)
        else:
            angles[angle_name] = float("nan")

    # Composite (midpoint-based) angles
    for angle_name, (names_a, names_b, names_c) in COMPOSITE_ANGLES.items():
        all_names = names_a + names_b + names_c
        if _keypoints_confident(confidences, all_names, min_confidence):
            a = _midpoint(keypoints, names_a)
            b = _midpoint(keypoints, names_b)
            c = _midpoint(keypoints, names_c)
            angles[angle_name] = _compute_angle(a, b, c)
        else:
            angles[angle_name] = float("nan")

    # Averaged bilateral angles (for trick matching that says "knee" without left/right)
    for joint in ["knee", "hip", "elbow", "shoulder"]:
        left = angles.get(f"left_{joint}", float("nan"))
        right = angles.get(f"right_{joint}", float("nan"))
        if not math.isnan(left) and not math.isnan(right):
            angles[joint] = (left + right) / 2.0
        elif not math.isnan(left):
            angles[joint] = left
        elif not math.isnan(right):
            angles[joint] = right
        else:
            angles[joint] = float("nan")

    return angles


def get_joint_velocities(
    angles_sequence: list[dict[str, float]],
    timestamps_ms: list[float],
) -> list[dict[str, float]]:
    """Compute angular velocity (degrees/second) for each joint across frames.

    Returns a list of dicts (one per frame) mapping joint name → velocity.
    First frame gets velocity 0.0. NaN propagates.
    """
    if len(angles_sequence) < 2:
        return [{k: 0.0 for k in angles_sequence[0]}] if angles_sequence else []

    velocities: list[dict[str, float]] = []

    # First frame: zero velocity
    velocities.append({k: 0.0 for k in angles_sequence[0]})

    for i in range(1, len(angles_sequence)):
        dt_s = (timestamps_ms[i] - timestamps_ms[i - 1]) / 1000.0
        if dt_s <= 0:
            velocities.append({k: 0.0 for k in angles_sequence[i]})
            continue

        frame_vel: dict[str, float] = {}
        for joint in angles_sequence[i]:
            curr = angles_sequence[i].get(joint, float("nan"))
            prev = angles_sequence[i - 1].get(joint, float("nan"))
            if math.isnan(curr) or math.isnan(prev):
                frame_vel[joint] = float("nan")
            else:
                frame_vel[joint] = (curr - prev) / dt_s

        velocities.append(frame_vel)

    return velocities


def frame_result_to_analysis(
    frame: FrameResult,
    velocities: dict[str, float] | None = None,
) -> FrameAnalysis:
    """Convert a raw FrameResult into a FrameAnalysis with computed angles."""
    angles = get_joint_angles(frame.keypoints, frame.keypoint_confidences)

    return FrameAnalysis(
        frame_idx=frame.frame_idx,
        timestamp_ms=frame.timestamp_ms,
        keypoints=frame.keypoints,
        keypoint_confidences=frame.keypoint_confidences,
        angles=angles,
        velocities=velocities,
    )
