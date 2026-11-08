"""Run the person detector before pose, so an empty frame stays empty.

rtmlib's ``Body.__call__`` passes an empty box list straight to the pose model, which then
poses one box covering the whole frame. That invents a confident skeleton for frames with
nobody detected (2026-11-04 person-selection audit).
"""
from __future__ import annotations

import numpy as np


def detected_people(body, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pose every detected person: ((N, 17, 2) keypoints, (N, 17) scores), N may be 0."""
    boxes = body.det_model(frame)
    if boxes is None or len(boxes) == 0:
        return np.zeros((0, 17, 2), np.float32), np.zeros((0, 17), np.float32)
    keypoints, scores = body.pose_model(frame, bboxes=boxes)
    return np.asarray(keypoints)[:, :17], np.asarray(scores)[:, :17]
