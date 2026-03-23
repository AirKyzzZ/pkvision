"""YOLO11n-pose wrapper for pose estimation from video frames."""

from __future__ import annotations

from pathlib import Path
from typing import Generator

import cv2
import numpy as np

from core.models import FrameResult
from core.pose.constants import KEYPOINT_INDEX, KEYPOINT_NAMES


class PoseDetector:
    """Wraps YOLO11n-pose for skeleton extraction from video frames."""

    def __init__(self, model_path: str | None = None, confidence_threshold: float = 0.3):
        from ultralytics import YOLO

        self.model_path = model_path or "yolo26n-pose.pt"
        self.model = YOLO(self.model_path)
        self.confidence_threshold = confidence_threshold

    def process_frame(self, frame: np.ndarray, frame_idx: int = 0, timestamp_ms: float = 0.0) -> FrameResult | None:
        """Run pose estimation on a single frame.

        Returns FrameResult for the most confident person detected, or None if no person found.
        """
        results = self.model(frame, verbose=False)

        if not results or len(results) == 0:
            return None

        result = results[0]

        if result.keypoints is None or len(result.keypoints) == 0:
            return None

        # Get the most confident detection (first person)
        keypoints_data = result.keypoints.data  # (num_people, 17, 3) — x, y, conf

        if keypoints_data.shape[0] == 0:
            return None

        # Select person with highest average keypoint confidence
        avg_confs = keypoints_data[:, :, 2].mean(dim=1)
        best_idx = avg_confs.argmax().item()

        person_kps = keypoints_data[best_idx].cpu().numpy()  # (17, 3)

        keypoints = person_kps[:, :2]  # (17, 2) — x, y
        confidences = person_kps[:, 2]  # (17,) — confidence per keypoint

        h, w = frame.shape[:2]

        return FrameResult(
            frame_idx=frame_idx,
            timestamp_ms=timestamp_ms,
            keypoints=keypoints,
            keypoint_confidences=confidences,
            frame_shape=(h, w),
        )

    def process_video(self, video_path: Path | str) -> Generator[FrameResult, None, None]:
        """Process a video file frame by frame, yielding FrameResult for each frame with a detected person."""
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_idx = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                timestamp_ms = (frame_idx / fps) * 1000.0

                result = self.process_frame(frame, frame_idx=frame_idx, timestamp_ms=timestamp_ms)

                if result is not None:
                    yield result

                frame_idx += 1
        finally:
            cap.release()

    def process_webcam(self, max_frames: int | None = None) -> Generator[FrameResult, None, None]:
        """Process webcam feed, yielding FrameResult per frame."""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("Cannot open webcam")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_idx = 0

        try:
            while max_frames is None or frame_idx < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break

                timestamp_ms = (frame_idx / fps) * 1000.0

                result = self.process_frame(frame, frame_idx=frame_idx, timestamp_ms=timestamp_ms)

                if result is not None:
                    yield result

                frame_idx += 1
        finally:
            cap.release()
