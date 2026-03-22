"""Temporal model detection strategy — ST-GCN inference for complex tricks."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from core.models import FrameAnalysis, TrickConfig, TrickDetection


class TemporalModelStrategy:
    """Detects tricks using a trained ST-GCN model on keypoint sequences.

    This strategy feeds a sequence of keypoints into the trained model
    and returns the classification result with confidence.
    """

    def __init__(self, model_path: Path | str | None = None, min_confidence: float = 0.4):
        self.min_confidence = min_confidence
        self.model = None
        self.trick_classes: list[str] = []

        if model_path is not None:
            self.load_model(model_path)

    def load_model(self, model_path: Path | str) -> None:
        """Load a trained ST-GCN model from disk."""
        model_path = Path(model_path)
        if not model_path.exists():
            return

        try:
            import torch

            checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
            self.trick_classes = checkpoint.get("trick_classes", [])

            from ml.stgcn.model import STGCN

            num_classes = len(self.trick_classes) if self.trick_classes else 10
            self.model = STGCN(
                num_classes=num_classes,
                in_channels=3,
                num_joints=17,
            )
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.eval()
        except Exception:
            self.model = None

    def is_loaded(self) -> bool:
        return self.model is not None

    def run_inference(
        self,
        trick: TrickConfig,
        frames: list[FrameAnalysis],
    ) -> TrickDetection | None:
        """Run inference using the trained temporal model."""
        if self.model is None:
            return None

        if len(frames) < 10:
            return None

        try:
            import torch
        except ImportError:
            return None

        # Prepare input: (1, 3, T, 17) — channels are x, y, confidence
        keypoint_sequence = self._prepare_input(frames)

        if keypoint_sequence is None:
            return None

        tensor_input = torch.FloatTensor(keypoint_sequence).unsqueeze(0)

        with torch.no_grad():
            logits = self.model(tensor_input)
            probs = torch.softmax(logits, dim=1)

        # Check if the requested trick is in our class list
        if trick.trick_id not in self.trick_classes:
            return None

        trick_idx = self.trick_classes.index(trick.trick_id)
        confidence = float(probs[0, trick_idx])

        if confidence < self.min_confidence:
            return None

        return TrickDetection(
            trick_id=trick.trick_id,
            trick_name=trick.get_name("en"),
            confidence=confidence,
            start_frame=frames[0].frame_idx,
            end_frame=frames[-1].frame_idx,
            start_time_ms=frames[0].timestamp_ms,
            end_time_ms=frames[-1].timestamp_ms,
            strategy_used="temporal_model",
            phase_confidences={
                f"class_{i}": float(probs[0, i])
                for i in range(min(5, probs.shape[1]))
            },
        )

    def _prepare_input(self, frames: list[FrameAnalysis]) -> np.ndarray | None:
        """Convert frames to ST-GCN input format: (3, T, 17)."""
        T = len(frames)
        data = np.zeros((3, T, 17), dtype=np.float32)

        for t, frame in enumerate(frames):
            if frame.keypoints is None:
                continue

            kps = np.array(frame.keypoints, dtype=np.float32)
            confs = np.array(frame.keypoint_confidences, dtype=np.float32)

            if kps.shape[0] != 17:
                continue

            # Normalize keypoint coordinates to [0, 1]
            if kps.max() > 1.0:
                h, w = 720, 1280
                kps[:, 0] /= w
                kps[:, 1] /= h

            data[0, t, :] = kps[:, 0]  # x
            data[1, t, :] = kps[:, 1]  # y
            data[2, t, :] = confs       # confidence

        return data
