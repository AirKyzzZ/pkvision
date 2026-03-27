"""MLP inference strategy — classifies tricks using the trained MLP model.

Implements the DetectionStrategy protocol. Loads a checkpoint from disk,
extracts camera-invariant features from frames, and returns class probabilities.

Usage:
    strategy = MLPStrategy(checkpoint_path="data/models/mlp_v1.pt")
    detection = strategy.evaluate(trick_config, analyzed_frames)
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from core.models import FrameAnalysis, TrickConfig, TrickDetection

logger = logging.getLogger(__name__)

# Target frames for interpolation (must match training)
TARGET_FRAMES = 64


class MLPStrategy:
    """Trick detection via trained MLP classifier.

    Loads a trained TrickMLP checkpoint, extracts features from input frames,
    and returns class probabilities. The model runs on CPU for inference
    (fast enough — ~1ms per classification).
    """

    def __init__(
        self,
        checkpoint_path: Path | str | None = None,
        min_confidence: float = 0.3,
    ):
        """
        Args:
            checkpoint_path: Path to .pt checkpoint. None = no model loaded.
            min_confidence: Minimum probability to report a detection.
        """
        self.min_confidence = min_confidence
        self._model = None
        self._class_names: list[str] = []
        self._class_to_idx: dict[str, int] = {}
        self._loaded = False

        # Feature extraction cache
        self._cached_frames_id: int | None = None
        self._cached_flat: np.ndarray | None = None

        if checkpoint_path is not None:
            self.load(checkpoint_path)

    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def class_names(self) -> list[str]:
        return list(self._class_names)

    def load(self, checkpoint_path: Path | str) -> None:
        """Load a trained checkpoint."""
        import torch
        from ml.mlp.model import TrickMLP

        path = Path(checkpoint_path)
        if not path.exists():
            logger.warning("MLP checkpoint not found: %s", path)
            return

        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        metadata = checkpoint["metadata"]

        self._class_names = metadata["class_names"]
        self._class_to_idx = {name: i for i, name in enumerate(self._class_names)}

        model = TrickMLP(
            num_classes=metadata["num_classes"],
            input_size=metadata.get("input_size", TARGET_FRAMES * 60),
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.cpu()
        model.train(False)

        self._model = model
        self._loaded = True
        logger.info(
            "Loaded MLP model: %d classes, val_acc=%.1f%%",
            metadata["num_classes"],
            metadata.get("best_val_acc", 0) * 100,
        )

    def evaluate(
        self,
        trick: TrickConfig,
        frames: list[FrameAnalysis],
    ) -> TrickDetection | None:
        """Evaluate whether the trick matches using MLP classification.

        Returns None if:
        - No model loaded
        - Trick not in model's class list
        - Probability below min_confidence
        """
        if not self._loaded or self._model is None:
            return None

        if trick.trick_id not in self._class_to_idx:
            return None

        if not frames:
            return None

        flat = self._extract_features(frames)
        if flat is None:
            return None

        probabilities = self._predict(flat)
        if probabilities is None:
            return None

        class_idx = self._class_to_idx[trick.trick_id]
        confidence = float(probabilities[class_idx])

        if confidence < self.min_confidence:
            return None

        return TrickDetection(
            trick_id=trick.trick_id,
            trick_name=trick.get_name(),
            confidence=confidence,
            start_frame=frames[0].frame_idx,
            end_frame=frames[-1].frame_idx,
            start_time_ms=frames[0].timestamp_ms,
            end_time_ms=frames[-1].timestamp_ms,
            strategy_used="mlp",
        )

    def predict_all(self, frames: list[FrameAnalysis]) -> dict[str, float]:
        """Get probabilities for all classes at once.

        Returns dict mapping class_name to probability.
        Useful for ensemble scoring.
        """
        if not self._loaded or self._model is None:
            return {}

        flat = self._extract_features(frames)
        if flat is None:
            return {}

        probabilities = self._predict(flat)
        if probabilities is None:
            return {}

        return {name: float(probabilities[i]) for i, name in enumerate(self._class_names)}

    def _predict(self, flat: np.ndarray) -> np.ndarray | None:
        """Run inference on a flat feature array. Returns probability array."""
        import torch

        with torch.no_grad():
            x = torch.tensor(flat, dtype=torch.float32).unsqueeze(0)  # (1, 3840)
            probs = self._model.predict_proba(x)  # (1, num_classes)
            return probs.squeeze(0).numpy()

    def _extract_features(self, frames: list[FrameAnalysis]) -> np.ndarray | None:
        """Extract, normalize, interpolate, and flatten features.

        Caches results for repeated calls with the same frames.
        """
        frames_id = id(frames)
        if self._cached_frames_id == frames_id and self._cached_flat is not None:
            return self._cached_flat

        from core.pose.features import extract_features_from_frames

        seq = extract_features_from_frames(frames)
        if seq.n_frames == 0:
            return None

        flat = seq.to_flat_array(target_frames=TARGET_FRAMES, normalize=True)

        # Replace NaN with 0 for model input
        np.nan_to_num(flat, copy=False, nan=0.0)

        self._cached_frames_id = frames_id
        self._cached_flat = flat
        return flat
