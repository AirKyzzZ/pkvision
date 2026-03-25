"""TCN inference strategy — classifies tricks using the trained TCN model.

Implements the same interface as MLPStrategy for drop-in replacement.
Loads a checkpoint from disk, extracts temporal features from frames,
and returns class probabilities.

Usage:
    strategy = TCNStrategy(checkpoint_path="data/models/tcn_v1.pt")
    detection = strategy.evaluate(trick_config, analyzed_frames)
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from core.models import FrameAnalysis, TrickConfig, TrickDetection
from core.pose.features import FEATURES_PER_FRAME, extract_features_from_frames

logger = logging.getLogger(__name__)

# Target frames for interpolation (must match training)
TARGET_FRAMES = 64


class TCNStrategy:
    """TCN-based trick classification strategy.

    Follows the same interface as MLPStrategy for drop-in replacement.
    """

    def __init__(
        self,
        checkpoint_path: Path | str | None = None,
        min_confidence: float = 0.3,
        device: str = "auto",
    ):
        """
        Args:
            checkpoint_path: Path to .pt checkpoint. None = no model loaded.
            min_confidence: Minimum probability to report a detection.
            device: "auto" picks CUDA/MPS if available, otherwise CPU.
        """
        self.min_confidence = min_confidence
        self._device = self._resolve_device(device)
        self._model = None
        self._class_names: list[str] = []
        self._class_to_idx: dict[str, int] = {}
        self._loaded = False

        # Feature extraction cache
        self._cached_frames_id: int | None = None
        self._cached_temporal: np.ndarray | None = None

        if checkpoint_path is not None:
            self.load(checkpoint_path)

    @staticmethod
    def _resolve_device(device: str) -> str:
        import torch

        if device != "auto":
            return device
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def class_names(self) -> list[str]:
        return list(self._class_names)

    def load(self, checkpoint_path: Path | str) -> None:
        """Load a trained checkpoint."""
        import torch
        from ml.tcn.model import TrickTCN

        path = Path(checkpoint_path)
        if not path.exists():
            logger.warning("TCN checkpoint not found: %s", path)
            return

        checkpoint = torch.load(path, map_location=self._device, weights_only=False)
        metadata = checkpoint["metadata"]

        self._class_names = metadata["class_names"]
        self._class_to_idx = {name: i for i, name in enumerate(self._class_names)}

        model = TrickTCN(
            num_classes=metadata["num_classes"],
            n_features=metadata.get("n_features", FEATURES_PER_FRAME),
            hidden_channels=tuple(metadata.get("hidden_channels", (128, 128, 128, 128))),
            kernel_size=metadata.get("kernel_size", 3),
            dropout=metadata.get("dropout", 0.2),
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self._device)
        model.train(False)

        self._model = model
        self._loaded = True
        logger.info(
            "Loaded TCN model on %s: %d classes, val_acc=%.1f%%",
            self._device,
            metadata["num_classes"],
            metadata.get("best_val_acc", 0) * 100,
        )

    def evaluate(
        self,
        trick: TrickConfig,
        frames: list[FrameAnalysis],
    ) -> TrickDetection | None:
        """Evaluate whether the trick matches using TCN classification.

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

        temporal = self._extract_features(frames)
        if temporal is None:
            return None

        probabilities = self._predict(temporal)
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
            strategy_used="tcn",
        )

    def predict_all(self, frames: list[FrameAnalysis]) -> dict[str, float]:
        """Get probabilities for all classes at once.

        Returns dict mapping class_name to probability.
        """
        if not self._loaded or self._model is None:
            return {}

        temporal = self._extract_features(frames)
        if temporal is None:
            return {}

        probabilities = self._predict(temporal)
        if probabilities is None:
            return {}

        return {name: float(probabilities[i]) for i, name in enumerate(self._class_names)}

    def _predict(self, temporal: np.ndarray) -> np.ndarray | None:
        """Run inference on a temporal feature array. Returns probability array."""
        import torch

        with torch.no_grad():
            x = torch.tensor(temporal, dtype=torch.float32).unsqueeze(0)  # (1, 75, T)
            x = x.to(self._device)
            probs = self._model.predict_proba(x)  # (1, num_classes)
            return probs.squeeze(0).cpu().numpy()

    def _extract_features(self, frames: list[FrameAnalysis]) -> np.ndarray | None:
        """Extract, normalize, interpolate to temporal array (75, 64).

        Caches results for repeated calls with the same frames.
        """
        frames_id = id(frames)
        if self._cached_frames_id == frames_id and self._cached_temporal is not None:
            return self._cached_temporal

        seq = extract_features_from_frames(frames)
        if seq.n_frames == 0:
            return None

        temporal = seq.to_temporal_array(target_frames=TARGET_FRAMES, normalize=True)

        # Replace NaN with 0 for model input
        np.nan_to_num(temporal, copy=False, nan=0.0)

        self._cached_frames_id = frames_id
        self._cached_temporal = temporal
        return temporal
