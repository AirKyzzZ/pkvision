"""Trick classifier — loads catalog and dispatches to detection strategies."""

from __future__ import annotations

import json
from pathlib import Path

from core.models import DetectionMethod, FrameAnalysis, TrickConfig, TrickDetection
from core.recognition.confidence import compute_multi_factor_confidence
from core.recognition.sequence import SequenceAnalyzer, TrickWindow
from core.recognition.strategies.angle import AngleThresholdStrategy
from core.recognition.strategies.temporal import TemporalModelStrategy


class TrickClassifier:
    """Loads the trick catalog and classifies frame sequences against all known tricks."""

    def __init__(
        self,
        catalog_dir: Path | str,
        model_path: Path | str | None = None,
        language: str = "en",
        confidence_threshold: float = 0.5,
    ):
        self.catalog_dir = Path(catalog_dir)
        self.language = language
        self.confidence_threshold = confidence_threshold

        # Load trick catalog
        self.tricks = self._load_catalog()

        # Initialize strategies
        self.angle_strategy = AngleThresholdStrategy()
        self.temporal_strategy = TemporalModelStrategy(model_path=model_path)

        # Sequence analyzer for windowing
        self.sequence_analyzer = SequenceAnalyzer()

    def _load_catalog(self) -> list[TrickConfig]:
        """Load all trick definitions from the catalog directory."""
        lang_dir = self.catalog_dir / self.language
        if not lang_dir.exists():
            lang_dir = self.catalog_dir / "en"

        if not lang_dir.exists():
            return []

        tricks: list[TrickConfig] = []
        for path in sorted(lang_dir.glob("*.json")):
            with open(path) as f:
                data = json.load(f)
            tricks.append(TrickConfig(**data))

        return tricks

    def classify(self, frames: list[FrameAnalysis]) -> list[TrickDetection]:
        """Classify all trick windows in a frame sequence.

        Returns all detected tricks with confidence scores.
        """
        if not frames:
            return []

        # Find trick windows
        windows = self.sequence_analyzer.find_trick_windows(frames)

        detections: list[TrickDetection] = []

        for window in windows:
            window_detections = self._classify_window(window, frames)
            detections.extend(window_detections)

        # Refine confidence scores
        for i, det in enumerate(detections):
            detections[i].confidence = compute_multi_factor_confidence(det, frames)

        # Filter by confidence threshold
        detections = [d for d in detections if d.confidence >= self.confidence_threshold]

        return detections

    def _classify_window(
        self, window: TrickWindow, all_frames: list[FrameAnalysis]
    ) -> list[TrickDetection]:
        """Try all tricks against a single trick window."""
        detections: list[TrickDetection] = []

        for trick in self.tricks:
            strategy = self._get_strategy(trick)
            if strategy is None:
                continue

            if isinstance(strategy, TemporalModelStrategy):
                detection = strategy.run_inference(trick, window.frames)
            else:
                detection = strategy.evaluate(trick, window.frames)

            if detection is not None:
                detections.append(detection)

        return detections

    def _get_strategy(self, trick: TrickConfig):
        """Select the appropriate detection strategy for a trick."""
        if trick.detection_method == DetectionMethod.TEMPORAL_MODEL:
            if self.temporal_strategy.is_loaded():
                return self.temporal_strategy
            # Fall back to angle threshold if no model loaded
            return self.angle_strategy

        return self.angle_strategy

    def get_trick_by_id(self, trick_id: str) -> TrickConfig | None:
        """Look up a trick by its ID."""
        for trick in self.tricks:
            if trick.trick_id == trick_id:
                return trick
        return None

    def list_tricks(self, lang: str | None = None) -> list[dict]:
        """List all tricks with localized names."""
        target_lang = lang or self.language
        return [
            {
                "trick_id": t.trick_id,
                "name": t.get_name(target_lang),
                "category": t.category,
                "difficulty": t.difficulty,
                "detection_method": t.detection_method.value,
                "tags": t.tags,
            }
            for t in self.tricks
        ]
