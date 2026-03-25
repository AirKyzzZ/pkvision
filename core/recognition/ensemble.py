"""Ensemble classifier — combines DTW and MLP strategy outputs.

Weighted combination of DTW similarity scores and MLP class probabilities.
With 3-4 reference clips per trick: ~85-90% accuracy on known tricks.

Usage:
    ensemble = EnsembleStrategy(
        dtw=DTWStrategy(references_dir="data/references"),
        mlp=MLPStrategy(checkpoint_path="data/models/mlp_v1.pt"),
    )
    detection = ensemble.evaluate(trick_config, frames)
"""

from __future__ import annotations

import logging

from core.models import FrameAnalysis, TrickConfig, TrickDetection

logger = logging.getLogger(__name__)


class EnsembleStrategy:
    """Combines DTW and MLP strategy outputs via weighted scoring.

    final_score = mlp_weight * mlp_probability + dtw_weight * dtw_confidence

    Falls back gracefully:
    - If only DTW has references for a trick: uses DTW alone
    - If only MLP is loaded: uses MLP alone
    - If neither available: returns None
    """

    def __init__(
        self,
        dtw=None,
        mlp=None,
        mlp_weight: float = 0.6,
        dtw_weight: float = 0.4,
        min_confidence: float = 0.3,
    ):
        """
        Args:
            dtw: DTWStrategy instance (or None if unavailable).
            mlp: MLPStrategy instance (or None if unavailable).
            mlp_weight: Weight for MLP probability in ensemble.
            dtw_weight: Weight for DTW confidence in ensemble.
            min_confidence: Minimum ensemble confidence to report a detection.
        """
        self.dtw = dtw
        self.mlp = mlp
        self.mlp_weight = mlp_weight
        self.dtw_weight = dtw_weight
        self.min_confidence = min_confidence

    def evaluate(
        self,
        trick: TrickConfig,
        frames: list[FrameAnalysis],
    ) -> TrickDetection | None:
        """Evaluate a trick using the DTW+MLP ensemble.

        Combines confidences from both strategies. If only one is available
        for this trick, uses that one alone (with full weight).
        """
        if not frames:
            return None

        dtw_conf = self._get_dtw_confidence(trick, frames)
        mlp_conf = self._get_mlp_confidence(trick, frames)

        # Combine scores
        confidence = self._combine(dtw_conf, mlp_conf)

        if confidence is None or confidence < self.min_confidence:
            return None

        # Determine which strategy contributed
        strategies_used = []
        if dtw_conf is not None:
            strategies_used.append("dtw")
        if mlp_conf is not None:
            strategies_used.append("mlp")
        strategy_label = "ensemble:" + "+".join(strategies_used)

        return TrickDetection(
            trick_id=trick.trick_id,
            trick_name=trick.get_name(),
            confidence=confidence,
            start_frame=frames[0].frame_idx,
            end_frame=frames[-1].frame_idx,
            start_time_ms=frames[0].timestamp_ms,
            end_time_ms=frames[-1].timestamp_ms,
            strategy_used=strategy_label,
        )

    def _combine(
        self,
        dtw_conf: float | None,
        mlp_conf: float | None,
    ) -> float | None:
        """Combine DTW and MLP confidences with configured weights.

        If only one source is available, uses it at full weight.
        """
        if dtw_conf is not None and mlp_conf is not None:
            # Both available: weighted combination
            total_weight = self.mlp_weight + self.dtw_weight
            score = (self.mlp_weight * mlp_conf + self.dtw_weight * dtw_conf) / total_weight
            return score

        if mlp_conf is not None:
            return mlp_conf

        if dtw_conf is not None:
            return dtw_conf

        return None

    def _get_dtw_confidence(
        self, trick: TrickConfig, frames: list[FrameAnalysis]
    ) -> float | None:
        if self.dtw is None:
            return None
        detection = self.dtw.evaluate(trick, frames)
        return detection.confidence if detection is not None else None

    def _get_mlp_confidence(
        self, trick: TrickConfig, frames: list[FrameAnalysis]
    ) -> float | None:
        if self.mlp is None:
            return None
        if not self.mlp.is_loaded():
            return None
        detection = self.mlp.evaluate(trick, frames)
        return detection.confidence if detection is not None else None
