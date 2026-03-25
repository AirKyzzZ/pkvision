"""Tests for ensemble strategy (DTW + MLP combination)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from core.models import FrameAnalysis, TrickConfig, TrickDetection, TrickPhase
from core.pose.angles import get_joint_angles
from core.pose.features import extract_features
from core.recognition.ensemble import EnsembleStrategy


# ── Helpers ──────────────────────────────────────────────────────────


def _make_frames(n: int = 20, seed: int = 0) -> list[FrameAnalysis]:
    base = np.array([
        [320, 100], [310, 90], [330, 90], [300, 95], [340, 95],
        [280, 200], [360, 200], [260, 300], [380, 300],
        [240, 400], [400, 400], [290, 400], [350, 400],
        [290, 540], [350, 540], [290, 680], [350, 680],
    ], dtype=np.float32)
    confs = np.full(17, 0.95, dtype=np.float32)
    rng = np.random.default_rng(seed)

    frames = []
    for i in range(n):
        t = i / max(n - 1, 1)
        bend = math.sin(t * math.pi) * 120
        kps = base.copy()
        kps[13, 1] -= bend * 0.3
        kps[14, 1] -= bend * 0.3
        kps += rng.normal(0, 1, kps.shape).astype(np.float32)
        angles = get_joint_angles(kps, confs)
        frames.append(FrameAnalysis(
            frame_idx=i, timestamp_ms=i * 33.33,
            keypoints=kps, keypoint_confidences=confs, angles=angles,
        ))
    return frames


def _make_trick(trick_id: str = "test") -> TrickConfig:
    return TrickConfig(
        trick_id=trick_id, category="flip", difficulty=3.0,
        phases=[TrickPhase(name="exec", duration_range_ms=(200, 1000))],
        names={"en": "Test"},
    )


class FakeDTW:
    """Fake DTW strategy that returns configurable confidences."""

    def __init__(self, confidences: dict[str, float]):
        self._confidences = confidences

    def evaluate(self, trick: TrickConfig, frames: list[FrameAnalysis]) -> TrickDetection | None:
        if trick.trick_id not in self._confidences:
            return None
        conf = self._confidences[trick.trick_id]
        return TrickDetection(
            trick_id=trick.trick_id, trick_name=trick.get_name(),
            confidence=conf, start_frame=0, end_frame=len(frames) - 1,
            start_time_ms=0, end_time_ms=1000, strategy_used="dtw",
        )


class FakeMLP:
    """Fake MLP strategy that returns configurable confidences."""

    def __init__(self, confidences: dict[str, float]):
        self._confidences = confidences
        self._loaded = True

    def is_loaded(self) -> bool:
        return self._loaded

    def evaluate(self, trick: TrickConfig, frames: list[FrameAnalysis]) -> TrickDetection | None:
        if trick.trick_id not in self._confidences:
            return None
        conf = self._confidences[trick.trick_id]
        return TrickDetection(
            trick_id=trick.trick_id, trick_name=trick.get_name(),
            confidence=conf, start_frame=0, end_frame=len(frames) - 1,
            start_time_ms=0, end_time_ms=1000, strategy_used="mlp",
        )


# ── Tests ────────────────────────────────────────────────────────────


class TestEnsembleStrategy:
    def test_weighted_combination(self):
        """Ensemble should weight DTW and MLP according to configured weights."""
        dtw = FakeDTW({"trick": 0.8})
        mlp = FakeMLP({"trick": 0.6})

        ensemble = EnsembleStrategy(
            dtw=dtw, mlp=mlp,
            mlp_weight=0.6, dtw_weight=0.4,
            min_confidence=0.0,
        )

        frames = _make_frames()
        trick = _make_trick("trick")
        detection = ensemble.evaluate(trick, frames)

        assert detection is not None
        # Expected: (0.6 * 0.6 + 0.4 * 0.8) / (0.6 + 0.4) = (0.36 + 0.32) / 1.0 = 0.68
        expected = (0.6 * 0.6 + 0.4 * 0.8) / 1.0
        assert abs(detection.confidence - expected) < 0.01
        assert "dtw" in detection.strategy_used
        assert "mlp" in detection.strategy_used

    def test_dtw_only_fallback(self):
        """If MLP is None, should use DTW alone."""
        dtw = FakeDTW({"trick": 0.75})
        ensemble = EnsembleStrategy(dtw=dtw, mlp=None, min_confidence=0.0)

        frames = _make_frames()
        trick = _make_trick("trick")
        detection = ensemble.evaluate(trick, frames)

        assert detection is not None
        assert detection.confidence == 0.75
        assert "dtw" in detection.strategy_used
        assert "mlp" not in detection.strategy_used

    def test_mlp_only_fallback(self):
        """If DTW has no references for this trick, should use MLP alone."""
        dtw = FakeDTW({})  # no references for "trick"
        mlp = FakeMLP({"trick": 0.9})
        ensemble = EnsembleStrategy(dtw=dtw, mlp=mlp, min_confidence=0.0)

        frames = _make_frames()
        trick = _make_trick("trick")
        detection = ensemble.evaluate(trick, frames)

        assert detection is not None
        assert detection.confidence == 0.9

    def test_neither_available(self):
        """If both strategies return None, ensemble returns None."""
        dtw = FakeDTW({})
        mlp = FakeMLP({})
        ensemble = EnsembleStrategy(dtw=dtw, mlp=mlp, min_confidence=0.0)

        frames = _make_frames()
        trick = _make_trick("trick")
        assert ensemble.evaluate(trick, frames) is None

    def test_both_none(self):
        """No strategies at all."""
        ensemble = EnsembleStrategy(dtw=None, mlp=None)
        frames = _make_frames()
        trick = _make_trick("trick")
        assert ensemble.evaluate(trick, frames) is None

    def test_below_threshold(self):
        """Low combined confidence should return None."""
        dtw = FakeDTW({"trick": 0.1})
        mlp = FakeMLP({"trick": 0.1})
        ensemble = EnsembleStrategy(dtw=dtw, mlp=mlp, min_confidence=0.5)

        frames = _make_frames()
        trick = _make_trick("trick")
        assert ensemble.evaluate(trick, frames) is None

    def test_empty_frames(self):
        dtw = FakeDTW({"trick": 0.8})
        mlp = FakeMLP({"trick": 0.8})
        ensemble = EnsembleStrategy(dtw=dtw, mlp=mlp)
        trick = _make_trick("trick")
        assert ensemble.evaluate(trick, []) is None

    def test_unloaded_mlp(self):
        """MLP that is not loaded should not contribute."""
        dtw = FakeDTW({"trick": 0.7})
        mlp = FakeMLP({"trick": 0.9})
        mlp._loaded = False

        ensemble = EnsembleStrategy(dtw=dtw, mlp=mlp, min_confidence=0.0)
        frames = _make_frames()
        trick = _make_trick("trick")
        detection = ensemble.evaluate(trick, frames)

        assert detection is not None
        assert detection.confidence == 0.7  # DTW only

    def test_multiple_tricks_ranked(self):
        """Ensemble should give different scores to different tricks."""
        dtw = FakeDTW({"back_flip": 0.9, "front_flip": 0.3})
        mlp = FakeMLP({"back_flip": 0.8, "front_flip": 0.4})
        ensemble = EnsembleStrategy(dtw=dtw, mlp=mlp, min_confidence=0.0)

        frames = _make_frames()
        det_back = ensemble.evaluate(_make_trick("back_flip"), frames)
        det_front = ensemble.evaluate(_make_trick("front_flip"), frames)

        assert det_back is not None
        assert det_front is not None
        assert det_back.confidence > det_front.confidence

    def test_custom_weights(self):
        """Custom weight ratio should be honored."""
        dtw = FakeDTW({"t": 1.0})
        mlp = FakeMLP({"t": 0.0})

        # DTW-heavy ensemble
        ensemble = EnsembleStrategy(dtw=dtw, mlp=mlp, mlp_weight=0.1, dtw_weight=0.9, min_confidence=0.0)
        frames = _make_frames()
        detection = ensemble.evaluate(_make_trick("t"), frames)

        assert detection is not None
        # (0.1 * 0.0 + 0.9 * 1.0) / 1.0 = 0.9
        assert abs(detection.confidence - 0.9) < 0.01
