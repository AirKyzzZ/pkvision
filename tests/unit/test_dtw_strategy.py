"""Tests for DTW detection strategy."""

from __future__ import annotations

import math

import numpy as np
import pytest

from core.models import FrameAnalysis, TrickConfig, TrickPhase
from core.pose.features import FEATURES_PER_FRAME
from core.pose.angles import get_joint_angles
from core.pose.features import extract_features
from core.recognition.strategies.dtw import (
    DTWStrategy,
    _dtw_distance,
    _dtw_distance_numpy,
    _normalize_features,
    save_reference,
)


# ── Helpers ──────────────────────────────────────────────────────────


def _make_skeleton(bend: float = 0.0, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Create a standing skeleton with controllable knee bend."""
    rng = np.random.default_rng(seed)
    base = np.array([
        [320, 100], [310, 90], [330, 90], [300, 95], [340, 95],
        [280, 200], [360, 200], [260, 300], [380, 300],
        [240, 400], [400, 400], [290, 400], [350, 400],
        [290, 540], [350, 540], [290, 680], [350, 680],
    ], dtype=np.float32)
    # Bend knees
    base[13, 1] -= bend * 0.3
    base[14, 1] -= bend * 0.3
    # Small jitter
    base += rng.normal(0, 1, base.shape).astype(np.float32)
    confs = np.full(17, 0.95, dtype=np.float32)
    return base, confs


def _make_feature_sequence(
    n_frames: int = 30,
    movement: str = "squat",
    seed: int = 0,
) -> tuple[np.ndarray, list[FrameAnalysis]]:
    """Create a feature array and corresponding FrameAnalysis list.

    movement="squat": knees bend and extend (simulates a squat-like trick)
    movement="static": standing still
    movement="wave": arms move up and down
    """
    kps_seq, conf_seq, timestamps = [], [], []

    for i in range(n_frames):
        t = i / max(n_frames - 1, 1)
        if movement == "squat":
            bend = math.sin(t * math.pi) * 120
        elif movement == "wave":
            bend = math.sin(t * 2 * math.pi) * 60
        else:
            bend = 0.0

        kps, confs = _make_skeleton(bend=bend, seed=seed + i)
        kps_seq.append(kps)
        conf_seq.append(confs)
        timestamps.append(i * 33.33)

    # Extract features
    features = extract_features(kps_seq, conf_seq, timestamps)
    arr = features.to_array()

    # Build FrameAnalysis objects
    frame_analyses = []
    for i in range(n_frames):
        angles = get_joint_angles(kps_seq[i], conf_seq[i])
        frame_analyses.append(FrameAnalysis(
            frame_idx=i,
            timestamp_ms=timestamps[i],
            keypoints=kps_seq[i],
            keypoint_confidences=conf_seq[i],
            angles=angles,
            velocities=None,
        ))

    return arr, frame_analyses


def _make_trick_config(trick_id: str = "test_trick") -> TrickConfig:
    return TrickConfig(
        trick_id=trick_id,
        category="flip",
        difficulty=3.0,
        phases=[TrickPhase(name="execution", duration_range_ms=(200, 1000))],
        names={"en": "Test Trick"},
    )


# ── Unit Tests: DTW Distance ────────────────────────────────────────


class TestDTWDistance:
    def test_identical_sequences_zero_distance(self):
        s = np.random.default_rng(0).random((20, 10)).astype(np.float64)
        d = _dtw_distance(s, s)
        assert d < 1e-6

    def test_different_sequences_positive_distance(self):
        rng = np.random.default_rng(0)
        s1 = rng.random((20, 10)).astype(np.float64)
        s2 = rng.random((20, 10)).astype(np.float64)
        d = _dtw_distance(s1, s2)
        assert d > 0

    def test_symmetric(self):
        rng = np.random.default_rng(1)
        s1 = rng.random((15, 5)).astype(np.float64)
        s2 = rng.random((20, 5)).astype(np.float64)
        d1 = _dtw_distance(s1, s2)
        d2 = _dtw_distance(s2, s1)
        assert abs(d1 - d2) < 1e-6

    def test_different_lengths(self):
        rng = np.random.default_rng(2)
        s1 = rng.random((10, 5)).astype(np.float64)
        s2 = rng.random((30, 5)).astype(np.float64)
        d = _dtw_distance(s1, s2)
        assert d > 0

    def test_window_constraint(self):
        rng = np.random.default_rng(3)
        s1 = rng.random((20, 5)).astype(np.float64)
        s2 = rng.random((20, 5)).astype(np.float64)
        d_full = _dtw_distance(s1, s2, window=None)
        d_narrow = _dtw_distance(s1, s2, window=3)
        # Window constraint can only increase or equal the distance
        assert d_narrow >= d_full - 1e-6


class TestDTWNumpyFallback:
    def test_matches_identical(self):
        s = np.random.default_rng(0).random((10, 5)).astype(np.float64)
        d = _dtw_distance_numpy(s, s)
        assert d < 1e-6

    def test_positive_for_different(self):
        rng = np.random.default_rng(0)
        s1 = rng.random((10, 5)).astype(np.float64)
        s2 = rng.random((10, 5)).astype(np.float64)
        d = _dtw_distance_numpy(s1, s2)
        assert d > 0

    def test_with_window(self):
        rng = np.random.default_rng(1)
        s1 = rng.random((10, 3)).astype(np.float64)
        s2 = rng.random((10, 3)).astype(np.float64)
        d = _dtw_distance_numpy(s1, s2, window=3)
        assert d > 0


# ── Unit Tests: Feature Normalization ────────────────────────────────


class TestNormalizeFeatures:
    def test_angles_normalized(self):
        arr = np.zeros((5, FEATURES_PER_FRAME), dtype=np.float32)
        arr[:, :9] = 90.0  # 90 degrees
        normed = _normalize_features(arr)
        assert np.allclose(normed[:, :9], 0.5)

    def test_velocities_clipped(self):
        arr = np.zeros((5, FEATURES_PER_FRAME), dtype=np.float32)
        arr[:, 9:18] = 2000.0  # exceeds clip range
        normed = _normalize_features(arr)
        assert np.allclose(normed[:, 9:18], 1.0)

    def test_nan_replaced_with_zero(self):
        arr = np.full((5, FEATURES_PER_FRAME), np.nan, dtype=np.float32)
        normed = _normalize_features(arr)
        assert not np.any(np.isnan(normed))
        assert np.all(normed == 0.0)

    def test_does_not_modify_input(self):
        arr = np.ones((3, FEATURES_PER_FRAME), dtype=np.float32) * 90.0
        original = arr.copy()
        _normalize_features(arr)
        np.testing.assert_array_equal(arr, original)


# ── Unit Tests: DTWStrategy ──────────────────────────────────────────


class TestDTWStrategy:
    def test_no_references_returns_none(self, tmp_path):
        strategy = DTWStrategy(references_dir=tmp_path)
        trick = _make_trick_config()
        _, frames = _make_feature_sequence()
        assert strategy.evaluate(trick, frames) is None

    def test_empty_frames_returns_none(self, tmp_path):
        strategy = DTWStrategy(references_dir=tmp_path)
        trick = _make_trick_config()
        assert strategy.evaluate(trick, []) is None

    def test_detects_matching_trick(self, tmp_path):
        """Same movement as reference should get high confidence."""
        ref_arr, frames = _make_feature_sequence(movement="squat", seed=0)

        strategy = DTWStrategy(references_dir=tmp_path, min_confidence=0.1)
        strategy.add_references("test_trick", [ref_arr])

        trick = _make_trick_config("test_trick")
        detection = strategy.evaluate(trick, frames)

        assert detection is not None
        assert detection.trick_id == "test_trick"
        assert detection.confidence > 0.5
        assert detection.strategy_used == "dtw"

    def test_low_confidence_for_different_movement(self, tmp_path):
        """Different movement should get lower confidence than matching one."""
        squat_arr, _ = _make_feature_sequence(movement="squat", seed=0)
        _, wave_frames = _make_feature_sequence(movement="wave", seed=100)

        strategy = DTWStrategy(references_dir=tmp_path, min_confidence=0.0)
        strategy.add_references("squat_trick", [squat_arr])

        trick = _make_trick_config("squat_trick")

        # Evaluate squat reference against wave input
        detection = strategy.evaluate(trick, wave_frames)
        wave_conf = detection.confidence if detection else 0.0

        # Now evaluate against matching squat input
        _, squat_frames = _make_feature_sequence(movement="squat", seed=0)
        detection2 = strategy.evaluate(trick, squat_frames)
        squat_conf = detection2.confidence if detection2 else 0.0

        assert squat_conf > wave_conf, (
            f"Matching should score higher: squat={squat_conf:.3f} vs wave={wave_conf:.3f}"
        )

    def test_multiple_references_uses_best(self, tmp_path):
        """With multiple references, the best match (lowest distance) should be used."""
        ref1, _ = _make_feature_sequence(movement="squat", seed=0)
        ref2, _ = _make_feature_sequence(movement="squat", seed=10)
        ref3, _ = _make_feature_sequence(movement="wave", seed=20)  # dissimilar

        strategy = DTWStrategy(references_dir=tmp_path, min_confidence=0.0)
        strategy.add_references("test_trick", [ref1, ref2, ref3])

        trick = _make_trick_config("test_trick")
        _, squat_frames = _make_feature_sequence(movement="squat", seed=0)
        detection = strategy.evaluate(trick, squat_frames)

        assert detection is not None
        assert detection.confidence > 0.5

    def test_below_threshold_returns_none(self, tmp_path):
        """Confidence below min_confidence should return None."""
        ref_arr, _ = _make_feature_sequence(movement="squat", seed=0)
        _, diff_frames = _make_feature_sequence(movement="wave", seed=100)

        strategy = DTWStrategy(references_dir=tmp_path, min_confidence=0.99)
        strategy.add_references("test_trick", [ref_arr])

        trick = _make_trick_config("test_trick")
        detection = strategy.evaluate(trick, diff_frames)
        assert detection is None

    def test_detection_fields(self, tmp_path):
        """Verify all TrickDetection fields are populated correctly."""
        ref_arr, frames = _make_feature_sequence(n_frames=20, movement="squat", seed=0)

        strategy = DTWStrategy(references_dir=tmp_path, min_confidence=0.0)
        strategy.add_references("test_trick", [ref_arr])

        trick = _make_trick_config("test_trick")
        detection = strategy.evaluate(trick, frames)

        assert detection is not None
        assert detection.trick_id == "test_trick"
        assert detection.trick_name == "Test Trick"
        assert 0.0 <= detection.confidence <= 1.0
        assert detection.start_frame == frames[0].frame_idx
        assert detection.end_frame == frames[-1].frame_idx
        assert detection.start_time_ms == frames[0].timestamp_ms
        assert detection.end_time_ms == frames[-1].timestamp_ms
        assert detection.strategy_used == "dtw"

    def test_feature_caching(self, tmp_path):
        """Multiple evaluate() calls on the same frames should reuse cached features."""
        ref_arr, frames = _make_feature_sequence(movement="squat", seed=0)

        strategy = DTWStrategy(references_dir=tmp_path, min_confidence=0.0)
        strategy.add_references("trick_a", [ref_arr])
        strategy.add_references("trick_b", [ref_arr])

        trick_a = _make_trick_config("trick_a")
        trick_b = _make_trick_config("trick_b")

        # First call extracts features
        strategy.evaluate(trick_a, frames)
        assert strategy._cached_frames_id == id(frames)

        # Second call with same frames list should use cache
        cached_id_before = strategy._cached_frames_id
        strategy.evaluate(trick_b, frames)
        assert strategy._cached_frames_id == cached_id_before

    def test_loaded_tricks(self, tmp_path):
        strategy = DTWStrategy(references_dir=tmp_path)
        assert strategy.loaded_tricks == []

        ref_arr, _ = _make_feature_sequence()
        strategy.add_references("back_flip", [ref_arr])
        strategy.add_references("front_flip", [ref_arr])

        assert sorted(strategy.loaded_tricks) == ["back_flip", "front_flip"]


# ── Tests: Loading from Disk ─────────────────────────────────────────


class TestDiskReferences:
    def test_load_from_npy_files(self, tmp_path):
        """Strategy should load .npy references from disk."""
        ref_arr, _ = _make_feature_sequence(movement="squat", seed=0)

        # Save references
        trick_dir = tmp_path / "test_trick"
        trick_dir.mkdir()
        np.save(trick_dir / "ref_001.npy", ref_arr)
        np.save(trick_dir / "ref_002.npy", ref_arr)

        strategy = DTWStrategy(references_dir=tmp_path)
        assert "test_trick" in strategy.loaded_tricks
        assert len(strategy._references["test_trick"]) == 2

    def test_skips_invalid_shapes(self, tmp_path):
        """Files with wrong shapes should be skipped, not crash."""
        trick_dir = tmp_path / "bad_trick"
        trick_dir.mkdir()
        # Wrong number of features
        np.save(trick_dir / "bad.npy", np.zeros((10, 30)))
        # Valid
        np.save(trick_dir / "good.npy", np.zeros((10, FEATURES_PER_FRAME)))

        strategy = DTWStrategy(references_dir=tmp_path)
        assert len(strategy._references.get("bad_trick", [])) == 1

    def test_missing_directory_no_crash(self, tmp_path):
        """Non-existent references directory should not crash."""
        strategy = DTWStrategy(references_dir=tmp_path / "nonexistent")
        assert strategy.loaded_tricks == []

    def test_empty_trick_directory_skipped(self, tmp_path):
        (tmp_path / "empty_trick").mkdir()
        strategy = DTWStrategy(references_dir=tmp_path)
        assert "empty_trick" not in strategy.loaded_tricks


# ── Tests: save_reference Utility ────────────────────────────────────


class TestSaveReference:
    def test_saves_and_loads(self, tmp_path):
        arr = np.random.default_rng(0).random((20, FEATURES_PER_FRAME)).astype(np.float32)
        path = save_reference(arr, "back_flip", references_dir=tmp_path)

        assert path.exists()
        loaded = np.load(path)
        np.testing.assert_array_equal(loaded, arr)

    def test_auto_naming(self, tmp_path):
        arr = np.zeros((10, FEATURES_PER_FRAME), dtype=np.float32)
        p1 = save_reference(arr, "trick_a", references_dir=tmp_path)
        p2 = save_reference(arr, "trick_a", references_dir=tmp_path)
        assert p1.name == "ref_001.npy"
        assert p2.name == "ref_002.npy"

    def test_custom_name(self, tmp_path):
        arr = np.zeros((10, FEATURES_PER_FRAME), dtype=np.float32)
        path = save_reference(arr, "trick_b", references_dir=tmp_path, name="my_clip")
        assert path.name == "my_clip.npy"

    def test_creates_directories(self, tmp_path):
        arr = np.zeros((10, FEATURES_PER_FRAME), dtype=np.float32)
        path = save_reference(arr, "new_trick", references_dir=tmp_path / "deep" / "refs")
        assert path.exists()


# ── Integration Test: End-to-End ─────────────────────────────────────


class TestEndToEnd:
    def test_save_load_evaluate_cycle(self, tmp_path):
        """Full workflow: extract features → save reference → load → evaluate."""
        # Create reference data
        ref_arr, frames = _make_feature_sequence(movement="squat", seed=42)

        # Save as reference
        save_reference(ref_arr, "my_trick", references_dir=tmp_path)

        # Load strategy from disk
        strategy = DTWStrategy(references_dir=tmp_path, min_confidence=0.1)
        assert "my_trick" in strategy.loaded_tricks

        # Evaluate with similar input
        trick = _make_trick_config("my_trick")
        detection = strategy.evaluate(trick, frames)

        assert detection is not None
        assert detection.trick_id == "my_trick"
        assert detection.confidence > 0.5

    def test_classify_multiple_tricks(self, tmp_path):
        """Should correctly rank the matching trick higher."""
        squat_arr, _ = _make_feature_sequence(movement="squat", seed=0)
        wave_arr, _ = _make_feature_sequence(movement="wave", seed=0)

        strategy = DTWStrategy(references_dir=tmp_path, min_confidence=0.0)
        strategy.add_references("squat_trick", [squat_arr])
        strategy.add_references("wave_trick", [wave_arr])

        # Input is a squat
        _, squat_frames = _make_feature_sequence(movement="squat", seed=0)

        squat_config = _make_trick_config("squat_trick")
        wave_config = _make_trick_config("wave_trick")

        det_squat = strategy.evaluate(squat_config, squat_frames)
        det_wave = strategy.evaluate(wave_config, squat_frames)

        squat_conf = det_squat.confidence if det_squat else 0.0
        wave_conf = det_wave.confidence if det_wave else 0.0

        assert squat_conf > wave_conf, (
            f"Squat input should match squat ref better: {squat_conf:.3f} vs {wave_conf:.3f}"
        )
