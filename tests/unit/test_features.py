"""Tests for camera-invariant feature extraction pipeline."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from core.pose.constants import KEYPOINT_INDEX
from core.pose.features import (
    ANGLE_NAMES,
    FEATURES_PER_FRAME,
    LIMB_RATIO_NAMES,
    FeatureSequence,
    _compute_com,
    _compute_limb_ratios,
    _compute_relative_positions,
    _compute_torso_length,
    extract_features,
    extract_frame_features,
)
from core.pose.angles import get_joint_angles

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture
def standing_pose() -> tuple[np.ndarray, np.ndarray]:
    with open(FIXTURES_DIR / "sample_keypoints.json") as f:
        data = json.load(f)
    kps = np.array(data["standing_pose"]["keypoints"], dtype=np.float32)
    confs = np.array(data["standing_pose"]["confidences"], dtype=np.float32)
    return kps, confs


@pytest.fixture
def low_confidence_pose() -> tuple[np.ndarray, np.ndarray]:
    with open(FIXTURES_DIR / "sample_keypoints.json") as f:
        data = json.load(f)
    kps = np.array(data["low_confidence_pose"]["keypoints"], dtype=np.float32)
    confs = np.array(data["low_confidence_pose"]["confidences"], dtype=np.float32)
    return kps, confs


def _make_symmetric_standing(scale: float = 1.0, offset_x: float = 0.0, offset_y: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Create a synthetic standing skeleton with controllable scale and position.

    Returns (keypoints (17, 2), confidences (17,)).
    Scale multiplies all coordinates; offset shifts the entire skeleton.
    """
    # Symmetric standing person facing camera
    base_kps = np.array([
        [320, 100],   # 0  nose
        [310, 90],    # 1  left_eye
        [330, 90],    # 2  right_eye
        [300, 95],    # 3  left_ear
        [340, 95],    # 4  right_ear
        [280, 200],   # 5  left_shoulder
        [360, 200],   # 6  right_shoulder
        [260, 300],   # 7  left_elbow
        [380, 300],   # 8  right_elbow
        [240, 400],   # 9  left_wrist
        [400, 400],   # 10 right_wrist
        [290, 400],   # 11 left_hip
        [350, 400],   # 12 right_hip
        [290, 540],   # 13 left_knee
        [350, 540],   # 14 right_knee
        [290, 680],   # 15 left_ankle
        [350, 680],   # 16 right_ankle
    ], dtype=np.float32)

    # Apply scale around center, then offset
    center = base_kps.mean(axis=0)
    kps = (base_kps - center) * scale + center + np.array([offset_x, offset_y])

    confs = np.full(17, 0.95, dtype=np.float32)
    return kps, confs


def _make_sequence(
    n_frames: int = 10,
    speed_factor: float = 1.0,
    scale: float = 1.0,
) -> tuple[list[np.ndarray], list[np.ndarray], list[float]]:
    """Create a synthetic keypoint sequence simulating knee bend over time.

    The knees go from ~180° (straight) to ~90° (deep squat) and back.
    speed_factor < 1 = slower (more real time per frame).
    """
    kps_seq = []
    conf_seq = []
    timestamps = []

    base_kps, confs = _make_symmetric_standing(scale=scale)

    for i in range(n_frames):
        t = i / max(n_frames - 1, 1)
        # Move knees and ankles to simulate a squat (bend at knee + hip)
        bend = math.sin(t * math.pi) * 140 * scale  # pixels of vertical movement

        frame_kps = base_kps.copy()
        # Lower the ankles and knees to simulate bending
        for idx in [13, 14]:  # knees
            frame_kps[idx, 1] = base_kps[idx, 1] - bend * 0.3
        for idx in [15, 16]:  # ankles stay mostly in place
            frame_kps[idx, 1] = base_kps[idx, 1]

        kps_seq.append(frame_kps)
        conf_seq.append(confs.copy())
        # Frame interval: 33ms at speed_factor=1.0 (30fps)
        timestamps.append(i * 33.33 / speed_factor)

    return kps_seq, conf_seq, timestamps


# ── Unit Tests: Individual Feature Components ────────────────────────


class TestComputeCOM:
    def test_hip_midpoint(self, standing_pose):
        kps, confs = standing_pose
        com = _compute_com(kps, confs, 0.3)
        left_hip = kps[KEYPOINT_INDEX["left_hip"]]
        right_hip = kps[KEYPOINT_INDEX["right_hip"]]
        expected = (left_hip + right_hip) / 2.0
        np.testing.assert_allclose(com, expected)

    def test_fallback_on_low_confidence_hips(self):
        kps = np.ones((17, 2), dtype=np.float32) * 100.0
        confs = np.full(17, 0.9, dtype=np.float32)
        # Set hips to low confidence
        confs[KEYPOINT_INDEX["left_hip"]] = 0.1
        confs[KEYPOINT_INDEX["right_hip"]] = 0.1
        com = _compute_com(kps, confs, 0.3)
        # Falls back to mean of confident keypoints
        valid = confs >= 0.3
        expected = kps[valid].mean(axis=0)
        np.testing.assert_allclose(com, expected)


class TestComputeTorsoLength:
    def test_standing_pose(self, standing_pose):
        kps, confs = standing_pose
        length = _compute_torso_length(kps, confs, 0.3)
        assert not math.isnan(length)
        assert length > 0

    def test_scales_with_skeleton(self):
        kps1, confs1 = _make_symmetric_standing(scale=1.0)
        kps2, confs2 = _make_symmetric_standing(scale=2.0)
        len1 = _compute_torso_length(kps1, confs1, 0.3)
        len2 = _compute_torso_length(kps2, confs2, 0.3)
        assert abs(len2 / len1 - 2.0) < 0.01

    def test_nan_on_low_confidence(self, low_confidence_pose):
        kps, confs = low_confidence_pose
        length = _compute_torso_length(kps, confs, 0.3)
        assert math.isnan(length)


class TestRelativePositions:
    def test_com_is_at_origin(self, standing_pose):
        kps, confs = standing_pose
        com = _compute_com(kps, confs, 0.3)
        torso = _compute_torso_length(kps, confs, 0.3)
        rel = _compute_relative_positions(kps, confs, com, torso, 0.3)

        # Hip midpoint (COM) should be close to (0, 0) in relative coords
        left_hip_rel = rel[KEYPOINT_INDEX["left_hip"]]
        right_hip_rel = rel[KEYPOINT_INDEX["right_hip"]]
        hip_mid_rel = (left_hip_rel + right_hip_rel) / 2.0
        np.testing.assert_allclose(hip_mid_rel, [0, 0], atol=0.01)

    def test_nan_on_bad_torso(self):
        kps = np.ones((17, 2), dtype=np.float32) * 100.0
        confs = np.full(17, 0.9, dtype=np.float32)
        com = np.array([100.0, 100.0])
        rel = _compute_relative_positions(kps, confs, com, float("nan"), 0.3)
        assert np.all(np.isnan(rel))

    def test_low_confidence_keypoint_is_nan(self):
        kps, confs = _make_symmetric_standing()
        confs[0] = 0.1  # nose below threshold
        com = _compute_com(kps, confs, 0.3)
        torso = _compute_torso_length(kps, confs, 0.3)
        rel = _compute_relative_positions(kps, confs, com, torso, 0.3)
        assert np.all(np.isnan(rel[0]))  # nose should be NaN
        assert not np.any(np.isnan(rel[5]))  # left_shoulder should be valid


class TestLimbRatios:
    def test_all_ratios_computed(self, standing_pose):
        kps, confs = standing_pose
        torso = _compute_torso_length(kps, confs, 0.3)
        ratios = _compute_limb_ratios(kps, confs, torso, 0.3)
        assert ratios.shape == (8,)
        # All should be positive for a valid skeleton
        for i, name in enumerate(LIMB_RATIO_NAMES):
            assert not math.isnan(ratios[i]), f"{name} is NaN"
            assert ratios[i] > 0, f"{name} should be positive"

    def test_ratios_invariant_to_scale(self):
        """Limb ratios should be identical regardless of skeleton size."""
        kps1, confs1 = _make_symmetric_standing(scale=1.0)
        kps2, confs2 = _make_symmetric_standing(scale=3.0)
        t1 = _compute_torso_length(kps1, confs1, 0.3)
        t2 = _compute_torso_length(kps2, confs2, 0.3)
        r1 = _compute_limb_ratios(kps1, confs1, t1, 0.3)
        r2 = _compute_limb_ratios(kps2, confs2, t2, 0.3)
        np.testing.assert_allclose(r1, r2, atol=1e-4)

    def test_nan_on_bad_torso(self):
        kps, confs = _make_symmetric_standing()
        ratios = _compute_limb_ratios(kps, confs, float("nan"), 0.3)
        assert np.all(np.isnan(ratios))


# ── Unit Tests: Frame Feature Extraction ─────────────────────────────


class TestExtractFrameFeatures:
    def test_output_shapes(self, standing_pose):
        kps, confs = standing_pose
        angles = get_joint_angles(kps, confs)
        features = extract_frame_features(kps, confs, angles)

        assert features.joint_angles.shape == (9,)
        assert features.angular_velocities.shape == (9,)
        assert features.relative_positions.shape == (17, 2)
        assert features.limb_ratios.shape == (8,)
        assert features.angular_accelerations.shape == (9,)
        assert isinstance(features.body_tilt, float)
        assert isinstance(features.left_right_symmetry, float)

    def test_velocities_zero_when_none(self, standing_pose):
        kps, confs = standing_pose
        angles = get_joint_angles(kps, confs)
        features = extract_frame_features(kps, confs, angles, velocities=None)
        np.testing.assert_array_equal(features.angular_velocities, np.zeros(9))

    def test_low_confidence_produces_nans(self, low_confidence_pose):
        kps, confs = low_confidence_pose
        angles = get_joint_angles(kps, confs, min_confidence=0.3)
        features = extract_frame_features(kps, confs, angles, min_confidence=0.3)
        # All angles should be NaN
        assert np.all(np.isnan(features.joint_angles))
        # All relative positions should be NaN (torso can't be computed)
        assert np.all(np.isnan(features.relative_positions))


# ── Unit Tests: Full Sequence Extraction ─────────────────────────────


class TestExtractFeatures:
    def test_basic_sequence(self):
        kps_seq, conf_seq, timestamps = _make_sequence(n_frames=10)
        features = extract_features(kps_seq, conf_seq, timestamps)
        assert features.n_frames == 10
        assert features.n_features_per_frame == FEATURES_PER_FRAME

    def test_empty_input(self):
        features = extract_features([], [], [])
        assert features.n_frames == 0

    def test_single_frame(self):
        kps, confs = _make_symmetric_standing()
        features = extract_features([kps], [confs], [0.0])
        assert features.n_frames == 1
        # First frame velocities should be zero
        np.testing.assert_array_equal(features.frames[0].angular_velocities, np.zeros(9))

    def test_to_array_shape(self):
        kps_seq, conf_seq, timestamps = _make_sequence(n_frames=20)
        features = extract_features(kps_seq, conf_seq, timestamps)
        arr = features.to_array()
        assert arr.shape == (20, FEATURES_PER_FRAME)
        assert arr.dtype == np.float32

    def test_to_array_normalized(self):
        kps_seq, conf_seq, timestamps = _make_sequence(n_frames=10)
        features = extract_features(kps_seq, conf_seq, timestamps)
        arr = features.to_array(normalize=True)
        # Angles should be in [0, 1] (NaN excluded)
        angle_cols = arr[:, :9]
        valid = ~np.isnan(angle_cols)
        if valid.any():
            assert angle_cols[valid].min() >= -0.01
            assert angle_cols[valid].max() <= 1.01
        # Velocities should be in [-1, 1]
        vel_cols = arr[:, 9:18]
        valid_v = ~np.isnan(vel_cols)
        if valid_v.any():
            assert vel_cols[valid_v].min() >= -1.01
            assert vel_cols[valid_v].max() <= 1.01


# ── Unit Tests: Interpolation ────────────────────────────────────────


class TestInterpolation:
    def test_same_length_returns_self(self):
        kps_seq, conf_seq, timestamps = _make_sequence(n_frames=10)
        features = extract_features(kps_seq, conf_seq, timestamps)
        interpolated = features.interpolate(10)
        assert interpolated.n_frames == 10

    def test_upsample(self):
        kps_seq, conf_seq, timestamps = _make_sequence(n_frames=10)
        features = extract_features(kps_seq, conf_seq, timestamps)
        interpolated = features.interpolate(20)
        assert interpolated.n_frames == 20
        assert interpolated.n_features_per_frame == FEATURES_PER_FRAME

    def test_downsample(self):
        kps_seq, conf_seq, timestamps = _make_sequence(n_frames=20)
        features = extract_features(kps_seq, conf_seq, timestamps)
        interpolated = features.interpolate(10)
        assert interpolated.n_frames == 10

    def test_to_flat_array(self):
        kps_seq, conf_seq, timestamps = _make_sequence(n_frames=30)
        features = extract_features(kps_seq, conf_seq, timestamps)
        flat = features.to_flat_array(target_frames=64)
        assert flat.shape == (64 * FEATURES_PER_FRAME,)

    def test_empty_interpolation(self):
        features = FeatureSequence(frames=[])
        interpolated = features.interpolate(10)
        assert interpolated.n_frames == 0


# ── Invariance Tests ─────────────────────────────────────────────────


class TestHeightInvariance:
    """Features should be approximately the same for different athlete heights."""

    def test_joint_angles_invariant(self):
        """Joint angles are pure geometry — independent of scale."""
        kps1, confs1 = _make_symmetric_standing(scale=1.0)
        kps2, confs2 = _make_symmetric_standing(scale=2.5)
        angles1 = get_joint_angles(kps1, confs1)
        angles2 = get_joint_angles(kps2, confs2)

        for name in ANGLE_NAMES:
            if not math.isnan(angles1[name]) and not math.isnan(angles2[name]):
                assert abs(angles1[name] - angles2[name]) < 0.1, (
                    f"{name}: {angles1[name]:.1f}° vs {angles2[name]:.1f}°"
                )

    def test_limb_ratios_invariant(self):
        """Limb ratios normalized by torso length should be identical."""
        kps1, confs1 = _make_symmetric_standing(scale=1.0)
        kps2, confs2 = _make_symmetric_standing(scale=2.5)
        f1 = extract_frame_features(kps1, confs1, get_joint_angles(kps1, confs1))
        f2 = extract_frame_features(kps2, confs2, get_joint_angles(kps2, confs2))
        np.testing.assert_allclose(f1.limb_ratios, f2.limb_ratios, atol=1e-4)

    def test_relative_positions_invariant(self):
        """Relative positions (normalized by COM + torso) should be identical."""
        kps1, confs1 = _make_symmetric_standing(scale=1.0)
        kps2, confs2 = _make_symmetric_standing(scale=2.5)
        f1 = extract_frame_features(kps1, confs1, get_joint_angles(kps1, confs1))
        f2 = extract_frame_features(kps2, confs2, get_joint_angles(kps2, confs2))
        # Filter NaN
        valid = ~np.isnan(f1.relative_positions) & ~np.isnan(f2.relative_positions)
        if valid.any():
            np.testing.assert_allclose(
                f1.relative_positions[valid],
                f2.relative_positions[valid],
                atol=0.01,
            )

    def test_full_sequence_invariant(self):
        """Full feature arrays should be nearly identical at different scales."""
        kps1, conf1, ts1 = _make_sequence(n_frames=10, scale=1.0)
        kps2, conf2, ts2 = _make_sequence(n_frames=10, scale=2.0)
        # Same timestamps (same speed)
        f1 = extract_features(kps1, conf1, ts1)
        f2 = extract_features(kps2, conf2, ts2)

        arr1 = f1.to_array()
        arr2 = f2.to_array()

        # Angles (cols 0-8) should be identical
        valid = ~np.isnan(arr1[:, :9]) & ~np.isnan(arr2[:, :9])
        if valid.any():
            np.testing.assert_allclose(arr1[:, :9][valid], arr2[:, :9][valid], atol=0.5)

        # Limb ratios (cols 52-60) should be identical
        valid_r = ~np.isnan(arr1[:, 52:60]) & ~np.isnan(arr2[:, 52:60])
        if valid_r.any():
            np.testing.assert_allclose(arr1[:, 52:60][valid_r], arr2[:, 52:60][valid_r], atol=0.01)


class TestMirrorInvariance:
    """Mirrored skeletons should produce equivalent features with left/right swapped."""

    def _mirror_keypoints(self, kps: np.ndarray) -> np.ndarray:
        """Mirror horizontally: flip x and swap left/right pairs."""
        mirrored = kps.copy()
        # Flip x around center
        cx = kps[:, 0].mean()
        mirrored[:, 0] = 2 * cx - kps[:, 0]
        # Swap left/right pairs
        swap_pairs = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]
        for left, right in swap_pairs:
            mirrored[left], mirrored[right] = mirrored[right].copy(), mirrored[left].copy()
        return mirrored

    def test_angles_swap_left_right(self):
        """Mirrored pose should swap left/right angle values."""
        kps, confs = _make_symmetric_standing()
        mirrored_kps = self._mirror_keypoints(kps)

        f_orig = extract_frame_features(kps, confs, get_joint_angles(kps, confs))
        f_mirror = extract_frame_features(mirrored_kps, confs, get_joint_angles(mirrored_kps, confs))

        # left_knee of original should equal right_knee of mirrored
        pairs = [
            ("left_knee", "right_knee"),
            ("left_hip", "right_hip"),
            ("left_elbow", "right_elbow"),
            ("left_shoulder", "right_shoulder"),
        ]
        for left_name, right_name in pairs:
            left_idx = ANGLE_NAMES.index(left_name)
            right_idx = ANGLE_NAMES.index(right_name)
            orig_left = f_orig.joint_angles[left_idx]
            mirror_right = f_mirror.joint_angles[right_idx]
            if not math.isnan(orig_left) and not math.isnan(mirror_right):
                assert abs(orig_left - mirror_right) < 1.0, (
                    f"{left_name} orig={orig_left:.1f}° vs {right_name} mirror={mirror_right:.1f}°"
                )

    def test_limb_ratios_swap(self):
        """Mirrored limb ratios should swap left/right."""
        kps, confs = _make_symmetric_standing()
        mirrored_kps = self._mirror_keypoints(kps)

        f_orig = extract_frame_features(kps, confs, get_joint_angles(kps, confs))
        f_mirror = extract_frame_features(mirrored_kps, confs, get_joint_angles(mirrored_kps, confs))

        limb_pairs = [
            ("left_upper_arm", "right_upper_arm"),
            ("left_forearm", "right_forearm"),
            ("left_thigh", "right_thigh"),
            ("left_shin", "right_shin"),
        ]
        for left_name, right_name in limb_pairs:
            left_idx = LIMB_RATIO_NAMES.index(left_name)
            right_idx = LIMB_RATIO_NAMES.index(right_name)
            assert abs(f_orig.limb_ratios[left_idx] - f_mirror.limb_ratios[right_idx]) < 0.01


class TestSpeedInvariance:
    """Joint angles and positions should be the same regardless of playback speed."""

    def test_angles_same_at_different_speeds(self):
        """Same movement at different speeds should have identical angles per phase."""
        kps_fast, conf_fast, ts_fast = _make_sequence(n_frames=10, speed_factor=1.0)
        kps_slow, conf_slow, ts_slow = _make_sequence(n_frames=10, speed_factor=0.5)

        f_fast = extract_features(kps_fast, conf_fast, ts_fast)
        f_slow = extract_features(kps_slow, conf_slow, ts_slow)

        # Angles at same frame indices should be identical (same keypoints)
        for i in range(10):
            valid = ~np.isnan(f_fast.frames[i].joint_angles) & ~np.isnan(f_slow.frames[i].joint_angles)
            if valid.any():
                np.testing.assert_allclose(
                    f_fast.frames[i].joint_angles[valid],
                    f_slow.frames[i].joint_angles[valid],
                    atol=0.1,
                )

    def test_velocities_scale_with_speed(self):
        """Angular velocities should be proportional to speed factor."""
        kps1, conf1, ts1 = _make_sequence(n_frames=10, speed_factor=1.0)
        kps2, conf2, ts2 = _make_sequence(n_frames=10, speed_factor=0.5)

        f1 = extract_features(kps1, conf1, ts1)
        f2 = extract_features(kps2, conf2, ts2)

        # At half speed (speed_factor=0.5), timestamps are 2x farther apart,
        # so velocities should be ~half (same angle change over 2x time)
        for i in range(2, 8):  # skip edges where effects are less stable
            v1 = f1.frames[i].angular_velocities
            v2 = f2.frames[i].angular_velocities
            valid = ~np.isnan(v1) & ~np.isnan(v2) & (np.abs(v1) > 1.0)
            if valid.any():
                ratio = v2[valid] / v1[valid]
                # Speed_factor 0.5 means real time is 2x longer → velocity is 0.5x
                np.testing.assert_allclose(ratio, 0.5, atol=0.15)

    def test_positions_same_at_different_speeds(self):
        """Relative positions should be identical regardless of speed."""
        kps1, conf1, ts1 = _make_sequence(n_frames=10, speed_factor=1.0)
        kps2, conf2, ts2 = _make_sequence(n_frames=10, speed_factor=2.0)

        f1 = extract_features(kps1, conf1, ts1)
        f2 = extract_features(kps2, conf2, ts2)

        for i in range(10):
            valid = ~np.isnan(f1.frames[i].relative_positions) & ~np.isnan(f2.frames[i].relative_positions)
            if valid.any():
                np.testing.assert_allclose(
                    f1.frames[i].relative_positions[valid],
                    f2.frames[i].relative_positions[valid],
                    atol=0.01,
                )


# ── Integration Tests ────────────────────────────────────────────────


class TestEndToEnd:
    def test_full_pipeline_produces_valid_output(self):
        """Smoke test: full pipeline from keypoints to flat MLP input."""
        kps_seq, conf_seq, timestamps = _make_sequence(n_frames=30)
        features = extract_features(kps_seq, conf_seq, timestamps)

        # Check basic properties
        assert features.n_frames == 30
        assert features.n_features_per_frame == FEATURES_PER_FRAME

        # Flatten for TCN/MLP
        flat = features.to_flat_array(target_frames=64, normalize=True)
        assert flat.shape == (64 * FEATURES_PER_FRAME,)
        assert flat.dtype == np.float32

        # No infinities
        assert not np.any(np.isinf(flat))

    def test_array_roundtrip_preserves_values(self):
        """Converting to array and back should preserve feature values."""
        kps_seq, conf_seq, timestamps = _make_sequence(n_frames=15)
        features = extract_features(kps_seq, conf_seq, timestamps)
        arr = features.to_array()

        # Reconstruct manually
        for i in range(15):
            row = arr[i]
            np.testing.assert_allclose(row[:9], features.frames[i].joint_angles, equal_nan=True)
            np.testing.assert_allclose(row[9:18], features.frames[i].angular_velocities, equal_nan=True)
            np.testing.assert_allclose(
                row[18:52].reshape(17, 2),
                features.frames[i].relative_positions,
                equal_nan=True,
            )
            np.testing.assert_allclose(row[52:60], features.frames[i].limb_ratios, equal_nan=True)
            # World-frame features
            assert row[60] == pytest.approx(features.frames[i].body_tilt, nan_ok=True)
            assert row[61] == pytest.approx(features.frames[i].body_tilt_velocity, nan_ok=True)
            assert row[62] == pytest.approx(features.frames[i].vertical_com, nan_ok=True)
            assert row[63] == pytest.approx(features.frames[i].vertical_com_velocity, nan_ok=True)
            assert row[64] == pytest.approx(features.frames[i].cumulative_rotation, nan_ok=True)
            assert row[65] == pytest.approx(features.frames[i].left_right_symmetry, nan_ok=True)
            np.testing.assert_allclose(row[66:75], features.frames[i].angular_accelerations, equal_nan=True)

    def test_npy_save_load_roundtrip(self, tmp_path):
        """Features should survive save/load as .npy files (for DTW references)."""
        kps_seq, conf_seq, timestamps = _make_sequence(n_frames=20)
        features = extract_features(kps_seq, conf_seq, timestamps)
        arr = features.to_array()

        path = tmp_path / "reference.npy"
        np.save(path, arr)
        loaded = np.load(path)
        np.testing.assert_array_equal(arr, loaded)


# ── World-Frame Feature Tests ───────────────────────────────────────


class TestBodyTilt:
    """Tests for body tilt (torso angle vs vertical)."""

    def test_upright_tilt_is_zero(self):
        """A standing person should have body_tilt ~0°."""
        kps, confs = _make_symmetric_standing()
        features = extract_features([kps], [confs], [0.0])
        assert abs(features.frames[0].body_tilt) < 1.0  # ~0° upright

    def test_tilt_invariant_to_scale(self):
        """Body tilt should be the same at different scales."""
        kps1, confs1 = _make_symmetric_standing(scale=1.0)
        kps2, confs2 = _make_symmetric_standing(scale=3.0)
        f1 = extract_features([kps1], [confs1], [0.0])
        f2 = extract_features([kps2], [confs2], [0.0])
        assert abs(f1.frames[0].body_tilt - f2.frames[0].body_tilt) < 0.1

    def test_tilt_invariant_to_position(self):
        """Body tilt should be the same at different positions."""
        kps1, confs1 = _make_symmetric_standing(offset_x=0, offset_y=0)
        kps2, confs2 = _make_symmetric_standing(offset_x=200, offset_y=-100)
        f1 = extract_features([kps1], [confs1], [0.0])
        f2 = extract_features([kps2], [confs2], [0.0])
        assert abs(f1.frames[0].body_tilt - f2.frames[0].body_tilt) < 0.1


class TestLeftRightSymmetry:
    """Tests for left/right shoulder symmetry."""

    def test_symmetric_standing_is_zero(self):
        """Symmetric standing pose should have ~0 L/R symmetry."""
        kps, confs = _make_symmetric_standing()
        features = extract_features([kps], [confs], [0.0])
        assert abs(features.frames[0].left_right_symmetry) < 0.01

    def test_asymmetric_pose_nonzero(self):
        """Asymmetric pose (one shoulder raised) should have nonzero symmetry."""
        kps, confs = _make_symmetric_standing()
        # Raise left shoulder (lower y in image coords)
        from core.pose.constants import KEYPOINT_INDEX
        kps[KEYPOINT_INDEX["left_shoulder"], 1] -= 50
        features = extract_features([kps], [confs], [0.0])
        # Left shoulder higher → positive symmetry
        assert features.frames[0].left_right_symmetry > 0.1


class TestVerticalCOM:
    """Tests for vertical center of mass trajectory."""

    def test_stationary_is_zero(self):
        """Standing still should have vertical_com ~0."""
        kps, confs = _make_symmetric_standing()
        features = extract_features([kps, kps.copy()], [confs, confs.copy()], [0.0, 33.33])
        assert abs(features.frames[0].vertical_com) < 0.01
        assert abs(features.frames[1].vertical_com) < 0.01

    def test_jump_positive(self):
        """Moving up (lower y in image coords) should give positive vertical_com."""
        kps1, confs = _make_symmetric_standing()
        kps2 = kps1.copy()
        kps2[:, 1] -= 100  # Move everything up
        features = extract_features([kps1, kps2], [confs, confs.copy()], [0.0, 33.33])
        assert features.frames[1].vertical_com > 0  # Went up


class TestCumulativeRotation:
    """Tests for cumulative rotation tracking."""

    def test_no_rotation_is_zero(self):
        """No body tilt change → cumulative rotation stays 0."""
        kps, confs = _make_symmetric_standing()
        features = extract_features([kps, kps.copy(), kps.copy()],
                                    [confs, confs.copy(), confs.copy()],
                                    [0.0, 33.33, 66.66])
        assert abs(features.frames[-1].cumulative_rotation) < 1.0


class TestTemporalArray:
    """Tests for the TCN-oriented temporal array output."""

    def test_to_temporal_array_shape(self):
        kps_seq, conf_seq, timestamps = _make_sequence(n_frames=30)
        features = extract_features(kps_seq, conf_seq, timestamps)
        tcn_arr = features.to_temporal_array(target_frames=64)
        assert tcn_arr.shape == (FEATURES_PER_FRAME, 64)
        assert tcn_arr.dtype == np.float32

    def test_to_temporal_array_is_transpose_of_flat(self):
        kps_seq, conf_seq, timestamps = _make_sequence(n_frames=30)
        features = extract_features(kps_seq, conf_seq, timestamps)
        tcn = features.to_temporal_array(target_frames=64, normalize=True)
        flat = features.to_flat_array(target_frames=64, normalize=True)
        # TCN is (F, T), flat is (F*T,) — should be same data
        np.testing.assert_allclose(tcn.flatten(order='F'), flat, atol=1e-5)
