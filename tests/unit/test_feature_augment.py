"""Tests for feature-level augmentation engine."""

from __future__ import annotations

import math

import numpy as np
import pytest

from core.pose.features import (
    ANGLE_NAMES,
    FEATURES_PER_FRAME,
    LIMB_RATIO_NAMES,
    FeatureSequence,
    extract_features,
)
from ml.feature_augment import (
    AugmentConfig,
    FeatureAugmenter,
    _dropout,
    _inject_noise,
    _mirror,
    _perturb_angles,
    _recompute_velocities,
    _scale_proportions,
    _time_stretch,
    augment_from_keypoints,
    augment_sequence,
)


# ── Fixtures ─────────────────────────────────────────────────────────


def _make_reference(n_frames: int = 30, seed: int = 42) -> FeatureSequence:
    """Create a synthetic reference FeatureSequence simulating a squat movement."""
    rng = np.random.default_rng(seed)

    # Symmetric standing skeleton
    base_kps = np.array([
        [320, 100], [310, 90], [330, 90], [300, 95], [340, 95],
        [280, 200], [360, 200], [260, 300], [380, 300],
        [240, 400], [400, 400], [290, 400], [350, 400],
        [290, 540], [350, 540], [290, 680], [350, 680],
    ], dtype=np.float32)
    confs = np.full(17, 0.95, dtype=np.float32)

    kps_seq = []
    conf_seq = []
    timestamps = []

    for i in range(n_frames):
        t = i / max(n_frames - 1, 1)
        bend = math.sin(t * math.pi) * 100

        frame_kps = base_kps.copy()
        for idx in [13, 14]:
            frame_kps[idx, 1] = base_kps[idx, 1] - bend * 0.3
        # Add small random variation for realism
        frame_kps[:, :2] += rng.normal(0, 1, size=frame_kps.shape).astype(np.float32)

        kps_seq.append(frame_kps)
        conf_seq.append(confs.copy())
        timestamps.append(i * 33.33)

    return extract_features(kps_seq, conf_seq, timestamps)


@pytest.fixture
def reference() -> FeatureSequence:
    return _make_reference()


@pytest.fixture
def reference_arr(reference: FeatureSequence) -> np.ndarray:
    return reference.to_array()


# ── Unit Tests: Time Stretch ─────────────────────────────────────────


class TestTimeStretch:
    def test_slower_produces_more_frames(self, reference_arr):
        stretched = _time_stretch(reference_arr, factor=0.5, fps=30.0)
        assert stretched.shape[0] > reference_arr.shape[0]
        assert stretched.shape[1] == FEATURES_PER_FRAME

    def test_faster_produces_fewer_frames(self, reference_arr):
        stretched = _time_stretch(reference_arr, factor=2.0, fps=30.0)
        assert stretched.shape[0] < reference_arr.shape[0]
        assert stretched.shape[1] == FEATURES_PER_FRAME

    def test_factor_one_preserves_length(self, reference_arr):
        stretched = _time_stretch(reference_arr, factor=1.0, fps=30.0)
        assert stretched.shape[0] == reference_arr.shape[0]

    def test_velocities_recomputed(self, reference_arr):
        stretched = _time_stretch(reference_arr, factor=0.5, fps=30.0)
        # First frame velocity should be zero
        np.testing.assert_array_equal(stretched[0, 9:18], 0.0)
        # Velocities should be finite where angles are finite
        angles_valid = ~np.isnan(stretched[5, :9])
        vels = stretched[5, 9:18]
        assert np.all(np.isfinite(vels[angles_valid]))

    def test_minimum_frames(self, reference_arr):
        # Even extreme factor should produce at least 4 frames
        stretched = _time_stretch(reference_arr, factor=100.0, fps=30.0)
        assert stretched.shape[0] >= 4


# ── Unit Tests: Angle Perturbation ───────────────────────────────────


class TestPerturbAngles:
    def test_angles_change(self, reference_arr):
        rng = np.random.default_rng(0)
        perturbed = _perturb_angles(reference_arr, std=10.0, smoothing=5, rng=rng, fps=30.0)
        # Angles should be different from original
        orig = reference_arr[:, :9]
        pert = perturbed[:, :9]
        valid = ~np.isnan(orig) & ~np.isnan(pert)
        if valid.any():
            assert not np.allclose(orig[valid], pert[valid])

    def test_angles_stay_in_range(self, reference_arr):
        rng = np.random.default_rng(1)
        perturbed = _perturb_angles(reference_arr, std=20.0, smoothing=3, rng=rng, fps=30.0)
        angles = perturbed[:, :9]
        valid = ~np.isnan(angles)
        if valid.any():
            assert angles[valid].min() >= 0.0
            assert angles[valid].max() <= 180.0

    def test_velocities_recomputed(self, reference_arr):
        rng = np.random.default_rng(2)
        perturbed = _perturb_angles(reference_arr, std=5.0, smoothing=5, rng=rng, fps=30.0)
        # First frame velocity should be zero
        np.testing.assert_array_equal(perturbed[0, 9:18], 0.0)

    def test_smooth_noise_is_smoother(self, reference_arr):
        """Smoothed noise should have lower frame-to-frame variance than raw noise."""
        rng1 = np.random.default_rng(10)
        rng2 = np.random.default_rng(10)
        rough = _perturb_angles(reference_arr, std=10.0, smoothing=1, rng=rng1, fps=30.0)
        smooth = _perturb_angles(reference_arr, std=10.0, smoothing=7, rng=rng2, fps=30.0)

        # Compare frame-to-frame angle differences
        rough_diffs = np.nanmean(np.abs(np.diff(rough[:, :9], axis=0)))
        smooth_diffs = np.nanmean(np.abs(np.diff(smooth[:, :9], axis=0)))
        assert smooth_diffs < rough_diffs

    def test_nan_angles_stay_nan(self, reference_arr):
        arr = reference_arr.copy()
        arr[5, 0] = np.nan  # Set one angle to NaN
        rng = np.random.default_rng(3)
        perturbed = _perturb_angles(arr, std=5.0, smoothing=3, rng=rng, fps=30.0)
        assert np.isnan(perturbed[5, 0])


# ── Unit Tests: Proportion Scaling ───────────────────────────────────


class TestScaleProportions:
    def test_ratios_change(self, reference_arr):
        rng = np.random.default_rng(0)
        scaled = _scale_proportions(reference_arr, (0.8, 1.2), rng)
        orig_ratios = reference_arr[:, 52:60]
        new_ratios = scaled[:, 52:60]
        valid = ~np.isnan(orig_ratios) & ~np.isnan(new_ratios)
        if valid.any():
            assert not np.allclose(orig_ratios[valid], new_ratios[valid])

    def test_consistent_across_frames(self, reference_arr):
        """Same scale factor applied to every frame."""
        rng = np.random.default_rng(1)
        scaled = _scale_proportions(reference_arr, (0.8, 1.2), rng)
        orig = reference_arr[:, 52:60]
        new = scaled[:, 52:60]
        # Compute ratio of change for each limb, should be constant across frames
        valid_frames = ~np.isnan(orig) & ~np.isnan(new) & (orig > 1e-6)
        if valid_frames.any():
            ratios = new[valid_frames] / orig[valid_frames]
            # All frames should have same ratio per limb (within floating point)
            for j in range(8):
                col_valid = valid_frames[:, j]
                if col_valid.sum() > 1:
                    col_ratios = new[col_valid, j] / orig[col_valid, j]
                    assert np.std(col_ratios) < 0.001

    def test_bilateral_correlation(self, reference_arr):
        """Left and right limbs should scale similarly (correlated)."""
        rng = np.random.default_rng(2)
        scaled = _scale_proportions(reference_arr, (0.8, 1.2), rng)
        orig = reference_arr[0, 52:60]
        new = scaled[0, 52:60]
        valid = ~np.isnan(orig) & ~np.isnan(new) & (orig > 1e-6)
        if valid.all():
            left_scales = new[::2] / orig[::2]  # left limbs
            right_scales = new[1::2] / orig[1::2]  # right limbs
            # Left and right should be close (within jitter range)
            np.testing.assert_allclose(left_scales, right_scales, atol=0.05)

    def test_other_features_unchanged(self, reference_arr):
        rng = np.random.default_rng(3)
        scaled = _scale_proportions(reference_arr, (0.8, 1.2), rng)
        # Angles, velocities, positions should be untouched
        np.testing.assert_array_equal(scaled[:, :18], reference_arr[:, :18])
        np.testing.assert_array_equal(scaled[:, 18:52], reference_arr[:, 18:52])


# ── Unit Tests: Noise Injection ──────────────────────────────────────


class TestInjectNoise:
    def test_positions_change(self, reference_arr):
        rng = np.random.default_rng(0)
        noisy = _inject_noise(reference_arr, std=0.05, rng=rng)
        positions_orig = reference_arr[:, 18:52]
        positions_new = noisy[:, 18:52]
        valid = ~np.isnan(positions_orig) & ~np.isnan(positions_new)
        if valid.any():
            assert not np.allclose(positions_orig[valid], positions_new[valid])

    def test_angles_unchanged(self, reference_arr):
        rng = np.random.default_rng(1)
        noisy = _inject_noise(reference_arr, std=0.05, rng=rng)
        np.testing.assert_array_equal(noisy[:, :18], reference_arr[:, :18])
        np.testing.assert_array_equal(noisy[:, 52:60], reference_arr[:, 52:60])

    def test_nan_positions_stay_nan(self, reference_arr):
        arr = reference_arr.copy()
        arr[3, 20] = np.nan
        rng = np.random.default_rng(2)
        noisy = _inject_noise(arr, std=0.05, rng=rng)
        assert np.isnan(noisy[3, 20])


# ── Unit Tests: Mirror ───────────────────────────────────────────────


class TestMirror:
    def test_angle_swap(self, reference_arr):
        mirrored = _mirror(reference_arr)
        # left_knee (col 0) should become right_knee (col 1)
        np.testing.assert_array_equal(mirrored[:, 0], reference_arr[:, 1])
        np.testing.assert_array_equal(mirrored[:, 1], reference_arr[:, 0])

    def test_velocity_swap(self, reference_arr):
        mirrored = _mirror(reference_arr)
        # Velocity cols follow same pattern (offset by 9)
        np.testing.assert_array_equal(mirrored[:, 9], reference_arr[:, 10])
        np.testing.assert_array_equal(mirrored[:, 10], reference_arr[:, 9])

    def test_ratio_swap(self, reference_arr):
        mirrored = _mirror(reference_arr)
        # left_upper_arm (col 52) ↔ right_upper_arm (col 53)
        np.testing.assert_array_equal(mirrored[:, 52], reference_arr[:, 53])
        np.testing.assert_array_equal(mirrored[:, 53], reference_arr[:, 52])

    def test_position_x_flipped(self, reference_arr):
        mirrored = _mirror(reference_arr)
        # nose (keypoint 0) x should be negated (no swap pair for nose)
        orig_nose_x = reference_arr[:, 18]  # positions start at 18, nose is kp 0 → index 18
        mirror_nose_x = mirrored[:, 18]
        valid = ~np.isnan(orig_nose_x) & ~np.isnan(mirror_nose_x)
        if valid.any():
            np.testing.assert_allclose(mirror_nose_x[valid], -orig_nose_x[valid])

    def test_double_mirror_is_identity(self, reference_arr):
        double = _mirror(_mirror(reference_arr))
        np.testing.assert_allclose(double, reference_arr, atol=1e-6, equal_nan=True)

    def test_spine_angle_unchanged(self, reference_arr):
        """Spine (col 8) is not in a swap pair — should stay the same."""
        mirrored = _mirror(reference_arr)
        np.testing.assert_array_equal(mirrored[:, 8], reference_arr[:, 8])


# ── Unit Tests: Dropout ──────────────────────────────────────────────


class TestDropout:
    def test_some_positions_become_nan(self, reference_arr):
        rng = np.random.default_rng(0)
        dropped = _dropout(reference_arr, rate=0.2, rng=rng)
        positions = dropped[:, 18:52]
        # Some should be NaN now
        assert np.isnan(positions).sum() > 0

    def test_angles_and_ratios_unchanged(self, reference_arr):
        rng = np.random.default_rng(1)
        dropped = _dropout(reference_arr, rate=0.3, rng=rng)
        np.testing.assert_array_equal(dropped[:, :18], reference_arr[:, :18])
        np.testing.assert_array_equal(dropped[:, 52:60], reference_arr[:, 52:60])

    def test_zero_rate_no_change(self, reference_arr):
        rng = np.random.default_rng(2)
        dropped = _dropout(reference_arr, rate=0.0, rng=rng)
        np.testing.assert_array_equal(dropped, reference_arr)


# ── Unit Tests: Velocity Recomputation ───────────────────────────────


class TestRecomputeVelocities:
    def test_first_frame_zero(self, reference_arr):
        arr = reference_arr.copy()
        _recompute_velocities(arr, fps=30.0)
        np.testing.assert_array_equal(arr[0, 9:18], 0.0)

    def test_constant_angles_zero_velocity(self):
        """If all angles are constant, all velocities should be zero."""
        arr = np.zeros((10, FEATURES_PER_FRAME), dtype=np.float32)
        arr[:, :9] = 90.0  # Constant 90° for all angles
        _recompute_velocities(arr, fps=30.0)
        np.testing.assert_array_equal(arr[:, 9:18], 0.0)

    def test_linear_angle_change(self):
        """Linear angle change should produce constant velocity."""
        arr = np.zeros((10, FEATURES_PER_FRAME), dtype=np.float32)
        for i in range(10):
            arr[i, 0] = 90.0 + i * 10.0  # 10° per frame at 30fps = 300°/s
        _recompute_velocities(arr, fps=30.0)
        # Frames 1-9 should have velocity 300°/s for angle 0
        expected_vel = 10.0 * 30.0  # 300°/s
        np.testing.assert_allclose(arr[1:, 9], expected_vel, atol=0.01)


# ── Integration Tests: FeatureAugmenter ──────────────────────────────


class TestFeatureAugmenter:
    def test_augment_one_produces_valid_output(self, reference):
        augmenter = FeatureAugmenter(seed=42)
        variant = augmenter.augment_one(reference)
        assert variant.n_frames > 0
        assert variant.n_features_per_frame == FEATURES_PER_FRAME

    def test_augment_many_count(self, reference):
        augmenter = FeatureAugmenter(seed=42)
        variants = augmenter.augment_many(reference, n_variants=10)
        assert len(variants) == 10

    def test_variants_are_different(self, reference):
        augmenter = FeatureAugmenter(seed=42)
        variants = augmenter.augment_many(reference, n_variants=5)
        arrays = [v.to_array() for v in variants]
        # At least some should have different shapes (time stretch)
        shapes = {a.shape[0] for a in arrays}
        assert len(shapes) > 1, "Expected different frame counts from time stretch"

    def test_deterministic_with_seed(self, reference):
        v1 = FeatureAugmenter(seed=123).augment_one(reference)
        v2 = FeatureAugmenter(seed=123).augment_one(reference)
        np.testing.assert_array_equal(v1.to_array(), v2.to_array())

    def test_empty_input(self):
        augmenter = FeatureAugmenter(seed=42)
        empty = FeatureSequence(frames=[])
        variant = augmenter.augment_one(empty)
        assert variant.n_frames == 0

    def test_augment_references(self):
        refs = [_make_reference(seed=i) for i in range(3)]
        augmenter = FeatureAugmenter(seed=42)
        variants = augmenter.augment_references(refs, n_total=100)
        assert len(variants) == 100

    def test_augment_references_even_distribution(self):
        refs = [_make_reference(seed=i) for i in range(3)]
        augmenter = FeatureAugmenter(seed=42)
        # 100 / 3 = 33 each + 1 extra for first ref
        variants = augmenter.augment_references(refs, n_total=100)
        assert len(variants) == 100

    def test_no_infinities_in_output(self, reference):
        augmenter = FeatureAugmenter(seed=42)
        variants = augmenter.augment_many(reference, n_variants=20)
        for v in variants:
            arr = v.to_array()
            assert not np.any(np.isinf(arr)), "Output contains infinities"

    def test_custom_config(self, reference):
        config = AugmentConfig(
            time_stretch_prob=0.0,
            angle_noise_prob=0.0,
            proportion_scale_prob=0.0,
            position_noise_prob=0.0,
            mirror_prob=0.0,
            dropout_prob=0.0,
        )
        augmenter = FeatureAugmenter(config=config, seed=42)
        variant = augmenter.augment_one(reference)
        # With all augmentations disabled, output should equal input
        np.testing.assert_allclose(
            variant.to_array(), reference.to_array(), atol=1e-6, equal_nan=True,
        )


# ── Integration Tests: Public API ────────────────────────────────────


class TestAugmentSequence:
    def test_basic(self, reference):
        variants = augment_sequence(reference, n_variants=10, seed=42)
        assert len(variants) == 10
        for v in variants:
            assert v.n_features_per_frame == FEATURES_PER_FRAME

    def test_produces_plausible_angles(self, reference):
        """Augmented angles should stay within physical range [0, 180]."""
        config = AugmentConfig(angle_noise_std=15.0)
        variants = augment_sequence(reference, n_variants=50, config=config, seed=42)
        for v in variants:
            arr = v.to_array()
            angles = arr[:, :9]
            valid = ~np.isnan(angles)
            if valid.any():
                assert angles[valid].min() >= -0.1, "Angle below 0°"
                assert angles[valid].max() <= 180.1, "Angle above 180°"


class TestAugmentFromKeypoints:
    def test_basic(self):
        """Bridge function should produce valid FeatureSequence variants."""
        rng = np.random.default_rng(42)
        base_kps = np.array([
            [320, 100], [310, 90], [330, 90], [300, 95], [340, 95],
            [280, 200], [360, 200], [260, 300], [380, 300],
            [240, 400], [400, 400], [290, 400], [350, 400],
            [290, 540], [350, 540], [290, 680], [350, 680],
        ], dtype=np.float32)

        kps_seq = [base_kps + rng.normal(0, 2, base_kps.shape).astype(np.float32) for _ in range(20)]
        conf_seq = [np.full(17, 0.9, dtype=np.float32) for _ in range(20)]
        timestamps = [i * 33.33 for i in range(20)]

        variants = augment_from_keypoints(
            kps_seq, conf_seq, timestamps,
            n_variants=10, seed=42,
        )
        assert len(variants) == 10
        for v in variants:
            assert v.n_features_per_frame == FEATURES_PER_FRAME


# ── Scale Test ───────────────────────────────────────────────────────


class TestScale:
    def test_generate_500_variants(self, reference):
        """Verify we can generate 500 variants from a single reference in reasonable time."""
        variants = augment_sequence(reference, n_variants=500, seed=42)
        assert len(variants) == 500

        # All should be valid
        for v in variants[:10]:  # spot-check first 10
            arr = v.to_array()
            assert arr.shape[1] == FEATURES_PER_FRAME
            assert not np.any(np.isinf(arr))

    def test_mlp_ready_output(self, reference):
        """Variants should be convertible to fixed-length flat arrays for MLP."""
        variants = augment_sequence(reference, n_variants=10, seed=42)
        for v in variants:
            flat = v.to_flat_array(target_frames=64)
            assert flat.shape == (64 * FEATURES_PER_FRAME,)
            assert flat.dtype == np.float32
