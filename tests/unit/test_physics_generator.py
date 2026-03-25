"""Tests for physics-based feature sequence generator."""

from __future__ import annotations

import numpy as np
import pytest

from core.pose.features import FEATURES_PER_FRAME
from ml.physics_generator import PhysicsFeatureGenerator
from ml.trick_physics import TRICK_DEFINITIONS, RotationAxis


TARGET_FRAMES = 64


@pytest.fixture
def generator():
    return PhysicsFeatureGenerator(target_frames=TARGET_FRAMES, seed=42)


class TestGenerateAll:
    def test_generates_all_tricks_plus_no_trick(self, generator):
        data = generator.generate_all(samples_per_trick=3)
        assert "no_trick" in data
        for trick_id in TRICK_DEFINITIONS:
            assert trick_id in data, f"Missing trick: {trick_id}"

    def test_correct_shape(self, generator):
        data = generator.generate_all(samples_per_trick=2)
        for name, arrays in data.items():
            for arr in arrays:
                assert arr.shape == (TARGET_FRAMES, FEATURES_PER_FRAME), (
                    f"{name}: expected ({TARGET_FRAMES}, {FEATURES_PER_FRAME}), got {arr.shape}"
                )

    def test_correct_sample_count(self, generator):
        n = 7
        data = generator.generate_all(samples_per_trick=n)
        for name, arrays in data.items():
            assert len(arrays) == n, f"{name}: expected {n}, got {len(arrays)}"

    def test_no_nans_or_infs(self, generator):
        data = generator.generate_all(samples_per_trick=5)
        for name, arrays in data.items():
            for i, arr in enumerate(arrays):
                assert not np.any(np.isinf(arr)), f"{name}[{i}] has inf"
                nan_frac = np.isnan(arr).mean()
                assert nan_frac < 0.1, f"{name}[{i}] has {nan_frac:.0%} NaN"


class TestTrickPhysics:
    """Verify physics properties of generated tricks."""

    def test_backflip_cumulative_rotation(self, generator):
        """Back flip should have ~360° cumulative rotation."""
        trick_def = TRICK_DEFINITIONS["back_flip"]
        samples = generator.generate(trick_def, n=10)
        for arr in samples:
            cum_rot = arr[-1, 64]  # cumulative_rotation is column 64
            # Should be roughly ±360° (direction depends on convention)
            assert abs(abs(cum_rot) - 360) < 60, f"Back flip cumrot={cum_rot:.0f}°"

    def test_double_back_more_rotation(self, generator):
        """Double back should rotate more than single back flip."""
        single = TRICK_DEFINITIONS["back_flip"]
        double = TRICK_DEFINITIONS["double_back"]
        single_samples = generator.generate(single, n=10)
        double_samples = generator.generate(double, n=10)

        single_rot = np.mean([abs(s[-1, 64]) for s in single_samples])
        double_rot = np.mean([abs(s[-1, 64]) for s in double_samples])
        assert double_rot > single_rot * 1.3, (
            f"Double ({double_rot:.0f}°) should be > single ({single_rot:.0f}°)"
        )

    def test_no_trick_minimal_rotation(self, generator):
        """No-trick class should have minimal cumulative rotation."""
        data = generator.generate_all(samples_per_trick=10)
        no_trick = data["no_trick"]
        for arr in no_trick:
            cum_rot = abs(arr[-1, 64])
            assert cum_rot < 45, f"No-trick has {cum_rot:.0f}° rotation"

    def test_no_trick_minimal_vertical(self, generator):
        """No-trick class should not have large vertical COM displacement."""
        data = generator.generate_all(samples_per_trick=10)
        no_trick = data["no_trick"]
        for arr in no_trick:
            max_vert = np.max(np.abs(arr[:, 62]))  # vertical_com
            assert max_vert < 2.0, f"No-trick max vertical COM = {max_vert:.1f}"

    def test_side_flip_has_asymmetry(self, generator):
        """Side flip should have significant L/R asymmetry."""
        trick_def = TRICK_DEFINITIONS["side_flip"]
        samples = generator.generate(trick_def, n=10)
        for arr in samples:
            max_asym = np.max(np.abs(arr[:, 65]))  # left_right_symmetry
            assert max_asym > 0.05, f"Side flip max L/R asymmetry = {max_asym:.3f}"


class TestVariability:
    """Verify that generated samples have meaningful variation."""

    def test_samples_are_different(self, generator):
        """Different samples of the same trick should not be identical."""
        trick_def = TRICK_DEFINITIONS["front_flip"]
        samples = generator.generate(trick_def, n=5)
        for i in range(len(samples) - 1):
            diff = np.mean(np.abs(samples[i] - samples[i + 1]))
            assert diff > 0.01, "Consecutive samples are too similar"

    def test_deterministic_with_seed(self):
        """Same seed should produce identical results."""
        gen1 = PhysicsFeatureGenerator(seed=123)
        gen2 = PhysicsFeatureGenerator(seed=123)
        trick_def = TRICK_DEFINITIONS["back_flip"]
        s1 = gen1.generate(trick_def, n=3)
        s2 = gen2.generate(trick_def, n=3)
        for a, b in zip(s1, s2):
            np.testing.assert_array_equal(a, b)

    def test_different_seeds_differ(self):
        """Different seeds should produce different results."""
        gen1 = PhysicsFeatureGenerator(seed=1)
        gen2 = PhysicsFeatureGenerator(seed=2)
        trick_def = TRICK_DEFINITIONS["back_flip"]
        s1 = gen1.generate(trick_def, n=1)[0]
        s2 = gen2.generate(trick_def, n=1)[0]
        assert not np.allclose(s1, s2)
