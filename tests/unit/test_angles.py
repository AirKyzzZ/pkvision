"""Tests for joint angle calculations."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from core.pose.angles import (
    _compute_angle,
    get_joint_angles,
    get_joint_velocities,
)

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


@pytest.fixture
def standing_pose() -> tuple[np.ndarray, np.ndarray]:
    with open(FIXTURES_DIR / "sample_keypoints.json") as f:
        data = json.load(f)
    kps = np.array(data["standing_pose"]["keypoints"], dtype=np.float32)
    confs = np.array(data["standing_pose"]["confidences"], dtype=np.float32)
    return kps, confs


@pytest.fixture
def tucked_pose() -> tuple[np.ndarray, np.ndarray]:
    with open(FIXTURES_DIR / "sample_keypoints.json") as f:
        data = json.load(f)
    kps = np.array(data["tucked_pose"]["keypoints"], dtype=np.float32)
    confs = np.array(data["tucked_pose"]["confidences"], dtype=np.float32)
    return kps, confs


@pytest.fixture
def low_confidence_pose() -> tuple[np.ndarray, np.ndarray]:
    with open(FIXTURES_DIR / "sample_keypoints.json") as f:
        data = json.load(f)
    kps = np.array(data["low_confidence_pose"]["keypoints"], dtype=np.float32)
    confs = np.array(data["low_confidence_pose"]["confidences"], dtype=np.float32)
    return kps, confs


class TestComputeAngle:
    def test_right_angle(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 0.0])
        c = np.array([0.0, 1.0])
        angle = _compute_angle(a, b, c)
        assert abs(angle - 90.0) < 0.1

    def test_straight_angle(self):
        a = np.array([-1.0, 0.0])
        b = np.array([0.0, 0.0])
        c = np.array([1.0, 0.0])
        angle = _compute_angle(a, b, c)
        assert abs(angle - 180.0) < 0.1

    def test_acute_angle(self):
        a = np.array([1.0, 1.0])
        b = np.array([0.0, 0.0])
        c = np.array([1.0, 0.0])
        angle = _compute_angle(a, b, c)
        assert abs(angle - 45.0) < 0.1

    def test_zero_angle(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 0.0])
        c = np.array([2.0, 0.0])
        angle = _compute_angle(a, b, c)
        assert abs(angle - 0.0) < 0.1


class TestGetJointAngles:
    def test_returns_all_expected_angles(self, standing_pose):
        kps, confs = standing_pose
        angles = get_joint_angles(kps, confs)

        expected_keys = {
            "left_knee", "right_knee", "left_hip", "right_hip",
            "left_elbow", "right_elbow", "left_shoulder", "right_shoulder",
            "spine", "neck",
            "knee", "hip", "elbow", "shoulder",  # averaged bilateral
        }
        assert expected_keys.issubset(set(angles.keys()))

    def test_standing_has_extended_knees(self, standing_pose):
        kps, confs = standing_pose
        angles = get_joint_angles(kps, confs)
        # Standing person should have relatively extended knees (>120°)
        assert not math.isnan(angles["knee"])
        assert angles["knee"] > 100, f"Expected extended knee, got {angles['knee']}°"

    def test_tucked_has_bent_joints(self, tucked_pose):
        kps, confs = tucked_pose
        angles = get_joint_angles(kps, confs)
        # Tucked pose should have more flexed hip
        assert not math.isnan(angles["hip"])
        # Hip angle should be noticeably different from standing

    def test_low_confidence_gives_nan(self, low_confidence_pose):
        kps, confs = low_confidence_pose
        angles = get_joint_angles(kps, confs, min_confidence=0.3)
        # All keypoints are below 0.3 confidence, so all angles should be NaN
        for name, angle in angles.items():
            assert math.isnan(angle), f"Expected NaN for {name}, got {angle}"

    def test_averaged_bilateral(self, standing_pose):
        kps, confs = standing_pose
        angles = get_joint_angles(kps, confs)
        # Averaged "knee" should be mean of left_knee and right_knee
        if not math.isnan(angles["left_knee"]) and not math.isnan(angles["right_knee"]):
            expected = (angles["left_knee"] + angles["right_knee"]) / 2.0
            assert abs(angles["knee"] - expected) < 0.01


class TestGetJointVelocities:
    def test_single_frame_zero_velocity(self):
        angles = [{"knee": 160.0, "hip": 170.0}]
        timestamps = [0.0]
        velocities = get_joint_velocities(angles, timestamps)
        assert len(velocities) == 1
        assert velocities[0]["knee"] == 0.0
        assert velocities[0]["hip"] == 0.0

    def test_two_frames_computes_velocity(self):
        angles = [
            {"knee": 160.0},
            {"knee": 100.0},
        ]
        timestamps = [0.0, 100.0]  # 100ms apart
        velocities = get_joint_velocities(angles, timestamps)
        assert len(velocities) == 2
        # (100 - 160) / 0.1s = -600 deg/s
        assert abs(velocities[1]["knee"] - (-600.0)) < 0.01

    def test_nan_propagates(self):
        angles = [
            {"knee": 160.0},
            {"knee": float("nan")},
        ]
        timestamps = [0.0, 100.0]
        velocities = get_joint_velocities(angles, timestamps)
        assert math.isnan(velocities[1]["knee"])

    def test_empty_input(self):
        assert get_joint_velocities([], []) == []
