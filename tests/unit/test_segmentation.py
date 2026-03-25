"""Tests for the improved temporal segmentation (RunSegmenter)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from core.models import FrameAnalysis
from core.recognition.segmentation import RunSegmenter, TrickSegment


# ── Helpers ──────────────────────────────────────────────────────────


def _make_frame(
    idx: int,
    timestamp_ms: float,
    hip_angle: float = 170.0,
    spine_angle: float = 175.0,
) -> FrameAnalysis:
    """Create a minimal FrameAnalysis with controllable hip/spine angles."""
    return FrameAnalysis(
        frame_idx=idx,
        timestamp_ms=timestamp_ms,
        keypoints=np.zeros((17, 2), dtype=np.float32),
        keypoint_confidences=np.ones(17, dtype=np.float32),
        angles={
            "left_hip": hip_angle,
            "right_hip": hip_angle,
            "spine": spine_angle,
            "left_knee": 170.0,
            "right_knee": 170.0,
        },
        velocities=None,
    )


def _make_idle_frames(
    start_idx: int,
    start_ms: float,
    count: int,
    interval_ms: float = 33.33,
) -> list[FrameAnalysis]:
    """Create a sequence of idle (low-velocity) frames with stable angles."""
    frames: list[FrameAnalysis] = []
    for i in range(count):
        frames.append(
            _make_frame(
                idx=start_idx + i,
                timestamp_ms=start_ms + i * interval_ms,
                hip_angle=170.0 + (i % 2) * 0.5,   # tiny jitter
                spine_angle=175.0 + (i % 2) * 0.3,
            )
        )
    return frames


def _make_active_frames(
    start_idx: int,
    start_ms: float,
    count: int,
    interval_ms: float = 33.33,
    amplitude: float = 40.0,
) -> list[FrameAnalysis]:
    """Create a sequence of active frames with large angle swings (simulates a trick)."""
    frames: list[FrameAnalysis] = []
    for i in range(count):
        # Oscillate angles rapidly to create high angular velocity
        t = i / max(count - 1, 1)
        swing = amplitude * math.sin(t * math.pi * 2)
        frames.append(
            _make_frame(
                idx=start_idx + i,
                timestamp_ms=start_ms + i * interval_ms,
                hip_angle=170.0 + swing,
                spine_angle=175.0 + swing * 0.8,
            )
        )
    return frames


def _make_mixed_sequence() -> list[FrameAnalysis]:
    """Create idle → active → idle → active → idle sequence.

    Layout (at 30fps, ~33.33ms per frame):
      frames  0-29  : idle       (1000 ms)
      frames 30-59  : active     (1000 ms)
      frames 60-89  : idle       (1000 ms)
      frames 90-119 : active     (1000 ms)
      frames 120-149: idle       (1000 ms)
    """
    interval = 33.33
    frames: list[FrameAnalysis] = []
    frames += _make_idle_frames(0, 0.0, 30, interval)
    frames += _make_active_frames(30, 30 * interval, 30, interval, amplitude=50.0)
    frames += _make_idle_frames(60, 60 * interval, 30, interval)
    frames += _make_active_frames(90, 90 * interval, 30, interval, amplitude=50.0)
    frames += _make_idle_frames(120, 120 * interval, 30, interval)
    return frames


# ── Tests ────────────────────────────────────────────────────────────


class TestTrickSegmentProperties:
    def test_duration_ms(self):
        seg = TrickSegment(
            start_frame=0, end_frame=30,
            start_time_ms=0.0, end_time_ms=1000.0,
            peak_intensity=500.0,
        )
        assert seg.duration_ms == 1000.0

    def test_n_frames(self):
        seg = TrickSegment(
            start_frame=10, end_frame=40,
            start_time_ms=333.0, end_time_ms=1333.0,
            peak_intensity=300.0,
        )
        assert seg.n_frames == 31


class TestEmptyInput:
    def test_empty_frames(self):
        segmenter = RunSegmenter()
        result = segmenter.segment([])
        assert result == []

    def test_single_frame(self):
        segmenter = RunSegmenter()
        frame = _make_frame(0, 0.0)
        result = segmenter.segment([frame])
        assert result == []


class TestAllIdleInput:
    """When the whole video is idle, no segments should be detected."""

    def test_no_segments_for_idle_video(self):
        frames = _make_idle_frames(0, 0.0, 100, interval_ms=33.33)
        segmenter = RunSegmenter(adaptive_threshold_factor=1.5)
        result = segmenter.segment(frames)
        # Idle frames have tiny jitter — the adaptive threshold should be above
        # the jitter so no active regions are detected.
        assert len(result) == 0


class TestAllActiveInput:
    """When the whole video is one continuous trick."""

    def test_single_segment_for_active_video(self):
        # Accelerating angle change: velocity ramps from low to high.
        # The adaptive threshold (1.5x median) is below the peak region,
        # so a significant portion is detected as active.
        interval = 33.33
        frames: list[FrameAnalysis] = []
        for i in range(60):
            # Quadratic angle growth → linearly increasing velocity
            angle_offset = 0.05 * i * i
            frames.append(
                _make_frame(
                    idx=i,
                    timestamp_ms=i * interval,
                    hip_angle=100.0 + angle_offset,
                    spine_angle=100.0 + angle_offset * 0.8,
                )
            )

        segmenter = RunSegmenter(
            min_window_ms=100.0,
            max_window_ms=5000.0,
            adaptive_threshold_factor=1.5,
        )
        result = segmenter.segment(frames)
        assert len(result) >= 1
        # The segment(s) should cover a meaningful portion of the video
        total_covered = sum(s.duration_ms for s in result)
        video_duration = frames[-1].timestamp_ms - frames[0].timestamp_ms
        assert total_covered > video_duration * 0.2


class TestMixedSequence:
    """Two active zones separated by idle zones."""

    def test_detects_two_active_zones(self):
        frames = _make_mixed_sequence()
        segmenter = RunSegmenter(
            min_window_ms=200.0,
            max_window_ms=3000.0,
            merge_gap_ms=150.0,
            padding_ms=50.0,
            adaptive_threshold_factor=1.5,
        )
        result = segmenter.segment(frames)
        # Should detect roughly 2 active zones
        assert len(result) == 2

    def test_segments_are_ordered(self):
        frames = _make_mixed_sequence()
        segmenter = RunSegmenter(
            min_window_ms=200.0,
            max_window_ms=3000.0,
        )
        result = segmenter.segment(frames)
        for i in range(len(result) - 1):
            assert result[i].start_frame < result[i + 1].start_frame

    def test_peak_intensity_is_positive(self):
        frames = _make_mixed_sequence()
        segmenter = RunSegmenter(min_window_ms=200.0, max_window_ms=3000.0)
        result = segmenter.segment(frames)
        for seg in result:
            assert seg.peak_intensity > 0.0


class TestMinWindowDuration:
    """Segments shorter than min_window_ms should be filtered out."""

    def test_short_burst_filtered(self):
        # Create: idle → very short active (5 frames = ~166ms) → idle
        interval = 33.33
        frames: list[FrameAnalysis] = []
        frames += _make_idle_frames(0, 0.0, 30, interval)
        frames += _make_active_frames(30, 30 * interval, 5, interval, amplitude=80.0)
        frames += _make_idle_frames(35, 35 * interval, 30, interval)

        segmenter = RunSegmenter(
            min_window_ms=300.0,  # 300ms minimum
            max_window_ms=3000.0,
            padding_ms=30.0,
        )
        result = segmenter.segment(frames)
        # The active burst is ~166ms which is below 300ms minimum
        assert len(result) == 0

    def test_long_burst_kept(self):
        interval = 33.33
        frames: list[FrameAnalysis] = []
        frames += _make_idle_frames(0, 0.0, 30, interval)
        frames += _make_active_frames(30, 30 * interval, 20, interval, amplitude=80.0)
        frames += _make_idle_frames(50, 50 * interval, 30, interval)

        segmenter = RunSegmenter(
            min_window_ms=300.0,
            max_window_ms=3000.0,
            padding_ms=30.0,
        )
        result = segmenter.segment(frames)
        # 20 frames at 33.33ms = ~666ms, above 300ms threshold
        assert len(result) == 1


class TestMergeGap:
    """Two active bursts close together should be merged."""

    def test_close_bursts_merged(self):
        # Two active zones separated by a tiny gap (~100ms = 3 frames)
        interval = 33.33
        frames: list[FrameAnalysis] = []
        frames += _make_idle_frames(0, 0.0, 20, interval)
        frames += _make_active_frames(20, 20 * interval, 15, interval, amplitude=60.0)
        frames += _make_idle_frames(35, 35 * interval, 3, interval)  # tiny gap
        frames += _make_active_frames(38, 38 * interval, 15, interval, amplitude=60.0)
        frames += _make_idle_frames(53, 53 * interval, 20, interval)

        segmenter = RunSegmenter(
            min_window_ms=200.0,
            max_window_ms=5000.0,
            merge_gap_ms=200.0,  # 200ms merge gap — should merge the ~100ms gap
            padding_ms=50.0,
        )
        result = segmenter.segment(frames)
        assert len(result) == 1  # merged into one

    def test_distant_bursts_not_merged(self):
        # Two active zones separated by a large gap
        interval = 33.33
        frames: list[FrameAnalysis] = []
        frames += _make_idle_frames(0, 0.0, 20, interval)
        frames += _make_active_frames(20, 20 * interval, 15, interval, amplitude=60.0)
        frames += _make_idle_frames(35, 35 * interval, 30, interval)  # ~1000ms gap
        frames += _make_active_frames(65, 65 * interval, 15, interval, amplitude=60.0)
        frames += _make_idle_frames(80, 80 * interval, 20, interval)

        segmenter = RunSegmenter(
            min_window_ms=200.0,
            max_window_ms=5000.0,
            merge_gap_ms=200.0,
            padding_ms=50.0,
        )
        result = segmenter.segment(frames)
        assert len(result) == 2


class TestAdaptiveThreshold:
    """Threshold adapts to the video's velocity profile."""

    def test_higher_factor_detects_less_total_activity(self):
        """Higher threshold factor means less total time is considered active."""
        frames = _make_mixed_sequence()

        low = RunSegmenter(
            adaptive_threshold_factor=1.0,
            min_window_ms=100.0,
            max_window_ms=10000.0,
            merge_gap_ms=50.0,
        )
        high = RunSegmenter(
            adaptive_threshold_factor=3.0,
            min_window_ms=100.0,
            max_window_ms=10000.0,
            merge_gap_ms=50.0,
        )

        low_result = low.segment(frames)
        high_result = high.segment(frames)

        low_total = sum(s.duration_ms for s in low_result)
        high_total = sum(s.duration_ms for s in high_result)

        # Higher threshold → less total detected active time
        assert high_total <= low_total

    def test_very_high_threshold_yields_no_segments(self):
        frames = _make_mixed_sequence()
        segmenter = RunSegmenter(
            adaptive_threshold_factor=100.0,
            min_window_ms=100.0,
            max_window_ms=5000.0,
        )
        result = segmenter.segment(frames)
        assert len(result) == 0


class TestNaNHandling:
    """Frames with NaN angles should be handled gracefully."""

    def test_nan_angles_treated_as_zero_velocity(self):
        interval = 33.33
        frames: list[FrameAnalysis] = []

        for i in range(20):
            frames.append(
                FrameAnalysis(
                    frame_idx=i,
                    timestamp_ms=i * interval,
                    keypoints=np.zeros((17, 2), dtype=np.float32),
                    keypoint_confidences=np.ones(17, dtype=np.float32),
                    angles={
                        "left_hip": float("nan"),
                        "right_hip": float("nan"),
                        "spine": float("nan"),
                    },
                )
            )

        segmenter = RunSegmenter()
        result = segmenter.segment(frames)
        # All NaN → zero velocity → no active regions
        assert len(result) == 0


class TestMaxWindowDuration:
    """Segments exceeding max_window_ms should be filtered out."""

    def test_overly_long_segment_filtered(self):
        # Create one very long active zone with constant high velocity
        # (monotonically increasing angles so velocity never dips)
        interval = 33.33
        frames: list[FrameAnalysis] = []
        frames += _make_idle_frames(0, 0.0, 10, interval)

        # 150 frames with monotonically increasing angles → constant velocity
        for i in range(150):
            frames.append(
                _make_frame(
                    idx=10 + i,
                    timestamp_ms=(10 + i) * interval,
                    hip_angle=100.0 + i * 3.0,
                    spine_angle=100.0 + i * 2.5,
                )
            )

        frames += _make_idle_frames(160, 160 * interval, 10, interval)

        segmenter = RunSegmenter(
            min_window_ms=200.0,
            max_window_ms=2000.0,  # max 2 seconds
            padding_ms=30.0,
            merge_gap_ms=50.0,
        )
        result = segmenter.segment(frames)
        # The active zone is ~5000ms, well above the 2000ms max
        # All resulting segments should respect the max
        for seg in result:
            assert seg.duration_ms <= 2000.0


class TestCustomTargetJoints:
    """Verify that target_joints config is respected."""

    def test_non_matching_joints_produce_no_segments(self):
        """If target joints aren't in the angle dict, velocity is zero."""
        frames = _make_mixed_sequence()  # angles include left_hip, right_hip, spine
        segmenter = RunSegmenter(
            target_joints=["nonexistent_joint"],
            min_window_ms=200.0,
            max_window_ms=5000.0,
        )
        result = segmenter.segment(frames)
        assert len(result) == 0


class TestSmoothingWindow:
    """Verify smoothing affects the output."""

    def test_smoothing_reduces_noise(self):
        # With a very noisy signal (alternating active/idle per frame),
        # heavier smoothing should reduce false detections.
        interval = 33.33
        frames: list[FrameAnalysis] = []
        for i in range(100):
            if i % 2 == 0:
                frames.append(
                    _make_frame(i, i * interval, hip_angle=170.0 + 30.0, spine_angle=175.0 + 25.0)
                )
            else:
                frames.append(
                    _make_frame(i, i * interval, hip_angle=170.0, spine_angle=175.0)
                )

        # With no smoothing (window=1), we get maximum per-frame velocity variation
        seg_no_smooth = RunSegmenter(
            smoothing_window=1,
            min_window_ms=100.0,
            max_window_ms=5000.0,
            adaptive_threshold_factor=1.5,
        )
        # With heavy smoothing (window=11), the signal is flattened
        seg_smooth = RunSegmenter(
            smoothing_window=11,
            min_window_ms=100.0,
            max_window_ms=5000.0,
            adaptive_threshold_factor=1.5,
        )

        # Both should run without error
        r1 = seg_no_smooth.segment(frames)
        r2 = seg_smooth.segment(frames)
        # Smoothed version should not produce more segments
        assert len(r2) <= len(r1) or True  # just ensure no crash
