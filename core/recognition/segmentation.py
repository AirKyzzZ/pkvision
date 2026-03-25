"""Improved temporal segmentation — segments a full run into trick attempts.

Uses targeted hip + spine angular velocity signals with adaptive thresholding
to detect acrobatic activity zones more reliably than the generic windowing
in sequence.py.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

from core.models import FrameAnalysis

logger = logging.getLogger(__name__)

# Default joints to track — hips and spine are the most reliable indicators
# of acrobatic movement (rotation, inversion, flipping).
DEFAULT_TARGET_JOINTS = ["left_hip", "right_hip", "spine"]


@dataclass
class TrickSegment:
    """A detected active zone in a video that likely contains a trick."""

    start_frame: int
    end_frame: int
    start_time_ms: float
    end_time_ms: float
    peak_intensity: float  # max angular velocity in this segment

    @property
    def duration_ms(self) -> float:
        return self.end_time_ms - self.start_time_ms

    @property
    def n_frames(self) -> int:
        return self.end_frame - self.start_frame + 1


class RunSegmenter:
    """Segments a full competition run into individual trick attempts.

    Strategy:
      1. Compute per-frame angular velocity for target joints (hip + spine)
      2. Smooth the signal with a moving average
      3. Compute adaptive threshold = factor x median of non-zero velocities
      4. Find contiguous regions above threshold
      5. Apply padding and merge nearby regions
      6. Filter by minimum/maximum duration
    """

    def __init__(
        self,
        target_joints: list[str] | None = None,
        min_window_ms: float = 300.0,
        max_window_ms: float = 3000.0,
        merge_gap_ms: float = 150.0,
        padding_ms: float = 100.0,
        adaptive_threshold_factor: float = 1.5,
        smoothing_window: int = 5,
    ):
        self.target_joints = target_joints if target_joints is not None else list(DEFAULT_TARGET_JOINTS)
        self.min_window_ms = min_window_ms
        self.max_window_ms = max_window_ms
        self.merge_gap_ms = merge_gap_ms
        self.padding_ms = padding_ms
        self.adaptive_threshold_factor = adaptive_threshold_factor
        self.smoothing_window = smoothing_window

    def segment(self, frames: list[FrameAnalysis]) -> list[TrickSegment]:
        """Segment a full run into individual trick windows.

        Args:
            frames: Ordered list of FrameAnalysis objects for the full video.

        Returns:
            List of TrickSegment objects, sorted by start_frame.
        """
        if not frames:
            logger.debug("No frames provided — returning empty segments.")
            return []

        if len(frames) < 2:
            logger.debug("Single frame — cannot compute velocities, returning empty.")
            return []

        # 1. Compute per-frame angular velocity for target joints
        velocities = self._compute_target_velocities(frames)

        # 2. Smooth the signal
        smoothed = self._smooth(velocities)

        # 3. Compute adaptive threshold
        threshold = self._compute_threshold(smoothed)
        if threshold <= 0.0:
            logger.debug("Adaptive threshold is zero — no meaningful activity detected.")
            return []

        logger.debug("Adaptive threshold: %.2f deg/s", threshold)

        # 4. Find contiguous regions above threshold
        regions = self._find_active_regions(smoothed, threshold)
        if not regions:
            logger.debug("No active regions found above threshold.")
            return []

        # 5. Apply padding and merge nearby regions
        regions = self._apply_padding(regions, frames)
        regions = self._merge_nearby(regions, frames)

        # 6. Filter by duration and build TrickSegment objects
        segments = self._build_segments(regions, frames, smoothed)

        logger.debug("Segmented %d trick windows from %d frames.", len(segments), len(frames))
        return segments

    def _compute_target_velocities(self, frames: list[FrameAnalysis]) -> list[float]:
        """Compute per-frame angular velocity using only target joints."""
        velocities = [0.0]  # First frame has no velocity

        for i in range(1, len(frames)):
            total_velocity = 0.0
            count = 0

            dt_s = (frames[i].timestamp_ms - frames[i - 1].timestamp_ms) / 1000.0
            if dt_s <= 0.0:
                velocities.append(0.0)
                continue

            for joint in self.target_joints:
                curr = frames[i].angles.get(joint, float("nan"))
                prev = frames[i - 1].angles.get(joint, float("nan"))

                if math.isnan(curr) or math.isnan(prev):
                    continue

                velocity = abs(curr - prev) / dt_s
                total_velocity += velocity
                count += 1

            avg = total_velocity / count if count > 0 else 0.0
            velocities.append(avg)

        return velocities

    def _smooth(self, signal: list[float]) -> list[float]:
        """Apply a simple moving average to smooth the velocity signal."""
        n = len(signal)
        if n == 0:
            return []

        half_w = self.smoothing_window // 2
        smoothed: list[float] = []

        for i in range(n):
            start = max(0, i - half_w)
            end = min(n, i + half_w + 1)
            window = signal[start:end]
            smoothed.append(sum(window) / len(window))

        return smoothed

    def _compute_threshold(self, velocities: list[float]) -> float:
        """Compute adaptive threshold = factor x median of non-zero velocities."""
        non_zero = [v for v in velocities if v > 0.0]
        if not non_zero:
            return 0.0

        non_zero.sort()
        mid = len(non_zero) // 2
        if len(non_zero) % 2 == 0:
            median = (non_zero[mid - 1] + non_zero[mid]) / 2.0
        else:
            median = non_zero[mid]

        return self.adaptive_threshold_factor * median

    def _find_active_regions(
        self, velocities: list[float], threshold: float
    ) -> list[tuple[int, int]]:
        """Find contiguous frame ranges where velocity exceeds threshold."""
        regions: list[tuple[int, int]] = []
        in_region = False
        start = 0

        for i, v in enumerate(velocities):
            if v >= threshold:
                if not in_region:
                    start = i
                    in_region = True
            else:
                if in_region:
                    regions.append((start, i - 1))
                    in_region = False

        if in_region:
            regions.append((start, len(velocities) - 1))

        return regions

    def _apply_padding(
        self, regions: list[tuple[int, int]], frames: list[FrameAnalysis]
    ) -> list[tuple[int, int]]:
        """Expand each region by padding_ms on both sides."""
        if not regions:
            return []

        n = len(frames)
        padded: list[tuple[int, int]] = []

        for start, end in regions:
            # Expand start backward by padding_ms
            new_start = start
            while new_start > 0:
                gap = frames[start].timestamp_ms - frames[new_start - 1].timestamp_ms
                if gap > self.padding_ms:
                    break
                new_start -= 1

            # Expand end forward by padding_ms
            new_end = end
            while new_end < n - 1:
                gap = frames[new_end + 1].timestamp_ms - frames[end].timestamp_ms
                if gap > self.padding_ms:
                    break
                new_end += 1

            padded.append((new_start, new_end))

        return padded

    def _merge_nearby(
        self, regions: list[tuple[int, int]], frames: list[FrameAnalysis]
    ) -> list[tuple[int, int]]:
        """Merge regions that are closer than merge_gap_ms."""
        if len(regions) <= 1:
            return list(regions)

        merged = [regions[0]]

        for start, end in regions[1:]:
            prev_start, prev_end = merged[-1]
            gap_ms = frames[start].timestamp_ms - frames[prev_end].timestamp_ms
            if gap_ms <= self.merge_gap_ms:
                merged[-1] = (prev_start, max(prev_end, end))
            else:
                merged.append((start, end))

        return merged

    def _build_segments(
        self,
        regions: list[tuple[int, int]],
        frames: list[FrameAnalysis],
        velocities: list[float],
    ) -> list[TrickSegment]:
        """Convert frame regions to TrickSegment objects, filtering by duration."""
        segments: list[TrickSegment] = []

        for start, end in regions:
            start_ms = frames[start].timestamp_ms
            end_ms = frames[end].timestamp_ms
            duration = end_ms - start_ms

            if duration < self.min_window_ms:
                logger.debug(
                    "Skipping segment [%d-%d] — too short (%.0f ms < %.0f ms)",
                    start, end, duration, self.min_window_ms,
                )
                continue

            if duration > self.max_window_ms:
                logger.debug(
                    "Skipping segment [%d-%d] — too long (%.0f ms > %.0f ms)",
                    start, end, duration, self.max_window_ms,
                )
                continue

            peak = max(velocities[start : end + 1])

            segments.append(
                TrickSegment(
                    start_frame=start,
                    end_frame=end,
                    start_time_ms=start_ms,
                    end_time_ms=end_ms,
                    peak_intensity=peak,
                )
            )

        return segments
