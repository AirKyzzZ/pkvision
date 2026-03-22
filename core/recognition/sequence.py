"""Temporal sequence analysis — groups frames into trick windows."""

from __future__ import annotations

import math

from core.models import FrameAnalysis


class TrickWindow:
    """A contiguous sequence of frames that likely contains a trick."""

    def __init__(self, frames: list[FrameAnalysis], start_idx: int, end_idx: int):
        self.frames = frames
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.start_time_ms = frames[0].timestamp_ms if frames else 0.0
        self.end_time_ms = frames[-1].timestamp_ms if frames else 0.0
        self.duration_ms = self.end_time_ms - self.start_time_ms

    def __len__(self) -> int:
        return len(self.frames)


class SequenceAnalyzer:
    """Groups frames into trick windows based on movement intensity.

    Identifies segments where significant movement occurs (angular velocity spikes)
    and separates them from idle/walking frames.
    """

    def __init__(
        self,
        velocity_threshold: float = 100.0,
        min_window_frames: int = 8,
        max_gap_frames: int = 5,
        padding_frames: int = 3,
    ):
        self.velocity_threshold = velocity_threshold
        self.min_window_frames = min_window_frames
        self.max_gap_frames = max_gap_frames
        self.padding_frames = padding_frames

    def find_trick_windows(self, frames: list[FrameAnalysis]) -> list[TrickWindow]:
        """Identify trick windows from a sequence of analyzed frames."""
        if len(frames) < self.min_window_frames:
            # If the whole sequence is short, treat it as one window
            if frames:
                return [TrickWindow(frames, 0, len(frames) - 1)]
            return []

        # Compute movement intensity per frame
        intensities = self._compute_intensities(frames)

        # Find active regions (above threshold)
        active_regions = self._find_active_regions(intensities)

        # Merge nearby regions and apply padding
        merged = self._merge_regions(active_regions, len(frames))

        # Convert to TrickWindows
        windows: list[TrickWindow] = []
        for start, end in merged:
            if end - start + 1 >= self.min_window_frames:
                window_frames = frames[start : end + 1]
                windows.append(TrickWindow(window_frames, start, end))

        # If no windows found, return the entire sequence as one window
        if not windows:
            return [TrickWindow(frames, 0, len(frames) - 1)]

        return windows

    def _compute_intensities(self, frames: list[FrameAnalysis]) -> list[float]:
        """Compute a scalar movement intensity per frame based on angle changes."""
        intensities = [0.0]

        for i in range(1, len(frames)):
            total_change = 0.0
            count = 0

            for joint in frames[i].angles:
                curr = frames[i].angles.get(joint, float("nan"))
                prev = frames[i - 1].angles.get(joint, float("nan"))

                if math.isnan(curr) or math.isnan(prev):
                    continue

                dt_s = (frames[i].timestamp_ms - frames[i - 1].timestamp_ms) / 1000.0
                if dt_s > 0:
                    velocity = abs(curr - prev) / dt_s
                    total_change += velocity
                    count += 1

            avg_velocity = total_change / count if count > 0 else 0.0
            intensities.append(avg_velocity)

        return intensities

    def _find_active_regions(self, intensities: list[float]) -> list[tuple[int, int]]:
        """Find contiguous regions where intensity exceeds threshold."""
        regions: list[tuple[int, int]] = []
        in_region = False
        start = 0

        for i, intensity in enumerate(intensities):
            if intensity >= self.velocity_threshold:
                if not in_region:
                    start = i
                    in_region = True
            else:
                if in_region:
                    regions.append((start, i - 1))
                    in_region = False

        if in_region:
            regions.append((start, len(intensities) - 1))

        return regions

    def _merge_regions(
        self, regions: list[tuple[int, int]], total_frames: int
    ) -> list[tuple[int, int]]:
        """Merge nearby regions and apply padding."""
        if not regions:
            return []

        # Add padding
        padded = []
        for start, end in regions:
            padded_start = max(0, start - self.padding_frames)
            padded_end = min(total_frames - 1, end + self.padding_frames)
            padded.append((padded_start, padded_end))

        # Merge overlapping/nearby regions
        merged = [padded[0]]
        for start, end in padded[1:]:
            prev_start, prev_end = merged[-1]
            if start <= prev_end + self.max_gap_frames:
                merged[-1] = (prev_start, max(prev_end, end))
            else:
                merged.append((start, end))

        return merged
