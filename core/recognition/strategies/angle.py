"""Angle threshold detection strategy — baseline/fallback for PoC."""

from __future__ import annotations

import math

from core.models import (
    AngleMatch,
    FrameAnalysis,
    TrickConfig,
    TrickDetection,
    TrickPhase,
)


class AngleThresholdStrategy:
    """Detects tricks by matching angle rules across frame sequences using a sliding window.

    For each phase of a trick, checks if the angle rules are satisfied
    in a contiguous block of frames. Reports confidence based on how
    many rules match and how closely.
    """

    def __init__(self, min_phase_confidence: float = 0.5):
        self.min_phase_confidence = min_phase_confidence

    def evaluate(
        self,
        trick: TrickConfig,
        frames: list[FrameAnalysis],
    ) -> TrickDetection | None:
        if not frames or not trick.phases:
            return None

        # Try to match trick phases sequentially using a sliding window
        best_match = self._find_best_match(trick, frames)

        if best_match is None:
            return None

        start_frame, end_frame, confidence, angle_matches, phase_confidences = best_match

        if confidence < self.min_phase_confidence:
            return None

        return TrickDetection(
            trick_id=trick.trick_id,
            trick_name=trick.get_name("en"),
            confidence=confidence,
            start_frame=frames[start_frame].frame_idx,
            end_frame=frames[end_frame].frame_idx,
            start_time_ms=frames[start_frame].timestamp_ms,
            end_time_ms=frames[end_frame].timestamp_ms,
            strategy_used="angle_threshold",
            angle_matches=angle_matches,
            phase_confidences=phase_confidences,
        )

    def _find_best_match(
        self,
        trick: TrickConfig,
        frames: list[FrameAnalysis],
    ) -> tuple[int, int, float, list[AngleMatch], dict[str, float]] | None:
        """Slide through frames trying to match all phases sequentially."""
        best: tuple[int, int, float, list[AngleMatch], dict[str, float]] | None = None

        num_frames = len(frames)
        phases = trick.phases

        # Estimate minimum frames needed per phase (at ~30fps)
        fps_estimate = 30.0
        if num_frames >= 2:
            dt = frames[1].timestamp_ms - frames[0].timestamp_ms
            if dt > 0:
                fps_estimate = 1000.0 / dt

        for start_idx in range(num_frames):
            result = self._try_match_from(trick, frames, start_idx, fps_estimate)
            if result is not None:
                _, _, conf, _, _ = result
                if best is None or conf > best[2]:
                    best = result

        return best

    def _try_match_from(
        self,
        trick: TrickConfig,
        frames: list[FrameAnalysis],
        start_idx: int,
        fps: float,
    ) -> tuple[int, int, float, list[AngleMatch], dict[str, float]] | None:
        """Try to match all phases starting from a given frame index."""
        current_idx = start_idx
        all_angle_matches: list[AngleMatch] = []
        phase_confidences: dict[str, float] = {}

        for phase in trick.phases:
            min_frames = max(1, int(phase.duration_range_ms[0] / 1000.0 * fps))
            max_frames = max(min_frames, int(phase.duration_range_ms[1] / 1000.0 * fps))

            if current_idx + min_frames > len(frames):
                return None

            # Find the best matching window within the allowed duration range
            phase_result = self._match_phase(
                phase, frames, current_idx, min_frames, max_frames
            )

            if phase_result is None:
                return None

            phase_end, phase_conf, phase_matches = phase_result
            phase_confidences[phase.name] = phase_conf
            all_angle_matches.extend(phase_matches)
            current_idx = phase_end + 1

        end_idx = current_idx - 1
        if end_idx >= len(frames):
            end_idx = len(frames) - 1

        overall_confidence = (
            sum(phase_confidences.values()) / len(phase_confidences)
            if phase_confidences
            else 0.0
        )

        return start_idx, end_idx, overall_confidence, all_angle_matches, phase_confidences

    def _match_phase(
        self,
        phase: TrickPhase,
        frames: list[FrameAnalysis],
        start_idx: int,
        min_frames: int,
        max_frames: int,
    ) -> tuple[int, float, list[AngleMatch]] | None:
        """Find the best matching window for a single phase."""
        if not phase.angle_rules:
            # Phase with no rules always matches
            end = min(start_idx + min_frames - 1, len(frames) - 1)
            return end, 1.0, []

        best_end = None
        best_conf = 0.0
        best_matches: list[AngleMatch] = []

        end_limit = min(start_idx + max_frames, len(frames))

        for end_idx in range(start_idx + min_frames - 1, end_limit):
            # Check angle rules across this window
            window = frames[start_idx : end_idx + 1]
            conf, matches = self._evaluate_angle_rules(phase, window)

            if conf > best_conf:
                best_conf = conf
                best_end = end_idx
                best_matches = matches

        if best_end is None or best_conf < self.min_phase_confidence:
            return None

        return best_end, best_conf, best_matches

    def _evaluate_angle_rules(
        self,
        phase: TrickPhase,
        window: list[FrameAnalysis],
    ) -> tuple[float, list[AngleMatch]]:
        """Evaluate angle rules across a window of frames.

        A rule is "matched" if ANY frame in the window satisfies it.
        Confidence is the fraction of rules matched, weighted by how
        close the angles are to the rule's midpoint.
        """
        if not phase.angle_rules:
            return 1.0, []

        matches: list[AngleMatch] = []
        total_score = 0.0

        for rule in phase.angle_rules:
            best_match = self._best_angle_match_in_window(rule.joint, rule.min, rule.max, window)
            matches.append(best_match)
            if best_match.matched:
                # Score based on how centered the angle is in the rule range
                midpoint = (rule.min + rule.max) / 2.0
                range_half = (rule.max - rule.min) / 2.0
                if range_half > 0:
                    distance = abs(best_match.measured - midpoint) / range_half
                    total_score += max(0.0, 1.0 - distance * 0.5)
                else:
                    total_score += 1.0

        confidence = total_score / len(phase.angle_rules) if phase.angle_rules else 0.0
        return min(confidence, 1.0), matches

    def _best_angle_match_in_window(
        self,
        joint: str,
        rule_min: float,
        rule_max: float,
        window: list[FrameAnalysis],
    ) -> AngleMatch:
        """Find the frame in the window that best matches an angle rule."""
        best_measured = float("nan")
        best_distance = float("inf")
        matched = False

        midpoint = (rule_min + rule_max) / 2.0

        for frame in window:
            angle = frame.angles.get(joint, float("nan"))
            if math.isnan(angle):
                continue

            if rule_min <= angle <= rule_max:
                distance = abs(angle - midpoint)
                if not matched or distance < best_distance:
                    best_measured = angle
                    best_distance = distance
                    matched = True
            elif not matched:
                # Track closest non-matching angle
                distance = min(abs(angle - rule_min), abs(angle - rule_max))
                if distance < best_distance:
                    best_measured = angle
                    best_distance = distance

        return AngleMatch(
            joint=joint,
            measured=best_measured if not math.isnan(best_measured) else -1.0,
            rule_min=rule_min,
            rule_max=rule_max,
            matched=matched,
        )
