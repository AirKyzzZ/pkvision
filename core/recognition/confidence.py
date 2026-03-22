"""Multi-factor confidence scoring for trick detections."""

from __future__ import annotations

import math

from core.models import FrameAnalysis, TrickDetection


def compute_multi_factor_confidence(
    detection: TrickDetection,
    frames: list[FrameAnalysis],
) -> float:
    """Compute a refined confidence score using multiple factors.

    Factors:
    1. Base detection confidence (from strategy)
    2. Keypoint quality (average confidence of keypoints in detection window)
    3. Temporal consistency (how stable are the angles across the window)
    4. Phase coverage (how many phases were detected)
    """
    factors: list[tuple[float, float]] = []  # (score, weight)

    # Factor 1: Base detection confidence
    factors.append((detection.confidence, 0.4))

    # Factor 2: Keypoint quality in detection window
    kp_quality = _keypoint_quality(frames, detection.start_frame, detection.end_frame)
    factors.append((kp_quality, 0.2))

    # Factor 3: Phase coverage
    if detection.phase_confidences:
        phase_coverage = sum(1 for v in detection.phase_confidences.values() if v > 0.3)
        phase_score = phase_coverage / max(len(detection.phase_confidences), 1)
        factors.append((phase_score, 0.2))

    # Factor 4: Angle match quality
    if detection.angle_matches:
        matched = sum(1 for m in detection.angle_matches if m.matched)
        match_ratio = matched / len(detection.angle_matches)
        factors.append((match_ratio, 0.2))

    # Weighted average
    total_weight = sum(w for _, w in factors)
    if total_weight == 0:
        return detection.confidence

    weighted_sum = sum(score * weight for score, weight in factors)
    return min(weighted_sum / total_weight, 1.0)


def _keypoint_quality(
    frames: list[FrameAnalysis],
    start_frame: int,
    end_frame: int,
) -> float:
    """Compute average keypoint confidence in the detection window."""
    total_conf = 0.0
    count = 0

    for frame in frames:
        if frame.frame_idx < start_frame or frame.frame_idx > end_frame:
            continue

        if frame.keypoint_confidences is not None:
            confs = frame.keypoint_confidences
            if hasattr(confs, '__len__'):
                valid_confs = [c for c in confs if not math.isnan(float(c))]
                if valid_confs:
                    total_conf += sum(valid_confs) / len(valid_confs)
                    count += 1

    return total_conf / count if count > 0 else 0.0
