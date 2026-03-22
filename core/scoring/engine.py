"""Scoring engine — selects top 3 tricks by difficulty and computes total score."""

from __future__ import annotations

from core.models import ScoreResult, ScoredTrick, TrickDetection


class ScoringEngine:
    """Selects top 3 tricks by difficulty from detections and computes scores."""

    def __init__(self, confidence_threshold: float = 0.5, top_n: int = 3):
        self.confidence_threshold = confidence_threshold
        self.top_n = top_n

    def score(
        self,
        detections: list[TrickDetection],
        trick_difficulties: dict[str, float],
    ) -> ScoreResult:
        """Score a list of detections.

        Args:
            detections: All detected tricks from the classifier.
            trick_difficulties: Mapping of trick_id → difficulty rating.

        Returns:
            ScoreResult with top 3 tricks and total score.
        """
        # Filter by confidence
        confident = [d for d in detections if d.confidence >= self.confidence_threshold]

        # Deduplicate overlapping detections (same trick in nearby time)
        deduped = self._deduplicate(confident)

        # Sort by difficulty (descending), break ties by confidence
        deduped.sort(key=lambda d: (trick_difficulties.get(d.trick_id, 0), d.confidence), reverse=True)

        # Select top N
        top = deduped[: self.top_n]

        # Build scored tricks
        scored: list[ScoredTrick] = []
        for det in top:
            difficulty = trick_difficulties.get(det.trick_id, 0.0)
            weighted = difficulty * det.confidence
            scored.append(
                ScoredTrick(
                    trick_id=det.trick_id,
                    trick_name=det.trick_name,
                    difficulty=difficulty,
                    confidence=det.confidence,
                    weighted_score=round(weighted, 3),
                    detection=det,
                )
            )

        total = sum(s.weighted_score for s in scored)
        max_possible = sum(s.difficulty for s in scored)

        return ScoreResult(
            top3=scored,
            total_score=round(total, 3),
            max_possible_score=round(max_possible, 3),
        )

    def _deduplicate(self, detections: list[TrickDetection]) -> list[TrickDetection]:
        """Remove overlapping detections of the same trick.

        If two detections of the same trick overlap by more than 50% in time,
        keep the one with higher confidence.
        """
        if not detections:
            return []

        # Group by trick_id
        by_trick: dict[str, list[TrickDetection]] = {}
        for det in detections:
            by_trick.setdefault(det.trick_id, []).append(det)

        deduped: list[TrickDetection] = []

        for trick_id, dets in by_trick.items():
            # Sort by confidence descending
            dets.sort(key=lambda d: d.confidence, reverse=True)

            kept: list[TrickDetection] = []
            for det in dets:
                overlaps = False
                for existing in kept:
                    if self._temporal_overlap(det, existing) > 0.5:
                        overlaps = True
                        break
                if not overlaps:
                    kept.append(det)

            deduped.extend(kept)

        return deduped

    @staticmethod
    def _temporal_overlap(a: TrickDetection, b: TrickDetection) -> float:
        """Compute IoU of two detection time ranges."""
        overlap_start = max(a.start_time_ms, b.start_time_ms)
        overlap_end = min(a.end_time_ms, b.end_time_ms)

        if overlap_start >= overlap_end:
            return 0.0

        overlap_duration = overlap_end - overlap_start
        a_duration = a.end_time_ms - a.start_time_ms
        b_duration = b.end_time_ms - b.start_time_ms
        union_duration = a_duration + b_duration - overlap_duration

        if union_duration <= 0:
            return 0.0

        return overlap_duration / union_duration
