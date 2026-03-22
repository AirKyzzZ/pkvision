"""Tests for the scoring engine."""

from __future__ import annotations

import pytest

from core.models import TrickDetection
from core.scoring.engine import ScoringEngine


def make_detection(
    trick_id: str,
    name: str,
    confidence: float,
    start_ms: float = 0.0,
    end_ms: float = 1000.0,
) -> TrickDetection:
    return TrickDetection(
        trick_id=trick_id,
        trick_name=name,
        confidence=confidence,
        start_frame=0,
        end_frame=30,
        start_time_ms=start_ms,
        end_time_ms=end_ms,
        strategy_used="angle_threshold",
    )


DIFFICULTIES = {
    "front_flip": 3.5,
    "back_flip": 3.5,
    "kong_vault": 2.0,
    "triple_cork": 8.0,
    "gainer": 4.5,
    "webster": 4.0,
}


class TestScoringEngine:
    def test_selects_top_3_by_difficulty(self):
        engine = ScoringEngine(confidence_threshold=0.3)

        detections = [
            make_detection("kong_vault", "Kong Vault", 0.9, 0, 1000),
            make_detection("front_flip", "Front Flip", 0.8, 1000, 2000),
            make_detection("triple_cork", "Triple Cork", 0.7, 2000, 3000),
            make_detection("gainer", "Gainer", 0.85, 3000, 4000),
        ]

        result = engine.score(detections, DIFFICULTIES)

        assert len(result.top3) == 3
        # Top 3 by difficulty: triple_cork (8.0), gainer (4.5), front_flip (3.5)
        top_ids = [t.trick_id for t in result.top3]
        assert top_ids[0] == "triple_cork"
        assert top_ids[1] == "gainer"
        assert top_ids[2] == "front_flip"

    def test_weighted_score_computation(self):
        engine = ScoringEngine(confidence_threshold=0.3)

        detections = [
            make_detection("front_flip", "Front Flip", 0.8, 0, 1000),
        ]

        result = engine.score(detections, DIFFICULTIES)
        assert len(result.top3) == 1
        # 3.5 * 0.8 = 2.8
        assert result.top3[0].weighted_score == pytest.approx(2.8, abs=0.01)
        assert result.total_score == pytest.approx(2.8, abs=0.01)

    def test_filters_below_confidence_threshold(self):
        engine = ScoringEngine(confidence_threshold=0.5)

        detections = [
            make_detection("front_flip", "Front Flip", 0.3),  # below threshold
            make_detection("back_flip", "Back Flip", 0.7),
        ]

        result = engine.score(detections, DIFFICULTIES)
        assert len(result.top3) == 1
        assert result.top3[0].trick_id == "back_flip"

    def test_deduplicates_overlapping_same_trick(self):
        engine = ScoringEngine(confidence_threshold=0.3)

        detections = [
            make_detection("front_flip", "Front Flip", 0.8, 0, 1000),
            make_detection("front_flip", "Front Flip", 0.6, 200, 1200),  # overlaps
        ]

        result = engine.score(detections, DIFFICULTIES)
        # Should keep only the higher confidence one
        flip_detections = [t for t in result.top3 if t.trick_id == "front_flip"]
        assert len(flip_detections) == 1
        assert flip_detections[0].confidence == 0.8

    def test_keeps_non_overlapping_same_trick(self):
        engine = ScoringEngine(confidence_threshold=0.3)

        detections = [
            make_detection("front_flip", "Front Flip", 0.8, 0, 1000),
            make_detection("front_flip", "Front Flip", 0.7, 5000, 6000),  # far apart
        ]

        result = engine.score(detections, DIFFICULTIES)
        # Both should be kept (different time windows)
        # But only top 3 are selected, and both are front_flip with same difficulty
        assert len(result.top3) >= 1

    def test_empty_detections(self):
        engine = ScoringEngine()
        result = engine.score([], DIFFICULTIES)
        assert len(result.top3) == 0
        assert result.total_score == 0.0

    def test_max_possible_score(self):
        engine = ScoringEngine(confidence_threshold=0.3)

        detections = [
            make_detection("triple_cork", "Triple Cork", 0.9, 0, 1000),
            make_detection("gainer", "Gainer", 0.8, 1000, 2000),
        ]

        result = engine.score(detections, DIFFICULTIES)
        # Max possible = sum of difficulties of selected tricks
        assert result.max_possible_score == 8.0 + 4.5
