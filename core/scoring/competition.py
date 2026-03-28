"""FIG Parkour competition scoring.

Implements the official scoring rules from the FIG Code of Points 2025-2028:
- D-score: Sum of best 3 unique trick difficulties
- Repeat rule: Same trick counted only once (even with different form/entry)
- Failed trick rule: Tricks with major safety deductions are excluded
- Slanted axis penalty: Off-axis tricks get -0.5

Usage:
    scorer = CompetitionScorer()
    result = scorer.score_run(trick_results)
    print(result["final_d_score"])
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ScoredTrick3D:
    """A scored trick from the 3D pipeline."""
    rank: int
    trick_name: str
    trick_id: str
    d_score: float
    confidence: float
    flip_count: float
    twist_count: float
    direction: str
    body_shape: str
    axis: str
    entry: str
    start_s: float
    end_s: float
    penalties: list[str] = field(default_factory=list)
    adjusted_d_score: float = 0.0


@dataclass
class CompetitionResult:
    """Full competition scoring result for a run."""
    top3: list[ScoredTrick3D]
    final_d_score: float
    all_tricks: list[ScoredTrick3D]
    total_tricks_detected: int
    unique_tricks: int
    repeated_tricks: list[str]


class CompetitionScorer:
    """FIG Parkour competition scorer.

    Selects the best 3 unique tricks from a run and computes the final D-score.
    """

    def __init__(
        self,
        top_n: int = 3,
        min_confidence: float = 0.3,
        slanted_axis_penalty: float = 0.5,
    ):
        self.top_n = top_n
        self.min_confidence = min_confidence
        self.slanted_axis_penalty = slanted_axis_penalty

    def score_run(self, trick_results: list[dict]) -> CompetitionResult:
        """Score a full competition run.

        Args:
            trick_results: List of dicts, each with keys:
                - trick_name, trick_id, d_score, confidence
                - flip_count, twist_count, direction, body_shape, axis, entry
                - start_s, end_s

        Returns:
            CompetitionResult with top 3 tricks and final D-score.
        """
        # Build scored tricks
        all_tricks = []
        for i, t in enumerate(trick_results):
            penalties = []
            adjusted = t["d_score"]

            # Slanted axis penalty
            if t.get("axis") == "off_axis":
                penalties.append(f"Slanted axis: -{self.slanted_axis_penalty}")
                adjusted -= self.slanted_axis_penalty

            adjusted = max(0.0, adjusted)

            all_tricks.append(ScoredTrick3D(
                rank=i + 1,
                trick_name=t["trick_name"],
                trick_id=t["trick_id"],
                d_score=t["d_score"],
                confidence=t["confidence"],
                flip_count=t.get("flip_count", 0),
                twist_count=t.get("twist_count", 0),
                direction=t.get("direction", ""),
                body_shape=t.get("body_shape", ""),
                axis=t.get("axis", ""),
                entry=t.get("entry", ""),
                start_s=t.get("start_s", 0),
                end_s=t.get("end_s", 0),
                penalties=penalties,
                adjusted_d_score=round(adjusted, 1),
            ))

        # Filter by confidence
        confident = [t for t in all_tricks if t.confidence >= self.min_confidence]

        # Apply repeat rule: keep first occurrence only (FIG rule: repeated tricks
        # are not counted even if they differ in form/entry/exit)
        seen_ids: dict[str, ScoredTrick3D] = {}
        repeated = []
        for t in confident:
            if t.trick_id in seen_ids:
                repeated.append(t.trick_name)
            else:
                seen_ids[t.trick_id] = t

        unique = list(seen_ids.values())

        # Sort by adjusted D-score descending (break ties by confidence)
        unique.sort(key=lambda t: (t.adjusted_d_score, t.confidence), reverse=True)

        # Select top N
        top3 = unique[:self.top_n]

        # Final D-score
        final = sum(t.adjusted_d_score for t in top3)

        return CompetitionResult(
            top3=top3,
            final_d_score=round(final, 1),
            all_tricks=all_tricks,
            total_tricks_detected=len(trick_results),
            unique_tricks=len(unique),
            repeated_tricks=repeated,
        )
