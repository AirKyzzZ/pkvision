"""Multi-model consensus engine for trick identification.

Sends the same clip to N models, aggregates results, and picks
the consensus answer. This is more reliable than any single model.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from core.vlm.base import VLMResponse, VLMTrickResult
from core.vlm.fig_matcher import FIGMatcher
from core.vlm.openrouter_provider import OpenRouterProvider


@dataclass
class ModelVote:
    """A single model's vote on a trick."""
    model: str
    trick_name: str
    fig_name: str
    d_score: float
    confidence: str
    reasoning: str
    flip_count: float = 0.0
    twist_count: float = 0.0


@dataclass
class ConsensusResult:
    """Aggregated result from multiple models."""
    consensus_name: str          # The agreed-upon trick name
    d_score: float
    agreement: float             # 0-1, fraction of models that agree
    confidence: str              # high (3/3), medium (2/3), low (no consensus)
    votes: list[ModelVote] = field(default_factory=list)
    total_input_tokens: int = 0
    total_output_tokens: int = 0


class ConsensusJudge:
    """Run multiple models and aggregate their trick identifications."""

    def __init__(
        self,
        models: list[str],
        api_key: str | None = None,
    ):
        self.models = models
        self.api_key = api_key
        self.matcher = FIGMatcher()

    def judge(self, video_path: Path) -> ConsensusResult:
        """Send clip to all models and return consensus."""
        video_path = Path(video_path)
        votes: list[ModelVote] = []
        total_in = 0
        total_out = 0

        for model_name in self.models:
            print(f"    [{model_name}]...", end=" ", flush=True)
            try:
                provider = OpenRouterProvider(
                    model=model_name,
                    api_key=self.api_key,
                )
                response: VLMResponse = provider.analyze_trick(video_path)
                total_in += response.input_tokens
                total_out += response.output_tokens

                if response.raw_text.startswith("ERROR:"):
                    print(f"error: {response.raw_text[:80]}")
                    continue

                for trick in response.tricks:
                    fig_match = self.matcher.match(
                        trick.trick_name,
                        flip_count=trick.flip_count,
                        twist_count=trick.twist_count,
                        direction=trick.direction,
                    )
                    vote = ModelVote(
                        model=model_name,
                        trick_name=trick.trick_name,
                        fig_name=fig_match.fig_name if fig_match else trick.trick_name,
                        d_score=fig_match.d_score if fig_match else 0.0,
                        confidence=trick.confidence,
                        reasoning=trick.reasoning,
                        flip_count=trick.flip_count,
                        twist_count=trick.twist_count,
                    )
                    votes.append(vote)
                    print(f"{vote.fig_name}", end=" ", flush=True)

            except Exception as e:
                print(f"failed: {e}")
                continue

            print()

        return self._aggregate(votes, total_in, total_out)

    def _aggregate(
        self,
        votes: list[ModelVote],
        total_in: int,
        total_out: int,
    ) -> ConsensusResult:
        """Pick consensus from model votes."""
        if not votes:
            return ConsensusResult(
                consensus_name="Unknown",
                d_score=0.0,
                agreement=0.0,
                confidence="none",
                votes=votes,
                total_input_tokens=total_in,
                total_output_tokens=total_out,
            )

        # Count votes by FIG name
        name_counts = Counter(v.fig_name for v in votes)
        total_votes = len(votes)

        # Pick the most common name
        winner_name, winner_count = name_counts.most_common(1)[0]
        agreement = winner_count / total_votes

        # Get D-score from any vote with the winning name
        winner_votes = [v for v in votes if v.fig_name == winner_name]
        d_score = winner_votes[0].d_score if winner_votes else 0.0

        # Confidence based on agreement level
        if agreement >= 0.8:
            confidence = "high"
        elif agreement >= 0.5:
            confidence = "medium"
        else:
            confidence = "low"

        return ConsensusResult(
            consensus_name=winner_name,
            d_score=d_score,
            agreement=agreement,
            confidence=confidence,
            votes=votes,
            total_input_tokens=total_in,
            total_output_tokens=total_out,
        )
