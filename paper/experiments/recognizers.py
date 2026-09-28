"""Recognizer protocol for the structured benchmark harness.

A Recognizer takes a video clip and returns `RecognizerOutput` — a ranked list
of FIG candidates with optional structured cues. This lets any recognizer
(VLM, heuristic, structured-decoder, future learned models) be evaluated by
the same harness.

The existing VLM runners in `run_benchmark.py` stay untouched. This module
adapts them to the protocol so they can be benchmarked alongside new
recognizers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


@dataclass
class Candidate:
    """One ranked FIG candidate."""
    fig_name: str
    score: float
    d_score: float = 0.0
    category: str = ""
    reasoning: str = ""


@dataclass
class RecognizerOutput:
    """Result of running a Recognizer on one clip.

    `candidates` must be sorted by descending score. `cues` is an optional
    bag of structured evidence the recognizer inferred — useful for error
    analysis even when the top-1 is wrong.
    """
    recognizer: str
    mode: str = ""
    candidates: list[Candidate] = field(default_factory=list)
    cues: dict[str, Any] = field(default_factory=dict)
    raw_text: str = ""
    latency_s: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    error: str = ""

    @property
    def top1(self) -> str:
        return self.candidates[0].fig_name if self.candidates else ""

    def topk_names(self, k: int = 3) -> list[str]:
        return [c.fig_name for c in self.candidates[:k]]


class Recognizer(Protocol):
    """Anything that can take a clip and rank FIG candidates."""

    name: str

    def recognize(self, clip: Path) -> RecognizerOutput:
        ...


# --------------------------------------------------------------------------- #
# VLM adapters — wrap the existing runners in run_benchmark.py
# --------------------------------------------------------------------------- #


class VLMRecognizer:
    """Adapter that delegates to an existing run_benchmark.py runner."""

    def __init__(self, model_id: str):
        self.model_id = model_id
        self.name = model_id
        # Lazy import — avoid pulling FIGMatcher at module load time.
        from paper.experiments.run_benchmark import dispatch  # noqa: WPS433
        from core.vlm.fig_matcher import FIGMatcher  # noqa: WPS433
        self._dispatch = dispatch
        self._matcher = FIGMatcher()
        self._fig_names = [t.name for t in self._matcher.get_all_tricks()]

    def recognize(self, clip: Path) -> RecognizerOutput:
        call = self._dispatch(self.model_id, clip, self._fig_names)
        candidates: list[Candidate] = []
        if call.tricks:
            parsed = call.tricks[0]
            match = self._matcher.match(
                parsed.trick_name,
                flip_count=parsed.flip_count,
                twist_count=parsed.twist_count,
                direction=parsed.direction,
            )
            if match:
                candidates.append(Candidate(
                    fig_name=match.fig_name,
                    score=float(match.confidence),
                    d_score=match.d_score,
                    category=match.category,
                    reasoning=f"vlm->{match.match_level}",
                ))
            else:
                # Raw VLM name that couldn't be FIG-matched.
                candidates.append(Candidate(
                    fig_name=parsed.trick_name,
                    score=0.3,
                    reasoning="vlm-unmatched",
                ))
        return RecognizerOutput(
            recognizer=call.model,
            mode=call.mode,
            candidates=candidates,
            cues={
                "flip": call.tricks[0].flip_count if call.tricks else 0.0,
                "twist": call.tricks[0].twist_count if call.tricks else 0.0,
                "direction": call.tricks[0].direction if call.tricks else "",
            },
            raw_text=call.raw_text,
            latency_s=call.latency_s,
            input_tokens=call.input_tokens,
            output_tokens=call.output_tokens,
            error=call.error,
        )


def build_recognizer(spec: str) -> Recognizer:
    """Factory: map a CLI spec string to a concrete Recognizer.

    Examples:
        - "heuristic"                         -> HeuristicRecognizer (pose-only)
        - "heuristic+attrs:path/to/ckpt.pt"   -> HeuristicRecognizer with
                                                 attribute-model upgrade
        - "heuristic+attrs"                   -> use default path
                                                 data/models/pkvision_attributes_v2.pt
        - "vlm:gemini-2.5-flash"              -> VLMRecognizer("gemini-2.5-flash")
        - "vlm:claude"                        -> VLMRecognizer("claude")
        - "vlm:openrouter/qwen/..."           -> VLMRecognizer("openrouter/...")
    """
    if spec == "heuristic":
        from core.recognition.heuristic_recognizer import HeuristicRecognizer
        return HeuristicRecognizer()
    if spec == "heuristic+attrs" or spec.startswith("heuristic+attrs:"):
        from core.recognition.heuristic_recognizer import HeuristicRecognizer
        from pathlib import Path as _Path
        default_ckpt = _Path(__file__).resolve().parents[2] / "data" / "models" / "pkvision_attributes_v3.pt"
        ckpt = default_ckpt if spec == "heuristic+attrs" else _Path(spec[len("heuristic+attrs:"):])
        return HeuristicRecognizer(attr_ckpt=ckpt)
    if spec.startswith("vlm:"):
        return VLMRecognizer(spec[len("vlm:"):])
    raise ValueError(f"Unknown recognizer spec: {spec!r}")
