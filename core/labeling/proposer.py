"""clip -> {trick, cues, confidence} proposal. LocalModelProposer is free ($0)."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from core.labeling.clip_ref import ClipRef
from core.vlm.fig_matcher import FIGMatcher


@dataclass
class Proposal:
    slug: str
    trick: str | None
    cues: dict
    d_score: float | None
    confidence: float
    source: str


class Proposer(ABC):
    @abstractmethod
    def propose(self, clip: ClipRef) -> Proposal: ...


class LocalModelProposer(Proposer):
    def __init__(self, model, matcher: FIGMatcher | None = None):
        self.model = model
        self.matcher = matcher or FIGMatcher()

    def propose(self, clip: ClipRef) -> Proposal:
        cues = self.model.predict(clip.get_skeleton())
        confs = [v for k, v in cues.items() if k.endswith("_conf")]
        confidence = float(sum(confs) / len(confs)) if confs else 0.0
        clean = {k: v for k, v in cues.items() if not k.endswith("_conf")}
        match = self.matcher.match(
            trick_name="",  # cue-only -> physics fallback (flip/twist/direction)
            flip_count=clean.get("flip", 0.0),
            twist_count=clean.get("twist", 0.0),
            direction=clean.get("direction"),
        )
        return Proposal(clip.slug, match.fig_name if match else None, clean,
                        match.d_score if match else None, confidence, "local")
