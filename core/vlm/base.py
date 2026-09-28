"""Base types and ABC for VLM providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class VLMTrickResult:
    """A single trick identified by the VLM."""
    trick_name: str
    category: str | None = None       # swing / wall / acrobatics / pk_basics
    direction: str | None = None      # forward / backward / side
    flip_count: float = 0.0
    twist_count: float = 0.0
    confidence: str = "medium"        # high / medium / low
    reasoning: str = ""


@dataclass
class VLMResponse:
    """Full response from a VLM analysis call."""
    tricks: list[VLMTrickResult] = field(default_factory=list)
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    raw_text: str = ""


class VLMProvider(ABC):
    """Abstract base for VLM trick analysis providers."""

    @abstractmethod
    def analyze_trick(
        self,
        video_path: Path,
        trick_list: str,
    ) -> VLMResponse:
        """Analyze a video clip and identify the parkour trick.

        Args:
            video_path: Path to the trick video clip.
            trick_list: Formatted FIG trick table for the prompt.

        Returns:
            VLMResponse with identified tricks.
        """
        ...
