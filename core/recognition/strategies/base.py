"""Detection strategy protocol — the interface all detection methods implement."""

from __future__ import annotations

from typing import Protocol

from core.models import FrameAnalysis, TrickConfig, TrickDetection


class DetectionStrategy(Protocol):
    """Protocol for trick detection strategies.

    Each strategy receives a trick definition and a sequence of analyzed frames,
    and returns a TrickDetection if the trick was found, or None.
    """

    def evaluate(
        self,
        trick: TrickConfig,
        frames: list[FrameAnalysis],
    ) -> TrickDetection | None:
        """Evaluate whether the given trick is present in the frame sequence.

        Args:
            trick: The trick configuration to look for.
            frames: Sequence of analyzed frames (with computed angles/velocities).

        Returns:
            TrickDetection if the trick is detected, None otherwise.
        """
        ...
