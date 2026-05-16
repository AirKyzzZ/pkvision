"""D-score lookup + equivalence. For an auto-SCORING system, confusing two
tricks with the same D-score is a zero-cost error. These helpers let the eval
report scoring-aware accuracy alongside raw trick top-1.
"""
from __future__ import annotations
import json
from pathlib import Path

from core.recognition.oracle_cues import _norm


class DScoreBook:
    def __init__(self, fig_path: str | Path = "data/fig_tricks_2025.json") -> None:
        raw = json.loads(Path(fig_path).read_text())
        self._d: dict[str, float] = {}
        for obj in raw["categories"].values():
            for t in obj["tricks"]:
                self._d[_norm(t["name"])] = float(t["score"])
                for a in t.get("aliases", []) or []:
                    self._d.setdefault(_norm(a), float(t["score"]))

    def d_score_for(self, trick: str) -> float:
        k = _norm(trick)
        if k not in self._d:
            raise KeyError(f"Unknown trick: {trick!r}")
        return self._d[k]

    def same_dscore(self, a: str, b: str, tol: float = 1e-9) -> bool:
        return abs(self.d_score_for(a) - self.d_score_for(b)) <= tol


def same_dscore(a: str, b: str, fig_path: str = "data/fig_tricks_2025.json") -> bool:
    return DScoreBook(fig_path).same_dscore(a, b)
