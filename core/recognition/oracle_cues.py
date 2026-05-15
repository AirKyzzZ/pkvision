"""Oracle cues = the true trick's structured attributes from the FIG ontology.

A perfect cue-extractor, on a clip that truly shows trick T, would produce
exactly these cues. Feeding them to the frozen decoder isolates decoder quality
from perception quality.
"""
from __future__ import annotations
import json
from pathlib import Path

_ONTOLOGY_CUE_KEYS = (
    "flip", "twist", "direction", "axis",
    "entry", "takeoff", "hand_contact", "kick",
)


class OracleCueError(KeyError):
    pass


def _norm(s: str) -> str:
    return " ".join(str(s).strip().lower().replace("-", " ").replace("_", " ").split())


class OracleCueBook:
    def __init__(self, fig_path: str | Path = "data/fig_tricks_2025.json") -> None:
        raw = json.loads(Path(fig_path).read_text())
        self._by_name: dict[str, dict] = {}
        self._context_of: dict[str, str] = {}
        for category, cat_obj in raw["categories"].items():
            for trick in cat_obj["tricks"]:
                key = _norm(trick["name"])
                self._by_name[key] = trick
                self._context_of[key] = category
                for alias in trick.get("aliases", []) or []:
                    self._by_name.setdefault(_norm(alias), trick)
                    self._context_of.setdefault(_norm(alias), category)

    def _lookup(self, trick_name: str) -> tuple[dict, str]:
        key = _norm(trick_name)
        if key not in self._by_name:
            raise OracleCueError(f"Unknown FIG trick: {trick_name!r}")
        return self._by_name[key], self._context_of[key]

    def cues_for(self, trick_name: str) -> dict:
        trick, context = self._lookup(trick_name)
        cues: dict = {"context": context}
        for k in _ONTOLOGY_CUE_KEYS:
            if k in trick:
                cues[k] = trick[k]
        return cues

    def d_score_for(self, trick_name: str) -> float:
        trick, _ = self._lookup(trick_name)
        return trick["score"]

    def canonical_name(self, trick_name: str) -> str:
        trick, _ = self._lookup(trick_name)
        return trick["name"]
