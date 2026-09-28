"""Fuzzy FIG trick name matching and D-score lookup.

Matches VLM output to official FIG trick names using a 5-level cascade:
1. Exact match (case-insensitive)
2. Alias match
3. Normalization (full→360, half→180, etc.)
4. Token overlap (Jaccard similarity)
5. Physics fallback (flip_count + twist_count + direction)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
FIG_PATH = ROOT / "data" / "fig_tricks_2025.json"

# Common name substitutions VLMs might use
NORMALIZE_MAP = {
    "full": "360",
    "half": "180",
    "double full": "720",
    "triple full": "1080",
    "quad full": "1440",
    "1.5": "540",
    "2.5": "900",
    "3.5": "1260",
}


@dataclass
class FIGMatch:
    """Result of matching a VLM trick name to the FIG database."""
    fig_name: str
    d_score: float
    category: str
    match_level: str   # exact / alias / normalized / token_overlap / physics
    confidence: float  # 0-1, how confident the match is


@dataclass
class _FIGEntry:
    name: str
    score: float
    flip: float
    twist: float
    direction: str | None
    axis: str | None
    category: str
    aliases: list[str]


class FIGMatcher:
    """Fuzzy matcher from VLM trick names to FIG D-scores."""

    def __init__(self, fig_path: str | Path = FIG_PATH):
        with open(fig_path) as f:
            data = json.load(f)
        self._entries: list[_FIGEntry] = []
        self._name_index: dict[str, _FIGEntry] = {}

        for cat_key, cat_data in data.get("categories", {}).items():
            for trick in cat_data.get("tricks", []):
                entry = _FIGEntry(
                    name=trick["name"],
                    score=trick.get("score", 0),
                    flip=trick.get("flip", 0),
                    twist=trick.get("twist", 0),
                    direction=trick.get("direction"),
                    axis=trick.get("axis"),
                    category=cat_key,
                    aliases=trick.get("aliases", []),
                )
                self._entries.append(entry)
                # Index by name and aliases (lowercase)
                self._name_index[entry.name.lower()] = entry
                for alias in entry.aliases:
                    self._name_index[alias.lower()] = entry

    def match(
        self,
        trick_name: str,
        flip_count: float = 0,
        twist_count: float = 0,
        direction: str | None = None,
    ) -> FIGMatch | None:
        """Match a VLM trick name to the FIG database.

        Tries 5 levels in order, returns first match.
        """
        name = trick_name.strip()
        name_lower = name.lower()

        # Level 1: Exact match
        if name_lower in self._name_index:
            entry = self._name_index[name_lower]
            return FIGMatch(entry.name, entry.score, entry.category, "exact", 1.0)

        # Level 2: Alias match (already in index from __init__)
        # — covered by level 1 since aliases are indexed

        # Level 3: Normalization
        normalized = self._normalize(name_lower)
        if normalized in self._name_index:
            entry = self._name_index[normalized]
            return FIGMatch(entry.name, entry.score, entry.category, "normalized", 0.9)

        # Also try normalizing all entries against the input
        for entry in self._entries:
            if self._normalize(entry.name.lower()) == normalized:
                return FIGMatch(entry.name, entry.score, entry.category, "normalized", 0.9)
            for alias in entry.aliases:
                if self._normalize(alias.lower()) == normalized:
                    return FIGMatch(entry.name, entry.score, entry.category, "normalized", 0.85)

        # Level 4: Token overlap (Jaccard)
        best_overlap = 0.0
        best_entry = None
        input_tokens = set(re.split(r"[\s\-_]+", name_lower))
        for entry in self._entries:
            for candidate in [entry.name] + entry.aliases:
                candidate_tokens = set(re.split(r"[\s\-_]+", candidate.lower()))
                intersection = input_tokens & candidate_tokens
                union = input_tokens | candidate_tokens
                jaccard = len(intersection) / len(union) if union else 0
                if jaccard > best_overlap:
                    best_overlap = jaccard
                    best_entry = entry

        if best_overlap >= 0.5 and best_entry:
            return FIGMatch(best_entry.name, best_entry.score, best_entry.category, "token_overlap", best_overlap)

        # Level 5: Physics fallback
        if flip_count > 0 or twist_count > 0:
            candidates = []
            for entry in self._entries:
                if entry.flip != flip_count:
                    continue
                twist_match = abs(entry.twist - twist_count) < 0.3
                dir_match = (
                    direction is None
                    or entry.direction is None
                    or entry.direction == direction
                )
                if twist_match and dir_match:
                    candidates.append(entry)

            if len(candidates) == 1:
                e = candidates[0]
                return FIGMatch(e.name, e.score, e.category, "physics", 0.7)
            elif candidates:
                # Multiple physics matches — pick highest D-score as most likely
                e = max(candidates, key=lambda x: x.score)
                return FIGMatch(e.name, e.score, e.category, "physics", 0.5)

        return None

    def _normalize(self, name: str) -> str:
        """Apply common substitutions."""
        result = name
        # Sort by length descending to match longer patterns first
        for old, new in sorted(NORMALIZE_MAP.items(), key=lambda x: -len(x[0])):
            result = result.replace(old, new)
        return result.strip()

    def get_all_tricks(self) -> list[_FIGEntry]:
        """Return all FIG trick entries."""
        return list(self._entries)
