"""Structured FIG decoder — rank FIG candidates from predicted cues.

Given a dict of predicted cues (flip, twist, direction, context, entry,
takeoff, hand_contact, axis, kick, body_shape), return a ranked list of
FIG tricks from `data/fig_tricks_2025.json`.

This is the counterpart to `core/vlm/fig_matcher.py`. That module does
fuzzy string matching from a free-form VLM name; this module does soft
structured scoring from predicted evidence.

Design principles:
    - Unknown cues are skipped (no penalty). Only agreements and
      disagreements count.
    - Cue weights are tuned for FIG disambiguation priorities: context
      and flip matter most, twist and direction next, entry/takeoff/
      hand_contact/kick are tie-breakers.
    - Disambiguation groups in the FIG JSON drive extra bonuses: when
      the predicted physics matches a group, distinguisher cues boost
      the in-group candidate whose distinguisher matches.
    - Returns top-k with per-cue breakdown so errors are inspectable.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
DEFAULT_FIG_PATH = REPO / "data" / "fig_tricks_2025.json"


# ── Cue weights ─────────────────────────────────────────────────────────────
# Tuned so the most reliable cues dominate, but no single cue can veto.
CUE_WEIGHTS: dict[str, float] = {
    "context": 3.0,        # swing/wall/acrobatics/pk_basics — usually reliable
    "flip": 3.0,           # flip count — reliable when measured
    "direction": 2.0,      # backward/forward/side — reliable
    "twist": 2.0,          # twist count — harder but important
    "axis": 1.5,           # lateral/longitudinal/off_axis/sagittal
    "entry": 1.5,          # standing/running/kong/caster/one_leg/wall
    "takeoff": 1.5,        # running_forward/standing/cheat — overlaps with entry
    "hand_contact": 1.5,   # bool — key for Backflip vs Backhandspring
    "kick": 1.0,           # bool — Frisbee vs Aerial
    "body_shape": 0.5,     # tuck/pike/layout — cosmetic
}

# Soft agreement functions per cue type
_NUMERIC_CUES = {"flip", "twist"}
_BOOL_CUES = {"hand_contact", "kick"}

# Canonical FIG names for physics clusters that contain duplicate entries in
# the 2025 table. These clusters are physically indistinguishable from
# structured cues alone (Kroc/Raiz/Cork/Dark Arabian all have
# flip=1/twist=0.5/backward/off_axis, Backflip 720/A-720/B-720 all have
# flip=1/twist=2/backward/lateral, etc.). A small bonus on the canonical name
# breaks ties toward the label the community and POOL-B ground truth use.
CANONICAL_NAMES: set[str] = {
    "Backflip", "Frontflip", "Sideflip", "Gainer",
    "Backflip 360", "Backflip 720", "Backflip 1080", "Backflip 1440",
    "Gainer 360", "Gainer 720", "Gainer 1080",
    "Kroc", "Double Kroc",
    "Double Cork", "Triple Cork",
    "Double Backflip", "Triple Backflip",
    "Aerial", "Cartwheel", "Webster",
    "Wall Backflip", "Wall Inward Frontflip", "Wall Flip",
    "Swing Gainer", "Swing 180",
}
CANONICAL_BONUS = 0.5


@dataclass
class FIGTrick:
    name: str
    category: str
    score: float
    flip: float
    twist: float
    direction: str | None
    axis: str | None
    entry: str | None
    takeoff: str | None
    hand_contact: bool | None
    kick: bool | None
    aliases: list[str] = field(default_factory=list)
    raw: dict = field(default_factory=dict)


@dataclass
class DecoderCandidate:
    trick: FIGTrick
    score: float
    breakdown: dict[str, float]
    group_bonus: float = 0.0

    @property
    def fig_name(self) -> str:
        return self.trick.name

    @property
    def d_score(self) -> float:
        return self.trick.score

    @property
    def category(self) -> str:
        return self.trick.category


class FIGDecoder:
    """Rank FIG tricks from a cue dict using soft weighted agreement."""

    def __init__(self, fig_path: str | Path = DEFAULT_FIG_PATH):
        with open(fig_path) as f:
            self._data = json.load(f)
        self._tricks: list[FIGTrick] = []
        for cat_key, cat in self._data.get("categories", {}).items():
            for t in cat.get("tricks", []):
                self._tricks.append(FIGTrick(
                    name=t["name"],
                    category=cat_key,
                    score=float(t.get("score", 0.0)),
                    flip=float(t.get("flip", 0.0) or 0.0),
                    twist=float(t.get("twist", 0.0) or 0.0),
                    direction=t.get("direction"),
                    axis=t.get("axis"),
                    entry=t.get("entry"),
                    takeoff=t.get("takeoff"),
                    hand_contact=t.get("hand_contact"),
                    kick=t.get("kick"),
                    aliases=list(t.get("aliases", []) or []),
                    raw=t,
                ))
        self._disamb = self._data.get("disambiguation_needed", {}) or {}

    @property
    def tricks(self) -> list[FIGTrick]:
        return list(self._tricks)

    # ── Scoring ──────────────────────────────────────────────────────────

    def rank(
        self,
        cues: dict[str, Any],
        k: int = 5,
        candidate_filter: list[str] | None = None,
        *,
        disable_canonical: bool = False,
        disable_group_bonus: bool = False,
    ) -> list[DecoderCandidate]:
        """Rank FIG tricks by cue agreement. Returns top-k.

        Ablation switches (default False — behavior byte-identical when omitted):
          disable_canonical:  suppress the CANONICAL_BONUS tie-break in scoring.
          disable_group_bonus: suppress _apply_group_bonus (group_bonus stays 0.0).
        """
        results: list[DecoderCandidate] = []
        for trick in self._tricks:
            if candidate_filter is not None and trick.name not in candidate_filter:
                continue
            score, breakdown = self._score_trick(trick, cues, disable_canonical=disable_canonical)
            results.append(DecoderCandidate(
                trick=trick,
                score=score,
                breakdown=breakdown,
            ))
        # Apply disambiguation group bonuses
        if not disable_group_bonus:
            self._apply_group_bonus(results, cues)
        results.sort(key=lambda c: c.score, reverse=True)
        return results[:k]

    def _score_trick(
        self,
        trick: FIGTrick,
        cues: dict[str, Any],
        *,
        disable_canonical: bool = False,
    ) -> tuple[float, dict[str, float]]:
        breakdown: dict[str, float] = {}
        total = 0.0
        for cue_name, weight in CUE_WEIGHTS.items():
            predicted = cues.get(cue_name)
            if predicted is None:
                continue
            agreement = self._cue_agreement(cue_name, trick, predicted)
            if agreement is None:
                continue  # FIG trick has no value for this cue — neutral
            contribution = agreement * weight
            breakdown[cue_name] = contribution
            total += contribution
        if not disable_canonical and trick.name in CANONICAL_NAMES:
            breakdown["canonical"] = CANONICAL_BONUS
            total += CANONICAL_BONUS
        return total, breakdown

    @staticmethod
    def _cue_agreement(
        cue: str,
        trick: FIGTrick,
        predicted: Any,
    ) -> float | None:
        """Return agreement in [-1, 1]. None if FIG trick has no value here.

        - Numeric cues (flip, twist): soft. exact=+1, ±0.5=+0.3, ±1=-0.5, else=-1.
        - Context: exact=+1, mismatch=-1.
        - Direction/axis/entry/takeoff: exact=+1, mismatch=-0.5 (soft).
        - Bool (hand_contact, kick): match=+1, mismatch=-1.
        """
        if cue == "context":
            if predicted == trick.category:
                return 1.0
            return -1.0

        if cue in _NUMERIC_CUES:
            fig_val = getattr(trick, cue)
            try:
                pred_val = float(predicted)
            except (TypeError, ValueError):
                return None
            delta = abs(fig_val - pred_val)
            if delta < 0.1:
                return 1.0
            if delta < 0.6:
                return 0.3
            if delta < 1.1:
                return -0.5
            return -1.0

        if cue in _BOOL_CUES:
            fig_val = getattr(trick, cue)
            if fig_val is None:
                return None
            pred_bool = bool(predicted)
            return 1.0 if bool(fig_val) == pred_bool else -1.0

        # String-categorical cues
        fig_val = getattr(trick, cue, None)
        if fig_val is None:
            return None  # FIG trick has no constraint on this cue
        if str(predicted).lower() == str(fig_val).lower():
            return 1.0
        return -0.5

    def _apply_group_bonus(
        self,
        results: list[DecoderCandidate],
        cues: dict[str, Any],
    ) -> None:
        """Add bonuses using `disambiguation_needed` groups when physics
        matches a known confusion cluster."""
        flip = cues.get("flip")
        twist = cues.get("twist")
        direction = cues.get("direction")
        if flip is None or twist is None or direction is None:
            return
        by_name = {c.trick.name: c for c in results}
        for gkey, group in self._disamb.items():
            if gkey.startswith("_"):
                continue
            if not isinstance(group, dict):
                continue
            phys = group.get("physics", {})
            try:
                if abs(float(phys.get("flip", -999)) - float(flip)) > 0.1:
                    continue
                if abs(float(phys.get("twist", -999)) - float(twist)) > 0.6:
                    continue
                if str(phys.get("direction")).lower() != str(direction).lower():
                    continue
            except (TypeError, ValueError):
                continue
            # Physics group matches. Boost any candidate whose distinguisher
            # matches an active cue.
            takeoff_cue = str(cues.get("takeoff") or "").lower()
            entry_cue = str(cues.get("entry") or "").lower()
            hand_cue = cues.get("hand_contact")
            for cand_spec in group.get("candidates", []):
                name = cand_spec.get("name")
                distinguisher = str(cand_spec.get("distinguisher") or "").lower()
                if name not in by_name:
                    continue
                cand = by_name[name]
                bonus = 0.0
                # Heuristic distinguisher → cue matching
                if distinguisher == "standing_takeoff" and takeoff_cue == "standing":
                    bonus = 2.0
                elif distinguisher == "running_forward_takeoff" and takeoff_cue in {"running_forward", "running"}:
                    bonus = 2.0
                elif distinguisher == "cheat_takeoff" and takeoff_cue == "cheat":
                    bonus = 2.0
                elif distinguisher == "caster_entry" and entry_cue == "caster":
                    bonus = 2.0
                elif distinguisher == "kong" and entry_cue == "kong":
                    bonus = 2.0
                elif distinguisher == "one_leg_takeoff" and (takeoff_cue == "one_leg" or entry_cue == "one_leg"):
                    bonus = 2.0
                elif distinguisher == "hand_contact_during_flip" and hand_cue is True:
                    bonus = 2.0
                elif distinguisher == "no_hand_contact" and hand_cue is False:
                    bonus = 1.0
                elif distinguisher == "one_hand_contact" and hand_cue is True:
                    bonus = 1.0
                if bonus > 0:
                    cand.score += bonus
                    cand.group_bonus = bonus
                    cand.breakdown[f"disamb:{distinguisher}"] = bonus
