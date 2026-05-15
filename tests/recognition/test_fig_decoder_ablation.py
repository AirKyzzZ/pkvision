"""Ablation switch tests for FIGDecoder.rank().

Exercises:
  - disable_canonical: suppresses CANONICAL_BONUS (0.5) from per-trick scoring.
  - disable_group_bonus: suppresses _apply_group_bonus, leaving group_bonus == 0.

CUES rationale: flip=1, twist=0, direction="backward", context="acrobatics",
takeoff="standing" places us in the '1_flip_0_twist_backward' disambiguation
group (data/fig_tricks_2025.json). In that group:
  - 'Backflip' is in CANONICAL_NAMES  → canonical bonus applies
  - 'Backflip' has distinguisher=standing_takeoff → group bonus of 2.0 applies
Both ablation switches are therefore genuinely observable.
"""

import pytest

from core.recognition.fig_decoder import CANONICAL_BONUS, FIGDecoder

# Physics-collision cues: backward 1-flip group, standing takeoff triggers
# Backflip's disambiguation distinguisher bonus AND Backflip is canonical.
CUES = {
    "context": "acrobatics",
    "flip": 1.0,
    "twist": 0.0,
    "direction": "backward",
    "takeoff": "standing",
}


def test_default_behavior_unchanged():
    """Two identical calls must return the same ranking (determinism check)."""
    d = FIGDecoder()
    a = d.rank(CUES, k=5)
    b = d.rank(CUES, k=5)
    assert [c.fig_name for c in a] == [c.fig_name for c in b]


def test_disable_canonical_drops_exactly_canonical_bonus():
    """disable_canonical=True must remove exactly CANONICAL_BONUS (0.5) from
    Backflip's score and must remove the 'canonical' key from its breakdown.

    Backflip is in CANONICAL_NAMES.  The group bonus from standing_takeoff is
    applied equally in both runs, so it cancels out: the only score delta
    between the full run and the no-canonical run must equal CANONICAL_BONUS.
    """
    d = FIGDecoder()
    full = d.rank(CUES, k=5)
    no_canon = d.rank(CUES, k=5, disable_canonical=True)
    assert isinstance(no_canon, list) and len(no_canon) > 0

    full_bf = next((c for c in full if c.fig_name == "Backflip"), None)
    no_canon_bf = next((c for c in no_canon if c.fig_name == "Backflip"), None)
    assert full_bf is not None, "Backflip must appear in full ranking for CUES"
    assert no_canon_bf is not None, "Backflip must appear in no-canonical ranking for CUES"

    # Exact score drop must equal the canonical bonus — not a disjunction.
    assert full_bf.score - no_canon_bf.score == pytest.approx(CANONICAL_BONUS), (
        f"Expected Backflip score to drop by exactly CANONICAL_BONUS={CANONICAL_BONUS}; "
        f"full={full_bf.score}, no_canon={no_canon_bf.score}, "
        f"delta={full_bf.score - no_canon_bf.score}"
    )
    # 'canonical' key must be present in full breakdown and absent when disabled.
    assert "canonical" in full_bf.breakdown, (
        f"Expected 'canonical' key in full breakdown; got {full_bf.breakdown}"
    )
    assert "canonical" not in no_canon_bf.breakdown, (
        f"Expected no 'canonical' key in no-canonical breakdown; got {no_canon_bf.breakdown}"
    )


def test_disable_group_bonus_zeros_group_component():
    """disable_group_bonus=True must leave all group_bonus fields at 0.0."""
    d = FIGDecoder()
    no_grp = d.rank(CUES, k=5, disable_group_bonus=True)
    assert all((c.group_bonus or 0) == 0 for c in no_grp)


def test_disable_group_bonus_lowers_score_vs_full():
    """With group bonus disabled the top-candidate's score must be lower than
    (or equal to) the full-mode score — it cannot gain points it never had."""
    d = FIGDecoder()
    full = d.rank(CUES, k=5)
    no_grp = d.rank(CUES, k=5, disable_group_bonus=True)
    # Find Backflip in both lists (it may shift position)
    full_bf = next((c for c in full if c.fig_name == "Backflip"), None)
    no_grp_bf = next((c for c in no_grp if c.fig_name == "Backflip"), None)
    if full_bf and no_grp_bf:
        assert no_grp_bf.score < full_bf.score, (
            f"Expected Backflip score to drop: full={full_bf.score}, "
            f"no_grp={no_grp_bf.score}"
        )


def test_both_disabled_is_ontology_only():
    """With both bonuses disabled the decoder runs in 'ontology-only' mode:
    pure cue-weight scoring, no canonical tie-break, no group disambiguation.

    Every returned candidate must have group_bonus == 0 and no 'canonical' key
    in its breakdown — confirming both bonus sources are fully suppressed.
    """
    d = FIGDecoder()
    ontology_only = d.rank(CUES, k=5, disable_canonical=True, disable_group_bonus=True)

    assert isinstance(ontology_only, list) and len(ontology_only) > 0, (
        "ontology-only rank must return a non-empty list"
    )
    for cand in ontology_only:
        assert (cand.group_bonus or 0) == 0, (
            f"Expected group_bonus==0 in ontology-only mode for {cand.fig_name}; "
            f"got group_bonus={cand.group_bonus}"
        )
        assert "canonical" not in cand.breakdown, (
            f"Expected no 'canonical' key in ontology-only breakdown for {cand.fig_name}; "
            f"got breakdown={cand.breakdown}"
        )
