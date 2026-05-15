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

from core.recognition.fig_decoder import FIGDecoder

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


def test_disable_canonical_runs_and_drops_canonical_bonus():
    """disable_canonical=True must remove the 0.5 canonical bonus from scoring.

    Backflip is canonical and sits at the top of this physics cluster, so its
    ranking score must drop when the bonus is removed (and/or the ordering
    changes).
    """
    d = FIGDecoder()
    full = d.rank(CUES, k=5)
    no_canon = d.rank(CUES, k=5, disable_canonical=True)
    assert isinstance(no_canon, list) and len(no_canon) > 0
    # The ordering or the top candidate's ranking score must differ
    assert (
        [c.fig_name for c in full] != [c.fig_name for c in no_canon]
        or full[0].score != no_canon[0].score
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
