from pathlib import Path
from scripts.p0_build_blind_set import (
    map_clip_to_fig, stratified_sample, ClipCandidate,
    physics_collision_groups, load_fig_grounded_candidates,
)
from core.recognition.oracle_cues import _norm
from core.data.decontam import is_contaminated


def test_map_clip_rejects_gibberish():
    assert map_clip_to_fig("zzz_not_a_trick_9999.mp4") is None


def test_physics_collision_groups_real_ontology():
    g = physics_collision_groups()
    assert _norm("Backflip") in g and _norm("Backhandspring") in g
    assert g[_norm("Backflip")] == g[_norm("Backhandspring")]
    assert len(g) >= 50               # ~106 hard tricks expected
    assert _norm("zzz not a trick") not in g


def test_fig_grounded_candidates_loaded_and_clean():
    cands = load_fig_grounded_candidates()
    assert 50 <= len(cands) <= 99
    assert all(c.proposed_fig_trick.strip() for c in cands)
    assert all(not is_contaminated(c.path) for c in cands)
    assert all(c.source == "fig_grounded" for c in cands)


def test_stratified_sample_excludes_contaminated_and_oversamples_hard():
    cands = [
        ClipCandidate(Path("data/parkourtheory_clips/gainer.mp4"), "Gainer", "g1", "filename"),
        ClipCandidate(Path("data/parkourtheory_clips/backflip.mp4"), "Backflip", "g1", "filename"),
        ClipCandidate(Path("data/final_clips/backflip.mp4"), "Backflip", "g1", "filename"),
        ClipCandidate(Path("data/parkourtheory_clips/stride.mp4"), "Stride", None, "filename"),
    ]
    picked = stratified_sample(cands, target=3, min_hard_fraction=0.6)
    paths = {str(c.path) for c in picked}
    assert "data/final_clips/backflip.mp4" not in paths
    hard = [c for c in picked if c.disambig_group is not None]
    assert len(hard) / len(picked) >= 0.6
