from pathlib import Path
from scripts.p0_build_blind_set import (
    map_clip_to_fig, stratified_sample, ClipCandidate,
)

def test_map_clip_uses_alias_and_rejects_ambiguous():
    # gibberish does not resolve
    assert map_clip_to_fig("zzz_not_a_trick_9999.mp4") is None

def test_stratified_sample_excludes_contaminated_and_oversamples_hard_groups():
    cands = [
        ClipCandidate(Path("data/parkourtheory_clips/gainer.mp4"), "Gainer", "1_flip_0_twist_backward"),
        ClipCandidate(Path("data/parkourtheory_clips/backflip.mp4"), "Backflip", "1_flip_0_twist_backward"),
        ClipCandidate(Path("data/final_clips/backflip.mp4"), "Backflip", "1_flip_0_twist_backward"),  # contaminated
        ClipCandidate(Path("data/parkourtheory_clips/stride.mp4"), "Stride", None),  # easy, no group
    ]
    picked = stratified_sample(cands, target=3, min_hard_fraction=0.6)
    paths = {str(c.path) for c in picked}
    assert "data/final_clips/backflip.mp4" not in paths           # decontam enforced
    hard = [c for c in picked if c.disambig_group is not None]
    assert len(hard) / len(picked) >= 0.6                          # hard-dominated
