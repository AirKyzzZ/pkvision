from pathlib import Path
from core.data.decontam import is_contaminated, filter_clean

CONTAMINATED = [
    "data/final_clips/backflip.mp4",
    "data/vlm_clips/IMG_5985/trick_01.mp4",
    "data/run_testing/test_run_2.mp4",
    "data/v5_full_training/frames/acrobatics/test_backflip.npy",
    "data/parkourtheory_clips/back_double_full_in_back_out.mp4",
]
CLEAN = [
    "data/parkourtheory_clips/gainer.mp4",
    "data/parkourtheory_clips/kong_gainer.mp4",
    "data/parkourtheory_clips_cropped/butterfly_twist.mp4",
]

def test_known_contaminated_are_flagged():
    for p in CONTAMINATED:
        assert is_contaminated(Path(p)) is True, p

def test_clean_clips_pass():
    for p in CLEAN:
        assert is_contaminated(Path(p)) is False, p

def test_filter_clean_removes_only_contaminated():
    allp = [Path(p) for p in CONTAMINATED + CLEAN]
    kept = filter_clean(allp)
    assert sorted(str(p) for p in kept) == sorted(CLEAN)

def test_test_prefix_anywhere_in_stem_is_flagged():
    assert is_contaminated(Path("data/parkourtheory_clips/test_foo.mp4")) is True
    assert is_contaminated(Path("data/x/contest_jump.mp4")) is False  # 'test' substring must not false-positive
