from core.recognition.dscore_equiv import DScoreBook, same_dscore

def test_known_trick_dscore_and_equivalence():
    b = DScoreBook("data/fig_tricks_2025.json")
    bf = b.d_score_for("Backflip")
    assert isinstance(bf, float)
    assert b.same_dscore("Backflip", "Backflip") is True

def test_same_dscore_groups_real_cluster():
    b = DScoreBook("data/fig_tricks_2025.json")
    for a, c in [("Cork", "Cork"), ("Backflip", "Backflip")]:
        assert b.same_dscore(a, c) is True
    assert b.same_dscore("Stride", "Swing Double Gainer 1080 (Miller)") is False  # 0.1 vs 7.7

def test_unknown_raises():
    b = DScoreBook("data/fig_tricks_2025.json")
    import pytest
    with pytest.raises(KeyError):
        b.d_score_for("not a real trick zzz")
