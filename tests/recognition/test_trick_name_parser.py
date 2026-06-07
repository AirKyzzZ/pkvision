from core.recognition.trick_name_parser import ParsedCues, parse_trick_name, tokenize


def test_returns_parsedcues_shape():
    out = parse_trick_name("frontflip")
    assert isinstance(out, ParsedCues)
    assert isinstance(out.cues, dict)
    assert isinstance(out.confidence, dict)
    assert isinstance(out.trace, list)
    assert isinstance(out.unparsed_tokens, list)


def test_non_string_raises():
    import pytest
    with pytest.raises(TypeError):
        parse_trick_name(123)


def test_gibberish_abstains_completely():
    out = parse_trick_name("qwxz_zzzz")
    assert out.cues == {}
    assert "qwxz" in out.unparsed_tokens


def test_tokenize_folds_multiword_and_splits():
    assert tokenize("back_one_and_a_half_full") == ["back", "one_and_a_half", "full"]
    assert tokenize("Dash Vault") == ["dash", "vault"]


def test_lexicon_loads_anchor_moves():
    from core.recognition.trick_name_parser import _load_lexicon
    _load_lexicon.cache_clear()
    lex = _load_lexicon()
    assert lex["moves"]["gainer"]["direction"] == "backward"
    assert lex["twist_words"]["full"] == 1.0
    assert "in" in lex["phase_boundaries"]


def test_segment_phases_splits_on_boundaries():
    from core.recognition.trick_name_parser import segment_phases
    assert segment_phases(["back", "full", "in", "full", "out"]) == [["back", "full"], ["full"], []]


def test_twist_sums_across_phases():
    out = parse_trick_name("back_full_in_full_out")
    assert out.cues["twist"] == 2.0


def test_double_full_is_two_twists():
    out = parse_trick_name("back_double_full")
    assert out.cues["twist"] == 2.0


def test_double_back_is_two_flips():
    out = parse_trick_name("double_backflip")
    assert out.cues["flip"] == 2.0


def test_one_and_a_half_is_flip():
    out = parse_trick_name("one_and_a_half_frontflip")
    assert out.cues["flip"] == 1.5


def test_numeric_routes_to_flip_for_somersault_family():
    out = parse_trick_name("1080_dive_roll")
    assert out.cues["flip"] == 3.0


def test_numeric_routes_to_twist_for_turning_family():
    out = parse_trick_name("180_cat")
    assert out.cues["twist"] == 0.5


def test_numeric_abstains_when_family_ambiguous():
    out = parse_trick_name("360")
    assert "flip" not in out.cues and "twist" not in out.cues
