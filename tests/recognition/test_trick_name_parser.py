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
