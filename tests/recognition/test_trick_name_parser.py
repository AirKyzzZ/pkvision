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
