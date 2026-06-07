from scripts.build_attribute_dataset import merge_layered


def test_parser_fills_missing_direction():
    base = {"flip": 1.0, "twist": 0.0, "direction": "none"}
    parsed = {"cues": {"direction": "backward"}, "confidence": {"direction": 0.9}}
    merged, prov, changes = merge_layered("back_x", base, "unified", parsed, conf_thresh=0.7)
    assert merged["direction"] == "backward"
    assert prov["direction"] == "parser"
    assert any(c["kind"] == "fill" for c in changes)


def test_parser_overrides_unified_when_confident():
    base = {"flip": 1.0, "twist": 0.0, "direction": "forward"}
    parsed = {"cues": {"direction": "backward"}, "confidence": {"direction": 0.9}}
    merged, prov, changes = merge_layered("g", base, "unified", parsed, conf_thresh=0.7)
    assert merged["direction"] == "backward"
    assert any(c["kind"] == "correct" for c in changes)


def test_fig_wins_but_logs_disagreement():
    base = {"flip": 1.0, "twist": 0.0, "direction": "forward"}
    parsed = {"cues": {"direction": "backward"}, "confidence": {"direction": 0.9}}
    merged, prov, changes = merge_layered("g", base, "fig", parsed, conf_thresh=0.7)
    assert merged["direction"] == "forward"
    assert prov["direction"] == "fig"
    assert any(c["kind"] == "fig_disagree" for c in changes)


def test_untrusted_cue_is_never_merged():
    base = {"context": "acrobatics", "flip": 1.0}
    parsed = {"cues": {"context": "wall", "flip": 1.0}, "confidence": {"context": 0.95, "flip": 0.95}}
    merged, prov, changes = merge_layered("x", base, "unified", parsed, conf_thresh=0.7)
    assert merged["context"] == "acrobatics"
    assert all(c["cue"] != "context" for c in changes)
