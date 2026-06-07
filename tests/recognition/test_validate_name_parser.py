from scripts.validate_name_parser import score_cue, load_gold_rows


def test_score_cue_counts_hits_and_abstains():
    pred = [{"flip": 1.0}, {}, {"flip": 2.0}]
    gold = [{"flip": 1.0}, {"flip": 1.0}, {"flip": 1.0}]
    s = score_cue("flip", pred, gold)
    assert s["fired"] == 2 and s["correct"] == 1 and s["abstained"] == 1


def test_load_gold_rows_parses_disambig_group():
    rows = load_gold_rows()
    assert len(rows) > 0
    r = rows[0]
    assert "slug" in r and "cues" in r
