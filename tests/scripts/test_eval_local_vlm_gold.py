import json

import pytest

from scripts.eval_local_vlm_gold import bin_count, is_aerial, order, report, run, signature
from scripts.run_vlm_blind import ARM_SPECS, KEYLESS_PROVIDERS
from scripts.score_vlm_blind import ALL_ARMS, ARMS, EXPLORATORY_ARMS, score


def test_bins_follow_the_attribute_dataset():
    assert [bin_count(v) for v in (0, 0.5, 1, 1.5, 2, 3, 2.5)] == ["0", "0.5", "1", "1.5", "2", "3+", "3+"]
    assert bin_count("1") == "1" and bin_count(None) is None and bin_count("many") is None


def test_signature_reads_as_a_trick():
    assert signature(1, 1, "backward") == "backward 1 flip 1 twist"
    assert signature(None, 0, None) == "none ? flip 0 twist"


def gold(n, aerial=True):
    return {f"t{i}": {"flip": 1.0, "twist": float(i % 3), "direction": "backward",
                      "context": "acrobatics" if aerial else "wall", "family": "flip" if aerial else "vault",
                      "source": "unified", "split": "test"} for i in range(n)}


def test_order_excludes_the_blind_queue_and_puts_aerial_flips_first():
    g = dict(gold(4), **{f"w{i}": v for i, v in enumerate(gold(3, aerial=False).values())})
    slugs = order(g, {"t0", "w0"})
    assert "t0" not in slugs and "w0" not in slugs and len(slugs) == 5
    assert [is_aerial(g[s]) for s in slugs] == [True, True, True, False, False]
    assert order(g, set(), aerial_only=True) == order(g, set())[:4]


class Oracle:
    """Answers the gold for twist, always 1 flip, so flip accuracy equals the share of 1-flip golds."""

    def __init__(self, g):
        self.g, self.i = g, 0

    def ask_images(self, images):
        return "", {"seconds": 0.1}


def test_run_resumes_and_report_scores_against_gold(tmp_path, monkeypatch):
    g = gold(6)
    out = tmp_path / "preds.jsonl"
    answers = iter([{"flip": 1, "twist": g[s]["twist"], "direction": "backward", "cork": False, "unsure": False} for s in sorted(g)])
    monkeypatch.setattr("scripts.eval_local_vlm_gold.ask_clip", lambda provider, path: dict(next(answers), usage={"seconds": 0.1}))
    summary = run(sorted(g), g, out, limit=4, provider=object())
    assert summary["made"] == 4 and summary["answered"] == 4
    summary = run(sorted(g), g, out, provider=object())
    assert summary["already"] == 4 and summary["made"] == 2
    rows = [json.loads(l) for l in out.read_text().splitlines()]
    rep = report(rows)
    assert rep["scored"] == 6 and rep["attributes"]["twist"]["accuracy"] == 1.0 and rep["attributes"]["flip"]["accuracy"] == 1.0
    assert rep["trick"]["exact_accuracy"] == 1.0 and rep["attributes"]["twist"]["beats_constant"]
    assert rep["by_group"]["aerial_flips"]["n"] == 6


def test_report_counts_malformed_as_wrong_and_shows_confusions():
    rows = [{"slug": "a", "gold": {"flip": 1, "twist": 1, "direction": "backward", "context": "acrobatics"},
             "flip": 1, "twist": 0, "direction": "backward", "cork": False, "unsure": False},
            {"slug": "b", "gold": {"flip": 1, "twist": 1, "direction": "backward", "context": "acrobatics"},
             "flip": None, "twist": None, "direction": None, "cork": None, "unsure": None, "raw": "sorry"},
            {"slug": "c", "gold": {"flip": 1, "twist": 0, "direction": "forward", "context": "wall"}, "error": "boom"}]
    rep = report(rows)
    assert rep["scored"] == 2 and rep["failed"] == 1 and rep["malformed"] == 1
    assert rep["attributes"]["twist"]["accuracy"] == 0.0 and rep["attributes"]["flip"]["accuracy"] == 0.5
    assert rep["confusion"]["twist"]["1"] == {"0": 1, "None": 1}
    assert rep["trick"]["top_confusions"][0]["gold"] == "backward 1 flip 1 twist"


def test_local_arm_is_exploratory_and_outside_the_verdict():
    assert "L-grid" in EXPLORATORY_ARMS and "L-grid" not in ARMS and set(ARM_SPECS) == set(ALL_ARMS)
    assert ARM_SPECS["L-grid"]["price"] == (0.0, 0.0) and ARM_SPECS["L-grid"]["provider"] in KEYLESS_PROVIDERS
    result = score({}, [], set(), accept_unlabeled=True)
    assert set(result["arms"]) == set(ARMS) and result["exploratory_arms"] == {}


def test_report_on_nothing():
    assert report([])["scored"] == 0
