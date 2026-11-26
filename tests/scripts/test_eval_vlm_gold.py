import json

from scripts import eval_vlm_gold as gold
from scripts.prepare_vlm_blind import file_id
from scripts.run_vlm_blind import FakeProvider


def test_predicted_uses_the_cue_model_bins_and_fails_closed():
    answer = {"flip": 1.0, "twist": 2.0, "direction": None, "unsure": False}
    assert [gold.predicted(answer, cue) for cue in gold.CUES] == [1, 3, 2]
    assert gold.predicted({**answer, "unsure": True}, "flip") == -1
    assert gold.predicted({"flip": None, "twist": None, "direction": None}, "twist") == -1
    assert gold.predicted({**answer, "direction": "diagonal"}, "direction") == -1
    assert gold.predicted(None, "flip") == -1


def test_report_scores_missing_votes_as_wrong():
    truth = {"a": {"flip": 1, "twist": 0, "direction": 0}, "b": {"flip": 1, "twist": 0, "direction": 0}}
    rows = [{"arm": "G-grid", "slug": "a", "flip": 1, "twist": 0, "direction": "backward", "unsure": False},
            {"arm": "G-grid", "slug": "b", "error": "timeout"}]
    arm = gold.report(rows, truth)["G-grid"]
    assert arm["answered"] == 1 and arm["failed_calls"] == 1
    assert arm["per_cue"]["flip"]["acc"] == 0.5 and arm["per_cue"]["flip"]["by_true_class"] == {"1": "1/2"}


def test_run_is_append_only_and_retries_only_failures(tmp_path):
    upload = tmp_path / "upload"
    upload.mkdir()
    for slug in ("a", "b"):
        for n in (1, 2):
            (upload / f"{file_id(slug, 'gold')}_grid{n}.jpg").write_bytes(b"g")
    votes = tmp_path / "votes.jsonl"
    votes.write_text(json.dumps({"arm": "G-grid", "slug": "a", "error": "timeout"}) + "\n")
    first = gold.run(["G-grid"], 2, 1.0, FakeProvider(), votes, upload, ["a", "b"])
    again = gold.run(["G-grid"], 2, 1.0, FakeProvider(), votes, upload, ["a", "b"])
    assert first["made"] == 2 and again["made"] == 0
    assert len(gold.read_rows(votes)) == 3
