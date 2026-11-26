import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.prepare_vlm_blind import file_id, opaque_id
from scripts.run_vlm_blind import ARM_SPECS, ARMS, FakeProvider, PROMPT, answered, cost_usd, plan, run
from scripts.score_vlm_blind import score, verify_stage, verify_vote_hashes

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/run_vlm_blind.py"
ORDER = [f"s{i}" for i in range(36)]


def labels(n):
    return {f"s{i}": {"flip": 1, "twist": float(i % 3), "cork": "no", "use": "one trick", "done": True} for i in range(n)}


def test_plan_sends_forward_only_for_scored_clips_and_reversed_only_for_control_ids():
    lab = labels(12)
    lab["s12"] = {"flip": 1, "twist": 0, "cork": "no", "use": "several tricks", "done": True}
    calls = plan(lab, ORDER, [], None, ["G-pro"])
    assert len(calls) == 12 and all(c["variant"] == "fwd" for c in calls)
    assert {c["id"] for c in calls} == {opaque_id(f"s{i}") for i in range(12)}
    with_control = plan(lab, ORDER, [], [opaque_id("s0"), opaque_id("s1")], ["G-pro"])
    assert sum(c["variant"] == "rev" for c in with_control) == 2
    assert all(c["file"] == file_id(ORDER[int(c["id"] == opaque_id("s1"))], "rev") for c in with_control if c["variant"] == "rev")


def test_plan_skips_answered_votes_but_retries_failed_calls():
    lab = labels(3)
    rows = [{"arm": "G-pro", "id": opaque_id("s0"), "variant": "fwd", "twist": 1.0},
            {"arm": "G-pro", "id": opaque_id("s1"), "variant": "fwd", "error": "timeout"}]
    calls = plan(lab, ORDER, rows, None, ["G-pro"])
    assert {c["id"] for c in calls} == {opaque_id("s1"), opaque_id("s2")}
    assert answered(rows) == {("G-pro", opaque_id("s0"), "fwd")}


def test_plan_refuses_control_ids_outside_the_scored_set():
    with pytest.raises(ValueError):
        plan(labels(3), ORDER, [], [opaque_id("s30")], ["G-pro"])
    with pytest.raises(ValueError):
        plan(labels(3), ORDER, [], ["c-not-a-clip"], ["G-pro"])


def test_plan_never_contains_a_slug():
    calls = plan(labels(5), ORDER, [], None, list(ARMS))
    assert len(calls) == 5 * len(ARMS) and not any(s in json.dumps(calls) for s in ORDER)


def test_frozen_prompt_matches_the_preregistration():
    prereg = (Path(__file__).resolve().parents[2] / "docs/science-superpowers/probes").glob("2026-11-04-vlm-blind-prereg*.md")
    text = next(prereg).read_text()
    assert PROMPT in text


def test_cost_uses_the_frozen_prices():
    assert cost_usd(ARM_SPECS["C-grid"], {"input_tokens": 1_000_000, "output_tokens": 0}) == 4.0
    assert cost_usd(ARM_SPECS["G-flash"], {"input_tokens": 0, "output_tokens": 1_000_000}) == 3.75


def test_fake_provider_is_deterministic_and_well_formed(tmp_path):
    f = FakeProvider()
    a, _ = f.ask(ARM_SPECS["G-grid"], [tmp_path / "x_grid1.jpg"])
    b, _ = f.ask(ARM_SPECS["G-grid"], [tmp_path / "x_grid1.jpg"])
    assert a == b and set(json.loads(a)) == {"flip", "twist", "direction", "cork", "unsure"}


def fake_upload(tmp_path, slugs, monkeypatch):
    upload = tmp_path / "upload"
    upload.mkdir()
    for slug in slugs:
        for variant in ("fwd", "rev"):
            fid = file_id(slug, variant)
            (upload / f"{fid}.mp4").write_bytes(b"v")
            for n in (1, 2):
                (upload / f"{fid}_grid{n}.jpg").write_bytes(b"g")
    monkeypatch.setattr("scripts.run_vlm_blind.UPLOAD", upload)
    return upload


def test_run_appends_rows_with_the_labels_hash_and_feeds_the_scorer(tmp_path, monkeypatch):
    lab = labels(12)
    fake_upload(tmp_path, lab, monkeypatch)
    votes = tmp_path / "votes.jsonl"
    calls = plan(lab, ORDER, [], None, ["G-pro", "C-grid"])
    summary = run(calls, "h" * 64, votes, 5.0, "fake")
    rows = [json.loads(l) for l in votes.read_text().splitlines()]
    assert summary["made"] == 24 and summary["answered"] == 24 and summary["spent_usd"] == 0
    assert all(r["labels_sha256"] == "h" * 64 and r["model"] == "fake" for r in rows)
    verify_vote_hashes(rows, {"h" * 64})
    with pytest.raises(ValueError):
        verify_vote_hashes(rows, {"other"})
    result = score(lab, rows, set(ORDER))
    assert result["arms"]["G-pro"]["verdict"] == "incomplete"  # 24 queue clips unlabeled
    assert result["arms"]["G-pro"]["n"] == 12
    assert plan(lab, ORDER, rows, None, ["G-pro", "C-grid"]) == []


def test_run_stops_at_the_cap(tmp_path, monkeypatch):
    lab = labels(4)
    fake_upload(tmp_path, lab, monkeypatch)
    votes = tmp_path / "votes.jsonl"
    votes.write_text(json.dumps({"arm": "G-pro", "id": "c0", "variant": "fwd", "cost_usd": 5.0}) + "\n")
    summary = run(plan(lab, ORDER, [], None, ["G-pro"]), "h" * 64, votes, 5.0, "fake")
    assert summary["made"] == 0 and "cap" in summary["stopped"]


def test_run_records_a_failure_as_a_row_without_an_answer(tmp_path, monkeypatch):
    lab = labels(2)
    fake_upload(tmp_path, lab, monkeypatch)
    (tmp_path / "upload" / f"{file_id('s0', 'fwd')}.mp4").unlink()
    votes = tmp_path / "votes.jsonl"
    summary = run(plan(lab, ORDER, [], None, ["G-pro"]), "h" * 64, votes, 5.0, "fake")
    rows = [json.loads(l) for l in votes.read_text().splitlines()]
    assert summary["failed"] == 1 and summary["answered"] == 1
    assert sum("error" in r for r in rows) == 1 and not any(s in votes.read_text() for s in ORDER)


def test_stage_must_be_unchanged_in_the_final_labels():
    stage = labels(12)
    final = labels(36)
    verify_stage(final, stage)
    final["s3"]["twist"] = 2.5
    with pytest.raises(ValueError):
        verify_stage(final, stage)


def cli(*args):
    return subprocess.run([sys.executable, str(SCRIPT), *map(str, args)], capture_output=True, text=True)


def test_cli_refuses_a_labels_file_that_does_not_match_its_hash(tmp_path):
    path = tmp_path / "labels.json"
    path.write_text("{}")
    out = cli("plan", "--labels", path, "--labels-sha256", "0" * 64)
    assert out.returncode != 0 and "does not match" in out.stderr
    ok = cli("plan", "--labels", path, "--labels-sha256", hashlib.sha256(b"{}").hexdigest())
    assert ok.returncode == 0 and json.loads(ok.stdout)["calls"] == 0


def test_cli_run_without_keys_needs_the_fake_provider(tmp_path, monkeypatch):
    for key in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    path = tmp_path / "labels.json"
    path.write_text("{}")
    out = cli("run", "--labels", path, "--labels-sha256", hashlib.sha256(b"{}").hexdigest(), "--votes", tmp_path / "v.jsonl")
    assert out.returncode != 0 and "no API key" in out.stderr
