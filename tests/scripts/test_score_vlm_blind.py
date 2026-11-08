import pytest

from scripts.prepare_vlm_blind import opaque_id
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from scripts.score_vlm_blind import ARMS, control_slugs, parse_reply, score, split_labels


def labels(n, twist=lambda i: float(i % 3)):
    return {f"s{i}": {"flip": 1, "twist": twist(i), "cork": "no", "use": "one trick", "done": True} for i in range(n)}


def votes(lab, arm="G-pro", right=lambda i: True, reverses=True):
    rows = []
    for i, (slug, truth) in enumerate(lab.items()):
        twist = truth["twist"] if right(i) else truth["twist"] + 1
        base = {"arm": arm, "id": opaque_id(slug), "flip": 1, "twist": twist, "cork": False, "unsure": False}
        rows.append({**base, "variant": "fwd", "direction": "forward"})
        rows.append({**base, "variant": "rev", "direction": "backward" if reverses else "forward"})
    return rows


def run(lab, rows, arm="G-pro"):
    return score(lab, rows, set(lab))["arms"][arm]


def test_only_finished_single_tricks_with_numeric_counts_are_scored():
    lab = labels(2)
    lab["several"] = {"flip": 1, "twist": 0, "use": "several tricks", "done": True}
    lab["caption"] = {"flip": 1, "twist": 0, "use": "name on screen", "done": True}
    lab["unsure"] = {"flip": 1, "twist": "unsure", "use": "one trick", "done": True}
    lab["early"] = {"flip": 1, "twist": 0, "use": "one trick", "done": False}
    truth, excluded = split_labels(lab, set(lab) | {"never"})
    assert set(truth) == {"s0", "s1"}
    assert excluded == {"several": "several tricks", "caption": "name on screen", "unsure": "unsure count",
                        "early": "not labeled", "never": "not labeled"}


def test_name_on_screen_without_counts_keeps_its_reason():
    assert split_labels({"s": {"use": "name on screen", "done": False}}, {"s"})[1] == {"s": "name on screen"}


def test_labels_outside_the_queue_stop_the_run():
    with pytest.raises(ValueError):
        split_labels(labels(2), {"s0"})


def test_accurate_arm_with_passing_control_is_promising():
    lab = labels(30)
    assert run(lab, votes(lab))["verdict"] == "promising"


def test_low_accuracy_is_killed():
    lab = labels(30)
    out = run(lab, votes(lab, right=lambda i: i < 15))
    assert out["verdict"] == "killed" and "twist below 75%" in out["kill_reasons"]


def test_arm_that_ignores_reversal_is_killed_despite_accuracy():
    lab = labels(30)
    out = run(lab, votes(lab, reverses=False))
    assert out["twist_exact"] == 30 and out["kill_reasons"] == ["reversed-clip control failed"]


def test_best_constant_can_be_a_nonzero_twist():
    lab = labels(30, twist=lambda i: 1.0 if i < 27 else 0.0)
    out = run(lab, votes(lab, right=lambda i: i < 27))
    assert out["always_zero"] == 3 and out["best_constant"] == 27
    assert out["kill_reasons"] == ["not above the best constant"]


def test_score_must_beat_the_best_constant():
    lab = labels(30, twist=lambda i: 0.0 if i < 27 else 1.0)
    out = run(lab, votes(lab, right=lambda i: i < 27))
    assert out["twist_exact"] == 27 and out["best_constant"] == 27
    assert out["kill_reasons"] == ["not above the best constant"]


def test_too_few_scored_clips_are_not_evaluable_even_when_perfect():
    lab = labels(19)
    assert run(lab, votes(lab))["verdict"] == "not evaluable"


def test_too_few_control_clips_is_not_evaluable_rather_than_killed():
    lab = labels(30, twist=lambda i: 0.0 if i < 5 else 0.5)
    rows = [{**r, "twist": lab[next(s for s in lab if opaque_id(s) == r["id"])]["twist"]} for r in votes(lab)]
    assert run(lab, rows)["verdict"] == "not evaluable"


def test_missing_forward_votes_are_incomplete_not_killed():
    lab = labels(30)
    dropped = next(s for s in lab if s not in control_slugs(split_labels(lab, set(lab))[0]))
    rows = [r for r in votes(lab) if not (r["id"] == opaque_id(dropped) and r["variant"] == "fwd")]
    out = run(lab, rows)
    assert out["unanswered"] == 1 and out["missing_reversed"] == 0 and out["verdict"] == "incomplete"


def test_failed_calls_are_incomplete_and_can_be_retried():
    lab = labels(30)
    failed = [{"arm": r["arm"], "id": r["id"], "variant": r["variant"], "error": "429"} for r in votes(lab)]
    assert run(lab, failed)["verdict"] == "incomplete"
    assert run(lab, failed + votes(lab))["verdict"] == "promising"


def test_incomplete_comes_before_not_evaluable_and_clears_kill_reasons():
    lab = labels(19)
    out = run(lab, votes(lab)[:4])
    assert out["verdict"] == "incomplete" and out["kill_reasons"] == []


@pytest.mark.parametrize("change", [
    {"use": "a typed clip name"}, {"use": "One Trick"}, {"use": False}, {"use": ["x"]}, {"done": "false"},
    {"flip": "1"}, {"flip": 7}, {"twist": True}, {"twist": 0.3}, {"cork": True},
])
def test_labels_the_page_cannot_write_stop_the_run(change):
    lab = labels(2)
    lab["s0"].update(change)
    with pytest.raises(ValueError) as error:
        split_labels(lab, set(lab))
    assert "typed clip name" not in str(error.value)


def test_reversed_clips_never_sent_are_incomplete_not_killed():
    lab = labels(30)
    out = run(lab, [r for r in votes(lab) if r["variant"] == "fwd"])
    assert out["verdict"] == "incomplete" and out["missing_reversed"] == 12


def test_unfinished_labels_give_no_verdict_unless_accepted():
    lab = labels(30)
    queue = set(lab) | {"unplayable"}
    assert score(lab, votes(lab), queue)["arms"]["G-pro"]["verdict"] == "incomplete"
    assert score(lab, votes(lab), queue, accept_unlabeled=True)["arms"]["G-pro"]["verdict"] == "promising"


def test_control_uses_only_whole_twists_with_a_flip():
    lab = labels(40, twist=lambda i: i % 2 * 0.5)
    lab["noflip"] = {"flip": 0, "twist": 0, "cork": "no", "use": "one trick", "done": True}
    chosen = control_slugs(split_labels(lab, set(lab))[0])
    assert len(chosen) == 12 and "noflip" not in chosen
    assert all(lab[s]["twist"] == 0 for s in chosen)


def test_unsure_reversed_passes_do_not_count_as_reversed():
    lab = labels(30)
    rows = [{**r, "unsure": True} if r["variant"] == "rev" else r for r in votes(lab)]
    assert not run(lab, rows)["control"]["passed"]


def test_too_few_directional_answers_fails_the_control():
    lab = labels(30)
    rows = [{**r, "direction": None} for r in votes(lab)]
    assert run(lab, rows)["control"] == {"answered": 0, "reversed": 0, "passed": False}


@pytest.mark.parametrize("change", [
    {"flip": "banana"}, {"cork": "maybe"}, {"unsure": None}, {"twist": 1.009}, {"twist": 10 ** 400},
    {"twist": True}, {"twist": float("nan")}, {"twist": -1}, {"direction": ["forward"]}, {"direction": "up"},
])
def test_malformed_answers_count_as_wrong(change):
    lab = labels(30)
    out = run(lab, [{**r, **change} for r in votes(lab)])
    assert out["twist_exact"] == 0 and out["malformed"] == 30 and out["verdict"] == "killed"


def test_missing_fields_count_as_malformed():
    lab = labels(30)
    rows = [{k: r[k] for k in ("arm", "id", "variant", "twist", "direction")} for r in votes(lab)]
    assert run(lab, rows)["malformed"] == 30


def test_duplicate_votes_stop_the_run_rather_than_overwrite():
    lab = labels(30)
    with pytest.raises(ValueError):
        run(lab, votes(lab, right=lambda i: False) + votes(lab))


@pytest.mark.parametrize("change", [{"id": "c0_fwd"}, {"variant": "forward"}, {"arm": "G-typo"}])
def test_runner_mistakes_stop_the_run_rather_than_look_like_model_failure(change):
    lab = labels(30)
    with pytest.raises(ValueError):
        run(lab, [{**r, **change} for r in votes(lab)])


def test_non_object_row_stops_the_run():
    with pytest.raises(ValueError):
        run(labels(2), [["not", "a", "row"]])


def test_non_string_id_stops_the_run():
    with pytest.raises(ValueError):
        run(labels(2), [{"arm": "G-pro", "id": ["x"], "variant": "fwd"}])


def test_bad_label_shape_stops_the_run():
    with pytest.raises(ValueError):
        split_labels({"s": "not an object"}, {"s"})


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/score_vlm_blind.py"
EMPTY_HASH = hashlib.sha256(b"{}").hexdigest()


def cli(*args):
    return subprocess.run([sys.executable, str(SCRIPT), *map(str, args)], capture_output=True, text=True)


def empty_labels(tmp_path):
    (tmp_path / "labels.json").write_text("{}")
    (tmp_path / "votes.jsonl").write_text("")
    return tmp_path / "labels.json", tmp_path / "votes.jsonl"


def test_cli_prints_no_clip_names(tmp_path):
    queue = json.loads((SCRIPT.parents[1] / "data/labeling/retest_2026-11/queue.json").read_text())
    labels_path, votes_path = empty_labels(tmp_path)
    run_ok = cli("score", labels_path, votes_path, "--labels-sha256", EMPTY_HASH)
    assert run_ok.returncode == 0 and not any(c["slug"] in run_ok.stdout for c in queue)
    assert json.loads(run_ok.stdout)["excluded_by_reason"] == {"not labeled": len(queue)}
    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps({"arm": "G-pro", "id": queue[0]["slug"], "variant": "fwd"}))
    failed = cli("score", labels_path, bad, "--labels-sha256", EMPTY_HASH)
    assert failed.returncode != 0 and queue[0]["slug"] not in failed.stdout + failed.stderr


def test_cli_requires_and_enforces_the_labels_hash(tmp_path):
    labels_path, votes_path = empty_labels(tmp_path)
    assert cli("score", labels_path, votes_path).returncode != 0
    wrong = cli("score", labels_path, votes_path, "--labels-sha256", "0" * 64)
    assert wrong.returncode != 0 and "does not match" in wrong.stderr
    assert cli("score", labels_path, votes_path, "--labels-sha256", EMPTY_HASH.upper()).returncode == 0


def test_control_ids_refuses_unfinished_labels_and_sets_without_a_verdict(tmp_path):
    labels_path, _ = empty_labels(tmp_path)
    unfinished = cli("control-ids", labels_path, "--labels-sha256", EMPTY_HASH)
    assert unfinished.returncode != 0 and "unfinished" in unfinished.stderr
    hopeless = cli("control-ids", labels_path, "--labels-sha256", EMPTY_HASH, "--accept-unlabeled")
    assert hopeless.returncode != 0 and "cannot reach a verdict" in hopeless.stderr


def test_accuracy_bar_is_inclusive_at_75_percent():
    lab = labels(20)
    assert run(lab, votes(lab, right=lambda i: i < 15))["verdict"] == "promising"
    assert run(lab, votes(lab, right=lambda i: i < 14))["kill_reasons"] == ["twist below 75%"]


def test_exactly_20_scored_clips_are_evaluable():
    lab = labels(20)
    assert run(lab, votes(lab))["verdict"] == "promising"


def test_exactly_6_control_clips_are_evaluable():
    lab = labels(20, twist=lambda i: 0.0 if i < 6 else 0.5)
    assert len(control_slugs(split_labels(lab, set(lab))[0])) == 6
    assert run(lab, votes(lab))["verdict"] == "promising"


def test_control_bar_is_inclusive_at_75_percent():
    lab = labels(30)
    chosen = {opaque_id(s) for s in control_slugs(split_labels(lab, set(lab))[0])}

    def flipped_after(limit):
        seen, rows = 0, []
        for row in votes(lab):
            if row["variant"] == "rev" and row["id"] in chosen:
                seen += 1
                if seen > limit:
                    row = {**row, "direction": "forward"}
            rows.append(row)
        return rows

    assert run(lab, flipped_after(9))["control"] == {"answered": 12, "reversed": 9, "passed": True}
    assert run(lab, flipped_after(8))["control"] == {"answered": 12, "reversed": 8, "passed": False}


def test_parse_reply_keeps_model_types_and_strips_a_fence():
    reply = '```json\n{"flip": 1, "twist": "2", "direction": null, "cork": false, "unsure": false}\n```'
    assert parse_reply(reply) == {"flip": 1, "twist": "2", "direction": None, "cork": False, "unsure": False}


@pytest.mark.parametrize("reply", ["I think two twists", "[1, 2]", '{"flip": 1}', ""])
def test_parse_reply_rejects_what_it_cannot_read(reply):
    with pytest.raises(ValueError):
        parse_reply(reply)


def test_missing_arms_are_reported():
    lab = labels(30)
    arms = score(lab, votes(lab), set(lab))["arms"]
    assert set(arms) == set(ARMS) and arms["C-grid"]["verdict"] == "missing"


def test_breakdowns_use_owner_labels():
    lab = labels(30, twist=lambda i: 2.0 if i < 3 else 0.0)
    lab["s0"]["cork"] = "yes"
    out = run(lab, votes(lab))
    assert out["twist_1.5_plus"] == "3/3" and out["cork"] == "1/1" and out["by_owner_twist"] == {"0.0": "27/27", "2.0": "3/3"}
