import pytest

from scripts.prepare_vlm_blind import opaque_id
from scripts.score_vlm_blind import ARMS, control_slugs, score, split_labels


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


def test_score_must_beat_the_best_constant():
    lab = labels(30, twist=lambda i: 0.0 if i < 27 else 1.0)
    out = run(lab, votes(lab, right=lambda i: i < 27))
    assert out["twist_exact"] == 27 and out["best_constant"] == 27
    assert out["kill_reasons"] == ["not above the best constant"]


def test_too_few_scored_clips_are_not_evaluable_even_when_perfect():
    lab = labels(19)
    assert run(lab, votes(lab))["verdict"] == "not evaluable"


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


def test_missing_arms_are_reported():
    lab = labels(30)
    arms = score(lab, votes(lab), set(lab))["arms"]
    assert set(arms) == set(ARMS) and arms["C-grid"]["verdict"] == "missing"


def test_breakdowns_use_owner_labels():
    lab = labels(30, twist=lambda i: 2.0 if i < 3 else 0.0)
    lab["s0"]["cork"] = "yes"
    out = run(lab, votes(lab))
    assert out["twist_1.5_plus"] == "3/3" and out["cork"] == "1/1" and out["by_owner_twist"] == {"0.0": "27/27", "2.0": "3/3"}
