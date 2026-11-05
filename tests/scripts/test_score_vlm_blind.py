from scripts.prepare_vlm_blind import opaque_id
from scripts.score_vlm_blind import scored_truth, score


def labels(n, twist=lambda i: float(i % 3)):
    return {f"s{i}": {"flip": 1, "twist": twist(i), "cork": "no", "use": "one trick", "done": True} for i in range(n)}


def votes(lab, arm="a", right=lambda i: True, reverses=True):
    rows = []
    for i, (slug, truth) in enumerate(lab.items()):
        twist = truth["twist"] if right(i) else truth["twist"] + 1
        base = {"arm": arm, "id": opaque_id(slug), "flip": 1, "twist": twist, "cork": False, "unsure": False}
        rows.append({**base, "variant": "fwd", "direction": "forward"})
        rows.append({**base, "variant": "rev", "direction": "backward" if reverses else "forward"})
    return rows


def test_only_single_tricks_with_numeric_counts_are_scored():
    lab = labels(2)
    lab["several"] = {"flip": 1, "twist": 0, "use": "several tricks"}
    lab["unsure"] = {"flip": 1, "twist": "unsure", "use": "one trick"}
    assert set(scored_truth(lab)) == {"s0", "s1"}


def test_accurate_arm_with_passing_control_is_promising():
    lab = labels(30)
    assert score(lab, votes(lab), {})["a"]["verdict"] == "promising"


def test_low_accuracy_is_killed():
    lab = labels(30)
    out = score(lab, votes(lab, right=lambda i: i < 15), {})["a"]
    assert out["verdict"] == "killed" and "twist below 75%" in out["kill_reasons"]


def test_arm_that_ignores_reversal_is_killed_despite_accuracy():
    lab = labels(30)
    out = score(lab, votes(lab, reverses=False), {})["a"]
    assert out["twist_exact"] == 30 and out["kill_reasons"] == ["reversed-clip control failed"]


def test_score_must_beat_the_best_constant():
    lab = labels(30, twist=lambda i: 0.0 if i < 27 else 1.0)
    out = score(lab, votes(lab, right=lambda i: i < 27), {})["a"]
    assert out["twist_exact"] == 27 and out["best_constant"] == 27
    assert out["kill_reasons"] == ["not above the best constant"]


def test_unsure_and_missing_votes_count_as_wrong():
    lab = labels(4)
    rows = votes(lab)
    rows[0]["unsure"] = True
    rows = [r for r in rows if not (r["id"] == opaque_id("s1") and r["variant"] == "fwd")]
    assert score(lab, rows, {})["a"]["twist_exact"] == 2


def test_too_few_directional_answers_fails_the_control():
    lab = labels(30)
    rows = [{**r, "direction": None} for r in votes(lab)]
    assert score(lab, rows, {})["a"]["control"] == {"answered": 0, "reversed": 0, "passed": False}
