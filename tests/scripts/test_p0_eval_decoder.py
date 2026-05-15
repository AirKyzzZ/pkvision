from scripts.p0_eval_decoder import evaluate_rows, Row, CONFIGS


class _FakeCand:
    def __init__(self, fig_name, score, d_score, group_bonus=0.0, breakdown=None):
        self.fig_name = fig_name
        self.score = score
        self.d_score = d_score
        self.group_bonus = group_bonus
        self.breakdown = breakdown or {}


class _FakeDecoder:
    """Ranks the true trick #1 normally; ranks WRONG #1 when the canonical
    bonus is disabled — proving the harness records per-config differences and
    detects canonical-bonus dependence. Does NOT read cues internals."""
    def __init__(self, true_name, true_d):
        self.true_name = true_name
        self.true_d = true_d

    def rank(self, cues, k=5, candidate_filter=None, *,
             disable_canonical=False, disable_group_bonus=False):
        good = _FakeCand(self.true_name, 9.0, self.true_d)
        bad = _FakeCand("WRONG TRICK", 8.0, self.true_d + 5.0)
        order = [bad, good] if disable_canonical else [good, bad]
        return order[:k]


def test_metrics_and_config_differentiation():
    rows = [Row(clip="a.npy", true_trick="Gainer",
                cues={"context": "acrobatics", "flip": 1.0},
                d_score=2.0, disambig_group="g1")]
    res = evaluate_rows(rows, decoder=_FakeDecoder("Gainer", 2.0))
    assert set(res.keys()) >= set(CONFIGS)
    assert res["full"]["top1"] == 1.0
    assert res["full"]["top3"] == 1.0
    assert res["no_canonical"]["top1"] == 0.0          # canonical-dependence detected
    assert res["full"]["d_score_mae"] == 0.0           # correct trick -> 0 error
    assert res["no_canonical"]["d_score_mae"] == 5.0   # WRONG trick d_score off by 5
    assert "g1" in res["full"]["per_group_top1"]
    assert res["full"]["per_group_top1"]["g1"] == 1.0
