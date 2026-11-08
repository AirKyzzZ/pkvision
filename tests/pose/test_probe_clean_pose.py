import numpy as np

from scripts.probe_clean_pose_cue import interp_missing, verdict


def seq(frames):
    out = np.zeros((len(frames), 17, 3), np.float32)
    for t, v in enumerate(frames):
        if v is not None:
            out[t, :, :2] = v
            out[t, :, 2] = 0.9
    return out


def test_interp_fills_gaps_and_keeps_zero_confidence():
    out = interp_missing(seq([0.0, None, 2.0, None]))
    assert out[1, 0, 0] == 1.0 and out[3, 0, 0] == 2.0  # held at the last detection
    assert out[1, 0, 2] == 0.0
    assert out[0, 0, 2] > 0.8


def test_interp_leaves_empty_clip_alone():
    assert interp_missing(seq([None, None])).sum() == 0


def test_verdict_thresholds():
    assert verdict([0.05, 0.04, 0.03, 0.06, -0.01]) == "helps"
    assert verdict([-0.05, -0.04, -0.03, -0.06, 0.01]) == "hurts"
    assert verdict([0.05, 0.05, 0.05, -0.01, -0.01]) == "no detectable effect"  # only 3 of 5
    assert verdict([0.02, 0.02, 0.02, 0.02, 0.02]) == "no detectable effect"  # under 0.03
