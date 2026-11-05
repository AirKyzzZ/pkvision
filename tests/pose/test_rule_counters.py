"""Synthetic-only tests: an orthographic skeleton with known flip and twist."""
import numpy as np
import pytest

from core.pose.rule_counters import L_HIP, L_SH, NOSE, R_HIP, R_SH, count, count_rising_edges

FPS = 30


def _body(flips, twists, view_yaw=np.pi / 2, n=30, still=15, facing=1.0):
    """Side view by default: body faces image +x, lateral axis points at the camera."""
    frames = []
    total = still + n + still
    for t in range(total):
        p = np.clip((t - still) / n, 0, 1)
        tw = 2 * np.pi * twists * p
        fl = 2 * np.pi * flips * p * facing
        pts = {NOSE: (0.1 * facing, 0, 1.15), L_SH: (0, 0.2, 1), R_SH: (0, -0.2, 1),
               L_HIP: (0, 0.22, 0), R_HIP: (0, -0.22, 0)}
        kp = np.zeros((17, 3))
        for j, (x, y, z) in pts.items():
            x1, y1 = x * np.cos(tw) - y * np.sin(tw), x * np.sin(tw) + y * np.cos(tw)
            x2, z2 = x1 * np.cos(fl) + z * np.sin(fl), -x1 * np.sin(fl) + z * np.cos(fl)
            x3 = x2 * np.cos(view_yaw - np.pi / 2) + y1 * np.sin(view_yaw - np.pi / 2)
            kp[j] = (100 * x3 + 500, 500 - 100 * z2, 0.9)
        frames.append(kp)
    return np.array(frames, dtype=np.float32)


@pytest.mark.parametrize("twists", [0.5, 1, 2, 3])
def test_side_view_counts_half_twists(twists):
    assert count(_body(1, twists), FPS).twist == twists


@pytest.mark.parametrize("flips", [1, 2])
def test_side_view_flip_count_no_twist(flips):
    out = count(_body(flips, 0), FPS)
    assert out.flip == flips
    assert out.twist == 0


def test_direction_forward_when_head_rotates_toward_facing():
    assert count(_body(1, 0), FPS).direction == "forward"
    assert count(_body(1, 0, facing=-1.0), FPS).direction == "forward"


def test_direction_backward_when_rotation_opposes_facing():
    body = _body(1, 0)
    body[..., 0] = 1000 - body[..., 0]
    body[:, NOSE, 0] = 1000 - body[:, NOSE, 0]
    assert count(body, FPS).direction == "backward"


def test_still_body_counts_zero():
    out = count(_body(0, 0), FPS)
    assert (out.flip, out.twist) == (0.0, 0.0)


def test_low_confidence_abstains():
    body = _body(1, 1)
    body[:, [L_SH, R_SH, L_HIP, R_HIP], 2] = 0.0
    out = count(body, FPS)
    assert out.twist is None and out.flip is None


def test_lr_swap_does_not_add_twists_but_is_logged():
    body = _body(0, 1, view_yaw=0.0)
    clean = count(body, FPS).twist
    mid = len(body) // 2
    body[mid, [L_HIP, R_HIP]] = body[mid, [R_HIP, L_HIP]]
    swapped = count(body, FPS)
    assert swapped.twist == clean
    assert swapped.swap_events >= 1


def test_hysteresis_ignores_noise_inside_the_band():
    assert count_rising_edges(np.array([0.1, 0.5, 0.30, 0.5, 0.1, 0.5]), 0.3125, 0.294) == 2


def test_bad_shape_raises():
    with pytest.raises(ValueError):
        count(np.zeros((5, 16, 3)), FPS)
