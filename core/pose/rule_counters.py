"""Training-free flip/twist/direction counters on COCO-17 2D keypoints (NS-AQA style).

Pre-registered in docs/science-superpowers/probes/2026-11-04-rule-counter-prereg.md.
Do not tune thresholds against the probe clips.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

NOSE, L_SH, R_SH, L_HIP, R_HIP = 0, 5, 6, 11, 12
CONF_VALID = 0.3
OUTER, INNER, VALID = 1 / 3.2, 1 / 3.4, 1 / 1.5
ANG_SPEED_DEG_S = 200.0
H_RATE_PER_S = 1.5
MERGE_GAP_S = 0.2
PAD_S = 0.1
MIN_VALID_FRAC = 0.6
MIN_DIRECTION_DEG = 180.0


@dataclass
class Counts:
    flip: float | None
    twist: float | None
    twist_shoulder: float | None
    direction: str | None
    window: tuple[int, int] | None
    valid_frac: float
    swap_events: int
    net_angle_deg: float | None


def _interp(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    t = np.arange(len(values))
    return np.stack([np.interp(t, t[valid], values[valid, i]) for i in range(values.shape[1])], 1)


def _median3(x: np.ndarray) -> np.ndarray:
    if len(x) < 3:
        return x.copy()
    padded = np.r_[x[0], x, x[-1]]
    return np.median(np.lib.stride_tricks.sliding_window_view(padded, 3), axis=1)


def _largest_run(mask: np.ndarray, gap: int) -> tuple[int, int] | None:
    idx = np.flatnonzero(mask)
    if not len(idx):
        return None
    runs, start, prev = [], idx[0], idx[0]
    for i in idx[1:]:
        if i - prev > gap + 1:
            runs.append((start, prev))
            start = i
        prev = i
    runs.append((start, prev))
    return max(runs, key=lambda r: r[1] - r[0])


def count_rising_edges(values: np.ndarray, outer: float, inner: float, reject: float | None = None) -> int:
    """Hysteresis lobes: a count needs a drop below inner, then a rise above outer."""
    armed, edges = None, 0
    for v in values:
        if reject is not None and v > reject:
            continue
        if armed is None:
            armed = v < inner
        elif armed and v > outer:
            edges += 1
            armed = False
        elif not armed and v < inner:
            armed = True
    return edges


def activity_window(angle_deg: np.ndarray, h_len: np.ndarray, fps: float) -> tuple[int, int] | None:
    speed = np.abs(np.diff(angle_deg, prepend=angle_deg[0])) * fps
    rate = np.abs(np.diff(h_len, prepend=h_len[0])) * fps
    run = _largest_run((_median3(speed) > ANG_SPEED_DEG_S) | (_median3(rate) > H_RATE_PER_S),
                       int(round(MERGE_GAP_S * fps)))
    if run is None:
        return None
    pad = int(round(PAD_S * fps))
    return max(0, run[0] - pad), min(len(angle_deg) - 1, run[1] + pad)


def count(kps: np.ndarray, fps: float) -> Counts:
    if kps.ndim != 3 or kps.shape[1:] != (17, 3):
        raise ValueError(f"expected (T, 17, 3), got {kps.shape}")
    torso_idx = [L_SH, R_SH, L_HIP, R_HIP]
    valid = (kps[:, torso_idx, 2] >= CONF_VALID).all(1)
    if valid.sum() < 2:
        return Counts(None, None, None, None, None, float(valid.mean()) if len(kps) else 0.0, 0, None)

    xy = _interp(kps[:, torso_idx, :2].reshape(len(kps), -1), valid).reshape(len(kps), 4, 2)
    mid_sh, mid_hip = xy[:, :2].mean(1), xy[:, 2:].mean(1)
    v = mid_sh - mid_hip
    scale = float(np.median(np.linalg.norm(v, axis=1)))
    if scale <= 0:
        return Counts(None, None, None, None, None, float(valid.mean()), 0, None)

    angle = np.degrees(np.unwrap(np.arctan2(v[:, 0], -v[:, 1])))
    h = xy[:, 3] - xy[:, 2]
    s = xy[:, 1] - xy[:, 0]
    h_len = _median3(np.linalg.norm(h, axis=1) / scale)
    s_len = _median3(np.linalg.norm(s, axis=1) / scale)

    window = activity_window(angle, h_len, fps)
    if window is None:
        return Counts(0.0, 0.0, 0.0, None, None, float(valid.mean()), 0, 0.0)
    a, b = window
    valid_frac = float(valid[a:b + 1].mean())
    if valid_frac < MIN_VALID_FRAC:
        return Counts(None, None, None, None, window, valid_frac, 0, None)

    net = float(np.median(angle[b - 1:b + 1]) - np.median(angle[a:a + 2]))
    flip = round(abs(net) / 360 * 2) / 2

    twist = count_rising_edges(h_len[a:b + 1], OUTER, INNER, reject=VALID) * 0.5
    ratio = float(np.median(s_len[a:b + 1]) / max(np.median(h_len[a:b + 1]), 1e-6))
    twist_s = count_rising_edges(s_len[a:b + 1], OUTER * ratio, INNER * ratio) * 0.5

    direction = None
    start_nose = kps[a, NOSE]
    if abs(net) >= MIN_DIRECTION_DEG and start_nose[2] >= CONF_VALID:
        facing = np.sign(start_nose[0] - mid_sh[a, 0])
        if facing != 0:
            direction = "forward" if np.sign(net) == facing else "backward"

    above = h_len[a:b + 1] > OUTER
    sign = np.sign(h[a:b + 1, 0])
    swaps = int(((sign[1:] != sign[:-1]) & above[1:] & above[:-1]).sum())
    return Counts(flip, twist, twist_s, direction, window, valid_frac, swaps, net)
