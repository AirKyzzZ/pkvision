#!/usr/bin/env python3
"""Disclosure only: rule-counter scores with the activity window as the pre-registration text reads.

The frozen counter median-filters angular speed and hip rate when finding the window. This
removes those filters for the window only, without changing core/pose/rule_counters.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import core.pose.rule_counters as rc
from scripts.probe_rule_counter import ARMS, OUT, _hit, decision_12, reference_set

FROZEN_WINDOW = rc.activity_window


def literal_window(kps: np.ndarray, fps: float) -> tuple[int, int] | None:
    idx = [rc.L_SH, rc.R_SH, rc.L_HIP, rc.R_HIP]
    valid = (kps[:, idx, 2] >= rc.CONF_VALID).all(1)
    if valid.sum() < 2:
        return None
    xy = rc._interp(kps[:, idx, :2].reshape(len(kps), -1), valid).reshape(len(kps), 4, 2)
    torso = xy[:, :2].mean(1) - xy[:, 2:].mean(1)
    scale = float(np.median(np.linalg.norm(torso, axis=1)))
    if scale <= 0:
        return None
    angle = np.degrees(np.unwrap(np.arctan2(torso[:, 0], -torso[:, 1])))
    hips = np.linalg.norm(xy[:, 3] - xy[:, 2], axis=1) / scale
    speed = np.abs(np.diff(angle, prepend=angle[0])) * fps
    rate = np.abs(np.diff(hips, prepend=hips[0])) * fps
    run = rc._largest_run((speed > rc.ANG_SPEED_DEG_S) | (rate > rc.H_RATE_PER_S), int(round(rc.MERGE_GAP_S * fps)))
    if run is None:
        return None
    pad = int(round(rc.PAD_S * fps))
    return max(0, run[0] - pad), min(len(angle) - 1, run[1] + pad)


def main() -> None:
    ref = reference_set()
    d12 = {s for group in decision_12(ref).values() for s in group}
    for arm in ARMS:
        for mode in ("frozen", "literal"):
            decision = twist = flip = covered = 0
            for slug, truth in ref.items():
                with np.load(OUT / f"{slug}.npz") as z:
                    kps, fps = z[arm], float(z["fps"])
                if mode == "literal":
                    window = literal_window(kps, fps)
                    rc.activity_window = lambda *_, w=window: w
                else:
                    rc.activity_window = FROZEN_WINDOW
                c = rc.count(kps, fps)
                hit = _hit(c.twist, truth["twist"])
                decision += slug in d12 and hit
                twist += hit
                flip += _hit(c.flip, truth["flip"])
                covered += c.twist is not None
            print(f"{arm} {mode}: decision12 {decision}, twist {twist}, flip {flip}, coverage {covered}")
    rc.activity_window = FROZEN_WINDOW


if __name__ == "__main__":
    main()
