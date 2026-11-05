"""Pick the performing athlete from per-frame multi-person keypoints.

Tracks people across frames, then keeps the track that rotates and travels most.
Size plays no part in the choice, so a large, still spectator cannot win.
Frames outside the chosen track are zero (invalid), not filled with someone else.
Frozen before its first score on real clips. Rationale:
docs/science-superpowers/probes/2026-11-04-person-selection-audit.md
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

CONF_VALID = 0.3
MIN_VALID_KPS = 6
MAX_GAP_S = 0.5  # the detector can lose an inverted athlete for a few frames
MAX_STEP_PER_FRAME = 0.6  # body diagonals per elapsed frame
MAX_STEP = 1.5  # body diagonals, whatever the gap: farther than this is another person or a cut
ROT_DEADBAND_DEG_S = 150.0  # a standing or walking torso stays under this; a flip does not
MAX_ROT_STEP_DEG = 60.0  # larger steps are keypoint noise
ROT_SPAN_S = 0.1  # longest gap across which torso rotation is measured
TRAVEL_WEIGHT_DEG = 90.0  # one body diagonal of travel counts as this much rotation
L_SH, R_SH, L_HIP, R_HIP = 5, 6, 11, 12


@dataclass
class _Track:
    frames: list[int] = field(default_factory=list)
    joints: list[np.ndarray] = field(default_factory=list)
    centers: list[np.ndarray] = field(default_factory=list)
    diags: list[float] = field(default_factory=list)


def _box(joints: np.ndarray) -> tuple[np.ndarray, float] | None:
    pts = joints[joints[:, 2] > CONF_VALID, :2]
    if len(pts) < MIN_VALID_KPS:
        return None
    low, high = pts.min(0), pts.max(0)
    diag = float(np.hypot(*(high - low)))
    return ((low + high) / 2, diag) if diag > 0 else None


def _torso_angle(joints: np.ndarray) -> float | None:
    if (joints[[L_SH, R_SH, L_HIP, R_HIP], 2] < CONF_VALID).any():
        return None
    v = joints[[L_SH, R_SH], :2].mean(0) - joints[[L_HIP, R_HIP], :2].mean(0)
    return float(np.degrees(np.arctan2(v[0], -v[1])))


def build_tracks(frames: list[np.ndarray], fps: float) -> list[_Track]:
    """Greedy nearest-centre association. Each element of frames is (N, 17, 3)."""
    max_gap = max(1, round(MAX_GAP_S * fps))
    tracks: list[_Track] = []
    for t, people in enumerate(frames):
        boxes = [(p, b) for p in people if (b := _box(p)) is not None]
        pairs = []
        for ti, tr in enumerate(tracks):
            gap = t - tr.frames[-1]
            if gap > max_gap:
                continue
            for pi, (_, (center, diag)) in enumerate(boxes):
                step = float(np.linalg.norm(center - tr.centers[-1])) / max(diag, tr.diags[-1])
                if step <= min(MAX_STEP_PER_FRAME * gap, MAX_STEP):
                    pairs.append((step, ti, pi))
        used_t, used_p = set(), set()
        for _, ti, pi in sorted(pairs):
            if ti in used_t or pi in used_p:
                continue
            used_t.add(ti), used_p.add(pi)
            _extend(tracks[ti], t, boxes[pi])
        for pi, item in enumerate(boxes):
            if pi not in used_p:
                tracks.append(_Track())
                _extend(tracks[-1], t, item)
    return tracks


def _extend(track: _Track, t: int, item) -> None:
    joints, (center, diag) = item
    track.frames.append(t), track.joints.append(joints), track.centers.append(center), track.diags.append(diag)


def track_activity(track: _Track, fps: float) -> float:
    """Rotation in degrees above the deadband, plus weighted travel in body diagonals."""
    frames = np.array(track.frames)
    if len(frames) < 2:
        return 0.0
    step = np.linalg.norm(np.diff(np.array(track.centers), axis=0), axis=1)
    travel = float((step / np.median(track.diags)).sum())
    rotation, prev_t, prev_a = 0.0, None, None
    for t, joints in zip(frames, track.joints):
        angle = _torso_angle(joints)
        if angle is None:
            continue
        if prev_a is not None and (t - prev_t) / fps <= ROT_SPAN_S:
            delta = abs((angle - prev_a + 180.0) % 360.0 - 180.0)
            if ROT_DEADBAND_DEG_S * (t - prev_t) / fps < delta <= MAX_ROT_STEP_DEG * (t - prev_t):
                rotation += delta
        prev_t, prev_a = t, angle
    return rotation + TRAVEL_WEIGHT_DEG * travel


def select_athlete(frames: list[np.ndarray], fps: float) -> np.ndarray:
    """(T, 17, 3) keypoints of the most active track, zero where it is absent."""
    out = np.zeros((len(frames), 17, 3), np.float32)
    tracks = build_tracks(frames, fps)
    if not tracks:
        return out
    best = max(tracks, key=lambda tr: track_activity(tr, fps))
    for t, joints in zip(best.frames, best.joints):
        out[t] = joints
    return out
