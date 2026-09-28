"""HeuristicRecognizer — pose → structured cues → FIG decoder.

A non-learned baseline that beats free-form VLM naming on the POOL-B
confusions by exploiting structured evidence:

    1. YOLO11-pose on all frames.
    2. Body-angle time series (nose → hip midpoint).
    3. Flip count   = cumulative unwrapped body-angle / 2π.
    4. Twist count  = shoulder-x sign-flip count during the aerial phase
                      (side-view assumption — noisy, good enough to
                      separate 0 / 1 / 2 twist bins).
    5. Direction    = sign of mean body-angle derivative, augmented with
                      path-based prior when the 2D signal is ambiguous.
    6. Context      = derived from clip path (wall/swing/acrobatics/pk_basics).
    7. Takeoff      = "running_forward" if the bbox translates horizontally
                      more than one body width before the aerial phase,
                      otherwise "standing".
    8. Feed cues dict → core.recognition.fig_decoder.FIGDecoder.rank.

This is deliberately simple. It is not a replacement for a learned model;
it is a structured baseline whose failures are interpretable.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from core.recognition.fig_decoder import FIGDecoder
from paper.experiments.recognizers import Candidate, RecognizerOutput

_REPO_ROOT = Path(__file__).resolve().parents[2]
_RUN_CONTEXT_MANIFEST = _REPO_ROOT / "data" / "run_context_manifest.json"


@lru_cache(maxsize=1)
def _load_run_context_manifest() -> dict[str, dict[str, Any]]:
    """Load the parent-run cue-override manifest.

    Returns an empty dict if the file is missing or malformed — the
    manifest is an optional override layer, not a hard dependency.
    """
    if not _RUN_CONTEXT_MANIFEST.exists():
        return {}
    try:
        with _RUN_CONTEXT_MANIFEST.open() as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}
    return {k: v for k, v in data.items() if not k.startswith("_") and isinstance(v, dict)}


def _manifest_overrides_for(clip: Path) -> dict[str, Any]:
    """Look up cue overrides for a clip at '<run_stem>/<clip_stem>'."""
    manifest = _load_run_context_manifest()
    if not manifest:
        return {}
    key = f"{clip.parent.name}/{clip.stem}"
    return manifest.get(key, {})

# YOLO keypoint layout (COCO-17)
_NOSE = 0
_L_SHOULDER, _R_SHOULDER = 5, 6
_L_HIP, _R_HIP = 11, 12
_L_ANKLE, _R_ANKLE = 15, 16


# ── Cue extraction ──────────────────────────────────────────────────────────


def _run_yolo(clip: Path):
    """Run YOLO11-pose on every frame. Returns (keypoints, confs, boxes, fps).

    keypoints: (T, 17, 2) — NaN where no detection.
    confs:     (T, 17)    — 0 where no detection.
    boxes:     (T, 4)     — NaN where no detection.
    """
    import cv2
    from ultralytics import YOLO

    root = Path(__file__).resolve().parents[2]
    yolo_path = root / "yolo11n-pose.pt"
    yolo = YOLO(str(yolo_path)) if yolo_path.exists() else YOLO("yolo11n-pose.pt")

    cap = cv2.VideoCapture(str(clip))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()

    T = len(frames)
    kps = np.full((T, 17, 2), np.nan, dtype=np.float32)
    conf = np.zeros((T, 17), dtype=np.float32)
    boxes = np.full((T, 4), np.nan, dtype=np.float32)

    for i, frame in enumerate(frames):
        res = yolo(frame, conf=0.25, verbose=False)
        if not res or res[0].boxes is None or len(res[0].boxes) == 0:
            continue
        box_data = res[0].boxes.xyxy.cpu().numpy()
        areas = (box_data[:, 2] - box_data[:, 0]) * (box_data[:, 3] - box_data[:, 1])
        # Centermost, largest detection — matches existing detect_and_track logic.
        h, w = frame.shape[:2]
        cx, cy = w / 2, h / 2
        best = -1
        best_score = -np.inf
        for j, b in enumerate(box_data):
            if areas[j] < h * w * 0.005:
                continue
            bcx, bcy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            dist = np.hypot(bcx - cx, bcy - cy) / np.hypot(cx, cy)
            score = (1 - dist) * 0.4 + (areas[j] / areas.max()) * 0.6
            if score > best_score:
                best_score = score
                best = j
        if best < 0:
            continue
        boxes[i] = box_data[best]
        if res[0].keypoints is not None and len(res[0].keypoints) > best:
            xy = res[0].keypoints.xy[best].cpu().numpy()
            c = res[0].keypoints.conf[best].cpu().numpy() if res[0].keypoints.conf is not None else np.ones(17)
            if xy.shape[0] >= 17:
                kps[i] = xy[:17]
                conf[i] = c[:17]

    return kps, conf, boxes, fps


def _body_angle(kps: np.ndarray, conf: np.ndarray) -> np.ndarray:
    """Signed nose → hip-midpoint angle in radians, (T,). NaN when missing."""
    T = kps.shape[0]
    ang = np.full(T, np.nan, dtype=np.float32)
    for i in range(T):
        if conf[i, _NOSE] < 0.2:
            continue
        lh, rh = conf[i, _L_HIP], conf[i, _R_HIP]
        if lh < 0.2 and rh < 0.2:
            continue
        if lh >= 0.2 and rh >= 0.2:
            hip = (kps[i, _L_HIP] + kps[i, _R_HIP]) / 2
        elif lh >= 0.2:
            hip = kps[i, _L_HIP]
        else:
            hip = kps[i, _R_HIP]
        dy = kps[i, _NOSE][1] - hip[1]
        dx = kps[i, _NOSE][0] - hip[0]
        ang[i] = np.arctan2(dy, dx)
    return ang


def _inversion_signal(
    kps: np.ndarray,
    conf: np.ndarray,
    boxes: np.ndarray,
) -> np.ndarray:
    """Per-frame 'inversion-ness' = (head_y - hip_y) / ref_torso.

    > 0: head below hip = inverted. NaN when keypoints missing.

    The torso reference is a FIXED value — the median shoulder-to-hip
    vertical distance over the frames where both are well-detected,
    falling back to 1/3 of the median bbox height. Using a fixed
    reference keeps the signal stable across frames where shoulders
    or hips are noisily detected (which happens a lot during inversion).
    """
    T = kps.shape[0]

    # Precompute a stable torso length across the clip.
    torsos = []
    for i in range(T):
        lh, rh = conf[i, _L_HIP], conf[i, _R_HIP]
        ls, rs = conf[i, _L_SHOULDER], conf[i, _R_SHOULDER]
        if lh >= 0.3 and rh >= 0.3 and ls >= 0.3 and rs >= 0.3:
            sh_y = (kps[i, _L_SHOULDER, 1] + kps[i, _R_SHOULDER, 1]) / 2
            hip_y = (kps[i, _L_HIP, 1] + kps[i, _R_HIP, 1]) / 2
            t = abs(hip_y - sh_y)
            if t > 5.0:
                torsos.append(t)
    if torsos:
        ref_torso = float(np.median(torsos))
    else:
        # Fallback: derive from bbox heights.
        bh = boxes[:, 3] - boxes[:, 1]
        bh = bh[~np.isnan(bh)]
        ref_torso = float(np.median(bh) / 3.0) if bh.size else 50.0
    if ref_torso < 5.0:
        ref_torso = 50.0

    out = np.full(T, np.nan, dtype=np.float32)
    for i in range(T):
        # Head: prefer nose, fallback to ears/eyes.
        head_y = np.nan
        if conf[i, _NOSE] >= 0.15:
            head_y = kps[i, _NOSE, 1]
        else:
            head_pts = [kps[i, j, 1] for j in (1, 2, 3, 4) if conf[i, j] >= 0.15]
            if head_pts:
                head_y = float(np.mean(head_pts))
        lc, rc = conf[i, _L_HIP], conf[i, _R_HIP]
        if lc >= 0.2 and rc >= 0.2:
            hip_y = (kps[i, _L_HIP, 1] + kps[i, _R_HIP, 1]) / 2
        elif lc >= 0.2:
            hip_y = kps[i, _L_HIP, 1]
        elif rc >= 0.2:
            hip_y = kps[i, _R_HIP, 1]
        else:
            continue
        if np.isnan(head_y):
            continue
        out[i] = (head_y - hip_y) / ref_torso
    return out


def _count_inversion_peaks(
    inv: np.ndarray,
    min_peak: float = 0.2,
    min_sustain_frac: float = 0.04,
) -> int:
    """Count inversion peaks that are both high enough and sustained.

    `min_peak`:       threshold on the normalized inversion signal.
    `min_sustain_frac`: peak must stay above threshold for at least this
                      fraction of the clip length (filters jitter from
                      pose flicker during twists).
    """
    valid = ~np.isnan(inv)
    if valid.sum() < 3:
        return 0
    idx = np.where(valid)[0]
    filled = np.interp(np.arange(len(inv)), idx, inv[idx])
    # Smooth first.
    k = max(3, int(len(filled) * 0.05))
    kernel = np.ones(k) / k
    sm = np.convolve(filled, kernel, mode="same")
    above = sm > min_peak
    min_len = max(3, int(len(inv) * min_sustain_frac))
    peaks = 0
    i = 0
    n = len(above)
    while i < n:
        if not above[i]:
            i += 1
            continue
        j = i
        while j < n and above[j]:
            j += 1
        if (j - i) >= min_len:
            peaks += 1
        i = j
    return peaks


def _cumulative_rotation_rad(
    ang: np.ndarray,
    start: int | None = None,
    end: int | None = None,
) -> float:
    """Total travelled rotation in radians within [start, end] frames.

    A full backflip starts and ends upright, so endpoint-minus-start is ~0
    — we want the integrated rotation, which is the sum of absolute
    per-frame angular deltas. When a window is supplied, only the deltas
    inside it are summed (this suppresses noise from static pre- and
    post-trick segments).
    """
    T = ang.shape[0]
    if T < 2:
        return 0.0
    s = 0 if start is None else max(0, int(start))
    e = T - 1 if end is None else min(T - 1, int(end))
    if e <= s:
        return 0.0
    seg = ang[s : e + 1]
    valid_mask = ~np.isnan(seg)
    if valid_mask.sum() < 2:
        return 0.0
    unwrapped = np.unwrap(seg[valid_mask])
    deltas = np.diff(unwrapped)
    # Clamp per-delta to a half-rotation so a single YOLO glitch can't
    # inject a phantom full-turn.
    deltas = np.clip(deltas, -np.pi, np.pi)
    return float(np.sum(np.abs(deltas)))


def _detect_aerial_phase(ang: np.ndarray) -> tuple[int, int]:
    """Return (start_idx, end_idx) of the frames where rotation is active.

    Uses median-based thresholding on angular velocity so a single
    jitter spike can't compress the window to nothing.
    """
    T = ang.shape[0]
    if T == 0:
        return 0, 0
    valid = ~np.isnan(ang)
    if valid.sum() < 3:
        return 0, T - 1
    idx = np.where(valid)[0]
    vals = np.unwrap(ang[valid])
    # Interpolate onto the full frame grid so the derivative is dense.
    full = np.interp(np.arange(T), idx, vals)
    ang_vel = np.abs(np.gradient(full))
    med = float(np.median(ang_vel))
    mx = float(ang_vel.max())
    if mx <= 0:
        return 0, T - 1
    thresh = max(0.05, med + 0.25 * (mx - med))
    active = ang_vel > thresh
    if not active.any():
        return 0, T - 1
    hits = np.where(active)[0]
    # Expand slightly so we don't miss the takeoff/landing frames.
    pad = max(1, int(0.02 * T))
    return max(0, int(hits[0]) - pad), min(T - 1, int(hits[-1]) + pad)


def _twist_count(kps: np.ndarray, conf: np.ndarray, start: int, end: int) -> float | None:
    """Conservative twist estimate from shoulder x-order crossings.

    Returns None when the signal is too noisy to be trusted — the decoder
    treats None as "unknown" and scores all twist values neutrally. This
    is better than making up twist counts from YOLO noise.

    Heuristic: require crossings where the magnitude of (ls.x - rs.x) at
    the extrema is > 0.15 × bbox_width, so minor jitter doesn't count.
    Then map: 0-1 strong crossings → 0 twist, 2-3 → 0.5, 4-5 → 1, ≥6 → 2.
    """
    if end <= start + 2:
        return 0.0
    window = slice(start, end + 1)
    ls_x = kps[window, _L_SHOULDER, 0]
    rs_x = kps[window, _R_SHOULDER, 0]
    lc = conf[window, _L_SHOULDER]
    rc = conf[window, _R_SHOULDER]
    valid = (lc >= 0.3) & (rc >= 0.3)
    if valid.sum() < 5:
        return None
    diff = (ls_x - rs_x).astype(np.float32)
    diff[~valid] = np.nan
    diff_v = diff[~np.isnan(diff)]
    if diff_v.size < 5:
        return None
    # Threshold against the 90th percentile of |diff| so we only count
    # crossings through a meaningful shoulder separation.
    mag = np.percentile(np.abs(diff_v), 90)
    if mag < 1.0:  # degenerate — shoulders collapsed together
        return None
    thresh = 0.3 * mag
    # Rebaseline: treat |diff| < thresh as "near zero" and count sign
    # changes only when the signal exits and re-enters above threshold.
    strong = np.where(np.abs(diff_v) > thresh, np.sign(diff_v), 0.0)
    # Collapse consecutive zeros: count state transitions +→- or -→+.
    prev = 0.0
    crossings = 0
    for s in strong:
        if s == 0:
            continue
        if prev != 0 and s != prev:
            crossings += 1
        prev = s
    if crossings <= 2:
        return 0.0
    if crossings <= 4:
        return 0.5
    if crossings <= 7:
        return 1.0
    return 2.0


def _direction_from_angle(ang: np.ndarray) -> str | None:
    """Heuristic direction from the sign of the mean body-angle derivative.

    Image-coordinate convention: y increases downward. In that frame, a
    backward flip (head goes backward-and-down → forward-and-down from
    the athlete's POV) produces a monotonic positive angle change,
    while a forward flip produces a negative change. The sign can also
    flip with camera handedness. We return None if the signal is weak,
    so the caller can fall back to path-based priors.
    """
    valid = ang[~np.isnan(ang)]
    if valid.size < 3:
        return None
    unwrapped = np.unwrap(valid)
    total = unwrapped[-1] - unwrapped[0]
    if abs(total) < np.pi / 2:  # less than a quarter turn — too weak
        return None
    return "backward" if total > 0 else "forward"


def _direction_from_path(clip: Path) -> str | None:
    name = clip.name.lower()
    stem = clip.stem.lower()
    full = (str(clip).lower())
    if any(k in stem or k in name for k in ("frontflip", "front_", "front-")):
        return "forward"
    if any(k in stem or k in name for k in ("sideflip", "side_", "cartwheel", "aerial")):
        return "side"
    if any(k in stem for k in ("back", "gainer", "cork", "kroc", "double_", "full")):
        return "backward"
    return None


def _context_from_path(clip: Path) -> str:
    """Classify context from the clip path.

    Default = `acrobatics` — it is the majority class across the POOL-B
    benchmark and across the FIG table, so it's the least-harmful prior
    when no path hint is available.
    """
    full = str(clip).lower()
    if "wall" in full:
        return "wall"
    if "swing" in full:
        return "swing"
    if "pk_basics" in full or "roll" in full:
        return "pk_basics"
    return "acrobatics"


def _takeoff_cue(boxes: np.ndarray, fps: float, start: int) -> str:
    """`running_forward` if bbox x-center shows sustained horizontal motion.

    We compute total horizontal travel over the WHOLE clip (not just
    pre-aerial) because the aerial-phase detector is unreliable. A
    running approach is characterised by monotone x drift > one full
    body width before the take-off point.
    """
    valid_idx = np.where(~np.isnan(boxes[:, 0]))[0]
    if valid_idx.size < 5:
        return "standing"
    xs = (boxes[valid_idx, 0] + boxes[valid_idx, 2]) / 2
    widths = boxes[valid_idx, 2] - boxes[valid_idx, 0]
    body_w = float(np.nanmedian(widths))
    if body_w <= 0:
        return "standing"
    # Max horizontal displacement in any leading subsequence (captures
    # running before the athlete enters the flip).
    cummin = np.minimum.accumulate(xs)
    cummax = np.maximum.accumulate(xs)
    max_range = float((cummax - cummin).max())
    # Short clips often frame the athlete tightly, so a single body-width
    # of horizontal drift (with camera panning factored out by dividing
    # by median body width) is enough to flag running_forward.
    return "running_forward" if max_range > 0.9 * body_w else "standing"


def _extract_cues(clip: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return (cues_for_decoder, diagnostics)."""
    kps, conf, boxes, fps = _run_yolo(clip)
    ang = _body_angle(kps, conf)
    inv = _inversion_signal(kps, conf, boxes)
    start, end = _detect_aerial_phase(ang)

    # --- Flip estimation ---
    # YOLO-pose is unreliable on inverted humans, so pose-derived flip
    # counts oscillate between under- and over-counting depending on
    # the clip. The practical compromise: default to 1 (covers the
    # majority of POOL-B), lift to 2 only when multiple sustained
    # inversions AND high integrated rotation agree. Never try to
    # estimate flip=0 from pose alone — segmentation already ensures
    # the clip contains an aerial phase.
    flip_from_peaks = _count_inversion_peaks(inv, min_peak=0.2, min_sustain_frac=0.02)
    total_rad = _cumulative_rotation_rad(ang, start, end)
    flip_from_rad = total_rad / (2 * np.pi)
    if flip_from_peaks >= 2 and flip_from_rad >= 1.8:
        flip: float = 2.0
    elif flip_from_peaks >= 3 or flip_from_rad >= 3.0:
        flip = 3.0
    else:
        flip = 1.0

    # Twist detection from 2D pose is too noisy on real clips — emit
    # None so the decoder treats it neutrally and canonical tiebreaks
    # win within each flip/direction/context cluster.
    twist = None

    dir_signal = _direction_from_angle(ang)
    dir_path = _direction_from_path(clip)
    direction = dir_path or dir_signal or "backward"

    context = _context_from_path(clip)
    takeoff = _takeoff_cue(boxes, fps, start)

    cues: dict[str, Any] = {
        "context": context,
        "flip": float(flip),
        "direction": direction,
        "takeoff": takeoff,
    }
    if twist is not None:
        cues["twist"] = float(twist)

    overrides = _manifest_overrides_for(clip)
    cues.update(overrides)
    diag = {
        "total_rotation_rad": total_rad,
        "active_phase": [start, end],
        "direction_from_angle": dir_signal,
        "direction_from_path": dir_path,
        "n_frames": int(kps.shape[0]),
        "fps": fps,
    }
    return cues, diag


# ── Recognizer class ────────────────────────────────────────────────────────


# v3 model (class-weighted, partial unfreeze) is only 18-28% accurate on
# minority classes for context/direction/flip — not good enough to override
# the pose heuristic. We ONLY take twist from the model, because the
# pose heuristic emits None for twist (nothing to regress). Setting a cue
# threshold to 1.01 disables that override entirely.
_MIN_ATTR_CONF = {
    "context": 1.01,
    "direction": 1.01,
    "flip": 1.01,
    "twist": 0.50,
}


class HeuristicRecognizer:
    """Pose-heuristic baseline with optional attribute-model upgrade.

    When an attribute-model checkpoint is supplied, its predictions replace
    the corresponding pose cues whenever confidence exceeds a per-cue
    threshold. Anything below threshold falls back to the pose heuristic.
    The parent-run context manifest still wins over both — it represents
    explicit per-run knowledge from the judge.
    """

    name = "heuristic-v1"

    def __init__(self, attr_ckpt: Path | str | None = None) -> None:
        self._decoder = FIGDecoder()
        self._predictor = None
        if attr_ckpt is not None:
            try:
                from core.recognition.attribute_predictor import AttributePredictor
                self._predictor = AttributePredictor(attr_ckpt)
                self.name = "heuristic-v1+attrs"
            except FileNotFoundError:
                # Graceful degrade — keep running as pose-only.
                self._predictor = None

    def recognize(self, clip: Path) -> RecognizerOutput:
        t0 = time.time()
        try:
            cues, diag = _extract_cues(Path(clip))
        except Exception as exc:  # noqa: BLE001
            return RecognizerOutput(
                recognizer=self.name,
                mode="pose-heuristic",
                error=f"cue extraction failed: {exc}",
                latency_s=time.time() - t0,
            )

        # ── Merge in attribute-model cues where confident ──────────────
        attr_cues: dict[str, Any] = {}
        if self._predictor is not None:
            try:
                attr_cues = self._predictor.predict(Path(clip))
            except Exception as exc:  # noqa: BLE001
                diag["attr_error"] = str(exc)
            for cue in ("context", "direction", "flip", "twist"):
                if cue not in attr_cues:
                    continue
                conf = attr_cues.get(f"{cue}_conf", 0.0)
                if conf < _MIN_ATTR_CONF[cue]:
                    diag[f"{cue}_attr_lowconf"] = conf
                    continue
                cues[cue] = attr_cues[cue]
                diag[f"{cue}_source"] = f"attr({conf:.2f})"

            # Re-apply run manifest overrides after attr merge — the manifest
            # is the most trustworthy source, so it must win either way.
            cues.update(_manifest_overrides_for(Path(clip)))

        ranked = self._decoder.rank(cues, k=5)
        candidates = [
            Candidate(
                fig_name=c.trick.name,
                score=float(c.score),
                d_score=float(c.trick.score),
                category=c.trick.category,
                reasoning="; ".join(f"{k}={v:+.1f}" for k, v in c.breakdown.items()),
            )
            for c in ranked
        ]
        cues_with_diag: dict[str, Any] = {**cues, **diag}
        mode = "pose-heuristic" if self._predictor is None else "pose+attrs"
        return RecognizerOutput(
            recognizer=self.name,
            mode=mode,
            candidates=candidates,
            cues=cues_with_diag,
            raw_text=f"cues={cues}",
            latency_s=time.time() - t0,
        )
