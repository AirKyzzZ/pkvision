#!/usr/bin/env python3
"""Rule-counter probe, pre-registered in docs/science-superpowers/probes/2026-11-04-rule-counter-prereg.md.

  probe_rule_counter.py select                  print the frozen truth sets
  probe_rule_counter.py extract [--limit N]     pose arms A/B/C per clip (resumable, GPU)
  probe_rule_counter.py score                   apply counters, write the results JSON
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.pose.rule_counters import L_HIP, L_SH, R_HIP, R_SH, count
from core.recognition.trick_name_parser import parse_trick_name

VOTES = ROOT / "data/labeling/claude_probe/gold99_votes.jsonl"
MANIFEST = ROOT / "data/v5_attribute_training/attribute_manifest.json"
CLIPS = ROOT / "data/parkourtheory_clips_cropped"
OWN = ROOT / "data/final_clips"
OUT = ROOT / "data/keypoints/probe_2026-11"
RESULTS = ROOT / "docs/science-superpowers/probes/2026-11-04-rule-counter-results.json"
OWN_TRUTH = {"backflip": 0.0, "frontflip": 0.0, "gainer": 0.0, "back_double_full": 2.0, "double_cork": None}
ARMS = ("A", "B", "C")
CONF_VALID = 0.3


def reference_set() -> dict[str, dict]:
    manifest = {c["slug"]: c for c in json.loads(MANIFEST.read_text())["clips"]}
    out = {}
    for line in VOTES.read_text().splitlines():
        row = json.loads(line)
        cons, prov, slug = row["consensus"], row["provenance"], row["slug"]
        if cons.get("multiple_tricks"):
            continue
        if prov.get("flip") != "unanimous" or prov.get("twist") != "unanimous":
            continue
        man, parsed = manifest[slug], parse_trick_name(slug).cues
        agree = all(
            abs(float(cons[c]) - float(man[c])) < 0.01
            and (c not in parsed or abs(float(parsed[c]) - float(cons[c])) < 0.01)
            for c in ("flip", "twist")
        )
        if agree:
            out[slug] = {"flip": float(cons["flip"]), "twist": float(cons["twist"]),
                         "direction": cons.get("direction"), "axis": cons.get("axis")}
    return out


def stratum(truth: dict) -> str | None:
    t, f = truth["twist"], truth["flip"]
    if t == 0:
        return "twist0_flip" if f >= 1 else None
    if t == 0.5:
        return "twist0.5"
    if t == 1:
        return "twist1"
    return "twist2+" if t >= 2 else None


def decision_12(ref: dict[str, dict]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for slug in sorted(ref, key=lambda s: hashlib.sha1(s.encode()).hexdigest()):
        key = stratum(ref[slug])
        if key:
            groups.setdefault(key, []).append(slug)
    return {k: groups[k][:3] for k in sorted(groups)}


def rotate_about(frame: np.ndarray, center, angle_deg: float, side: int):
    """Square canvas around center, rotated CCW by angle_deg. Returns (canvas, forward 2x3)."""
    import cv2
    m = cv2.getRotationMatrix2D((float(center[0]), float(center[1])), angle_deg, 1.0)
    m[:, 2] += (side / 2 - center[0], side / 2 - center[1])
    return cv2.warpAffine(frame, m, (side, side)), m


def apply_affine(points: np.ndarray, m: np.ndarray) -> np.ndarray:
    return points @ m[:, :2].T + m[:, 2]


def invert_affine(m: np.ndarray) -> np.ndarray:
    a = np.linalg.inv(m[:, :2])
    return np.c_[a, -a @ m[:, 2]]


def torso_angle(joints: np.ndarray) -> float | None:
    idx = [L_SH, R_SH, L_HIP, R_HIP]
    if (joints[idx, 2] < CONF_VALID).any():
        return None
    v = joints[[L_SH, R_SH], :2].mean(0) - joints[[L_HIP, R_HIP], :2].mean(0)
    return float(np.degrees(np.arctan2(v[0], -v[1])))


def _joints(kp: np.ndarray, sc: np.ndarray) -> np.ndarray:
    return np.c_[kp[0][:17], sc[0][:17]].astype(np.float32)


def _pose_rotated(pose_model, frame, box, angle_deg):
    x0, y0, x1, y1 = box
    center = ((x0 + x1) / 2, (y0 + y1) / 2)
    size = max(x1 - x0, y1 - y0)
    side = max(int(2 * size), 8)
    canvas, m = rotate_about(frame, center, angle_deg, side)
    pad = (side - size) / 2
    kp, sc = pose_model(canvas, bboxes=[[pad, pad, side - pad, side - pad]])
    joints = _joints(kp, sc)
    joints[:, :2] = apply_affine(joints[:, :2], invert_affine(m))
    return joints


def extract_clip(path: Path, body) -> dict:
    import cv2
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    seqs = {arm: [] for arm in ARMS}
    prev_angle = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        boxes = body.det_model(frame)
        if boxes is None or len(boxes) == 0:
            for arm in ARMS:
                seqs[arm].append(np.zeros((17, 3), np.float32))
            prev_angle = None
            continue
        box = max((list(map(float, b[:4])) for b in boxes), key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
        kp, sc = body.pose_model(frame, bboxes=[box])
        a = _joints(kp, sc)
        c = max([a] + [_pose_rotated(body.pose_model, frame, box, ang) for ang in (90, 180, 270)],
                key=lambda j: float(j[:, 2].mean()))
        b = c if prev_angle is None else _pose_rotated(body.pose_model, frame, box, prev_angle)
        prev_angle = torso_angle(b)
        seqs["A"].append(a), seqs["B"].append(b), seqs["C"].append(c)
    cap.release()
    return {**{arm: np.stack(s) if s else np.zeros((0, 17, 3), np.float32) for arm, s in seqs.items()},
            "fps": np.array(fps)}


def targets() -> list[tuple[str, Path]]:
    items = [(s, CLIPS / f"{s}.mp4") for s in sorted(reference_set())]
    for name in OWN_TRUTH:
        hit = sorted(OWN.glob(f"{name}.*"))
        if hit:
            items.append((f"own__{name}", hit[0]))
    return items


def cmd_extract(limit: int | None) -> None:
    import onnxruntime as ort
    from rtmlib import Body
    providers = ort.get_available_providers()
    print("providers", providers, flush=True)
    if "CUDAExecutionProvider" not in providers:
        sys.exit("CUDA provider missing: refusing a silent CPU run")
    body = Body(mode="performance", backend="onnxruntime", device="cuda")
    OUT.mkdir(parents=True, exist_ok=True)
    for i, (slug, path) in enumerate(targets()[:limit]):
        dest = OUT / f"{slug}.npz"
        if dest.exists():
            continue
        arrays = extract_clip(path, body)
        np.savez_compressed(dest, **arrays)
        print(f"{i + 1} {slug} frames={len(arrays['A'])}", flush=True)


def _hit(pred, truth) -> bool:
    return pred is not None and abs(float(pred) - float(truth)) < 0.01


def cmd_score() -> None:
    ref = reference_set()
    d12 = decision_12(ref)
    d12_slugs = [s for group in d12.values() for s in group]
    clips, summary = {}, {}
    for slug, truth in ref.items():
        with np.load(OUT / f"{slug}.npz") as z:
            fps = float(z["fps"])
            clips[slug] = {arm: vars(count(z[arm], fps)) for arm in ARMS}
    for slug, row in clips.items():
        for arm in ARMS:
            row[arm]["window"] = [int(x) for x in row[arm]["window"]] if row[arm]["window"] else None

    for arm in ARMS:
        def tw(s):
            return clips[s][arm]["twist"]
        groups = Counter(), Counter()
        for slug, truth in ref.items():
            key = stratum(truth) or "other"
            groups[1][key] += 1
            groups[0][key] += _hit(tw(slug), truth["twist"])
        scored_dir = [s for s, t in ref.items() if t["direction"] in ("forward", "backward")]
        dir_fired = [s for s in scored_dir if clips[s][arm]["direction"]]
        summary[arm] = {
            "decision12_exact": sum(_hit(tw(s), ref[s]["twist"]) for s in d12_slugs),
            "decision12_off_by_half": sum(tw(s) is not None and abs(tw(s) - ref[s]["twist"]) <= 0.5 for s in d12_slugs),
            "all_twist_exact": sum(_hit(tw(s), ref[s]["twist"]) for s in ref),
            "all_flip_exact": sum(_hit(clips[s][arm]["flip"], ref[s]["flip"]) for s in ref),
            "all_coverage": sum(tw(s) is not None for s in ref),
            "shoulder_twist_exact": sum(_hit(clips[s][arm]["twist_shoulder"], ref[s]["twist"]) for s in ref),
            "direction_correct": sum(clips[s][arm]["direction"] == ref[s]["direction"] for s in dir_fired),
            "direction_fired": len(dir_fired),
            "direction_scored": len(scored_dir),
            "swap_events_median": float(np.median([clips[s][arm]["swap_events"] for s in ref])),
            "twist_exact_by_stratum": {k: f"{groups[0][k]}/{groups[1][k]}" for k in sorted(groups[1])},
        }

    own = {}
    for name, truth in OWN_TRUTH.items():
        path = OUT / f"own__{name}.npz"
        if path.exists():
            with np.load(path) as z:
                fps = float(z["fps"])
                own[name] = {"truth_twist": truth, "fps": fps, **{
                    arm: {k: v for k, v in vars(count(z[arm], fps)).items() if k in ("flip", "twist", "direction")}
                    for arm in ARMS}}
    best = max(s["decision12_exact"] for s in summary.values())
    verdict = ("promising" if best >= 10 else "inspect" if best >= 7 else "killed")
    payload = {"n_reference": len(ref), "decision12": d12, "best_decision12": best, "verdict": verdict,
               "arms": summary, "own_footage": own, "clips": clips,
               "truth": {s: ref[s] for s in ref}}
    RESULTS.write_text(json.dumps(payload, indent=1, default=float))
    print(json.dumps({k: payload[k] for k in ("n_reference", "decision12", "best_decision12", "verdict", "arms", "own_footage")},
                     indent=1, default=float))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("select", "extract", "score"))
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    if args.cmd == "select":
        ref = reference_set()
        print(len(ref), json.dumps(decision_12(ref), indent=1))
    elif args.cmd == "extract":
        cmd_extract(args.limit)
    else:
        cmd_score()


if __name__ == "__main__":
    main()
