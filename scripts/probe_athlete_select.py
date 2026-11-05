#!/usr/bin/env python3
"""Exploratory: does motion-based athlete selection change the rule-counter score?

Runs on clips already seen, so nothing here can reverse the pre-registered kill.
Extra arm S: baseline pose on every detected person, then core.pose.athlete_select.

  probe_athlete_select.py extract    all-person pose for R47 + audit sample (resumable, GPU)
  probe_athlete_select.py score      compare arm A with arm S, write the results JSON
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.pose.athlete_select import select_athlete
from core.pose.rule_counters import count
from scripts.probe_rule_counter import CLIPS, OUT as ARM_DIR, _hit, decision_12, reference_set

AUDIT = ROOT / "docs/science-superpowers/probes/2026-11-04-person-selection-audit.json"
OUT = ROOT / "data/keypoints/probe_2026-11_multi"
RESULTS = ROOT / "docs/science-superpowers/probes/2026-11-04-person-selection-results.json"


def audit_slugs() -> list[str]:
    audit = json.loads(AUDIT.read_text())
    return [*audit["flagged"], *audit["unflagged"]]


def extract_clip(path: Path, body) -> dict:
    import cv2
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    index, people, t = [], [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        # body(frame) would pose a full-frame box when nobody is detected, inventing a person.
        boxes = body.det_model(frame)
        if boxes is not None and len(boxes):
            kp, sc = body.pose_model(frame, bboxes=boxes)
            people.append(np.concatenate([kp[:, :17], sc[:, :17, None]], axis=2).astype(np.float32))
            index.extend([t] * len(kp))
        t += 1
    cap.release()
    stacked = np.concatenate(people) if people else np.zeros((0, 17, 3), np.float32)
    return {"people": stacked, "frame": np.array(index, np.int32), "n_frames": np.array(t), "fps": np.array(fps)}


def load_selected(slug: str) -> tuple[np.ndarray, float]:
    with np.load(OUT / f"{slug}.npz") as z:
        fps, n = float(z["fps"]), int(z["n_frames"])
        frames = [z["people"][z["frame"] == t] for t in range(n)]
    return select_athlete(frames, fps), fps


def cmd_extract() -> None:
    import onnxruntime as ort
    from rtmlib import Body
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        sys.exit("CUDA provider missing: refusing a silent CPU run")
    body = Body(mode="performance", backend="onnxruntime", device="cuda")
    if "CUDAExecutionProvider" not in body.pose_model.session.get_providers():
        sys.exit("session fell back to CPU: check LD_LIBRARY_PATH for the CUDA libs")
    OUT.mkdir(parents=True, exist_ok=True)
    slugs = sorted({*reference_set(), *audit_slugs()})
    for i, slug in enumerate(slugs):
        dest = OUT / f"{slug}.npz"
        if not dest.exists():
            arrays = extract_clip(CLIPS / f"{slug}.mp4", body)
            np.savez_compressed(dest, **arrays)
            print(f"{i + 1}/{len(slugs)} {slug} frames={int(arrays['n_frames'])}", flush=True)


def cmd_score() -> None:
    ref = reference_set()
    d12 = [s for group in decision_12(ref).values() for s in group]
    clips = {}
    for slug, truth in ref.items():
        selected, fps = load_selected(slug)
        with np.load(ARM_DIR / f"{slug}.npz") as z:
            base = count(z["A"], fps)
        chosen = count(selected, fps)
        clips[slug] = {"truth_twist": truth["twist"], "truth_flip": truth["flip"],
                       "A_twist": base.twist, "S_twist": chosen.twist, "A_flip": base.flip, "S_flip": chosen.flip}

    def exact(slugs, arm, key):
        return sum(_hit(clips[s][f"{arm}_{key}"], clips[s][f"truth_{key}"]) for s in slugs)

    def coverage(slugs, arm):
        return sum(clips[s][f"{arm}_twist"] is not None for s in slugs)

    summary = {arm: {"decision12_twist": exact(d12, arm, "twist"), "all47_twist": exact(ref, arm, "twist"),
                     "all47_flip": exact(ref, arm, "flip"), "all47_coverage": coverage(ref, arm)} for arm in "AS"}
    RESULTS.write_text(json.dumps({"exploratory": True, "summary": summary, "decision12": d12, "clips": clips},
                                  indent=1, default=float))
    print(json.dumps(summary, indent=1))
    for slug in d12:
        print(slug, clips[slug])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("cmd", choices=("extract", "score"))
    {"extract": cmd_extract, "score": cmd_score}[parser.parse_args().cmd]()
