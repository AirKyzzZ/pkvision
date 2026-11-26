#!/usr/bin/env python3
"""Does the frozen rule counter reach its published diving accuracy with our own skeletons? Pre-registered 2026-11-26.

MTL-AQA dives carry official somersault and twist counts. Data stays under data/external/ (gitignored).

    python scripts/probe_diving_counter.py sample            # the frozen pilot list, no download
    python scripts/probe_diving_counter.py fetch             # dive segments from the source videos
    python scripts/probe_diving_counter.py pose              # skeletons, resumable, GPU
    python scripts/probe_diving_counter.py score             # verdict against the frozen rule
    python scripts/probe_diving_counter.py clean             # delete the downloaded segments
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import pickle
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.pose.rule_counters import count  # noqa: E402

DATA = ROOT / "data/external/mtl_aqa"
ANNOTATIONS = DATA / "ann/final_annotations_dict.pkl"
URLS = DATA / "ann/video_urls.json"
VIDEOS = DATA / "videos"
POSE = DATA / "pose"
RESULTS = ROOT / "docs/science-superpowers/probes/2026-11-26-diving-counter-results.json"
SOURCE = "https://raw.githubusercontent.com/ParitoshParmar/MTL-AQA/master/MTL-AQA_dataset_release/Ready_2_Use/MTL-AQA_split_0_data/final_annotations_dict.pkl"
VIDEOS_720P = (1, 2, 3, 4, 5, 6, 14, 17, 18)
QUOTA = {0: 60, 1: 20, 2: 25, 3: 35, 4: 25, 5: 30, 6: 11, 7: 5}
PILOT = 200
FPS = 25.0
MARGIN_S = 1.5
ARMS = ("C", "A")
PASS, FAIL, MIN_COVERAGE = 0.80, 0.60, 0.70


class NoGlobals(pickle.Unpickler):
    def find_class(self, module, name):
        raise pickle.UnpicklingError(f"refused {module}.{name}")


def annotations() -> dict[tuple[int, int], dict]:
    return NoGlobals(io.BytesIO(ANNOTATIONS.read_bytes())).load()


def order(key: tuple[int, int]) -> str:
    return hashlib.sha1(f"mtl-pilot:{key[0]}:{key[1]}".encode()).hexdigest()


def ranked(ann: dict) -> dict[int, list[tuple[int, int]]]:
    strata: dict[int, list] = {}
    for key in sorted((k for k in ann if k[0] in VIDEOS_720P), key=order):
        strata.setdefault(ann[key]["tw_no"], []).append(key)
    return strata


def pilot(ann: dict, unavailable: set = frozenset()) -> list[tuple[int, int]]:
    """Quota per twist count in hash order; a dive that cannot be fetched is replaced by the next of its stratum."""
    strata = {tw: [k for k in keys if k not in unavailable] for tw, keys in ranked(ann).items()}
    chosen = [k for tw, quota in QUOTA.items() for k in strata.get(tw, [])[:quota]]
    spare = [k for k in strata.get(0, [])[QUOTA[0]:]]
    return (chosen + spare[:max(0, PILOT - len(chosen))])[:PILOT]


def segment(key: tuple[int, int]) -> Path:
    return VIDEOS / f"{key[0]:02d}_{key[1]:03d}.mp4"


def fetch_one(key: tuple[int, int], ann: dict, urls: dict) -> bool:
    out = segment(key)
    if out.is_file() and out.stat().st_size > 10_000:
        return True
    t0 = max(0.0, ann[key]["start_frame"] / FPS - MARGIN_S)
    t1 = ann[key]["end_frame"] / FPS + MARGIN_S
    subprocess.run([os.environ["YT_DLP"], "--no-warnings", "--no-playlist", "-q", "--ffmpeg-location", os.environ["FFMPEG"],
                    "-f", "bv*[height<=720]", "--download-sections", f"*{t0:.2f}-{t1:.2f}", "--force-keyframes-at-cuts",
                    "--remux-video", "mp4", "-o", str(out), urls[str(key[0])]], capture_output=True, timeout=900)
    return out.is_file() and out.stat().st_size > 10_000


def cmd_fetch() -> None:
    VIDEOS.mkdir(parents=True, exist_ok=True)
    ann, urls = annotations(), json.loads(URLS.read_text())
    missing = set(json.loads((DATA / "unavailable.json").read_text())) if (DATA / "unavailable.json").exists() else set()
    missing = {tuple(k) for k in missing}
    while True:
        todo = [k for k in pilot(ann, missing) if not segment(k).is_file()]
        if not todo:
            break
        for key in todo:
            if not fetch_one(key, ann, urls):
                missing.add(key)
                (DATA / "unavailable.json").write_text(json.dumps(sorted(missing)))
            print(f"{sum(segment(k).is_file() for k in pilot(ann, missing))}/{PILOT}", flush=True)
    print("fetched", sum(segment(k).is_file() for k in pilot(ann, missing)), "unavailable", len(missing))


def final_pilot(ann: dict) -> list[tuple[int, int]]:
    missing = {tuple(k) for k in json.loads((DATA / "unavailable.json").read_text())} if (DATA / "unavailable.json").exists() else set()
    return pilot(ann, missing)


def cmd_pose() -> None:
    import cv2
    from rtmlib import Body

    from scripts.probe_rule_counter import _joints, _pose_rotated

    POSE.mkdir(parents=True, exist_ok=True)
    ann = annotations()
    body = Body(mode="performance", backend="onnxruntime", device="cuda")
    for key in final_pilot(ann):
        out = POSE / f"{key[0]:02d}_{key[1]:03d}.npz"
        if out.is_file():
            continue
        cap = cv2.VideoCapture(str(segment(key)))
        first = int(round(MARGIN_S * FPS)) if ann[key]["start_frame"] / FPS >= MARGIN_S else ann[key]["start_frame"]
        length = ann[key]["end_frame"] - ann[key]["start_frame"] + 1
        arms = {arm: [] for arm in ARMS}
        index = 0
        while True:
            ok, frame = cap.read()
            if not ok or index >= first + length:
                break
            index += 1
            if index <= first:
                continue
            boxes = body.det_model(frame)
            if boxes is None or len(boxes) == 0:
                for arm in ARMS:
                    arms[arm].append(np.zeros((17, 3), np.float32))
                continue
            box = max((list(map(float, b[:4])) for b in boxes), key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
            kp, sc = body.pose_model(frame, bboxes=[box])
            plain = _joints(kp, sc)
            best = max([plain] + [_pose_rotated(body.pose_model, frame, box, a) for a in (90, 180, 270)], key=lambda j: float(j[:, 2].mean()))
            arms["A"].append(plain)
            arms["C"].append(best)
        cap.release()
        np.savez(out, **{arm: np.stack(v) if v else np.zeros((0, 17, 3), np.float32) for arm, v in arms.items()})
        print(out.name, len(arms["A"]), flush=True)


def verdict(somersault: float, twist: float, coverage: float) -> str:
    if coverage >= MIN_COVERAGE and somersault >= PASS and twist >= PASS:
        return "counter validated"
    if coverage < 0.50 or somersault < FAIL or twist < FAIL:
        return "skeleton or counter is the problem"
    return "inconclusive"


def score(rows: list[dict]) -> dict:
    answered = [r for r in rows if r["flip"] is not None]
    if not answered:
        return {"n": len(rows), "answered": 0, "verdict": "skeleton or counter is the problem"}
    som = float(np.mean([r["flip"] == r["true_somersault"] for r in answered]))
    tw = float(np.mean([r["twist"] == r["true_twist"] for r in answered]))
    coverage = len(answered) / len(rows)
    by_twist = {}
    for value in sorted({r["true_twist"] for r in rows}):
        group = [r for r in answered if r["true_twist"] == value]
        by_twist[str(value)] = f"{sum(r['twist'] == value for r in group)}/{len(group)}"
    return {
        "n": len(rows), "answered": len(answered), "coverage": round(coverage, 3),
        "somersault_exact": round(som, 3), "twist_exact": round(tw, 3),
        "somersault_within_half": round(float(np.mean([abs(r["flip"] - r["true_somersault"]) <= 0.5 for r in answered])), 3),
        "twist_within_half": round(float(np.mean([abs(r["twist"] - r["true_twist"]) <= 0.5 for r in answered])), 3),
        "twist_exact_by_true_twist": by_twist,
        "constant_twist_0": round(float(np.mean([r["true_twist"] == 0 for r in answered])), 3),
        "constant_somersault": round(Counter(r["true_somersault"] for r in answered).most_common(1)[0][1] / len(answered), 3),
        "mean_valid_frac": round(float(np.mean([r["valid_frac"] for r in answered])), 3),
        "verdict": verdict(som, tw, coverage),
    }


def cmd_score() -> None:
    ann = annotations()
    report = {"arms": {}, "dives": 0}
    for arm in ARMS:
        rows = []
        for key in final_pilot(ann):
            path = POSE / f"{key[0]:02d}_{key[1]:03d}.npz"
            if not path.is_file():
                continue
            c = count(np.load(path)[arm], FPS)
            rows.append({"flip": c.flip, "twist": c.twist, "twist_shoulder": c.twist_shoulder, "valid_frac": c.valid_frac,
                         "true_somersault": ann[key]["ss_no"] / 2, "true_twist": ann[key]["tw_no"] / 2})
        report["arms"][arm] = score(rows)
        report["dives"] = len(rows)
    RESULTS.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cmd", choices=("sample", "fetch", "pose", "score", "clean"))
    args = parser.parse_args()
    if args.cmd == "sample":
        ann = annotations()
        chosen = pilot(ann)
        print(json.dumps({"dives": len(chosen), "by_twist": dict(sorted(Counter(ann[k]["tw_no"] / 2 for k in chosen).items())),
                          "by_somersault": dict(sorted(Counter(ann[k]["ss_no"] / 2 for k in chosen).items())),
                          "frames": sum(ann[k]["end_frame"] - ann[k]["start_frame"] + 1 for k in chosen)}, indent=1))
    elif args.cmd == "fetch":
        cmd_fetch()
    elif args.cmd == "pose":
        cmd_pose()
    elif args.cmd == "score":
        cmd_score()
    else:
        for path in VIDEOS.glob("*.mp4"):
            path.unlink()
        print("segments deleted")


if __name__ == "__main__":
    main()
