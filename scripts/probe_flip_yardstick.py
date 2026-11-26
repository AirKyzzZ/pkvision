from __future__ import annotations

import argparse
import glob
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.pose.rule_counters import CONF_VALID, L_HIP, L_SH, R_HIP, R_SH, count
from core.recognition.trick_name_parser import parse_trick_name

POSE_DIR = ROOT / "data/keypoints_detfirst/parkourtheory_pose"
MANIFEST = ROOT / "data/v5_attribute_training/manifest_baseline_noeval.json"
QUEUE = ROOT / "data/labeling/retest_2026-11/queue.json"
TORSO = [L_SH, R_SH, L_HIP, R_HIP]


def split_of(slug: str) -> str:
    return "dev" if int(hashlib.sha1(slug.encode()).hexdigest(), 16) % 2 == 0 else "test"


def half_step(value: float) -> float:
    return round(float(value) * 2) / 2


def load_clips(pose_dir: Path) -> dict[str, dict]:
    drop = {e["slug"] for e in json.loads(QUEUE.read_text(encoding="utf-8"))}
    meta = {c["name"]: c for c in json.loads((pose_dir / "manifest.json").read_text(encoding="utf-8"))["clips"]}
    clips: dict[str, dict] = {}
    for shard in sorted(glob.glob(str(pose_dir / "shard_*.npz"))):
        with np.load(shard) as z:
            for name in z.files:
                if name not in drop:
                    clips[name] = {"kps": z[name].astype(np.float32), "fps": float(meta[name]["fps"])}
    for shard in sorted(glob.glob(str(pose_dir / "people_*.npz"))):
        with np.load(shard) as z:
            for key in z.files:
                if not key.endswith("__frame"):
                    continue
                name = key[: -len("__frame")]
                if name in clips:
                    per_frame = np.bincount(z[key].astype(int), minlength=len(clips[name]["kps"]))
                    clips[name]["multi_frac"] = float((per_frame >= 2).mean())
    return clips


def load_labels() -> dict[str, dict]:
    labels = {}
    for c in json.loads(MANIFEST.read_text(encoding="utf-8"))["clips"]:
        parsed = parse_trick_name(c["slug"].replace("_", " ")).cues
        flip = half_step(c.get("flip") or 0)
        labels[c["slug"]] = {
            "flip": flip,
            "twist": half_step(c.get("twist") or 0),
            "flip_trusted": "flip" in parsed and half_step(parsed["flip"]) == flip,
        }
    return labels


def view_ratio(kps: np.ndarray, window: tuple[int, int] | None) -> float:
    valid = (kps[:, TORSO, 2] >= CONF_VALID).all(1)
    if window is not None:
        valid[window[0]:window[1] + 1] = False
    if valid.sum() < 3:
        return float("nan")
    xy = kps[valid][:, :, :2]
    shoulder = np.linalg.norm(xy[:, R_SH] - xy[:, L_SH], axis=1)
    torso = np.linalg.norm((xy[:, L_SH] + xy[:, R_SH]) / 2 - (xy[:, L_HIP] + xy[:, R_HIP]) / 2, axis=1)
    return float(np.median(shoulder / np.maximum(torso, 1e-6)))


def measure(clip: dict) -> dict:
    c = count(clip["kps"], clip["fps"])
    if c.flip is not None:
        status = "counted" if c.window is not None else "no_motion"
    else:
        status = "low_valid" if c.window is not None else "no_torso"
    residual = None
    if c.net_angle_deg is not None:
        turns = abs(c.net_angle_deg) / 360
        residual = abs(turns - round(turns * 2) / 2) * 360
    return {
        "status": status,
        "flip": c.flip,
        "twist": c.twist,
        "valid_frac": c.valid_frac,
        "residual_deg": residual,
        "multi_frac": clip.get("multi_frac", float("nan")),
        "view": view_ratio(clip["kps"], c.window),
        "frames": len(clip["kps"]),
    }


def rows_for(split: str, pose_dir: Path) -> list[dict]:
    clips, labels = load_clips(pose_dir), load_labels()
    rows = []
    for slug in sorted(set(clips) & set(labels)):
        if split_of(slug) == split:
            rows.append({**measure(clips[slug]), "true_flip": labels[slug]["flip"],
                         "true_twist": labels[slug]["twist"], "trusted": labels[slug]["flip_trusted"]})
    return rows


def accuracy(rows: list[dict], cue: str) -> dict:
    scored = [r for r in rows if r[cue] is not None]
    if not scored:
        return {"n": len(rows), "answered": 0}
    truth = [r[f"true_{cue}"] for r in scored]
    hits = sum(r[cue] == r[f"true_{cue}"] for r in scored)
    majority = Counter(truth).most_common(1)[0][1]
    return {"n": len(rows), "answered": len(scored), "coverage": round(len(scored) / len(rows), 3),
            "exact": round(hits / len(scored), 3), "constant": round(majority / len(scored), 3)}


def strata(rows: list[dict], key: str, edges: list[float]) -> dict:
    out = {}
    bounds = [-np.inf, *edges, np.inf]
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        group = [r for r in rows if r[key] is not None and lo < r[key] <= hi]
        out[f"({lo}, {hi}]"] = accuracy(group, "flip")
    missing = [r for r in rows if r[key] is None or np.isnan(r[key])]
    if missing:
        out["missing"] = accuracy(missing, "flip")
    return out


def confusion(rows: list[dict], cue: str) -> dict:
    cap = lambda v: min(v, 2.0)
    table = Counter((cap(r[f"true_{cue}"]), cap(r[cue])) for r in rows if r[cue] is not None)
    return {f"true {t} -> pred {p}": n for (t, p), n in sorted(table.items())}


def cmd_explore(pose_dir: Path, out: Path | None) -> None:
    rows = rows_for("dev", pose_dir)
    trusted = [r for r in rows if r["trusted"]]
    counted = [r for r in trusted if r["status"] == "counted" or r["status"] == "no_motion"]
    flip_right = [r for r in counted if r["flip"] == r["true_flip"]]
    report = {
        "split": "dev",
        "clips": len(rows),
        "flip_label_trusted": len(trusted),
        "true_flip": dict(sorted(Counter(r["true_flip"] for r in rows).items())),
        "true_twist": dict(sorted(Counter(min(r["true_twist"], 2.0) for r in rows).items())),
        "status": dict(Counter(r["status"] for r in rows)),
        "flip_all": accuracy(rows, "flip"),
        "flip_trusted": accuracy(trusted, "flip"),
        "flip_by_true": {str(v): accuracy([r for r in trusted if r["true_flip"] == v], "flip")
                         for v in sorted({r["true_flip"] for r in trusted})},
        "flip_by_valid_frac": strata(counted, "valid_frac", [0.8, 0.95, 0.999]),
        "flip_by_multi_person": strata(counted, "multi_frac", [0.0, 0.3]),
        "flip_by_view": strata(counted, "view", [0.25, 0.45, 0.65]),
        "flip_by_residual_deg": strata(counted, "residual_deg", [20, 45, 70]),
        "flip_confusion": confusion(trusted, "flip"),
        "twist_all": accuracy(rows, "twist"),
        "twist_when_flip_right": accuracy(flip_right, "twist"),
        "twist_confusion_when_flip_right": confusion(flip_right, "twist"),
    }
    print(json.dumps(report, indent=1))
    if out:
        out.write_text(json.dumps(report, indent=1), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["explore"])
    ap.add_argument("--pose-dir", type=Path, default=POSE_DIR)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    cmd_explore(args.pose_dir, args.out)


if __name__ == "__main__":
    main()
