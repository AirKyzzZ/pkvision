#!/usr/bin/env python3
"""Do invented skeletons in the corpus shards hurt the CueModel?

Pre-registration: docs/science-superpowers/probes/2026-11-08-clean-pose-cue-prereg.md

  probe_clean_pose_cue.py audit      invented-frame and jump rates, old shards vs detector-first
  probe_clean_pose_cue.py train      arm x seed runs, resumable, appended to the gitignored runs file
  probe_clean_pose_cue.py report     verdict, per-cue tables and confusion matrices

The 36 clips of the blind labeling queue are dropped from every step and never printed.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.labeling.cue_model import CUE_CLASSES, base_seq, featurize
from core.pose.athlete_select import select_athlete
from scripts.audit_person_continuity import count_jumps
from scripts.train_cue_model import _flip_bin, _twist_bin, fit, load_labels, macro_f1

OLD_DIR = ROOT / "data/keypoints/parkourtheory_pose"
NEW_DIR = ROOT / "data/keypoints_detfirst/parkourtheory_pose"
RUNS = ROOT / "data/keypoints_detfirst/cue_runs.jsonl"
QUEUE = ROOT / "data/labeling/retest_2026-11/queue.json"
TRAIN_MANIFEST = ROOT / "data/v5_attribute_training/manifest_baseline_noeval.json"
GOLD = ROOT / "data/v5_attribute_training/eval_gold_concordant.json"
PROBES = ROOT / "docs/science-superpowers/probes"

CUES = ("flip", "twist", "direction", "context")
ARMS = ("O", "D", "D0", "S", "T")
SEEDS = range(5)
MIN_DELTA, MIN_SEEDS = 0.03, 4
NUM_BOOT = 2000


def queue_slugs() -> set[str]:
    return {e["slug"] for e in json.loads(QUEUE.read_text())}


def interp_missing(seq: np.ndarray) -> np.ndarray:
    """Fill frames with no detection linearly from detected frames; confidence stays 0."""
    out = seq.copy()
    seen = (seq[..., 2] > 0).any(axis=1)
    if seen.all() or not seen.any():
        return out
    t, known = np.arange(len(seq)), np.flatnonzero(seen)
    for joint in range(17):
        for axis in range(2):
            out[~seen, joint, axis] = np.interp(t[~seen], known, seq[known, joint, axis])
    return out


def load_shards(directory: Path, drop: set[str]) -> dict[str, np.ndarray]:
    skel: dict[str, np.ndarray] = {}
    for shard in sorted(glob.glob(str(directory / "shard_*.npz"))):
        with np.load(shard) as z:
            for name in z.files:
                if name not in drop:
                    skel[name] = z[name].astype(np.float32)
    return skel


def load_selected(drop: set[str]) -> dict[str, np.ndarray]:
    """Arm S: the 2026-11-04 motion selector, frozen, over every detected person."""
    fps = {c["name"]: c["fps"] for c in json.loads((NEW_DIR / "manifest.json").read_text())["clips"]}
    n = {c["name"]: c["n_frames"] for c in json.loads((NEW_DIR / "manifest.json").read_text())["clips"]}
    out: dict[str, np.ndarray] = {}
    for shard in sorted(glob.glob(str(NEW_DIR / "people_*.npz"))):
        with np.load(shard) as z:
            for name in (f[: -len("__people")] for f in z.files if f.endswith("__people")):
                if name in drop:
                    continue
                people, index = z[f"{name}__people"], z[f"{name}__frame"]
                frames = [people[index == t] for t in range(n[name])]
                out[name] = select_athlete(frames, fps[name]).astype(np.float32)
    return out


def arm_skeletons(arm: str, drop: set[str]) -> dict[str, np.ndarray]:
    if arm == "O":
        return load_shards(OLD_DIR, drop)
    if arm == "D0":
        return load_shards(NEW_DIR, drop)
    if arm in ("D", "T"):
        return {s: interp_missing(v) for s, v in load_shards(NEW_DIR, drop).items()}
    if arm == "S":
        return {s: interp_missing(v) for s, v in load_selected(drop).items()}
    raise ValueError(arm)


def gold_truth() -> dict[str, dict[str, int]]:
    """Gold cue values as class indices at the model's bin granularity; absent cues stay absent."""
    truth: dict[str, dict[str, int]] = {}
    for slug, cues in json.loads(GOLD.read_text()).items():
        row = {}
        if "flip" in cues:
            row["flip"] = CUE_CLASSES["flip"].index(_flip_bin(cues["flip"]))
        if "twist" in cues:
            row["twist"] = CUE_CLASSES["twist"].index(_twist_bin(cues["twist"]))
        for cue in ("direction", "context"):
            if cue in cues:
                row[cue] = CUE_CLASSES[cue].index(cues[cue])
        truth[slug] = row
    return truth


def cmd_audit() -> None:
    drop = queue_slugs()
    old, new = load_shards(OLD_DIR, drop), load_shards(NEW_DIR, drop)
    if len(drop & set(new)):
        sys.exit("queue clip found in the new extraction")
    names = sorted(set(old) & set(new))
    frames = invented = clips_with = clips_over_10pct = 0
    jumps = {"old": [], "new": []}
    for name in names:
        a, b = old[name], new[name]
        missing = ~(b[..., 2] > 0).any(axis=1)
        made_up = int((a[missing, :, 2] > 0).any(axis=1).sum())
        frames += len(b)
        invented += made_up
        clips_with += int(made_up > 0)
        clips_over_10pct += int(missing.mean() > 0.1)
        jumps["old"].append(count_jumps(a))
        jumps["new"].append(count_jumps(b))
    report = {
        "clips": len(names),
        "frames": frames,
        "invented_frames": invented,
        "invented_rate": round(invented / frames, 4),
        "clips_with_invented_frames": clips_with,
        "clips_over_10pct_no_detection": clips_over_10pct,
        "jumps": {
            k: {
                ">=2": round(float((np.array(v) >= 2).mean()), 4),
                "1": round(float((np.array(v) == 1).mean()), 4),
                "0": round(float((np.array(v) == 0).mean()), 4),
            }
            for k, v in jumps.items()
        },
    }
    out = PROBES / "2026-11-08-clean-pose-cue-audit.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def cmd_train(arms: list[str], seeds: list[int]) -> None:
    import torch

    drop = queue_slugs()
    labels = {s: v for s, v in load_labels(TRAIN_MANIFEST).items() if s not in drop}
    truth = gold_truth()
    if set(truth) & (set(labels) | drop):
        sys.exit("gold clips leaked into training or the queue")
    done: set[tuple[str, int]] = set()
    if RUNS.exists():
        done = {(r["arm"], r["seed"]) for r in map(json.loads, RUNS.read_text().splitlines())}
    for arm in arms:
        todo = [s for s in seeds if (arm, s) not in done]
        if not todo:
            continue
        skel = arm_skeletons(arm, drop)
        train_labels = dict(labels)
        if arm == "T":
            detected = load_shards(NEW_DIR, drop)
            train_labels = {s: v for s, v in labels.items() if s in detected and count_jumps(detected[s]) < 2}
        for seed in todo:
            result = fit(skel, train_labels, seed=seed)
            core = result["model"].core.eval()
            device = next(core.parameters()).device
            preds: dict[str, dict[str, int]] = {}
            with torch.no_grad():
                for slug in truth:
                    x = torch.tensor(featurize(base_seq(skel[slug], 48))[None]).to(device)
                    out = core(x)
                    preds[slug] = {c: int(out[c][0].argmax()) for c in CUES}
            record = {
                "arm": arm,
                "seed": seed,
                "train_clips": len(set(train_labels) & set(skel)),
                "val_core_mf1": result["val_core_mf1"],
                "val_f1": result["val_f1"],
                "preds": preds,
            }
            with RUNS.open("a") as f:
                f.write(json.dumps(record) + "\n")
            print(f"[{arm} seed {seed}] val core-mF1 {result['val_core_mf1']:.3f}", flush=True)
            torch.cuda.empty_cache()


def verdict(deltas: list[float]) -> str:
    """Pre-registered rule on the per-seed D minus O gold macro-F1."""
    mean, up, down = np.mean(deltas), sum(d > 0 for d in deltas), sum(d < 0 for d in deltas)
    if mean >= MIN_DELTA and up >= MIN_SEEDS:
        return "helps"
    if mean <= -MIN_DELTA and down >= MIN_SEEDS:
        return "hurts"
    return "no detectable effect"


def seed_scores(record: dict, truth: dict) -> dict:
    per_cue = {}
    for cue in CUES:
        slugs = [s for s in truth if cue in truth[s]]
        y = np.array([truth[s][cue] for s in slugs])
        p = np.array([record["preds"][s][cue] for s in slugs])
        per_cue[cue] = {"acc": float((y == p).mean()), "macro_f1": macro_f1(y, p), "n": len(slugs)}
    return per_cue


def confusion(records: list[dict], truth: dict, cue: str) -> list[list[int]]:
    size = len(CUE_CLASSES[cue])
    grid = [[0] * size for _ in range(size)]
    for record in records:
        for slug, row in truth.items():
            if cue in row:
                grid[row[cue]][record["preds"][slug][cue]] += 1
    return grid


def majority_baseline(truth: dict) -> dict:
    """Train-majority class per cue, scored on the gold set."""
    drop = queue_slugs()
    labels = [v for s, v in load_labels(TRAIN_MANIFEST).items() if s not in drop]
    out = {}
    for cue in CUES:
        values = [v[cue] for v in labels if v[cue] is not None and v[cue] in CUE_CLASSES[cue]]
        top = CUE_CLASSES[cue].index(max(set(values), key=values.count))
        slugs = [s for s in truth if cue in truth[s]]
        y = np.array([truth[s][cue] for s in slugs])
        out[cue] = {"acc": float((y == top).mean()), "macro_f1": macro_f1(y, np.full_like(y, top))}
    return out


def bootstrap_delta(a: list[dict], b: list[dict], truth: dict) -> list[float]:
    """Paired clip bootstrap of mean accuracy, D minus O, across seeds and scored cues."""
    rng = np.random.default_rng(0)
    slugs = sorted(truth)

    def clip_acc(records: list[dict]) -> np.ndarray:
        return np.array([
            np.mean([np.mean([r["preds"][s][c] == truth[s][c] for c in truth[s]]) for r in records]) if truth[s] else np.nan
            for s in slugs
        ])

    delta = clip_acc(b) - clip_acc(a)
    delta = delta[~np.isnan(delta)]
    boots = [delta[rng.integers(0, len(delta), len(delta))].mean() for _ in range(NUM_BOOT)]
    return [float(delta.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]


def cmd_report() -> None:
    truth = gold_truth()
    runs: dict[str, dict[int, dict]] = defaultdict(dict)
    for record in map(json.loads, RUNS.read_text().splitlines()):
        runs[record["arm"]][record["seed"]] = record
    scores = {arm: {s: seed_scores(r, truth) for s, r in rs.items()} for arm, rs in runs.items()}
    summary, conf = {}, {}
    for arm, by_seed in scores.items():
        seeds = sorted(by_seed)
        summary[arm] = {
            "seeds": seeds,
            "train_clips": runs[arm][seeds[0]]["train_clips"],
            "gold_macro_f1_mean_of_cues": round(float(np.mean([np.mean([by_seed[s][c]["macro_f1"] for c in CUES]) for s in seeds])), 4),
            "per_seed": [round(float(np.mean([by_seed[s][c]["macro_f1"] for c in CUES])), 4) for s in seeds],
            "per_cue_acc": {c: round(float(np.mean([by_seed[s][c]["acc"] for s in seeds])), 4) for c in CUES},
            "per_cue_macro_f1": {c: round(float(np.mean([by_seed[s][c]["macro_f1"] for s in seeds])), 4) for c in CUES},
            "val_core_mf1_noisy_labels": round(float(np.mean([runs[arm][s]["val_core_mf1"] for s in seeds])), 4),
        }
        conf[arm] = {c: confusion(list(runs[arm].values()), truth, c) for c in CUES}
    deltas = [
        float(np.mean([scores["D"][s][c]["macro_f1"] for c in CUES]) - np.mean([scores["O"][s][c]["macro_f1"] for c in CUES]))
        for s in sorted(scores["D"])
    ]
    report = {
        "primary": {
            "comparison": "D minus O, mean macro-F1 over flip, twist, direction, context on gold, per seed",
            "per_seed_delta": [round(d, 4) for d in deltas],
            "mean_delta": round(float(np.mean(deltas)), 4),
            "verdict": verdict(deltas),
            "accuracy_delta_and_95ci": [round(x, 4) for x in bootstrap_delta(list(runs["O"].values()), list(runs["D"].values()), truth)],
        },
        "gold_clips_scored_per_cue": {c: sum(c in t for t in truth.values()) for c in CUES},
        "majority_baseline": majority_baseline(truth),
        "arms": summary,
        "confusion": conf,
        "cue_classes": {c: CUE_CLASSES[c] for c in CUES},
    }
    out = PROBES / "2026-11-08-clean-pose-cue-results.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("primary", "majority_baseline", "arms")}, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=["audit", "train", "report"])
    ap.add_argument("--arms", nargs="*", default=list(ARMS))
    ap.add_argument("--seeds", nargs="*", type=int, default=list(SEEDS))
    args = ap.parse_args()
    if args.command == "audit":
        cmd_audit()
    elif args.command == "train":
        cmd_train(args.arms, args.seeds)
    else:
        cmd_report()


if __name__ == "__main__":
    main()
