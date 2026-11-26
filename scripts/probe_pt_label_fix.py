from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.labeling.cue_model import CUE_CLASSES, base_seq, featurize
from scripts.probe_clean_pose_cue import (
    CUES, PROBES, TRAIN_MANIFEST, arm_skeletons, confusion, gold_truth, majority_baseline, queue_slugs, seed_scores,
)
from scripts.train_cue_model import fit, load_labels

PT_MOVES = ROOT / "data/parkourtheory_detailed.json"
BASELINE_RUNS = ROOT / "data/keypoints_detfirst/cue_runs.jsonl"
RUNS = ROOT / "data/keypoints_detfirst/pt_label_fix_runs.jsonl"
RESULTS = PROBES / "2026-11-26-pt-label-fix-results.json"
ARMS = {"P": (1.0, range(5)), "P50": (0.5, range(3)), "P25": (0.25, range(3))}
MIN_DELTA = 0.03
MIN_SEEDS = 4


def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def pt_categories() -> dict[str, set[str]]:
    moves = json.loads(PT_MOVES.read_text(encoding="utf-8"))
    moves = moves if isinstance(moves, list) else list(moves.values())
    return {slugify(m["name"]): set(m["type"].lower().split("/")) for m in moves if m.get("type")}


def corrected(labels: dict[str, dict]) -> tuple[dict[str, dict], Counter]:
    categories, changes, out = pt_categories(), Counter(), {}
    for slug, row in labels.items():
        row = dict(row)
        kinds = categories.get(slug)
        if kinds is None:
            changes["unmatched"] += 1
        else:
            if "flip" not in kinds:
                changes["flip_set_to_0"] += row["flip"] != 0
                changes["direction_set_to_none"] += row["direction"] != "none"
                row.update(flip=0, direction="none", axis=None)
            if "twist" in kinds and row["twist"] == 0:
                changes["twist_masked"] += 1
                row["twist"] = None
        out[slug] = row
    return out, changes


def train_labels() -> tuple[dict[str, dict], Counter]:
    drop = queue_slugs()
    return corrected({s: v for s, v in load_labels(TRAIN_MANIFEST).items() if s not in drop})


def cmd_plan() -> None:
    labels, changes = train_labels()
    print(json.dumps({"clips": len(labels), "changes": dict(changes),
                      "flip_bins": dict(sorted(Counter(v["flip"] for v in labels.values()).items())),
                      "direction": dict(sorted(Counter(v["direction"] for v in labels.values()).items()))}, indent=1))


def cmd_train(arms: list[str]) -> None:
    import torch

    drop = queue_slugs()
    labels, _ = train_labels()
    truth = gold_truth()
    if set(truth) & (set(labels) | drop):
        sys.exit("gold clips leaked into training or the queue")
    skel = arm_skeletons("D", drop)
    done = set()
    if RUNS.exists():
        done = {(r["arm"], r["seed"]) for r in map(json.loads, RUNS.read_text().splitlines())}
    for arm in arms:
        fraction, seeds = ARMS[arm]
        for seed in seeds:
            if (arm, seed) in done:
                continue
            slugs = sorted(set(labels) & set(skel))
            if fraction < 1:
                keep = np.random.default_rng(1000 + seed).choice(len(slugs), int(fraction * len(slugs)), replace=False)
                slugs = [slugs[i] for i in sorted(keep)]
            result = fit(skel, {s: labels[s] for s in slugs}, seed=seed)
            core = result["model"].core.eval()
            device = next(core.parameters()).device
            preds = {}
            with torch.no_grad():
                for slug in truth:
                    out = core(torch.tensor(featurize(base_seq(skel[slug], 48))[None]).to(device))
                    preds[slug] = {c: int(out[c][0].argmax()) for c in CUES}
            record = {"arm": arm, "seed": seed, "train_clips": len(slugs),
                      "val_core_mf1": result["val_core_mf1"], "val_f1": result["val_f1"], "preds": preds}
            with RUNS.open("a") as f:
                f.write(json.dumps(record) + "\n")
            print(f"[{arm} seed {seed}] done", flush=True)
            torch.cuda.empty_cache()


def gold_mf1(record: dict, truth: dict) -> float:
    return float(np.mean([v["macro_f1"] for v in seed_scores(record, truth).values()]))


def verdict(deltas: list[float]) -> str:
    mean, up, down = np.mean(deltas), sum(d > 0 for d in deltas), sum(d < 0 for d in deltas)
    if mean >= MIN_DELTA and up >= MIN_SEEDS:
        return "helps"
    if mean <= -MIN_DELTA and down >= MIN_SEEDS:
        return "hurts"
    return "no detectable effect"


def cmd_report() -> None:
    truth = gold_truth()
    read = lambda path: [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    baseline = {r["seed"]: r for r in read(BASELINE_RUNS) if r["arm"] == "D"}
    runs = read(RUNS)
    fixed = {r["seed"]: r for r in runs if r["arm"] == "P"}
    seeds = sorted(set(baseline) & set(fixed))
    if len(seeds) < 5:
        sys.exit(f"need 5 paired seeds, have {len(seeds)}")
    deltas = [gold_mf1(fixed[s], truth) - gold_mf1(baseline[s], truth) for s in seeds]

    def per_cue(records: list[dict]) -> dict:
        scores = [seed_scores(r, truth) for r in records]
        return {c: {"acc": float(np.mean([s[c]["acc"] for s in scores])),
                    "macro_f1": float(np.mean([s[c]["macro_f1"] for s in scores]))} for c in CUES}

    report = {
        "verdict": verdict(deltas),
        "per_seed_delta": deltas,
        "mean_delta": float(np.mean(deltas)),
        "gold_macro_f1": {"B": float(np.mean([gold_mf1(baseline[s], truth) for s in seeds])),
                          "P": float(np.mean([gold_mf1(fixed[s], truth) for s in seeds]))},
        "per_cue": {"B": per_cue([baseline[s] for s in seeds]), "P": per_cue([fixed[s] for s in seeds])},
        "constant": majority_baseline(truth),
        "confusion": {arm: {c: confusion(records, truth, c) for c in CUES}
                      for arm, records in (("B", [baseline[s] for s in seeds]), ("P", [fixed[s] for s in seeds]))},
        "classes": {c: [str(v) for v in CUE_CLASSES[c]] for c in CUES},
        "learning_curve_exploratory": {
            arm: {"train_clips": int(np.mean([r["train_clips"] for r in runs if r["arm"] == arm])),
                  "gold_macro_f1": [gold_mf1(r, truth) for r in runs if r["arm"] == arm]}
            for arm in ARMS if any(r["arm"] == arm for r in runs)
        },
        "label_changes": dict(train_labels()[1]),
    }
    RESULTS.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "confusion"}, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["plan", "train", "report"])
    ap.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    args = ap.parse_args()
    {"plan": cmd_plan, "train": lambda: cmd_train(args.arms), "report": cmd_report}[args.command]()


if __name__ == "__main__":
    main()
