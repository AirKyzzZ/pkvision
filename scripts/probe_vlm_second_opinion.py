#!/usr/bin/env python3
"""Does a Gemini second opinion on flip and direction improve the training labels? Pre-registered 2026-11-26.

    python scripts/probe_vlm_second_opinion.py prepare
    python scripts/probe_vlm_second_opinion.py collect [--workers 6] [--cap-usd 20]
    python scripts/probe_vlm_second_opinion.py train
    python scripts/probe_vlm_second_opinion.py report
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.labeling.cue_model import CUE_CLASSES, base_seq, featurize  # noqa: E402
from scripts import eval_vlm_gold as gold  # noqa: E402
from scripts.prepare_vlm_blind import CLIPS, SLOWDOWN, file_id, read_frames, to_h264, write_video  # noqa: E402
from scripts.probe_clean_pose_cue import CUES, PROBES, arm_skeletons, gold_truth, queue_slugs, seed_scores  # noqa: E402
from scripts.probe_pt_label_fix import RUNS as BASELINE_RUNS, gold_mf1, train_labels, verdict  # noqa: E402
from scripts.train_cue_model import fit  # noqa: E402

ARM = "G-flash"
OUT = ROOT / "data/labeling/vlm_train"
UPLOAD = OUT / "upload"
VOTES = OUT / "votes.jsonl"
RUNS = ROOT / "data/keypoints_detfirst/vlm_second_opinion_runs.jsonl"
RESULTS = PROBES / "2026-11-26-vlm-second-opinion-results.json"
CHECKED = ("flip", "direction")
MIN_COVERAGE = 0.95
SEEDS = range(5)


def train_slugs() -> list[str]:
    return sorted(s for s in train_labels()[0] if (CLIPS / f"{s}.mp4").is_file())


def prepare_one(slug: str) -> None:
    name = f"{file_id(slug, 'gold')}_slow{SLOWDOWN}"
    if (UPLOAD / f"{name}.mp4").is_file():
        return
    frames, fps = read_frames(CLIPS / f"{slug}.mp4")
    if frames:
        raw = OUT / f"{name}.raw.mp4"
        write_video(frames, fps / SLOWDOWN, raw)
        to_h264(raw, UPLOAD / f"{name}.mp4")


def cmd_prepare(workers: int) -> None:
    UPLOAD.mkdir(parents=True, exist_ok=True)
    slugs = train_slugs()
    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(prepare_one, slugs))
    print(f"{sum(1 for _ in UPLOAD.glob('*.mp4'))} of {len(slugs)} clips in {UPLOAD}")


def consensus(labels: dict[str, dict], votes: dict[str, dict]) -> tuple[dict[str, dict], Counter]:
    """Keep a flip or direction label only where the second opinion does not contradict it."""
    out, stats = {}, Counter()
    for slug, row in labels.items():
        row = dict(row)
        for cue in CHECKED:
            opinion = gold.predicted(votes.get(slug), cue)
            if opinion < 0:
                stats[f"{cue}_no_opinion"] += 1
            elif CUE_CLASSES[cue][opinion] == row[cue]:
                stats[f"{cue}_agree"] += 1
            else:
                stats[f"{cue}_masked"] += 1
                row[cue] = None
        out[slug] = row
    return out, stats


def read_votes() -> dict[str, dict]:
    return {r["slug"]: r for r in gold.read_rows(VOTES) if r["arm"] == ARM and "twist" in r}


def consensus_labels() -> tuple[dict[str, dict], Counter]:
    labels, votes = train_labels()[0], read_votes()
    coverage = len(set(labels) & set(votes)) / len(labels)
    if coverage < MIN_COVERAGE:
        sys.exit(f"votes cover {coverage:.1%} of the training clips, under {MIN_COVERAGE:.0%}")
    return consensus(labels, votes)


def cmd_train() -> None:
    import torch

    drop = queue_slugs()
    labels, _ = consensus_labels()
    truth = gold_truth()
    if set(truth) & (set(labels) | drop):
        sys.exit("gold clips leaked into training or the queue")
    skel = arm_skeletons("D", drop)
    done = {r["seed"] for r in gold.read_rows(RUNS)}
    for seed in SEEDS:
        if seed in done:
            continue
        result = fit(skel, labels, seed=seed)
        core = result["model"].core.eval()
        device = next(core.parameters()).device
        preds = {}
        with torch.no_grad():
            for slug in truth:
                out = core(torch.tensor(featurize(base_seq(skel[slug], 48))[None]).to(device))
                preds[slug] = {c: int(out[c][0].argmax()) for c in CUES}
        record = {"arm": "C", "seed": seed, "train_clips": len(set(labels) & set(skel)),
                  "val_core_mf1": result["val_core_mf1"], "val_f1": result["val_f1"], "preds": preds}
        with RUNS.open("a") as f:
            f.write(json.dumps(record) + "\n")
        print(f"[C seed {seed}] done", flush=True)
        torch.cuda.empty_cache()


def cmd_report() -> None:
    truth = gold_truth()
    baseline = {r["seed"]: r for r in gold.read_rows(BASELINE_RUNS) if r["arm"] == "P"}
    fixed = {r["seed"]: r for r in gold.read_rows(RUNS)}
    seeds = sorted(set(baseline) & set(fixed))
    if len(seeds) < 5:
        sys.exit(f"need 5 paired seeds, have {len(seeds)}")
    deltas = [gold_mf1(fixed[s], truth) - gold_mf1(baseline[s], truth) for s in seeds]

    def per_cue(records: list[dict]) -> dict:
        scores = [seed_scores(r, truth) for r in records]
        return {c: {"acc": float(np.mean([s[c]["acc"] for s in scores])),
                    "macro_f1": float(np.mean([s[c]["macro_f1"] for s in scores]))} for c in CUES}

    rows = gold.read_rows(VOTES)
    report = {
        "verdict": verdict(deltas),
        "per_seed_delta": deltas,
        "mean_delta": float(np.mean(deltas)),
        "gold_macro_f1": {"P": float(np.mean([gold_mf1(baseline[s], truth) for s in seeds])),
                          "C": float(np.mean([gold_mf1(fixed[s], truth) for s in seeds]))},
        "per_cue": {"P": per_cue([baseline[s] for s in seeds]), "C": per_cue([fixed[s] for s in seeds])},
        "second_opinion": dict(consensus_labels()[1]),
        "votes": {"answered": len(read_votes()), "unsure": sum(bool(v.get("unsure")) for v in read_votes().values()),
                  "failed_calls": sum("error" in r for r in rows),
                  "cost_usd": round(sum(r.get("usage", {}).get("cost_usd", 0.0) for r in rows), 2)},
    }
    RESULTS.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cmd", choices=("prepare", "collect", "train", "report"))
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--cap-usd", type=float, default=20.0)
    args = parser.parse_args()
    if args.cmd == "prepare":
        cmd_prepare(min(args.workers, 3))
    elif args.cmd == "collect":
        print(json.dumps(gold.run([ARM], args.workers, args.cap_usd, votes=VOTES, upload=UPLOAD, slugs=train_slugs()), indent=1))
    elif args.cmd == "train":
        cmd_train()
    else:
        cmd_report()


if __name__ == "__main__":
    main()
