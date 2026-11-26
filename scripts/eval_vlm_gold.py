#!/usr/bin/env python3
"""The four decision arms of the VLM blind probe on the concordant gold clips. Exploratory, no decision weight.

Same prompt, models and settings as run_vlm_blind.py, same gold and bins as the CueModel probes, so the
numbers sit next to the CueModel's. Blind-queue clips are never read here.

    python scripts/eval_vlm_gold.py prepare
    python scripts/eval_vlm_gold.py run [--arms ...] [--workers 6] [--cap-usd 15]
    python scripts/eval_vlm_gold.py report
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.labeling.cue_model import CUE_CLASSES  # noqa: E402
from scripts.prepare_vlm_blind import CLIPS, QUEUE, file_id, read_frames, write_inputs  # noqa: E402
from scripts.probe_clean_pose_cue import GOLD, gold_truth, majority_baseline  # noqa: E402
from scripts.run_vlm_blind import ARM_SPECS, PROVIDERS, inputs_for  # noqa: E402
from scripts.score_vlm_blind import ARMS, parse_reply  # noqa: E402
from scripts.train_cue_model import _flip_bin, _twist_bin, macro_f1  # noqa: E402

OUT = ROOT / "data/labeling/vlm_gold"
UPLOAD = OUT / "upload"
VOTES = OUT / "votes.jsonl"
RESULTS = ROOT / "docs/science-superpowers/probes/2026-11-26-vlm-gold-exploratory.json"
CUES = ("flip", "twist", "direction")
EMPTY = {"flip": None, "twist": None, "direction": None, "cork": None, "unsure": None}


def gold_slugs() -> list[str]:
    slugs = sorted(json.loads(GOLD.read_text()))
    if set(slugs) & {c["slug"] for c in json.loads(QUEUE.read_text())}:
        sys.exit("a gold clip is in the blind queue")
    return slugs


def read_rows(path: Path = VOTES) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def cmd_prepare() -> None:
    UPLOAD.mkdir(parents=True, exist_ok=True)
    for slug in gold_slugs():
        fid = file_id(slug, "gold")
        if (UPLOAD / f"{fid}_grid2.jpg").is_file():
            continue
        frames, fps = read_frames(CLIPS / f"{slug}.mp4")
        if not frames:
            sys.exit(f"could not decode {slug}")
        write_inputs(frames, fps, fid, UPLOAD)
    print(f"{len(gold_slugs())} clips in {UPLOAD}")


def ask(provider, arm: str, slug: str, upload: Path) -> dict:
    spec = ARM_SPECS[arm]
    row = {"arm": arm, "slug": slug, "model": spec["model"], "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}
    try:
        text, usage = provider.ask(spec, inputs_for(spec, file_id(slug, "gold"), upload))
        row["usage"] = usage
        try:
            row.update(parse_reply(text))
        except ValueError:
            row.update({**EMPTY, "raw": text[:300]})
    except Exception as error:  # noqa: BLE001 - a failed call is a missing vote, retried on the next run
        row["error"] = type(error).__name__ + ": " + str(error)[:200]
    return row


def run(arms: list[str], workers: int, cap_usd: float, provider=None, votes: Path = VOTES, upload: Path = UPLOAD,
        slugs: list[str] | None = None) -> dict:
    rows = read_rows(votes)
    done = {(r["arm"], r["slug"]) for r in rows if "twist" in r}
    todo = [(arm, slug) for arm in arms for slug in (slugs or gold_slugs()) if (arm, slug) not in done]
    providers = {arm: provider or PROVIDERS[ARM_SPECS[arm]["provider"]]() for arm in arms}
    state = {"spent": sum(r.get("usage", {}).get("cost_usd", 0.0) for r in rows), "made": 0, "failed": 0, "skipped": 0}
    lock = threading.Lock()
    votes.parent.mkdir(parents=True, exist_ok=True)

    def one(job: tuple[str, str]) -> None:
        with lock:
            if state["spent"] >= cap_usd:
                state["skipped"] += 1
                return
        row = ask(providers[job[0]], *job, upload)
        with lock:
            state["spent"] += row.get("usage", {}).get("cost_usd", 0.0)
            state["made"] += 1
            state["failed"] += "error" in row
            with votes.open("a") as out:
                out.write(json.dumps(row) + "\n")

    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(one, todo))
    return {**state, "spent": round(state["spent"], 4), "planned": len(todo)}


def predicted(row: dict | None, cue: str) -> int:
    """Class index at the CueModel's bins; -1 for a missing, malformed or unsure answer, which scores as wrong."""
    if row is None or row.get("twist") is None or row.get("unsure"):
        return -1
    try:
        if cue == "flip":
            return CUE_CLASSES["flip"].index(_flip_bin(row["flip"]))
        if cue == "twist":
            return CUE_CLASSES["twist"].index(_twist_bin(row["twist"]))
        return CUE_CLASSES["direction"].index(row["direction"] or "none")
    except (TypeError, ValueError):
        return -1


def report(rows: list[dict], truth: dict[str, dict]) -> dict:
    out = {}
    for arm in sorted({r["arm"] for r in rows}):
        votes = {r["slug"]: r for r in rows if r["arm"] == arm and "twist" in r}
        arm_rows = [r for r in rows if r["arm"] == arm]
        per_cue = {}
        for cue in CUES:
            slugs = [s for s in sorted(truth) if cue in truth[s]]
            y = np.array([truth[s][cue] for s in slugs])
            p = np.array([predicted(votes.get(s), cue) for s in slugs])
            per_cue[cue] = {"n": len(slugs), "acc": round(float((y == p).mean()), 3), "macro_f1": round(macro_f1(y, p), 3),
                            "by_true_class": {str(CUE_CLASSES[cue][k]): f"{int(((y == k) & (p == k)).sum())}/{int((y == k).sum())}"
                                              for k in np.unique(y)}}
        out[arm] = {
            "answered": len(votes),
            "unsure": sum(bool(v.get("unsure")) for v in votes.values()),
            "malformed": sum(v.get("twist") is None for v in votes.values()),
            "failed_calls": sum("error" in r for r in arm_rows),
            "cost_usd": round(sum(r.get("usage", {}).get("cost_usd", 0.0) for r in arm_rows), 3),
            "macro_f1_mean": round(float(np.mean([per_cue[c]["macro_f1"] for c in CUES])), 3),
            "per_cue": per_cue,
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cmd", choices=("prepare", "run", "report"))
    parser.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--cap-usd", type=float, default=15.0)
    args = parser.parse_args()
    if args.cmd == "prepare":
        cmd_prepare()
    elif args.cmd == "run":
        print(json.dumps(run(args.arms, args.workers, args.cap_usd), indent=1))
    else:
        truth = gold_truth()
        result = {"arms": report(read_rows(), truth),
                  "constant": {c: majority_baseline(truth)[c] for c in CUES}}
        RESULTS.write_text(json.dumps(result, indent=1), encoding="utf-8")
        print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
