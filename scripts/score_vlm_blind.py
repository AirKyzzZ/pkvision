#!/usr/bin/env python3
"""Apply the VLM blind probe's decision rule. Reads local files only; makes no network calls.

  score_vlm_blind.py LABELS VOTES

LABELS: blind_labels.json downloaded from label.html, keyed by slug.
VOTES: JSONL, one row per answer:
  {"arm", "id", "variant": "fwd"|"rev", "flip", "twist", "direction", "cork", "unsure"}
Design: docs/science-superpowers/probes/2026-11-04-vlm-blind-prereg-DRAFT.md
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.prepare_vlm_blind import QUEUE, opaque_id

MIN_ACCURACY = 0.75
CONTROL_CLIPS = 12
MIN_CONTROL_ANSWERS = 6
OPPOSITE = {"forward": "backward", "backward": "forward"}


def is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def scored_truth(labels: dict) -> dict[str, dict]:
    """Clips the owner marked as one trick with numeric flip and twist."""
    return {slug: {"flip": float(a["flip"]), "twist": float(a["twist"])}
            for slug, a in labels.items()
            if a.get("use") == "one trick" and is_number(a.get("flip")) and is_number(a.get("twist"))}


def _exact(vote: dict | None, key: str, truth: float) -> bool:
    return bool(vote) and not vote.get("unsure") and is_number(vote.get(key)) and abs(float(vote[key]) - truth) < 0.01


def control(arm_votes: dict, slugs: list[str]) -> dict:
    """Does predicted direction reverse when the clip plays backwards?"""
    chosen = sorted(slugs, key=lambda s: hashlib.sha1(s.encode()).hexdigest())[:CONTROL_CLIPS]
    answered = reversed_count = 0
    for slug in chosen:
        forward = arm_votes.get((opaque_id(slug), "fwd"), {}).get("direction")
        if forward in OPPOSITE:
            answered += 1
            reversed_count += arm_votes.get((opaque_id(slug), "rev"), {}).get("direction") == OPPOSITE[forward]
    # Too few directional answers means the control was not passed, not that it was waived.
    passed = answered >= MIN_CONTROL_ANSWERS and reversed_count / answered >= MIN_ACCURACY
    return {"answered": answered, "reversed": reversed_count, "passed": passed}


def score_arm(arm_votes: dict, truth: dict[str, dict], strata: dict[str, str]) -> dict:
    n = len(truth)
    hits = {s for s, t in truth.items() if _exact(arm_votes.get((opaque_id(s), "fwd")), "twist", t["twist"])}
    flips = sum(_exact(arm_votes.get((opaque_id(s), "fwd")), "flip", t["flip"]) for s, t in truth.items())
    counts = Counter(t["twist"] for t in truth.values())
    best_constant = counts.most_common(1)[0][1] if counts else 0
    per_stratum: dict[str, list[int]] = {}
    for slug in truth:
        row = per_stratum.setdefault(strata.get(slug, "unknown"), [0, 0])
        row[0] += slug in hits
        row[1] += 1
    ctrl = control(arm_votes, list(truth))
    reasons = []
    if n == 0:
        reasons.append("no scored clips")
    else:
        if len(hits) / n < MIN_ACCURACY:
            reasons.append("twist below 75%")
        if len(hits) <= best_constant:
            reasons.append("not above the best constant")
    if not ctrl["passed"]:
        reasons.append("reversed-clip control failed")
    return {"twist_exact": len(hits), "n": n, "flip_exact": flips, "always_zero": counts.get(0.0, 0),
            "best_constant": best_constant, "by_stratum": {k: f"{a}/{b}" for k, (a, b) in sorted(per_stratum.items())},
            "control": ctrl, "verdict": "killed" if reasons else "promising", "kill_reasons": reasons}


def score(labels: dict, rows: list[dict], strata: dict[str, str]) -> dict:
    truth = scored_truth(labels)
    arms: dict[str, dict] = {}
    for row in rows:
        arms.setdefault(row["arm"], {})[(row["id"], row["variant"])] = row
    return {arm: score_arm(votes, truth, strata) for arm, votes in sorted(arms.items())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("labels", type=Path)
    parser.add_argument("votes", type=Path)
    args = parser.parse_args()
    strata = {c["slug"]: c["stratum"] for c in json.loads(QUEUE.read_text())}
    rows = [json.loads(line) for line in args.votes.read_text().splitlines() if line.strip()]
    print(json.dumps(score(json.loads(args.labels.read_text()), rows, strata), indent=1))


if __name__ == "__main__":
    main()
