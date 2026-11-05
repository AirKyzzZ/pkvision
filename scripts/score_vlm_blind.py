#!/usr/bin/env python3
"""Apply the VLM blind probe's decision rule. Reads local files only; makes no network calls.

  score_vlm_blind.py control-ids LABELS         opaque ids the runner must also send reversed
  score_vlm_blind.py score LABELS VOTES         apply the rule; prints no clip names
  --accept-unlabeled                            the unlabeled clips are final, for example unplayable
  --show-excluded                               also list excluded clips by name; only after labeling

LABELS: blind_labels.json downloaded from label.html, keyed by slug.
VOTES: JSONL, one row per answer, with exactly one row per (arm, id, variant):
  {"arm", "id", "variant": "fwd"|"rev", "flip", "twist", "direction", "cork", "unsure"}
A row a model answered badly counts as wrong. A row the runner built badly stops the run, and
missing votes or labels give "incomplete" rather than a verdict. Check the malformed counts too:
a runner that stringifies numbers looks like a model failure.
Design: docs/science-superpowers/probes/2026-11-04-vlm-blind-prereg-DRAFT.md
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.prepare_vlm_blind import QUEUE, opaque_id

ARMS = ("G-pro", "G-flash", "G-grid", "C-grid")
VARIANTS = ("fwd", "rev")
DIRECTIONS = ("forward", "backward", "side", None)
MIN_ACCURACY = 0.75
MIN_SCORED = 20
CONTROL_CLIPS = 12
MIN_CONTROL_ANSWERS = 6
OPPOSITE = {"forward": "backward", "backward": "forward"}
ANSWER_FIELDS = ("flip", "twist", "direction", "cork", "unsure")
# The only values label.html can write.
LABEL_USES = ("one trick", "several tricks", "can't tell", "name on screen")
LABEL_CORKS = ("no", "yes", "unsure")
LABEL_FLIPS = (0, 0.5, 1, 1.5, 2, 2.5, 3)
LABEL_TWISTS = (0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4)


def is_half_step(value) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value) and value >= 0 and float(value * 2).is_integer()
    except OverflowError:
        return False


def well_formed(vote: dict) -> bool:
    return (is_half_step(vote.get("flip")) and is_half_step(vote.get("twist"))
            and isinstance(vote.get("cork"), bool) and isinstance(vote.get("unsure"), bool)
            and "direction" in vote and vote["direction"] in DIRECTIONS)


def usable(vote: dict | None) -> bool:
    return vote is not None and well_formed(vote) and not vote["unsure"]


def valid_label(answer: dict) -> bool:
    def count_ok(value, allowed) -> bool:
        return value is None or value == "unsure" or (is_half_step(value) and value in allowed)
    return (count_ok(answer.get("flip"), LABEL_FLIPS) and count_ok(answer.get("twist"), LABEL_TWISTS)
            and (answer.get("use") is None or answer.get("use") in LABEL_USES)
            and (answer.get("cork") is None or answer.get("cork") in LABEL_CORKS)
            and (answer.get("done") is None or isinstance(answer.get("done"), bool)))


def split_labels(labels: dict, queue_slugs: set[str]) -> tuple[dict[str, dict], dict[str, str]]:
    """Scored truth, and why each other queue clip is excluded. Unknown slugs stop the run."""
    if not isinstance(labels, dict) or not all(isinstance(a, dict) for a in labels.values()):
        raise ValueError("labels are not an object of objects")
    if not all(valid_label(a) for a in labels.values()):
        raise ValueError("a label has a value the page cannot write")  # never echo it
    unknown = set(labels) - queue_slugs
    if unknown:
        raise ValueError(f"{len(unknown)} labeled clips are not in the queue")
    truth, excluded = {}, {}
    for slug in sorted(queue_slugs):
        answer = labels.get(slug)
        if answer is None or answer.get("use") is None:
            excluded[slug] = "not labeled"
        elif answer["use"] != "one trick":
            excluded[slug] = str(answer["use"])
        elif not answer.get("done"):
            excluded[slug] = "not labeled"
        elif not (is_half_step(answer.get("flip")) and is_half_step(answer.get("twist"))):
            excluded[slug] = "unsure count"
        else:
            truth[slug] = {"flip": float(answer["flip"]), "twist": float(answer["twist"]), "cork": answer.get("cork")}
    return truth, excluded


def control_slugs(truth: dict[str, dict]) -> list[str]:
    """Clips with a flip and whole twists: only these should reverse apparent direction.

    An odd half twist makes the reversed clip start facing the other way, which cancels the reversal.
    The salted hash stops this order from repeating the queue's own selection order.
    """
    eligible = [s for s, t in truth.items() if t["flip"] >= 1 and t["twist"] % 1 == 0]
    return sorted(eligible, key=lambda s: hashlib.sha1(("control:" + s).encode()).hexdigest())[:CONTROL_CLIPS]


def control(arm_votes: dict, slugs: list[str]) -> dict:
    """Does predicted direction reverse when the clip plays backwards? Failing is informative; passing does not clear twist."""
    answered = reversed_count = 0
    for slug in slugs:
        forward, backward = arm_votes.get((opaque_id(slug), "fwd")), arm_votes.get((opaque_id(slug), "rev"))
        if usable(forward) and forward["direction"] in OPPOSITE:
            answered += 1
            reversed_count += usable(backward) and backward["direction"] == OPPOSITE[forward["direction"]]
    # Too few directional answers means the control was not passed, not that it was waived.
    passed = answered >= MIN_CONTROL_ANSWERS and reversed_count / answered >= MIN_ACCURACY
    return {"answered": answered, "reversed": reversed_count, "passed": passed}


def _rate(slugs: list[str], hits: set[str]) -> str:
    return f"{sum(s in hits for s in slugs)}/{len(slugs)}"


def score_arm(arm_votes: dict, truth: dict[str, dict], labels_final: bool = True) -> dict:
    n = len(truth)
    votes = {s: arm_votes.get((opaque_id(s), "fwd")) for s in truth}
    hits = {s for s, t in truth.items() if usable(votes[s]) and float(votes[s]["twist"]) == t["twist"]}
    flips = sum(usable(votes[s]) and float(votes[s]["flip"]) == t["flip"] for s, t in truth.items())
    counts = Counter(t["twist"] for t in truth.values())
    best_constant = counts.most_common(1)[0][1] if counts else 0
    chosen = control_slugs(truth)
    ctrl = control(arm_votes, chosen)
    missing_reversed = sum((opaque_id(s), "rev") not in arm_votes for s in chosen)
    unanswered = sum(votes[s] is None for s in truth)
    reasons = []
    if n and len(hits) / n < MIN_ACCURACY:
        reasons.append("twist below 75%")
    if n and len(hits) <= best_constant:
        reasons.append("not above the best constant")
    if not ctrl["passed"]:
        reasons.append("reversed-clip control failed")
    if not labels_final or unanswered or missing_reversed:
        verdict = "incomplete"
    elif n < MIN_SCORED or len(chosen) < MIN_CONTROL_ANSWERS:
        verdict = "not evaluable"
    else:
        verdict = "killed" if reasons else "promising"
    return {"twist_exact": len(hits), "n": n, "flip_exact": flips, "always_zero": counts.get(0.0, 0),
            "best_constant": best_constant,
            "unanswered": unanswered, "missing_reversed": missing_reversed,
            "malformed_reversed": sum((opaque_id(s), "rev") in arm_votes and not well_formed(arm_votes[(opaque_id(s), "rev")]) for s in chosen),
            "malformed": sum(votes[s] is not None and not well_formed(votes[s]) for s in truth),
            "unsure": sum(votes[s] is not None and well_formed(votes[s]) and votes[s]["unsure"] for s in truth),
            "twist_1.5_plus": _rate([s for s, t in truth.items() if t["twist"] >= 1.5], hits),
            "cork": _rate([s for s, t in truth.items() if t["cork"] == "yes"], hits),
            "by_owner_twist": {str(v): _rate([s for s, t in truth.items() if t["twist"] == v], hits) for v in sorted(counts)},
            "control": ctrl, "verdict": verdict, "kill_reasons": reasons if verdict == "killed" else []}


def index_votes(rows: list, known_ids: set[str]) -> dict[str, dict]:
    arms: dict[str, dict] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("vote row is not an object")
        arm, clip, variant = row.get("arm"), row.get("id"), row.get("variant")
        if not all(isinstance(x, str) for x in (arm, clip, variant)) or arm not in ARMS or clip not in known_ids or variant not in VARIANTS:
            raise ValueError("a vote row has an unknown arm, id or variant")  # never echo it: an id could be a clip name
        votes = arms.setdefault(arm, {})
        if not any(field in row for field in ANSWER_FIELDS):
            continue  # a failed call is a missing vote, and a retry may replace it
        key = (clip, variant)
        if key in votes:
            raise ValueError(f"duplicate vote in arm {arm} for clip {clip}, {variant}")
        votes[key] = row
    return arms


def score(labels: dict, rows: list, queue_slugs: set[str], accept_unlabeled: bool = False) -> dict:
    truth, excluded = split_labels(labels, queue_slugs)
    arms = index_votes(rows, {opaque_id(s) for s in queue_slugs})
    final = accept_unlabeled or "not labeled" not in excluded.values()
    return {"scored": len(truth), "excluded_by_reason": dict(Counter(excluded.values())), "excluded": excluded,
            "arms": {arm: score_arm(arms[arm], truth, final) if arm in arms else {"verdict": "missing"} for arm in ARMS}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cmd", choices=("control-ids", "score"))
    parser.add_argument("labels", type=Path)
    parser.add_argument("votes", type=Path, nargs="?")
    parser.add_argument("--accept-unlabeled", action="store_true")
    parser.add_argument("--show-excluded", action="store_true")
    args = parser.parse_args()
    slugs = {c["slug"] for c in json.loads(QUEUE.read_text())}
    raw_labels = args.labels.read_bytes()
    labels = json.loads(raw_labels)
    if args.cmd == "control-ids":
        truth, excluded = split_labels(labels, slugs)
        if "not labeled" in excluded.values() and not args.accept_unlabeled:
            raise ValueError("labeling is unfinished, and the control set depends on the final labels")
        print(json.dumps([opaque_id(s) for s in control_slugs(truth)], indent=1))
        return
    if args.votes is None:
        parser.error("score needs VOTES")
    rows = [json.loads(line) for line in args.votes.read_text().splitlines() if line.strip()]
    result = score(labels, rows, slugs, args.accept_unlabeled)
    if not args.show_excluded:
        del result["excluded"]
    # Compare with the hash recorded in the freeze commit: labels must not change after that.
    result["labels_sha256"] = hashlib.sha256(raw_labels).hexdigest()
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    try:
        main()
    except ValueError as error:
        sys.exit(f"stopped: {error}")
