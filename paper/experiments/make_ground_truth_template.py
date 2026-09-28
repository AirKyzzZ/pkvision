"""Generate a ground-truth template CSV for the VLM benchmark.

Enumerates single-trick candidate clips across the repo and writes one row per
clip with best-effort auto-filled FIG trick names. The human reviewer (Maxime)
fills the remaining rows and fixes auto-fills where wrong. D-scores are looked
up from data/fig_tricks_2025.json after names are confirmed.

Run:
    python paper/experiments/make_ground_truth_template.py

Writes:
    paper/experiments/ground_truth.csv
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "paper" / "experiments" / "ground_truth.csv"
FIG_JSON = REPO / "data" / "fig_tricks_2025.json"

# (directory, pool) pairs to enumerate for single-trick clips (POOL-B).
# POOL-A full competition runs are added at the bottom as explicit rows.
SINGLE_CLIP_DIRS = [
    (REPO / "data" / "final_clips", "POOL-B"),
    (REPO / "data" / "vlm_clips" / "IMG_5985", "POOL-B"),
    (REPO / "data" / "vlm_clips" / "test_run_2_zoomed", "POOL-B"),
]

# POOL-A: full competition runs (multi-trick). Listed so the reviewer can fill
# in the GT trick list manually --- they're evaluated differently from single
# clips (segmenter runs first, then each segment is judged).
POOL_A_RUNS = [
    REPO / "data" / "run_testing" / "IMG_5985.mov",
    REPO / "data" / "run_testing" / "IMG_4243.MOV",
    REPO / "data" / "run_testing" / "elis-final-run-japan.mp4",
    REPO / "data" / "run_testing" / "test_run_2.mp4",
]

# Filename keyword --> FIG trick name guess.
# Applied case-insensitively. First match wins.
NAME_GUESSES = [
    (re.compile(r"back[_\- ]?double[_\- ]?full", re.I), "Back Double Full"),
    (re.compile(r"back[_\- ]?full", re.I),             "Back Full"),
    (re.compile(r"back[_\- ]?flip|backflip", re.I),    "Back Flip"),
    (re.compile(r"double[_\- ]?cork", re.I),           "Double Cork"),
    (re.compile(r"triple[_\- ]?cork", re.I),           "Triple Cork"),
    (re.compile(r"\bcork\b", re.I),                    "Cork"),
    (re.compile(r"gainer[_\- ]?full", re.I),           "Gainer Full"),
    (re.compile(r"gainer", re.I),                      "Gainer"),
    (re.compile(r"front[_\- ]?flip|frontflip", re.I),  "Frontflip"),
    (re.compile(r"side[_\- ]?flip|sideflip", re.I),    "Sideflip"),
    (re.compile(r"webster", re.I),                     "Webster"),
    (re.compile(r"krok", re.I),                        "Krok"),
    (re.compile(r"raiz", re.I),                        "Raiz"),
]


def load_fig_index() -> dict[str, float]:
    """Return a case-insensitive name -> D-score lookup over all FIG tricks."""
    data = json.loads(FIG_JSON.read_text())
    idx: dict[str, float] = {}
    for cat in data["categories"].values():
        for trick in cat["tricks"]:
            idx[trick["name"].lower()] = float(trick["score"])
            for alias in trick.get("aliases", []) or []:
                idx.setdefault(alias.lower(), float(trick["score"]))
    return idx


def guess_name(filename: str) -> str:
    stem = Path(filename).stem.lower()
    for pat, canonical in NAME_GUESSES:
        if pat.search(stem):
            return canonical
    return ""


def main() -> None:
    fig_idx = load_fig_index()
    rows: list[dict[str, str]] = []

    # POOL-A full runs.
    for run in POOL_A_RUNS:
        if not run.exists():
            continue
        rows.append(
            {
                "clip_path": str(run.relative_to(REPO)),
                "pool": "POOL-A",
                "fig_name": "",           # multi-trick, fill manually as comma-list
                "d_score": "",
                "flip_count": "",
                "twist_count": "",
                "direction": "",
                "takeoff": "",
                "notes": "FULL RUN --- list GT tricks in fig_name as comma-separated, in order",
            }
        )

    # POOL-B single-trick clips.
    for base, pool in SINGLE_CLIP_DIRS:
        if not base.exists():
            continue
        for clip in sorted(base.glob("*.mp4")) + sorted(base.glob("*.MOV")) + sorted(base.glob("*.mov")):
            guess = guess_name(clip.name)
            d_score = fig_idx.get(guess.lower(), "") if guess else ""
            rows.append(
                {
                    "clip_path": str(clip.relative_to(REPO)),
                    "pool": pool,
                    "fig_name": guess,
                    "d_score": str(d_score) if d_score != "" else "",
                    "flip_count": "",
                    "twist_count": "",
                    "direction": "",
                    "takeoff": "",
                    "notes": "" if guess else "AUTO-GUESS FAILED --- fill manually",
                }
            )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "clip_path",
                "pool",
                "fig_name",
                "d_score",
                "flip_count",
                "twist_count",
                "direction",
                "takeoff",
                "notes",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {OUT.relative_to(REPO)}")
    auto = sum(1 for r in rows if r["fig_name"])
    print(f"  {auto} rows have an auto-filled fig_name, {len(rows) - auto} need manual entry.")


if __name__ == "__main__":
    main()
