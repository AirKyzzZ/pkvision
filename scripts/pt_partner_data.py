#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PT_MOVES = ROOT / "data/parkourtheory_detailed.json"
GOLD = ROOT / "data/v5_attribute_training/attribute_manifest_v2.json"
OUT_DIR = ROOT / "data/partner/parkourtheory"
SHEET = OUT_DIR / "move_sheet.csv"
LABELS = OUT_DIR / "labels.json"

CONTEXT_COLUMNS = ("pt_slug", "name", "alias", "pt_type", "pt_url", "pkvision_clip", "priority")
ENUMS = {
    "flip_dir": ("front", "back", "side", "none"),
    "travel": ("forward", "backward", "sideways", "up"),
    "axis": ("on", "off"),
    "shape": ("tuck", "pike", "straight", "straddle", "open", "any"),
    "takeoff": ("one_foot", "two_feet", "sequential", "hands", "swing"),
    "takeoff_from": ("ground", "wall", "bar", "obstacle", "ledge", "beam"),
    "hands": ("none", "obstacle", "wall", "bar", "ground"),
    "landing": ("two_feet", "one_foot", "hands", "roll", "bar"),
    "demo_is_canonical": ("yes", "variant", "no"),
}
MULTI_VALUED = {"hands"}
STEPS = {"flips": 4, "twists": 2}
TIMES = ("move_start_s", "move_end_s")
FILL_COLUMNS = ("flips", "flip_dir", "twists", "travel", "axis", "shape", "takeoff", "takeoff_from", "hands",
                "landing", "demo_is_canonical", "move_start_s", "move_end_s", "notes")
UNKNOWN = ("", "?")
GOLD_DIRECTION = {"front": "forward", "back": "backward", "side": "side", "none": "none"}
GOLD_SHAPE = {"tuck": "tuck", "pike": "pike", "straight": "layout", "open": "open"}


def slugify(text: str | None) -> str:
    return re.sub(r"[^a-z0-9]+", "_", (text or "").lower()).strip("_")


def pt_slug(move: dict) -> str:
    return (move.get("url") or "").rstrip("/").split("/")[-1]


def corpus_slug(move: dict, corpus: set[str]) -> str | None:
    for candidate in (pt_slug(move), move.get("name"), move.get("name_on_site"), move.get("alias")):
        slug = slugify(candidate)
        if slug in corpus:
            return slug
    return None


def priority(move: dict, clip: str | None) -> int:
    if not clip:
        return 4
    type_ = (move.get("type") or "").lower()
    return 1 if "twist" in type_ else 2 if "flip" in type_ else 3


def build_sheet(moves: list[dict], corpus: set[str]) -> list[dict]:
    rows = []
    for move in moves:
        if not move.get("url"):
            continue
        clip = corpus_slug(move, corpus)
        rows.append({
            "pt_slug": pt_slug(move),
            "name": move["name"],
            "alias": move.get("alias") or "",
            "pt_type": move.get("type") or "",
            "pt_url": move["url"],
            "pkvision_clip": clip or "",
            "priority": priority(move, clip),
            **dict.fromkeys(FILL_COLUMNS, ""),
        })
    return sorted(rows, key=lambda r: (r["priority"], r["name"].lower()))


def validate(row: dict) -> list[str]:
    errors = []
    for column, allowed in ENUMS.items():
        value = (row.get(column) or "").strip().lower()
        if value in UNKNOWN:
            continue
        parts = value.split("+") if column in MULTI_VALUED else [value]
        if any(part not in allowed for part in parts):
            errors.append(f"{column}={value!r} not in {'/'.join(allowed)}")
    for column, steps in STEPS.items():
        value = (row.get(column) or "").strip()
        if value in UNKNOWN:
            continue
        try:
            number = float(value)
        except ValueError:
            errors.append(f"{column}={value!r} is not a number")
            continue
        if number < 0 or abs(number * steps - round(number * steps)) > 1e-6:
            errors.append(f"{column}={value} must be a non-negative multiple of {1 / steps:g}")
    times = []
    for column in TIMES:
        value = (row.get(column) or "").strip()
        if value in UNKNOWN:
            continue
        try:
            times.append(float(value))
        except ValueError:
            errors.append(f"{column}={value!r} is not a number of seconds")
    if len(times) == 2 and not 0 <= times[0] < times[1]:
        errors.append("move_start_s must be before move_end_s")
    return errors


def number(value: str | None) -> float | None:
    value = (value or "").strip()
    return None if value in UNKNOWN else float(value)


def as_gold(row: dict) -> dict:
    flip_dir = (row.get("flip_dir") or "").strip().lower()
    axis = (row.get("axis") or "").strip().lower()
    shape = (row.get("shape") or "").strip().lower()
    return {
        "flip": number(row.get("flips")),
        "twist": number(row.get("twists")),
        "direction": GOLD_DIRECTION.get(flip_dir),
        "rotation_axis": "off_axis" if axis == "off" else None,
        "body_shape": GOLD_SHAPE.get(shape),
        "label_provenance": "parkourtheory",
    }


def compare(rows: list[dict], gold: dict[str, dict]) -> dict[str, dict]:
    result = {}
    for attribute in ("flip", "twist", "direction"):
        pairs = Counter()
        for row in rows:
            ours = gold.get(row.get("pkvision_clip") or "")
            theirs = as_gold(row)[attribute]
            if ours is None or theirs is None or ours.get(attribute) is None:
                continue
            pairs[(ours[attribute], theirs)] += 1
        compared = sum(pairs.values())
        agree = sum(n for (ours, theirs), n in pairs.items() if ours == theirs)
        result[attribute] = {"compared": compared, "agree": agree,
                             "top_disagreements": [[ours, theirs, n] for (ours, theirs), n in pairs.most_common()
                                                   if ours != theirs][:8]}
    return result


def load_moves() -> list[dict]:
    return json.loads(PT_MOVES.read_text(encoding="utf-8"))


def load_gold() -> dict[str, dict]:
    clips = json.loads(GOLD.read_text(encoding="utf-8"))["clips"]
    return {c["slug"].lower(): c for c in clips if c.get("source") != "fig"}


def cmd_template(_: argparse.Namespace) -> int:
    rows = build_sheet(load_moves(), set(load_gold()))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with SHEET.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CONTEXT_COLUMNS + FILL_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    counts = Counter(r["priority"] for r in rows)
    print(f"{SHEET.relative_to(ROOT)}: {len(rows)} moves, by priority {dict(sorted(counts.items()))}")
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    with Path(args.sheet).open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    missing = [c for c in ("pt_slug", "pkvision_clip") + FILL_COLUMNS if rows and c not in rows[0]]
    if missing:
        print(f"missing columns: {', '.join(missing)}")
        return 1
    bad = {line: errors for line, row in enumerate(rows, start=2) if (errors := validate(row))}
    for line, errors in bad.items():
        print(f"line {line}: {'; '.join(errors)}")
    filled = [row for row in rows if any((row.get(c) or "").strip() not in UNKNOWN for c in FILL_COLUMNS)]
    print(f"{len(rows)} rows, {len(filled)} filled, {len(bad)} with errors")
    if bad:
        return 1
    print(json.dumps(compare(filled, load_gold()), indent=2))
    if args.write:
        labels = {row["pkvision_clip"]: as_gold(row) for row in filled if row.get("pkvision_clip")}
        LABELS.write_text(json.dumps(labels, indent=2, sort_keys=True), encoding="utf-8")
        print(f"wrote {len(labels)} clip labels to {LABELS.relative_to(ROOT)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Move sheet for Parkour Theory: build it, then check what comes back.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("template", help="write data/partner/parkourtheory/move_sheet.csv").set_defaults(func=cmd_template)
    check = sub.add_parser("check", help="validate a returned sheet and compare it with our name-derived gold")
    check.add_argument("sheet")
    check.add_argument("--write", action="store_true", help="write data/partner/parkourtheory/labels.json (gitignored)")
    check.set_defaults(func=cmd_check)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
