"""Generate expanded TRICK_DEFINITIONS from unified_tricks.json.

Selects the most recognizable/common tricks and generates TrickDefinition
objects for ml/trick_physics.py. Prioritizes tricks that are:
1. From multiple sources (well-known)
2. Have explicit difficulty levels (well-documented)
3. Cover diverse physics combinations
4. Are core tricks in their family

Usage:
    python scripts/generate_trick_definitions.py
    python scripts/generate_trick_definitions.py --count 100
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


# Timing estimates based on physics
def estimate_timing(rot_count: float, twist_count: float, entry: str) -> dict:
    """Estimate takeoff/air/landing timing from physics."""
    base_duration = 0.6 + rot_count * 0.3 + twist_count * 0.1
    takeoff = 0.15 if entry == "standing" else 0.12
    landing = 0.20 + rot_count * 0.03
    air = base_duration - takeoff - landing
    return {
        "takeoff_duration": round(takeoff, 2),
        "air_duration": round(max(0.2, air), 2),
        "landing_duration": round(landing, 2),
    }


def estimate_height(rot_count: float, body_shape: str) -> dict:
    """Estimate jump height from rotation count and body shape."""
    base = 0.4 + rot_count * 0.4
    shape_mult = {"tuck": 0.9, "pike": 1.0, "layout": 1.1, "open": 1.0}
    typical = base * shape_mult.get(body_shape, 1.0)
    return {
        "min_height_m": round(max(0.3, typical * 0.6), 1),
        "typical_height_m": round(typical, 1),
    }


def score_trick(entry: dict) -> float:
    """Score a trick for inclusion in TRICK_DEFINITIONS.
    Higher = more important to include.
    """
    score = 0.0

    # Multi-source bonus (well-known trick)
    score += len(entry["sources"]) * 5.0

    # Has explicit difficulty (well-documented)
    if entry["difficulty"] is not None:
        score += 3.0

    # Has prerequisites/subsequents (connected in the graph)
    score += min(len(entry["related_tricks"]["prerequisites"]), 3) * 1.0
    score += min(len(entry["related_tricks"]["subsequents"]), 5) * 1.5

    # Has descriptions (more info available)
    score += min(len(entry["descriptions"]), 3) * 1.0

    # Penalize very obscure tricks (no descriptions, single source)
    if len(entry["sources"]) == 1 and not entry["descriptions"]:
        score -= 3.0

    # Penalize non-aerial families (rolls, vaults) — less relevant for recognition
    family = entry["physics"]["family"]
    if family in ("roll", "vault", "wall", "ground"):
        score -= 2.0
    if family == "bar":
        score -= 1.0

    return score


def make_trick_id(name: str) -> str:
    """Convert display name to snake_case trick_id."""
    tid = name.lower().replace("-", "_").replace(" ", "_")
    tid = "".join(c for c in tid if c.isalnum() or c == "_")
    tid = "_".join(part for part in tid.split("_") if part)
    return tid


def generate_definition_code(entry: dict) -> str:
    """Generate Python code for a TrickDefinition."""
    p = entry["physics"]
    tid = make_trick_id(entry["name"])
    name = entry["name"]

    timing = estimate_timing(p["rotation_count"], p["twist_count"], p["entry"])
    height = estimate_height(p["rotation_count"], p["body_shape"])

    axis_map = {"lateral": "RotationAxis.LATERAL", "longitudinal": "RotationAxis.LONGITUDINAL",
                "off_axis": "RotationAxis.OFF_AXIS", "sagittal": "RotationAxis.SAGITTAL"}
    dir_map = {"forward": "Direction.FORWARD", "backward": "Direction.BACKWARD",
               "left": "Direction.LEFT", "right": "Direction.RIGHT"}
    shape_map = {"tuck": "BodyShape.TUCK", "pike": "BodyShape.PIKE",
                 "layout": "BodyShape.LAYOUT", "open": "BodyShape.OPEN"}
    entry_map = {"standing": "EntryType.STANDING", "running": "EntryType.RUNNING",
                 "one_leg": "EntryType.ONE_LEG", "wall": "EntryType.WALL", "edge": "EntryType.EDGE"}

    parts = [
        f'    "{tid}": TrickDefinition("{tid}", "{name}",',
        f'        {axis_map[p["rotation_axis"]]}, {dir_map[p["direction"]]}, {p["rotation_count"]}',
    ]

    extras = []
    if p["twist_count"] > 0:
        extras.append(f'twist_count={p["twist_count"]}')
    if p["body_shape"] != "tuck":
        extras.append(f'body_shape={shape_map[p["body_shape"]]}')
    if p["entry"] != "standing":
        extras.append(f'entry={entry_map[p["entry"]]}')
    if height["typical_height_m"] != 1.0:
        extras.append(f'typical_height_m={height["typical_height_m"]}')
    if timing["air_duration"] > 0.7:
        extras.append(f'typical_duration_s={round(sum(timing.values()), 1)}')

    if extras:
        parts[-1] += ","
        parts.append(f'        {", ".join(extras)}),')
    else:
        parts[-1] += "),"

    return "\n".join(parts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--unified", default="data/unified_tricks.json")
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--output", default=None,
                        help="Output file (default: print to stdout)")
    args = parser.parse_args()

    unified = json.loads(Path(args.unified).read_text())

    # Score and rank
    scored = [(score_trick(e), e) for e in unified]
    scored.sort(key=lambda x: -x[0])

    # Select top N, ensuring physics diversity
    selected = []
    seen_combos = set()
    for score, entry in scored:
        p = entry["physics"]
        combo = (p["rotation_axis"], p["direction"], p["rotation_count"],
                 p["twist_count"], p["body_shape"], p["entry"])

        # Always include if new physics combo, or if high score
        if combo not in seen_combos or score > 10:
            selected.append(entry)
            seen_combos.add(combo)

        if len(selected) >= args.count:
            break

    # Sort by family then difficulty for readability
    family_order = {"flip": 0, "bar": 1, "vault": 2, "wall": 3, "ground": 4, "roll": 5, "kick": 6}
    selected.sort(key=lambda e: (family_order.get(e["physics"]["family"], 9), e.get("difficulty", 5)))

    # Generate code
    lines = []
    current_family = None
    for entry in selected:
        fam = entry["physics"]["family"]
        if fam != current_family:
            if current_family is not None:
                lines.append("")
            lines.append(f"    # --- {fam.title()} tricks ---")
            current_family = fam
        lines.append(generate_definition_code(entry))

    code = "\n".join(lines)

    if args.output:
        Path(args.output).write_text(code)
        print(f"Saved {len(selected)} definitions to {args.output}")
    else:
        print(code)

    # Stats
    from collections import Counter
    print(f"\n# Selected: {len(selected)} tricks")
    print(f"# Physics combos: {len(seen_combos)}")
    print(f"# By family: {dict(Counter(e['physics']['family'] for e in selected).most_common())}")


if __name__ == "__main__":
    main()
