"""Expand parsed_tricks.json from unified_tricks.json.

Takes the unified trick database and produces the physics-only format
consumed by the zero-shot matcher and MLP trainer.

Usage:
    python scripts/expand_parsed_tricks.py
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Expand parsed_tricks.json from unified database")
    parser.add_argument("--unified", default="data/unified_tricks.json")
    parser.add_argument("--output", default="data/parsed_tricks.json")
    args = parser.parse_args()

    unified = json.loads(Path(args.unified).read_text())
    print(f"Loaded {len(unified)} unified tricks")

    parsed = []
    for entry in unified:
        p = entry["physics"]
        parsed.append({
            "name": entry["name"],
            "rotation_axis": p["rotation_axis"],
            "direction": p["direction"],
            "rotation_count": p["rotation_count"],
            "twist_count": p["twist_count"],
            "body_shape": p["body_shape"],
            "entry": p["entry"],
            "family": p["family"],
            "alias": entry.get("metadata", {}).get("alias", ""),
        })

    # Save
    output_path = Path(args.output)
    with open(output_path, "w") as f:
        json.dump(parsed, f, indent=2, ensure_ascii=False)

    # Stats
    combos = set()
    for t in parsed:
        combo = (t["rotation_axis"], t["direction"], t["rotation_count"],
                 t["twist_count"], t["body_shape"], t["entry"])
        combos.add(combo)

    print(f"\nExpanded parsed_tricks.json:")
    print(f"  Tricks: {len(parsed)}")
    print(f"  Unique physics combos: {len(combos)}")

    families = Counter(t["family"] for t in parsed)
    axes = Counter(t["rotation_axis"] for t in parsed)
    print(f"\n  Families: {dict(families.most_common())}")
    print(f"  Axes: {dict(axes.most_common())}")
    print(f"\n  Rotation counts: {dict(sorted(Counter(t['rotation_count'] for t in parsed).items()))}")
    print(f"  Twist counts: {dict(sorted(Counter(t['twist_count'] for t in parsed).items()))}")
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
