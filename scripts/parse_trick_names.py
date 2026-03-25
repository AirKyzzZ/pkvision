"""Parse parkourtheory trick names into physics-based TrickDefinition parameters.

Parkour trick names are compositional — they encode the physics:
  "Running Gainer Double Full" = running entry + backward + 1 flip + 2 twists
  "Wall Back Cork Half" = wall entry + backward + off-axis + 0.5 twist
  "720 Dive Roll" = 2 rotations + diving entry + roll landing

This script parses each name into: rotation_axis, direction, rotation_count,
twist_count, body_shape, entry_type — matching our TrickDefinition format.

Usage:
    python scripts/parse_trick_names.py
    python scripts/parse_trick_names.py --output data/parsed_tricks.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


# -- Token-to-physics mappings ------------------------------------------------

# Rotation indicators and their flip counts
ROTATION_TOKENS = {
    # Degree-based
    "180": 0.5, "270": 0.75, "360": 1.0, "450": 1.25,
    "540": 1.5, "630": 1.75, "720": 2.0, "810": 2.25,
    "900": 2.5, "1080": 3.0,
    # Word-based
    "half": 0.5, "double": 2.0, "triple": 3.0, "quad": 4.0,
}

# Direction indicators
DIRECTION_TOKENS = {
    "front": "forward", "forward": "forward", "frog": "forward",
    "back": "backward", "backward": "backward",
    "side": "left", "lateral": "left",
}

# Axis indicators
AXIS_TOKENS = {
    "cork": "off_axis", "corkscrew": "off_axis",
    "flip": "lateral", "tuck": "lateral", "pike": "lateral", "layout": "lateral",
    "spin": "longitudinal", "twist": "longitudinal",
    "side": "sagittal", "aerial": "sagittal",
}

# Entry type indicators
ENTRY_TOKENS = {
    "wall": "wall", "two-step": "wall", "2-step": "wall",
    "running": "running", "punch": "running",
    "standing": "standing",
    "webster": "one_leg", "one-leg": "one_leg",
    "bar": "edge", "swing": "edge", "flyaway": "edge", "lache": "edge",
    "kong": "running", "dash": "running", "speed": "running",
    "drop": "edge", "dismount": "edge",
}

# Body shape indicators
SHAPE_TOKENS = {
    "tuck": "tuck",
    "pike": "pike",
    "layout": "layout", "whip": "layout",
    "open": "open", "straddle": "open",
}

# Twist indicators (these indicate twists ON TOP of the main rotation)
TWIST_TOKENS = {
    "full": 1.0,        # "full" = 1 twist
    "half-twist": 0.5,
}

# Special trick types that override defaults
SPECIAL_TRICKS = {
    "gainer": {"direction": "backward", "entry": "running", "axis": "lateral", "rotation": 1.0},
    "webster": {"direction": "forward", "entry": "one_leg", "axis": "lateral", "rotation": 1.0},
    "castaway": {"direction": "backward", "axis": "off_axis", "rotation": 1.5, "twist": 0.5},
    "arabian": {"direction": "forward", "axis": "lateral", "twist": 0.5},
    "pimp": {"direction": "backward", "axis": "lateral"},
    "raiz": {"direction": "backward", "axis": "off_axis", "rotation": 0.5, "twist": 0.5},
    "b-twist": {"direction": "left", "axis": "longitudinal", "twist": 1.0, "rotation": 0.0},
    "gaet": {"direction": "left", "axis": "sagittal"},
    "palm": {"entry": "wall"},
    "dive": {"direction": "forward"},
    "macaco": {"direction": "backward", "entry": "standing", "axis": "lateral"},
    "krok": {"direction": "forward", "axis": "lateral", "rotation": 0.5},
    "cody": {"direction": "backward", "entry": "edge", "axis": "lateral"},
    "devil": {"axis": "off_axis"},
}

# Movement family classification
VAULT_TOKENS = {"vault", "kong", "dash", "speed", "monkey", "reverse", "kash", "thief", "lazy"}
ROLL_TOKENS = {"roll", "bomb"}
BAR_TOKENS = {"flyaway", "lache", "swing", "dismount", "hang", "stall"}
WALL_TOKENS = {"wall", "palm", "tic-tac", "tunnel"}
GROUND_TOKENS = {"handstand", "cartwheel", "roundoff", "handspring", "macaco"}


# -- Trick family classification ----------------------------------------------

def classify_family(name_lower: str, tokens: list[str]) -> str:
    token_set = set(tokens)
    if token_set & VAULT_TOKENS:
        return "vault"
    if token_set & BAR_TOKENS:
        return "bar"
    if token_set & WALL_TOKENS and not (token_set & {"flip", "cork", "corkscrew"}):
        return "wall"
    if token_set & ROLL_TOKENS and not (token_set & {"flip", "cork", "gainer"}):
        return "roll"
    if token_set & GROUND_TOKENS:
        return "ground"
    return "flip"  # Default: most tricks are flips/rotations


# -- Main parser --------------------------------------------------------------

def parse_trick_name(name: str) -> dict:
    """Parse a trick name into physics parameters."""
    name_lower = name.lower().replace("-", " ")
    tokens = name_lower.split()

    result = {
        "name": name,
        "rotation_axis": "lateral",  # default
        "direction": "backward",     # default (most tricks are backward)
        "rotation_count": 1.0,       # default
        "twist_count": 0.0,
        "body_shape": "tuck",        # default
        "entry": "standing",         # default
        "family": "flip",
    }

    # Apply special trick overrides first
    for keyword, overrides in SPECIAL_TRICKS.items():
        if keyword in tokens:
            for key, val in overrides.items():
                if key == "rotation":
                    result["rotation_count"] = val
                elif key == "twist":
                    result["twist_count"] = val
                elif key == "axis":
                    result["rotation_axis"] = val
                elif key == "direction":
                    result["direction"] = val
                elif key == "entry":
                    result["entry"] = val

    # Parse direction
    for token in tokens:
        if token in DIRECTION_TOKENS:
            result["direction"] = DIRECTION_TOKENS[token]

    # Parse entry type
    for token in tokens:
        if token in ENTRY_TOKENS:
            result["entry"] = ENTRY_TOKENS[token]

    # Parse body shape
    for token in tokens:
        if token in SHAPE_TOKENS:
            result["body_shape"] = SHAPE_TOKENS[token]

    # Parse rotation axis
    for token in tokens:
        if token in AXIS_TOKENS:
            result["rotation_axis"] = AXIS_TOKENS[token]

    # Parse rotation count from degree tokens
    rotation_set = False
    for token in tokens:
        if token in ROTATION_TOKENS:
            val = ROTATION_TOKENS[token]
            if token in ("double", "triple", "quad"):
                # Multipliers apply to the base rotation
                if not rotation_set:
                    result["rotation_count"] = val
                    rotation_set = True
                else:
                    result["rotation_count"] *= val
            elif token == "half" and not rotation_set:
                result["rotation_count"] = 0.5
                rotation_set = True
            elif token.isdigit():
                # Degree-based: 360 = 1 flip, 720 = 2 flips
                result["rotation_count"] = val
                rotation_set = True

    # Parse twist count
    # "full" means 1 twist (360 deg around longitudinal axis)
    full_count = tokens.count("full")
    if full_count > 0:
        # Check if "full" modifies rotation or twist
        # If there's also a flip indicator, "full" = twist
        has_flip_indicator = any(t in tokens for t in ["flip", "gainer", "webster", "front", "back", "cork", "corkscrew", "arabian", "pimp"])
        if has_flip_indicator or result["rotation_count"] >= 1.0:
            result["twist_count"] = max(result["twist_count"], full_count * 1.0)
        elif full_count == 1 and result["rotation_count"] == 1.0:
            # Standalone "full" usually means back full (1 flip + 1 twist)
            result["twist_count"] = 1.0

    # "half" as twist modifier (when combined with other rotation)
    if "half" in tokens and result["rotation_count"] >= 1.0:
        # "half" after a rotation = 0.5 twist
        half_idx = tokens.index("half")
        # Check if it's at the end (twist modifier) vs beginning (rotation modifier)
        if half_idx > len(tokens) // 2:
            result["twist_count"] = max(result["twist_count"], 0.5)

    # Classify family
    result["family"] = classify_family(name_lower, tokens)

    # Fix logical inconsistencies
    # Cork always has at least 1 twist
    if "cork" in tokens and result["twist_count"] == 0:
        result["twist_count"] = 1.0
    # Corkscrew = cork variant
    if "corkscrew" in tokens and result["twist_count"] == 0:
        result["twist_count"] = 1.0

    # Wall tricks default to wall entry
    if "wall" in tokens:
        result["entry"] = "wall"

    return result


def main():
    parser = argparse.ArgumentParser(description="Parse parkourtheory trick names")
    parser.add_argument("--input", default="data/parkourtheory_tricks.json")
    parser.add_argument("--output", default="data/parsed_tricks.json")
    args = parser.parse_args()

    tricks = json.load(open(args.input))
    parsed = []

    for trick in tricks:
        result = parse_trick_name(trick["name"])
        result["alias"] = trick.get("alias", "")
        parsed.append(result)

    # Save
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(parsed, f, indent=2, ensure_ascii=False)

    # Stats
    from collections import Counter
    families = Counter(t["family"] for t in parsed)
    axes = Counter(t["rotation_axis"] for t in parsed)
    directions = Counter(t["direction"] for t in parsed)
    entries = Counter(t["entry"] for t in parsed)
    shapes = Counter(t["body_shape"] for t in parsed)

    print(f"Parsed {len(parsed)} tricks\n")
    print("Families:", dict(families))
    print("Axes:", dict(axes))
    print("Directions:", dict(directions))
    print("Entries:", dict(entries))
    print("Shapes:", dict(shapes))

    rot_counts = Counter(t["rotation_count"] for t in parsed)
    twist_counts = Counter(t["twist_count"] for t in parsed)
    print("Rotation counts:", dict(sorted(rot_counts.items())))
    print("Twist counts:", dict(sorted(twist_counts.items())))

    print(f"\nSaved to {args.output}")

    # Show some examples
    print("\nExample parses:")
    samples = ["Back Flip", "Double Cork Full", "Running Gainer", "Webster",
               "Wall Back Full", "720 Dive Roll", "Side Flip", "Triple Cork",
               "Castaway", "Kong Gainer", "Aerial", "B-Twist"]
    for name in samples:
        matching = [t for t in parsed if t["name"] == name]
        if matching:
            t = matching[0]
            print(f"  {name:30s} -> axis={t['rotation_axis']:12s} dir={t['direction']:10s} "
                  f"rot={t['rotation_count']:.1f} twist={t['twist_count']:.1f} "
                  f"shape={t['body_shape']:8s} entry={t['entry']:10s} family={t['family']}")


if __name__ == "__main__":
    main()
