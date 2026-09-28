#!/usr/bin/env python3
"""Mine FIG trick families from parkourtheory combo clip names.

Parkourtheory clips are named like "kong_gainer_full.mp4" — this contains a
"gainer" trick. By parsing all 1,618 clip names, we can find 5-250+ clips
per FIG trick family WITHOUT any manual labeling.

Outputs:
    data/trick_families.json — {trick_family: [clip_slugs]}
    data/v5_metric_training/manifest.json — training manifest for metric learning

Usage:
    python scripts/mine_trick_families.py
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CLIPS_DIR = ROOT / "data" / "parkourtheory_clips_cropped"
CLIPS_FALLBACK = ROOT / "data" / "parkourtheory_clips"
FIG_PATH = ROOT / "data" / "fig_tricks_2025.json"
FIG_MAP_PATH = ROOT / "data" / "fig_to_parkourtheory_map_v2.json"
OUTPUT_PATH = ROOT / "data" / "trick_families.json"
MANIFEST_PATH = ROOT / "data" / "v5_metric_training" / "manifest.json"

# FIG trick families — map FIG trick names to search patterns in clip names.
# A "family" groups tricks that share the same base motion (e.g., all gainer variants).
# Patterns are checked against clip slugs (lowercase, underscored).
TRICK_FAMILIES = {
    # === ACROBATICS (ground) ===
    "backflip": {
        "patterns": ["back_flip", "back_layout", "back_tuck", "back_pike"],
        "exclude": ["back_flip_precision", "handspring"],  # combos, not pure backflips
        "fig_tricks": ["Backflip"],
        "category": "acrobatics",
    },
    "frontflip": {
        "patterns": ["front_flip", "front_tuck", "front_pike"],
        "exclude": ["front_flip_precision"],
        "fig_tricks": ["Frontflip"],
        "category": "acrobatics",
    },
    "sideflip": {
        "patterns": ["side_flip"],
        "fig_tricks": ["Sideflip"],
        "category": "acrobatics",
    },
    "gainer": {
        "patterns": ["gainer"],
        "exclude": ["kong_gainer", "caster_gainer", "wall_gainer", "swing_gainer",
                     "cast_gainer", "sitting_dash_gainer", "handstand_gainer",
                     "double_gainer", "flyaway"],
        "fig_tricks": ["Gainer", "Running Gainer", "Cheat Gainer"],
        "category": "acrobatics",
    },
    "kong_gainer": {
        "patterns": ["kong_gainer"],
        "fig_tricks": ["Kong Gainer"],
        "category": "acrobatics",
    },
    "cartwheel": {
        "patterns": ["cartwheel"],
        "exclude": ["dive_wall_cartwheel"],
        "fig_tricks": ["Cartwheel"],
        "category": "acrobatics",
    },
    "aerial": {
        "patterns": ["aerial"],
        "exclude": ["aerial_twist"],  # aerial twist is different
        "fig_tricks": ["Aerial"],
        "category": "acrobatics",
    },
    "macaco": {
        "patterns": ["macaco"],
        "fig_tricks": ["Macaco", "Macaco-in"],
        "category": "acrobatics",
    },
    "webster": {
        "patterns": ["webster"],
        "fig_tricks": ["Webster"],
        "category": "acrobatics",
    },
    "arabian": {
        "patterns": ["arabian"],
        "exclude": ["dark_arabian", "lache_dark"],
        "fig_tricks": ["Arabian", "A-180"],
        "category": "acrobatics",
    },
    "cork": {
        "patterns": ["cork_zero", "cork_"],
        "exclude": ["corkscrew", "double_cork", "triple_cork", "wall_cork"],
        "fig_tricks": ["Cork"],
        "category": "acrobatics",
    },
    "corkscrew": {
        "patterns": ["corkscrew"],
        "fig_tricks": ["Kroc", "Double Kroc", "Triple Kroc"],
        "category": "acrobatics",
    },
    "double_cork": {
        "patterns": ["double_cork", "double_corkscrew"],
        "fig_tricks": ["Double Cork"],
        "category": "acrobatics",
    },
    "triple_cork": {
        "patterns": ["triple_cork", "triple_corkscrew"],
        "fig_tricks": ["Triple Cork"],
        "category": "acrobatics",
    },
    "b_twist": {
        "patterns": ["butterfly_twist", "b_twist"],
        "fig_tricks": ["B-Twist"],
        "category": "acrobatics",
    },
    "raiz": {
        "patterns": ["raiz"],
        "fig_tricks": ["Raiz"],
        "category": "acrobatics",
    },
    "frisbee": {
        "patterns": ["frisbee"],
        "exclude": ["double_frisbee"],
        "fig_tricks": ["Frisbee"],
        "category": "acrobatics",
    },
    "double_frisbee": {
        "patterns": ["double_frisbee"],
        "fig_tricks": ["Double Frisbee"],
        "category": "acrobatics",
    },
    "back_handspring": {
        "patterns": ["back_handspring"],
        "exclude": ["wall_back_handspring"],
        "fig_tricks": ["Backhandspring"],
        "category": "acrobatics",
    },
    "gumbi": {
        "patterns": ["gumbi"],
        "fig_tricks": ["Gumbi"],
        "category": "acrobatics",
    },
    "roll_bomb": {
        "patterns": ["roll_bomb"],
        "fig_tricks": ["Roll Bomb"],
        "category": "acrobatics",
    },
    "tunnel_flip": {
        "patterns": ["tunnel_flip", "tunnel"],
        "fig_tricks": ["Tunnel Flip"],
        "category": "acrobatics",
    },
    "tsukahara": {
        "patterns": ["tsukahara"],
        "fig_tricks": ["Tsukahara"],
        "category": "acrobatics",
    },
    "double_backflip": {
        "patterns": ["double_back_flip", "double_back"],
        "exclude": ["castaway_double_back", "swing_castaway_double_back",
                     "caster_wall_double_back", "wall_double_back"],
        "fig_tricks": ["Double Backflip"],
        "category": "acrobatics",
    },
    "double_frontflip": {
        "patterns": ["double_front_flip", "double_front"],
        "fig_tricks": ["Double Frontflip"],
        "category": "acrobatics",
    },
    # === WALL ===
    "wall_flip": {
        "patterns": ["wall_flip"],
        "exclude": ["wall_flip_back", "wall_flip_wall", "wall_flip_switch",
                     "wall_flip_palm", "wall_flip_down"],
        "fig_tricks": ["Wall Backflip"],
        "category": "wall",
    },
    "wall_spin": {
        "patterns": ["wall_spin"],
        "fig_tricks": ["Wallspin"],
        "category": "wall",
    },
    "wall_gainer": {
        "patterns": ["wall_gainer"],
        "fig_tricks": ["Wall Gainer"],
        "category": "wall",
    },
    "palm_flip": {
        "patterns": ["palm_flip", "palm_back"],
        "exclude": ["double_palm"],
        "fig_tricks": ["Palm Backflip"],
        "category": "wall",
    },
    "pimp_flip": {
        "patterns": ["pimp_flip", "pimp_back"],
        "exclude": ["gaet_pimp"],
        "fig_tricks": ["Pimp Backflip"],
        "category": "wall",
    },
    "gaet_pimp": {
        "patterns": ["gaet_pimp"],
        "fig_tricks": ["Gaet Pimp Backflip 360"],
        "category": "wall",
    },
    "angel_drop": {
        "patterns": ["angel_drop"],
        "fig_tricks": ["Angel Drop"],
        "category": "wall",
    },
    "devil_drop": {
        "patterns": ["devil_drop"],
        "fig_tricks": ["Devil Drop"],
        "category": "wall",
    },
    "wall_cork": {
        "patterns": ["wall_cork"],
        "exclude": ["wall_corkscrew"],
        "fig_tricks": ["Wall Cork"],
        "category": "wall",
    },
    "castaway_back": {
        "patterns": ["castaway_back"],
        "exclude": ["swing_castaway", "pop_castaway", "hang_castaway",
                     "castaway_double"],
        "fig_tricks": ["Castaway Backflip"],
        "category": "wall",
    },
    # === SWING (bar/beam) ===
    "flyaway": {
        "patterns": ["flyaway"],
        "exclude": ["flyaway_full", "flyaway_double", "flyaway_triple",
                     "flyaway_quadruple", "flyaway_in_", "flyaway_half",
                     "flyaway_layout", "flyaway_arabian", "flyaway_one_and",
                     "flyaway_precision", "flyaway_punch", "flyaway_kong",
                     "flyaway_spider", "flyaway_tic", "flyaway_walkdown",
                     "flyaway_palm", "flyaway_cat", "flyaway_catch",
                     "flyaway_ceiling", "flyaway_handstand", "flyaway_helicoptero",
                     "flyaway_x_out", "flyaway_zero"],
        "fig_tricks": ["Swing 180", "Swing Gainer"],
        "category": "swing",
    },
    "flyaway_full": {
        "patterns": ["flyaway_full"],
        "exclude": ["flyaway_full_in_", "flyaway_full_down", "flyaway_full_cat",
                     "flyaway_full_catch", "flyaway_full_double", "flyaway_full_straddle",
                     "flyaway_full_spider", "flyaway_full_unwind"],
        "fig_tricks": ["Swing Gainer 360"],
        "category": "swing",
    },
    "giant": {
        "patterns": ["giant"],
        "exclude": ["baby_knee_giant"],
        "fig_tricks": ["Giant"],
        "category": "swing",
    },
    "sole_circle": {
        "patterns": ["sole_circle"],
        "fig_tricks": ["(Straddle) Sole Circle"],
        "category": "swing",
    },
    # === PK BASICS ===
    "kong_vault": {
        "patterns": ["kong_vault", "kong_cat", "kong_precision"],
        "exclude": ["double_kong", "kong_gainer"],
        "fig_tricks": ["Kong Vault"],
        "category": "pk_basics",
    },
    "side_vault": {
        "patterns": ["side_vault"],
        "fig_tricks": ["Side Vault"],
        "category": "pk_basics",
    },
    "speed_vault": {
        "patterns": ["speed_vault"],
        "fig_tricks": ["Plyo"],
        "category": "pk_basics",
    },
    "climb_up": {
        "patterns": ["climb_up"],
        "fig_tricks": ["Climb Up"],
        "category": "pk_basics",
    },
    "precision": {
        "patterns": ["precision"],
        "exclude": ["360_precision", "back_flip_precision", "front_flip_precision",
                     "side_flip_precision", "kong_precision"],
        "fig_tricks": ["Stride"],
        "category": "pk_basics",
    },
    "tic_tac": {
        "patterns": ["tic_tac"],
        "fig_tricks": ["Tic Tac"],
        "category": "pk_basics",
    },
}


def slug_match(slug: str, patterns: list[str], exclude: list[str] | None = None) -> bool:
    """Check if a clip slug matches any pattern but not any exclude."""
    if exclude:
        for ex in exclude:
            if ex in slug:
                return False
    for pat in patterns:
        if pat in slug:
            return True
    return False


def main():
    clips_dir = CLIPS_DIR if CLIPS_DIR.exists() else CLIPS_FALLBACK
    clips = sorted(clips_dir.glob("*.mp4"))
    print(f"Mining {len(clips)} clips for FIG trick families...\n")

    # Mine families
    families: dict[str, list[str]] = {}
    clip_to_families: dict[str, list[str]] = defaultdict(list)

    for family_name, config in TRICK_FAMILIES.items():
        patterns = config["patterns"]
        exclude = config.get("exclude", [])
        matched = []

        for clip in clips:
            slug = clip.stem
            if slug_match(slug, patterns, exclude):
                matched.append(slug)
                clip_to_families[slug].append(family_name)

        families[family_name] = matched

    # Report
    print(f"{'Family':<25s} {'Clips':>6s}  {'FIG Tricks':<40s}  {'Category':<12s}")
    print(f"{'-'*25} {'-'*6}  {'-'*40}  {'-'*12}")

    total_assignments = 0
    usable_families = 0
    for family_name, matched in sorted(families.items(), key=lambda x: -len(x[1])):
        config = TRICK_FAMILIES[family_name]
        fig_tricks = ", ".join(config["fig_tricks"][:3])
        cat = config["category"]
        count = len(matched)
        total_assignments += count
        if count >= 2:
            usable_families += 1
        marker = "  " if count >= 3 else " !" if count >= 1 else " X"
        print(f"{marker}{family_name:<23s} {count:>6d}  {fig_tricks:<40s}  {cat:<12s}")

    print(f"\n  Families with 2+ clips: {usable_families}/{len(families)}")
    print(f"  Total clip assignments: {total_assignments}")

    # Multi-assigned clips (appear in multiple families)
    multi = {slug: fams for slug, fams in clip_to_families.items() if len(fams) > 1}
    if multi:
        print(f"  Multi-family clips: {len(multi)} (will use primary family only)")

    # Build manifest for metric learning
    # For metric learning, we need: anchor, positive (same family), negative (different family)
    # Manifest groups clips by family
    manifest_families = {}
    for family_name, matched in families.items():
        if len(matched) < 2:
            continue  # Need at least 2 for positive pairs
        config = TRICK_FAMILIES[family_name]
        manifest_families[family_name] = {
            "clips": matched,
            "fig_tricks": config["fig_tricks"],
            "category": config["category"],
            "count": len(matched),
        }

    # Save
    output = {
        "families": {k: v for k, v in families.items()},
        "configs": {k: {kk: vv for kk, vv in v.items() if kk != "patterns" and kk != "exclude"}
                    for k, v in TRICK_FAMILIES.items()},
    }
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved: {OUTPUT_PATH}")

    # Save metric learning manifest
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "type": "metric_learning",
        "families": manifest_families,
        "num_families": len(manifest_families),
        "total_clips": sum(f["count"] for f in manifest_families.values()),
        "clips_dir": str(clips_dir),
    }
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  Saved: {MANIFEST_PATH}")

    print(f"\n  Ready for metric learning: {len(manifest_families)} families, "
          f"{manifest['total_clips']} clips")
    print(f"  Next: python scripts/train_metric_modal.py")


if __name__ == "__main__":
    main()
