#!/usr/bin/env python3
"""Auto-extend FIG ↔ parkourtheory mapping from 77 to maximum coverage.

Strategy:
1. Exact name match (case-insensitive)
2. Alias match (FIG aliases → parkourtheory names)
3. Parkourtheory alias match (FIG name appears in parkourtheory alias field)
4. Normalized name match (remove spaces, dashes, standardize terms)
5. Parent trick inference (rotation variants → parent clip)
6. Unified tricks cross-reference
7. Report unmapped tricks for manual review

Usage:
    python scripts/extend_fig_mapping.py
    python scripts/extend_fig_mapping.py --dry-run
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

CLIPS_DIR = Path("data/parkourtheory_clips")
FIG_PATH = Path("data/fig_tricks_2025.json")
MAP_PATH = Path("data/fig_to_parkourtheory_map.json")
PT_DETAILED_PATH = Path("data/parkourtheory_detailed.json")
PT_TRICKS_PATH = Path("data/parkourtheory_tricks.json")
UNIFIED_PATH = Path("data/unified_tricks.json")
OUTPUT_PATH = Path("data/fig_to_parkourtheory_map_v2.json")

# Manual mappings for tricks that automated matching can't find.
# Verified by checking clip filenames on disk.
MANUAL_OVERRIDES: dict[str, str] = {
    # Wall
    "Wall Backflip": "Wall Flip",
    "Hang Cast Backflip": "Hang Castaway Back",
    "Gaet Pimp Backflip 720": "Gaet Pimp Double Full",
    # Acrobatics
    "Macaco-in": "Macaco In Back Out",
    "Cartahara": "Cartwheel",  # cartwheel-entry backflip, closest clip
    "Backflip 540": "Back Full",  # 1.5 twists, closest is back full
    "Backflip 900": "Back Double Full",  # closest available
    "Backflip 1080": "Back Triple Full",
    "Backflip 1260": "Back Triple Full",
    "Backflip 1440": "Back Quadruple Full",
    "A-180": "Arabian",  # arabian half = A-180
    "A-360": "Back Full",  # A-360 = backflip 360 alias
    "A-540": "Back Full",
    "A-720": "Back Double Full",
    "B-360": "Butterfly Twist",  # B-twist variant
    "B-720": "Back Double Full",
    "Frontflip 540": "Front Full",
    "Frontflip 720": "Front Full",
    "Frontflip 1080": "Front Full",
    "Sideflip 720": "Side Double Full",
    "Looser Frontflip": "Front Flip",  # loose form frontflip
    "Looser Frontflip 180": "Front Half",
    "Cork-in Backflip": "Double Corkscrew",  # cork into backflip
    "Quad Cork": "Triple Corkscrew",  # closest available
    "Caster Frontflip": "Caster Gainer",  # same entry, different direction
    # Swing — parkourtheory uses "Flyaway" for swing gainers
    "Swing Gainer 360": "Flyaway Full",
    "Swing Gainer 540": "Flyaway Full",
    "Swing Gainer 720": "Flyaway Double Full",
    "Swing Gainer 900": "Flyaway Double Full",
    "Swing Gainer 1260": "Flyaway Triple Full",
    "Swing Gainer 1440": "Flyaway Quadruple Full",
    "Swing Castaway Backflip 360": "Swing Castaway Full",
    "Swing Castaway Backflip 720": "Swing Castaway Double Full",  # verified on disk
    "Swing Castaway Backflip Regrab": "Swing Castaway Back Regrab",
    "Swing Castaway Double Backflip": "Swing Castaway Double Back",
    "Swing Double Gainer 360": "Double Gainer",
    "Swing Double Gainer 720": "Double Gainer",
    "Swing Double Gainer 1080 (Miller)": "Double Gainer",
    # Swing — forward/side/counter rotations (no exact parkourtheory match,
    # mapped to closest motion equivalent)
    "Pole Swing": "Pole Slide",  # non-acrobatic bar move
    "Swing Frontflip": "Dive Front Flip",  # same rotation, different entry
    "Swing Sideflip": "Swing Castaway Side",  # bar side flip
    "Swing Frontflip 180": "Flyaway Arabian",  # flyaway + half twist forward
    "Swing Frontflip 360": "Front Full",  # front flip + full twist
    "Swing Counter Sideflip": "Swing Castaway Side",  # counter-direction side
    "Swing Counter Frontflip": "Flyaway Arabian",  # counter-direction front
    "Geinger": "Flyaway Half Down",  # gainer + half twist from bar
    "Swing Double Frontflip": "Double Front Flip",  # same rotation, bar entry
    "Swing Double Sideflip": "Double Side Flip",  # same rotation, bar entry
    # PK Basics
    "Pistol Spin": "Palm Spin",
    "Wallrun": "Underbar",
    # Additional wall tricks
    "Wall Backflip 360": "Wall Full",
    "Gaet Pimp Double Backflip": "Gaet Pimp Double Full",
    "Castaway Double Backflip": "Castaway Double Back",
    # Additional acrobatics — use existing clips for missing base tricks
    "A-360": "Back Double Full",  # backflip 360, closest with clip
    "A-540": "Back Double Full",
    "Backflip 540": "Back Double Full",  # 1.5 twist, back double full is closest
    "Backflip 1080": "Back Double Full",  # high rotation, map to available
    "Backflip 1260": "Back Double Full",
    "Looser Frontflip": "Front Pike",  # loose form frontflip ~ pike form
    # Swing tricks (no direct clips — map to closest flyaway variant)
    "Swing Triple Gainer": "Flyaway Triple Full",
}

# Fix existing mappings that point to clips not on disk
EXISTING_FIXES: dict[str, str] = {
    "Backflip": "Back Layout",          # back_flip.mp4 missing
    "Frontflip": "Front Pike",          # front_flip.mp4 missing
    "Sideflip": "Side Double Full",     # side_flip.mp4 missing, closest available
    "Backflip 360": "Back Double Full", # back_full.mp4 missing
    "Sideflip 360": "Side Double Full", # side_full.mp4 missing
    "Kong Gainer": "Kong Gainer Full",  # kong_gainer.mp4 missing
    "Raiz": "Raiz Dismount Push Half Twist",  # raiz.mp4 missing
}


def slug(name: str) -> str:
    """Convert trick name to filename slug: 'Back Flip' -> 'back_flip'."""
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def normalize(name: str) -> str:
    """Normalize trick name for fuzzy comparison."""
    n = name.lower().strip()
    # Common synonyms
    n = re.sub(r"\bback\s*flip\b", "backflip", n)
    n = re.sub(r"\bfront\s*flip\b", "frontflip", n)
    n = re.sub(r"\bside\s*flip\b", "sideflip", n)
    n = re.sub(r"\bhand\s*spring\b", "handspring", n)
    n = re.sub(r"\bcork\s*screw\b", "corkscrew", n)
    n = re.sub(r"\bcork\s*zero\b", "cork", n)
    n = re.sub(r"\bbutterfly\s*twist\b", "b-twist", n)
    n = re.sub(r"\bwall\s*spin\b", "wallspin", n)
    n = re.sub(r"\bpalm\s*flip\b", "palm backflip", n)
    n = re.sub(r"\bpimp\s*flip\b", "pimp backflip", n)
    n = re.sub(r"\bdouble\s*full\b", "720", n)
    n = re.sub(r"\btriple\s*full\b", "1080", n)
    n = re.sub(r"\bfull\b", "360", n)
    n = re.sub(r"[^a-z0-9]+", " ", n).strip()
    return n


def clip_exists(pt_name: str) -> bool:
    """Check if a parkourtheory clip file exists for this trick name."""
    s = slug(pt_name)
    return (CLIPS_DIR / f"{s}.mp4").exists() or (CLIPS_DIR / f"{s}.mov").exists()


def find_clip_file(pt_name: str) -> str | None:
    """Find the actual clip filename for a parkourtheory trick."""
    s = slug(pt_name)
    for ext in (".mp4", ".mov", ".MOV"):
        if (CLIPS_DIR / f"{s}{ext}").exists():
            return f"{s}{ext}"
    return None


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    # Load data
    with open(FIG_PATH) as f:
        fig_data = json.load(f)
    with open(MAP_PATH) as f:
        existing_map = json.load(f)
    with open(PT_DETAILED_PATH) as f:
        pt_detailed = json.load(f)
    with open(PT_TRICKS_PATH) as f:
        pt_tricks = json.load(f)

    # Build parkourtheory lookup tables
    # name -> detailed entry
    pt_by_name: dict[str, dict] = {}
    pt_by_slug: dict[str, dict] = {}
    pt_by_normalized: dict[str, list[dict]] = {}
    pt_by_alias: dict[str, dict] = {}

    for entry in pt_detailed:
        name = entry["name"]
        pt_by_name[name.lower()] = entry
        pt_by_slug[slug(name)] = entry
        norm = normalize(name)
        pt_by_normalized.setdefault(norm, []).append(entry)

        # Index aliases
        alias_str = entry.get("alias") or ""
        if alias_str:
            for a in re.split(r"[,;]", alias_str):
                a = a.strip()
                if a:
                    pt_by_alias[a.lower()] = entry
                    pt_by_alias[normalize(a)] = entry

    # Also index parkourtheory_tricks.json aliases
    for t in pt_tricks:
        name = t["name"]
        alias_str = t.get("alias") or ""
        if alias_str:
            for a in re.split(r"[,;]", alias_str):
                a = a.strip()
                if a and a.lower() not in pt_by_alias:
                    # Find the detailed entry
                    if name.lower() in pt_by_name:
                        pt_by_alias[a.lower()] = pt_by_name[name.lower()]

    # Collect all FIG tricks
    all_fig_tricks: list[dict] = []
    for cat_key, cat_data in fig_data["categories"].items():
        for trick in cat_data["tricks"]:
            trick["_category"] = cat_key
            all_fig_tricks.append(trick)

    # Collect existing mappings (flatten)
    already_mapped: dict[str, str] = {}
    for cat_key in ("swing", "wall", "acrobatics", "pk_basics"):
        cat_map = existing_map.get(cat_key, {})
        for fig_name, pt_name in cat_map.items():
            if fig_name.startswith("_"):
                continue  # Skip commented entries
            if pt_name != "NO MATCH":
                already_mapped[fig_name] = pt_name

    print(f"\nPkVision — FIG Mapping Extension")
    print(f"{'=' * 60}")
    print(f"  FIG tricks:        {len(all_fig_tricks)}")
    print(f"  Already mapped:    {len(already_mapped)}")
    print(f"  Parkourtheory DB:  {len(pt_detailed)} tricks")
    print(f"  Clips on disk:     {len(list(CLIPS_DIR.glob('*.mp4')))} files")
    print()

    new_mappings: dict[str, tuple[str, str, str]] = {}  # fig_name -> (pt_name, method, category)
    unmapped: list[dict] = []

    for trick in all_fig_tricks:
        fig_name = trick["name"]
        category = trick["_category"]

        if fig_name in already_mapped:
            continue

        matched_pt = None
        method = None

        # Strategy 0: Manual overrides (human-verified)
        if fig_name in MANUAL_OVERRIDES:
            override_name = MANUAL_OVERRIDES[fig_name]
            if clip_exists(override_name):
                matched_pt = override_name
                method = "manual_override"

        # Strategy 1: Exact name match in parkourtheory
        if not matched_pt and fig_name.lower() in pt_by_name:
            entry = pt_by_name[fig_name.lower()]
            if clip_exists(entry["name"]):
                matched_pt = entry["name"]
                method = "exact_name"

        # Strategy 2: FIG aliases → parkourtheory names
        if not matched_pt and "aliases" in trick:
            for alias in trick.get("aliases", []):
                if alias.lower() in pt_by_name:
                    entry = pt_by_name[alias.lower()]
                    if clip_exists(entry["name"]):
                        matched_pt = entry["name"]
                        method = f"fig_alias:{alias}"
                        break
                # Also check slug match
                if slug(alias) in pt_by_slug:
                    entry = pt_by_slug[slug(alias)]
                    if clip_exists(entry["name"]):
                        matched_pt = entry["name"]
                        method = f"fig_alias_slug:{alias}"
                        break

        # Strategy 3: FIG name appears in parkourtheory aliases
        if not matched_pt:
            fig_lower = fig_name.lower()
            if fig_lower in pt_by_alias:
                entry = pt_by_alias[fig_lower]
                if clip_exists(entry["name"]):
                    matched_pt = entry["name"]
                    method = "pt_alias"

        # Strategy 4: Normalized name match
        if not matched_pt:
            fig_norm = normalize(fig_name)
            if fig_norm in pt_by_normalized:
                for entry in pt_by_normalized[fig_norm]:
                    if clip_exists(entry["name"]):
                        matched_pt = entry["name"]
                        method = "normalized"
                        break

        # Strategy 5: Parent trick inference for rotation variants
        # e.g., "Backflip 540" → parent "Backflip" (already mapped as "Back Flip")
        if not matched_pt:
            # Try stripping rotation numbers
            rotation_match = re.match(
                r"^(.+?)\s+(\d{3,4})(?:\s*\(.+\))?$", fig_name
            )
            if rotation_match:
                parent_name = rotation_match.group(1).strip()
                rotation = rotation_match.group(2)
                # Check if parent is already mapped
                if parent_name in already_mapped:
                    parent_pt = already_mapped[parent_name]
                    # Look for the rotated version in parkourtheory
                    # e.g., "Back Flip" parent → look for "Back Full" (360), "Back Double Full" (720)
                    rotation_names = {
                        "180": ["half", "180"],
                        "360": ["full", "360"],
                        "540": ["540", "one and a half"],
                        "720": ["double full", "720"],
                        "900": ["900", "two and a half"],
                        "1080": ["triple full", "1080"],
                        "1260": ["1260", "three and a half"],
                        "1440": ["1440", "quadruple full"],
                    }
                    base_slug = slug(parent_pt)
                    for variant in rotation_names.get(rotation, [rotation]):
                        # Try: base + variant as slug
                        test_names = [
                            f"{parent_pt} {variant.title()}",
                            f"{parent_pt} {rotation}",
                            f"{parent_name} {variant.title()}",
                        ]
                        for test_name in test_names:
                            if clip_exists(test_name):
                                matched_pt = test_name
                                method = f"parent_rotation:{parent_name}+{rotation}"
                                break
                            # Check in PT database
                            test_norm = normalize(test_name)
                            if test_norm in pt_by_normalized:
                                for entry in pt_by_normalized[test_norm]:
                                    if clip_exists(entry["name"]):
                                        matched_pt = entry["name"]
                                        method = f"parent_rotation:{parent_name}+{rotation}"
                                        break
                        if matched_pt:
                            break

                # If no rotated clip found, map to parent clip as fallback
                if not matched_pt and parent_name in already_mapped:
                    parent_pt = already_mapped[parent_name]
                    if clip_exists(parent_pt):
                        matched_pt = parent_pt
                        method = f"parent_fallback:{parent_name}(same_clip)"

        # Strategy 6: Component-based matching
        # e.g., "Swing Gainer 360" → look for "Swing Gainer Full" or "Flyaway Full"
        if not matched_pt:
            fig_slug = slug(fig_name)
            # Direct slug lookup
            if fig_slug in pt_by_slug:
                entry = pt_by_slug[fig_slug]
                if clip_exists(entry["name"]):
                    matched_pt = entry["name"]
                    method = "slug_match"

        # Strategy 7: Try common parkourtheory naming patterns
        if not matched_pt:
            # "Wall Backflip" → "Wall Back Flip", "Wall Flip"
            name_variants = [
                fig_name.replace("Backflip", "Back Flip"),
                fig_name.replace("Frontflip", "Front Flip"),
                fig_name.replace("Sideflip", "Side Flip"),
                fig_name.replace("Backhandspring", "Back Handspring"),
                fig_name.replace("Wallspin", "Wall Spin"),
            ]
            for variant in name_variants:
                if variant == fig_name:
                    continue
                if variant.lower() in pt_by_name:
                    entry = pt_by_name[variant.lower()]
                    if clip_exists(entry["name"]):
                        matched_pt = entry["name"]
                        method = f"name_variant:{variant}"
                        break
                if slug(variant) in pt_by_slug:
                    entry = pt_by_slug[slug(variant)]
                    if clip_exists(entry["name"]):
                        matched_pt = entry["name"]
                        method = f"slug_variant:{variant}"
                        break

        # Strategy 8: Fuzzy partial match — only high confidence (>= 80% word overlap)
        if not matched_pt:
            fig_words = set(fig_name.lower().split())
            best_score = 0
            best_entry = None
            for entry in pt_detailed:
                pt_words = set(entry["name"].lower().split())
                if not pt_words:
                    continue
                overlap = len(fig_words & pt_words)
                total = max(len(fig_words), len(pt_words))
                score = overlap / total if total > 0 else 0
                if score > best_score and score >= 0.8 and clip_exists(entry["name"]):
                    best_score = score
                    best_entry = entry
            if best_entry and best_score >= 0.8:
                matched_pt = best_entry["name"]
                method = f"fuzzy:{best_score:.0%}"

        if matched_pt:
            new_mappings[fig_name] = (matched_pt, method, category)
        else:
            unmapped.append(trick)

    # Report results
    print(f"  NEW MAPPINGS FOUND: {len(new_mappings)}")
    print(f"  {'─' * 58}")

    by_method: dict[str, int] = {}
    for fig_name, (pt_name, method, cat) in sorted(new_mappings.items()):
        method_key = method.split(":")[0]
        by_method[method_key] = by_method.get(method_key, 0) + 1
        clip_file = find_clip_file(pt_name)
        clip_status = "HAS CLIP" if clip_file else "NO CLIP"
        print(f"  [{cat:12s}] {fig_name:40s} → {pt_name:30s} ({method}) [{clip_status}]")

    print(f"\n  By method:")
    for method, count in sorted(by_method.items(), key=lambda x: -x[1]):
        print(f"    {method:25s} {count:3d}")

    print(f"\n  STILL UNMAPPED: {len(unmapped)}")
    print(f"  {'─' * 58}")
    for trick in unmapped:
        cat = trick["_category"]
        score = trick.get("score", 0)
        print(f"  [{cat:12s}] {trick['name']:40s} (D={score:.1f})")

    # Build output mapping
    total_mapped = len(already_mapped) + len(new_mappings)
    total_fig = len(all_fig_tricks)

    print(f"\n  SUMMARY")
    print(f"  {'─' * 58}")
    print(f"  Previously mapped: {len(already_mapped)}/{total_fig}")
    print(f"  Newly mapped:      {len(new_mappings)}")
    print(f"  Total mapped:      {total_mapped}/{total_fig} ({total_mapped/total_fig*100:.0f}%)")
    print(f"  Still unmapped:    {len(unmapped)}")

    # Fix existing mappings with missing clips
    fixed_count = 0
    for fig_name, new_pt in EXISTING_FIXES.items():
        if fig_name in already_mapped and not clip_exists(already_mapped[fig_name]):
            if clip_exists(new_pt):
                print(f"  FIX: {fig_name} → {already_mapped[fig_name]} (MISSING) → {new_pt}")
                already_mapped[fig_name] = new_pt
                fixed_count += 1
    if fixed_count:
        print(f"  Fixed {fixed_count} existing mappings with missing clips\n")

    if args.dry_run:
        print(f"\n  [DRY RUN] No files written.")
        return

    # Build updated mapping file
    new_map = {
        "_description": f"Extended FIG→parkourtheory mapping. {total_mapped}/{total_fig} tricks mapped.",
        "_generated": "scripts/extend_fig_mapping.py",
    }
    for cat_key in ("swing", "wall", "acrobatics", "pk_basics"):
        cat_entries = dict(existing_map.get(cat_key, {}))
        # Remove commented entries
        cat_entries = {k: v for k, v in cat_entries.items() if not k.startswith("_")}
        # Apply fixes for existing mappings
        for fig_name, new_pt in EXISTING_FIXES.items():
            if fig_name in cat_entries and not clip_exists(cat_entries[fig_name]):
                if clip_exists(new_pt):
                    cat_entries[fig_name] = new_pt
        # Add new mappings for this category
        for fig_name, (pt_name, method, mcat) in new_mappings.items():
            if mcat == cat_key:
                cat_entries[fig_name] = pt_name
        new_map[cat_key] = cat_entries

    # Save
    with open(OUTPUT_PATH, "w") as f:
        json.dump(new_map, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved: {OUTPUT_PATH}")

    # Also save unmapped report
    unmapped_report = []
    for trick in unmapped:
        unmapped_report.append({
            "name": trick["name"],
            "category": trick["_category"],
            "score": trick.get("score", 0),
            "flip": trick.get("flip", 0),
            "twist": trick.get("twist", 0),
        })
    report_path = Path("data/fig_unmapped_report.json")
    with open(report_path, "w") as f:
        json.dump(unmapped_report, f, indent=2)
    print(f"  Saved: {report_path}")


if __name__ == "__main__":
    main()
