#!/usr/bin/env python3
"""Build attribute-labeled dataset for hierarchical trick classification.

Labels parkourtheory clips with 6 attributes using 3-tier priority:
1. FIG Code of Points (gold standard, 149 tricks)
2. unified_tricks.json lookup (community-verified physics, 2,677 tricks)
3. Name-based inference (fallback)

Validates labels across sources and flags conflicts.

Usage:
    python scripts/build_attribute_dataset.py
    python scripts/build_attribute_dataset.py --stats
    python scripts/build_attribute_dataset.py --validate  # cross-check sources
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.recognition.trick_name_parser import parse_trick_name

TRUSTED_CUES = ("flip", "direction")
NUMERIC_CUES = ("flip", "twist")


def _cue_eq(cue, a, b) -> bool:
    if cue in NUMERIC_CUES:
        try:
            return abs(float(a) - float(b)) < 0.26
        except (TypeError, ValueError):
            return str(a) == str(b)
    return str(a) == str(b)


def merge_layered(slug, base, base_source, parsed, conf_thresh=0.8):
    cues = parsed["cues"]
    conf = parsed["confidence"]
    merged = dict(base)
    prov = {c: base_source for c in base}
    changes = []
    for cue in TRUSTED_CUES:
        if cue not in cues or conf.get(cue, 0.0) < conf_thresh:
            continue
        pv = cues[cue]
        cur = base.get(cue)
        missing = cue not in base or cur in (None, "none", "unknown")
        if not missing and not _cue_eq(cue, cur, pv):
            if base_source == "fig":
                changes.append({"slug": slug, "cue": cue, "old_value": cur,
                                "old_source": "fig", "parser_value": pv,
                                "parser_conf": conf[cue], "kind": "fig_disagree"})
                continue
            merged[cue] = pv
            prov[cue] = "parser"
            changes.append({"slug": slug, "cue": cue, "old_value": cur,
                            "old_source": base_source, "parser_value": pv,
                            "parser_conf": conf[cue], "kind": "correct"})
        elif missing:
            merged[cue] = pv
            prov[cue] = "parser"
            changes.append({"slug": slug, "cue": cue, "old_value": cur,
                            "old_source": base_source, "parser_value": pv,
                            "parser_conf": conf[cue], "kind": "fill"})
    return merged, prov, changes

ROOT = Path(__file__).resolve().parent.parent
FRAMES_DIR = ROOT / "data" / "v5_full_training" / "frames"
FIG_PATH = ROOT / "data" / "fig_tricks_2025.json"
FIG_MAP_PATH = ROOT / "data" / "fig_to_parkourtheory_map_v2.json"
UNIFIED_PATH = ROOT / "data" / "unified_tricks.json"
OUTPUT_PATH = ROOT / "data" / "v5_attribute_training" / "attribute_manifest.json"

# Direction normalization
DIR_MAP = {
    "backward": "backward",
    "forward": "forward",
    "left": "side",
    "right": "side",
    "side": "side",
}


# ── Source 1: FIG attributes ────────────────────────────────────────────


def load_fig_clip_attrs():
    """FIG trick → clip slug → attributes (gold standard)."""
    with open(FIG_PATH) as f:
        fig = json.load(f)
    with open(FIG_MAP_PATH) as f:
        fig_map = json.load(f)

    fig_attrs = {}
    for cat_name, cat in fig["categories"].items():
        for trick in cat["tricks"]:
            fig_attrs[trick["name"].lower()] = {
                "flip": trick.get("flip", 0),
                "twist": trick.get("twist", 0),
                "direction": DIR_MAP.get(trick.get("direction"), None),
                "context": cat_name,
                "entry": trick.get("entry"),
                "fig_name": trick["name"],
                "d_score": trick.get("score", 0),
            }

    clip_attrs = {}
    for cat, mappings in fig_map.items():
        if cat.startswith("_"):
            continue
        for fig_name, clip_name in mappings.items():
            slug = clip_name.lower().replace(" ", "_")
            if fig_name.lower() in fig_attrs:
                clip_attrs[slug] = fig_attrs[fig_name.lower()].copy()
                clip_attrs[slug]["source"] = "fig"
    return clip_attrs


# ── Source 2: unified_tricks lookup ─────────────────────────────────────


def load_unified_tricks():
    """Load unified_tricks.json into a slug → physics dict."""
    with open(UNIFIED_PATH, encoding="utf-8") as f:
        tricks = json.load(f)

    lookup = {}
    for trick in tricks:
        # Normalize name to slug format (spaces → underscores, lowercase)
        slug = trick.get("canonical_name", trick["name"]).lower().replace(" ", "_")
        physics = trick.get("physics", {})

        lookup[slug] = {
            "direction": DIR_MAP.get(physics.get("direction"), None),
            "flip": physics.get("rotation_count", 0),
            "twist": physics.get("twist_count", 0),
            "body_shape": physics.get("body_shape"),
            "entry": physics.get("entry"),
            "rotation_axis": physics.get("rotation_axis"),
            "family": physics.get("family"),
            "source": "unified",
        }

        # Also index by original name
        alt_slug = trick["name"].lower().replace(" ", "_")
        if alt_slug != slug:
            lookup[alt_slug] = lookup[slug]

    return lookup


# ── Context inference ───────────────────────────────────────────────────


def infer_context(slug, entry=None, family=None):
    """Infer context (ground/wall/swing/pk_basics) from name and attributes."""
    n = slug.lower()

    if any(x in n for x in ["wall_", "_wall", "angel_drop", "devil_drop", "tic_tac",
                              "palm_flip", "palm_back", "gaet_pimp", "pimp_"]):
        return "wall"
    if any(x in n for x in ["swing_", "flyaway", "giant", "sole_circle",
                              "hang_", "crucifix", "elbow_flyaway", "castaway"]):
        # Some castaways are wall moves
        if "wall" in n or entry == "wall":
            return "wall"
        return "swing"
    if any(x in n for x in ["vault", "precision", "speed_", "kong_vault",
                              "dash_vault", "lazy_vault", "reverse_vault",
                              "cat_leap", "climb", "descent", "muscle_up",
                              "pole_slide", "stall_"]):
        return "pk_basics"
    if family == "vault" or family == "roll":
        return "pk_basics"
    if family == "bar":
        return "swing"
    if family == "wall":
        return "wall"
    return "acrobatics"


# ── Binning ─────────────────────────────────────────────────────────────


def bin_flip(val):
    if val <= 0:
        return "0"
    if val <= 0.75:
        return "0.5"
    if val <= 1.25:
        return "1"
    if val <= 1.75:
        return "1.5"
    if val <= 2.25:
        return "2"
    return "3+"


def bin_twist(val):
    if val <= 0:
        return "0"
    if val <= 0.75:
        return "0.5"
    if val <= 1.25:
        return "1"
    if val <= 1.75:
        return "1.5"
    if val <= 2.25:
        return "2"
    return "3+"


# ── Main ─────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stats", action="store_true")
    parser.add_argument("--validate", action="store_true", help="Cross-validate sources")
    args = parser.parse_args()

    # Load all sources
    fig_attrs = load_fig_clip_attrs()
    unified = load_unified_tricks()
    print(f"  Sources: {len(fig_attrs)} FIG, {len(unified)} unified_tricks")

    # Find all clips with .npy frames
    all_clips = []
    for cat in ("acrobatics", "wall", "swing", "pk_basics"):
        cat_dir = FRAMES_DIR / cat
        if cat_dir.exists():
            for p in sorted(cat_dir.glob("*.npy")):
                if p.name.startswith("._"):
                    continue
                all_clips.append((p.stem, cat, str(p)))
    print(f"  Clips with frames: {len(all_clips)}")

    # Label each clip using priority: FIG > unified > name
    dataset = []
    dataset_v2 = []
    sources = Counter()
    conflicts = []
    all_changes = []

    for slug, frame_cat, npy_path in all_clips:
        fig = fig_attrs.get(slug)
        uni = unified.get(slug)

        if fig:
            # Priority 1: FIG gold standard
            attrs = {
                "direction": fig.get("direction") or "none",
                "flip": fig.get("flip", 0),
                "twist": fig.get("twist", 0),
                "body_shape": fig.get("body_shape") or (uni.get("body_shape") if uni else None) or "unknown",
                "entry": fig.get("entry") or (uni.get("entry") if uni else None) or "unknown",
                "context": fig.get("context", frame_cat),
                "source": "fig",
                "fig_name": fig.get("fig_name", ""),
                "d_score": fig.get("d_score", 0),
            }
            sources["fig"] += 1

            # Cross-validate with unified if available
            if uni and args.validate:
                if uni.get("direction") and attrs["direction"] != "none":
                    uni_dir = DIR_MAP.get(uni.get("direction"))
                    if uni_dir and uni_dir != attrs["direction"]:
                        conflicts.append(f"{slug}: direction FIG={attrs['direction']} vs unified={uni_dir}")
                if uni.get("flip", 0) != attrs["flip"]:
                    conflicts.append(f"{slug}: flip FIG={attrs['flip']} vs unified={uni.get('flip')}")

        elif uni:
            # Priority 2: unified_tricks verified physics
            context = infer_context(slug, uni.get("entry"), uni.get("family"))
            attrs = {
                "direction": DIR_MAP.get(uni.get("direction")) or "none",
                "flip": uni.get("flip", 0),
                "twist": uni.get("twist", 0),
                "body_shape": uni.get("body_shape") or "unknown",
                "entry": uni.get("entry") or "unknown",
                "context": context,
                "rotation_axis": uni.get("rotation_axis"),
                "family": uni.get("family"),
                "source": "unified",
            }
            sources["unified"] += 1

        else:
            # Priority 3: couldn't match - use frame category only
            attrs = {
                "direction": "none",
                "flip": 0,
                "twist": 0,
                "body_shape": "unknown",
                "entry": "unknown",
                "context": frame_cat,
                "source": "unmatched",
            }
            sources["unmatched"] += 1

        # Baseline (pre-merge) record — keeps attribute_manifest.json untouched by the parser.
        base_attrs = dict(attrs)
        base_attrs["flip_bin"] = bin_flip(base_attrs["flip"])
        base_attrs["twist_bin"] = bin_twist(base_attrs["twist"])
        dataset.append({"slug": slug, "npy_path": npy_path, "frame_cat": frame_cat, **base_attrs})

        # Parser-cleaned (v2) record — flip/direction filled/corrected, with provenance.
        parsed = parse_trick_name(slug)
        merged, prov, clip_changes = merge_layered(
            slug, attrs, attrs.get("source", "unmatched"),
            {"cues": parsed.cues, "confidence": parsed.confidence})
        merged["label_provenance"] = prov
        merged["flip_bin"] = bin_flip(merged["flip"])
        merged["twist_bin"] = bin_twist(merged["twist"])
        all_changes.extend(clip_changes)
        dataset_v2.append({"slug": slug, "npy_path": npy_path, "frame_cat": frame_cat, **merged})

    # Stats
    print(f"\n  Sources: FIG={sources['fig']}, unified={sources['unified']}, unmatched={sources['unmatched']}")

    if args.validate and conflicts:
        print(f"\n  Cross-validation conflicts ({len(conflicts)}):")
        for c in conflicts[:20]:
            print(f"    {c}")

    print(f"\n  Attribute distributions:")
    for attr in ["direction", "flip_bin", "twist_bin", "body_shape", "entry", "context"]:
        counts = Counter(d[attr] for d in dataset)
        print(f"\n  {attr}:")
        for val, cnt in sorted(counts.items(), key=lambda x: -x[1]):
            bar = "#" * (cnt * 30 // len(dataset))
            print(f"    {val:<12s} {cnt:>5d} ({cnt / len(dataset):>5.1%}) {bar}")

    if args.stats:
        return

    # Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    def _manifest(clips):
        return {
            "type": "attribute_classification",
            "total_clips": len(clips),
            "sources": dict(sources),
            "attribute_classes": {
                "direction": sorted(set(d["direction"] for d in clips)),
                "flip_bin": sorted(set(d["flip_bin"] for d in clips)),
                "twist_bin": sorted(set(d["twist_bin"] for d in clips)),
                "body_shape": sorted(set(d["body_shape"] for d in clips)),
                "entry": sorted(set(d["entry"] for d in clips)),
                "context": sorted(set(d["context"] for d in clips)),
            },
            "clips": clips,
        }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(_manifest(dataset), f, indent=2)

    v2 = OUTPUT_PATH.parent / "attribute_manifest_v2.json"
    with open(v2, "w") as f:
        json.dump(_manifest(dataset_v2), f, indent=2)
    changes_path = ROOT / "data" / "name_grammar" / "changes.json"
    changes_path.parent.mkdir(parents=True, exist_ok=True)
    changes_path.write_text(json.dumps(all_changes, indent=2))
    print(f"  v2 manifest: {v2}  | changes: {len(all_changes)} -> {changes_path}")

    print(f"\n  Saved: {OUTPUT_PATH}")
    print(f"  Clips: {len(dataset)}")


if __name__ == "__main__":
    main()
