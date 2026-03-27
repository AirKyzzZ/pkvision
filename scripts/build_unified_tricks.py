"""Build a unified trick database from 4 sources.

Merges: parkourtheory (1837), trickipedia (527), loopkicks (943), tricking_bible (45)
into a single deduplicated database with physics parameters, difficulty, and relationships.

Usage:
    python scripts/build_unified_tricks.py
    python scripts/build_unified_tricks.py --output data/unified_tricks.json
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


# ── Name normalization ──────────────────────────────────────────────

def normalize_name(name: str) -> str:
    """Normalize trick name for matching: lowercase, strip punctuation, collapse spaces."""
    n = name.lower().strip()
    n = n.replace("-", " ").replace("_", " ")
    n = re.sub(r"[^a-z0-9 ]", "", n)
    n = re.sub(r"\s+", " ", n).strip()
    return n


def generate_name_variants(name: str) -> set[str]:
    """Generate possible name variants for fuzzy matching.

    Handles:
    - "backflip" ↔ "back flip"
    - "sideflip" ↔ "side flip"
    - "frontflip" ↔ "front flip"
    - "kickflip" ↔ "kick flip"
    - "b-twist" ↔ "btwist" ↔ "b twist"
    """
    norm = normalize_name(name)
    variants = {norm}

    # Split compound words
    COMPOUNDS = {
        "backflip": "back flip", "frontflip": "front flip",
        "sideflip": "side flip", "kickflip": "kick flip",
        "handspring": "hand spring", "handstand": "hand stand",
        "roundoff": "round off", "cartwheel": "cart wheel",
        "corkscrew": "cork screw", "butterfly": "butter fly",
        "snapuswipe": "snapu swipe", "backswing": "back swing",
        "frontswing": "front swing", "sideswing": "side swing",
        "fulltwist": "full twist", "doubleleg": "double leg",
        "masterscoot": "master scoot", "moonkick": "moon kick",
        "flashkick": "flash kick", "jackknife": "jack knife",
        "touchdown": "touch down", "swingthrough": "swing through",
        "swingthru": "swing thru",
    }

    tokens = norm.split()
    # Try merging adjacent tokens
    for i in range(len(tokens) - 1):
        merged = tokens[i] + tokens[i + 1]
        rest_before = tokens[:i]
        rest_after = tokens[i + 2:]
        variant = " ".join(rest_before + [merged] + rest_after)
        variants.add(variant)

    # Try splitting known compound words
    new_tokens = []
    for tok in tokens:
        if tok in COMPOUNDS:
            new_tokens.extend(COMPOUNDS[tok].split())
        else:
            new_tokens.append(tok)
    variants.add(" ".join(new_tokens))

    # Try re-merging split versions
    for compound, split in COMPOUNDS.items():
        if split in norm:
            variants.add(norm.replace(split, compound))
        if compound in norm:
            variants.add(norm.replace(compound, split))

    return variants


# ── Physics parsing from descriptions ───────────────────────────────

ROTATION_WORDS = {
    "single": 1.0, "double": 2.0, "triple": 3.0, "quad": 4.0, "quadruple": 4.0,
    "half": 0.5,
    "180": 0.5, "360": 1.0, "540": 1.5, "720": 2.0, "900": 2.5,
    "1080": 3.0, "1260": 3.5, "1440": 4.0,
}

TWIST_WORDS = {
    "full twist": 1.0, "double twist": 2.0, "triple twist": 3.0,
    "half twist": 0.5, "full": 1.0,
}

DIRECTION_PATTERNS = {
    r"\bbackward\b": "backward", r"\bback\s?flip": "backward",
    r"\bforward\b": "forward", r"\bfront\s?flip": "forward",
    r"\bside\s?flip": "left", r"\blateral\b": "left",
}

BODY_SHAPE_PATTERNS = {
    r"\btuck(?:ed)?\b": "tuck", r"\bpike[d]?\b": "pike",
    r"\blayout\b": "layout", r"\bstraight\b": "layout",
    r"\bopen\b": "open", r"\bstraddle[d]?\b": "open",
}


def extract_physics_from_description(desc: str) -> dict:
    """Extract physics hints from a trick description."""
    if not desc:
        return {}

    desc_lower = desc.lower()
    hints = {}

    # Direction
    for pattern, direction in DIRECTION_PATTERNS.items():
        if re.search(pattern, desc_lower):
            hints["direction"] = direction
            break

    # Body shape
    for pattern, shape in BODY_SHAPE_PATTERNS.items():
        if re.search(pattern, desc_lower):
            hints["body_shape"] = shape
            break

    # Rotation count from description
    for word, count in ROTATION_WORDS.items():
        pattern = rf"\b{re.escape(word)}\b"
        if re.search(pattern, desc_lower):
            if word in ("double", "triple", "quad", "quadruple"):
                if "twist" not in desc_lower[max(0, desc_lower.index(word) - 5):desc_lower.index(word) + len(word) + 10]:
                    hints.setdefault("rotation_count", count)
            elif word.isdigit():
                hints.setdefault("rotation_count", count)

    # Twist count from description
    for phrase, count in sorted(TWIST_WORDS.items(), key=lambda x: -len(x[0])):
        if phrase in desc_lower:
            hints["twist_count"] = count
            break

    # Axis hints
    if re.search(r"\bcork(?:screw)?\b", desc_lower):
        hints["rotation_axis"] = "off_axis"
    elif re.search(r"\btwist(?:ing)?\b", desc_lower) and "flip" not in desc_lower:
        hints["rotation_axis"] = "longitudinal"
    elif re.search(r"\bflip(?:ping)?\b", desc_lower):
        hints["rotation_axis"] = "lateral"

    return hints


# ── Parkourtheory type field parsing ────────────────────────────────

def parse_parkourtheory_type(type_str: str) -> dict:
    """Parse the parkourtheory 'type' field into physics hints.

    Examples: "Flip", "Flip/Twist", "Wall/Flip", "Bar/Flip/Twist", "Roll/Twist"
    """
    if not type_str:
        return {}

    parts = [p.strip().lower() for p in type_str.split("/")]
    hints = {}

    if "flip" in parts:
        hints["rotation_axis"] = "lateral"
        hints.setdefault("rotation_count", 1.0)
    if "twist" in parts:
        hints.setdefault("twist_count", 1.0)
        if "flip" not in parts:
            hints["rotation_axis"] = "longitudinal"
    if "roll" in parts:
        hints["family"] = "roll"
    if "wall" in parts:
        hints["entry"] = "wall"
    if "bar" in parts:
        hints["entry"] = "edge"
        hints["family"] = "bar"
    if "vault" in parts:
        hints["family"] = "vault"
        hints["entry"] = "running"
    if "ground" in parts:
        hints["family"] = "ground"

    return hints


# ── Loopkicks category parsing ──────────────────────────────────────

LOOPKICKS_CATEGORY_MAP = {
    "backward-tricks": {"direction": "backward"},
    "forward-tricks": {"direction": "forward"},
    "vertical-kicks": {"rotation_axis": "longitudinal", "family": "kick"},
    "spin-kicks": {"rotation_axis": "longitudinal", "family": "kick"},
    "multiple-rotations": {},  # Don't guess rotation_count — let name parsing handle it
    "twists": {"rotation_axis": "longitudinal"},
    "ground-moves": {"family": "ground"},
    "transitions": {"family": "transition"},
}


def parse_loopkicks_category(category: str) -> dict:
    """Parse loopkicks category into physics hints."""
    return LOOPKICKS_CATEGORY_MAP.get(category, {})


# ── Tricking Bible parsing ──────────────────────────────────────────

TB_TYPE_MAP = {
    "kick": {"family": "kick", "rotation_axis": "longitudinal"},
    "flip": {"family": "flip", "rotation_axis": "lateral"},
    "twist": {"rotation_axis": "longitudinal"},
    "invert": {"family": "ground"},
    "invert/kick": {"family": "kick"},
    "invert/flip": {"family": "flip", "rotation_axis": "lateral"},
}

TB_DIFFICULTY_MAP = {"A": 2, "B": 4, "C": 7, "D": 9}


def parse_tricking_bible_type(type_str: str) -> dict:
    """Parse tricking bible type into physics hints."""
    return TB_TYPE_MAP.get(type_str.lower(), {})


# ── Difficulty unification ──────────────────────────────────────────

def unify_difficulty(trickipedia_level: int | None, tb_class: str | None) -> float | None:
    """Unify difficulty from trickipedia (1-10) and tricking_bible (A-D) to 1-10 scale."""
    scores = []
    if trickipedia_level is not None:
        scores.append(trickipedia_level)
    if tb_class and tb_class in TB_DIFFICULTY_MAP:
        scores.append(TB_DIFFICULTY_MAP[tb_class])
    if scores:
        return round(sum(scores) / len(scores), 1)
    return None


# ── Main merge logic ────────────────────────────────────────────────

def load_sources(data_dir: Path) -> tuple[list, list, list, list]:
    """Load all 4 data sources."""
    pt = json.loads((data_dir / "parkourtheory_detailed.json").read_text())
    tp = json.loads((data_dir / "trickipedia_tricks.json").read_text())
    os_data = json.loads((data_dir / "other_sources_tricks.json").read_text())
    lk = os_data["loopkicks"]
    tb = os_data["tricking_bible"]
    return pt, tp, lk, tb


def build_name_index(
    pt: list, tp: list, lk: list, tb: list,
) -> dict[str, dict]:
    """Build a normalized name → source data index for deduplication.

    Returns dict mapping canonical_name → {
        "canonical_name": str,
        "display_name": str,
        "sources": {"parkourtheory": {...}, "trickipedia": {...}, ...}
    }
    """
    # variant → canonical name mapping
    variant_to_canonical: dict[str, str] = {}
    # canonical name → merged entry
    entries: dict[str, dict] = {}

    def get_or_create(name: str, source_key: str) -> str:
        """Find existing canonical name or create new entry. Returns canonical name."""
        norm = normalize_name(name)
        variants = generate_name_variants(name)

        # Check if any variant matches an existing canonical
        for v in variants:
            if v in variant_to_canonical:
                return variant_to_canonical[v]

        # New entry — register all variants
        canonical = norm
        for v in variants:
            variant_to_canonical[v] = canonical

        entries[canonical] = {
            "canonical_name": canonical,
            "display_name": name.strip(),
            "sources": {},
        }
        return canonical

    # Process parkourtheory first (canonical source, most entries)
    for trick in pt:
        name = trick["name"]
        canonical = get_or_create(name, "parkourtheory")
        entries[canonical]["sources"]["parkourtheory"] = trick
        entries[canonical]["display_name"] = name  # PT names are best formatted

        # Also register alias
        if trick.get("alias"):
            alias_norm = normalize_name(trick["alias"])
            alias_variants = generate_name_variants(trick["alias"])
            for v in alias_variants:
                variant_to_canonical.setdefault(v, canonical)

    # Process trickipedia
    for trick in tp:
        name = trick["name"]
        canonical = get_or_create(name, "trickipedia")
        entries[canonical]["sources"]["trickipedia"] = trick
        # Use trickipedia name if no PT name
        if "parkourtheory" not in entries[canonical]["sources"]:
            entries[canonical]["display_name"] = name

    # Process loopkicks (ALL CAPS → title case)
    for trick in lk:
        name = trick["name"].title()  # "BACK PIKE" → "Back Pike"
        canonical = get_or_create(name, "loopkicks")
        entries[canonical]["sources"]["loopkicks"] = trick
        if not any(s in entries[canonical]["sources"] for s in ("parkourtheory", "trickipedia")):
            entries[canonical]["display_name"] = name

    # Process tricking bible
    for trick in tb:
        name = trick["name"]
        canonical = get_or_create(name, "tricking_bible")
        entries[canonical]["sources"]["tricking_bible"] = trick

    return entries


def build_unified_entry(canonical: str, entry: dict, parse_trick_name) -> dict:
    """Build a unified trick entry from merged source data."""
    sources = entry["sources"]
    display_name = entry["display_name"]

    # Start with name-based physics parsing
    physics = parse_trick_name(display_name)

    # Collect enrichments from each source (lower priority → higher priority)
    enrichments = []

    # Loopkicks category hints (lowest priority)
    if "loopkicks" in sources:
        lk = sources["loopkicks"]
        enrichments.append(parse_loopkicks_category(lk.get("category", "")))
        enrichments.append(extract_physics_from_description(lk.get("description", "")))

    # Tricking Bible type hints
    if "tricking_bible" in sources:
        tb = sources["tricking_bible"]
        enrichments.append(parse_tricking_bible_type(tb.get("type", "")))

    # Trickipedia description hints
    if "trickipedia" in sources:
        tp = sources["trickipedia"]
        enrichments.append(extract_physics_from_description(tp.get("description", "")))

    # Parkourtheory type + description hints (highest priority)
    if "parkourtheory" in sources:
        pt = sources["parkourtheory"]
        enrichments.append(parse_parkourtheory_type(pt.get("type", "")))
        enrichments.append(extract_physics_from_description(pt.get("description", "")))

    # Apply enrichments only for fields NOT explicitly set by name parsing
    explicit_fields = physics.pop("_explicit", set())
    for hints in enrichments:
        for key, val in hints.items():
            if key in physics and key not in explicit_fields:
                physics[key] = val

    # Build unified entry
    result = {
        "name": display_name,
        "canonical_name": canonical,
        "physics": {
            "rotation_axis": physics.get("rotation_axis", "lateral"),
            "direction": physics.get("direction", "backward"),
            "rotation_count": physics.get("rotation_count", 1.0),
            "twist_count": physics.get("twist_count", 0.0),
            "body_shape": physics.get("body_shape", "tuck"),
            "entry": physics.get("entry", "standing"),
            "family": physics.get("family", "flip"),
        },
        "difficulty": None,
        "sources": list(sources.keys()),
        "descriptions": {},
        "related_tricks": {
            "prerequisites": [],
            "subsequents": [],
            "related": [],
        },
        "metadata": {},
    }

    # Collect descriptions
    if "parkourtheory" in sources:
        pt = sources["parkourtheory"]
        result["descriptions"]["parkourtheory"] = pt.get("description", "")
        result["metadata"]["parkourtheory_type"] = pt.get("type", "")
        result["metadata"]["parkourtheory_url"] = pt.get("url", "")
        result["metadata"]["video_url"] = pt.get("video_url", "")
        result["metadata"]["alias"] = pt.get("alias", "")

    if "trickipedia" in sources:
        tp = sources["trickipedia"]
        result["descriptions"]["trickipedia"] = tp.get("description", "")
        result["metadata"]["trickipedia_id"] = tp.get("id", "")
        result["metadata"]["trickipedia_slug"] = tp.get("slug", "")
        result["metadata"]["subcategory"] = tp.get("subcategory", {}).get("name", "")
        result["metadata"]["master_category"] = (
            tp.get("subcategory", {}).get("master_category", {}).get("name", "")
        )
        result["metadata"]["step_by_step_guide"] = tp.get("step_by_step_guide", "")

    if "loopkicks" in sources:
        lk = sources["loopkicks"]
        result["descriptions"]["loopkicks"] = lk.get("description", "")
        result["metadata"]["loopkicks_url"] = lk.get("url", "")
        result["metadata"]["loopkicks_category"] = lk.get("category", "")

    if "tricking_bible" in sources:
        tb = sources["tricking_bible"]
        result["metadata"]["tricking_bible_type"] = tb.get("type", "")
        result["metadata"]["tricking_bible_origin"] = tb.get("origin", "")
        result["metadata"]["tricking_bible_difficulty"] = tb.get("difficulty_class", "")

    # Unify difficulty
    tp_level = None
    tb_class = None
    if "trickipedia" in sources:
        tp_level = sources["trickipedia"].get("difficulty_level")
    if "tricking_bible" in sources:
        tb_class = sources["tricking_bible"].get("difficulty_class")
    result["difficulty"] = unify_difficulty(tp_level, tb_class)

    # Collect relationships
    if "parkourtheory" in sources:
        pt = sources["parkourtheory"]
        result["related_tricks"]["prerequisites"] = [
            normalize_name(p) for p in pt.get("prerequisites", []) if p
        ]
        result["related_tricks"]["subsequents"] = [
            normalize_name(s) for s in pt.get("subsequents", []) if s
        ]
        result["related_tricks"]["related"] = [
            normalize_name(r) for r in pt.get("explore_related", []) if r
        ]

    if "trickipedia" in sources:
        tp = sources["trickipedia"]
        # Trickipedia uses UUIDs for prerequisites — resolve to names
        prereq_ids = tp.get("prerequisite_ids", [])
        if prereq_ids:
            result["metadata"]["trickipedia_prerequisite_ids"] = prereq_ids

    # Clean up empty metadata
    result["metadata"] = {k: v for k, v in result["metadata"].items() if v}

    return result


def physics_defaults() -> dict:
    """Default physics values (used to detect if a field was set by name parsing)."""
    return {
        "rotation_axis": "lateral",
        "direction": "backward",
        "rotation_count": 1.0,
        "twist_count": 0.0,
        "body_shape": "tuck",
        "entry": "standing",
        "family": "flip",
    }


def resolve_trickipedia_prerequisites(
    unified: list[dict], tp_tricks: list[dict],
) -> None:
    """Resolve trickipedia UUID prerequisites to canonical names."""
    # Build ID → name mapping
    id_to_name = {}
    for trick in tp_tricks:
        tid = trick.get("id", "")
        if tid:
            id_to_name[tid] = normalize_name(trick["name"])

    # Resolve for each unified trick that has trickipedia prereq IDs
    for entry in unified:
        prereq_ids = entry.get("metadata", {}).get("trickipedia_prerequisite_ids", [])
        for pid in prereq_ids:
            resolved = id_to_name.get(pid)
            if resolved and resolved not in entry["related_tricks"]["prerequisites"]:
                entry["related_tricks"]["prerequisites"].append(resolved)
        # Clean up the temp field
        entry.get("metadata", {}).pop("trickipedia_prerequisite_ids", None)


def estimate_difficulty_from_physics(unified: list[dict]) -> None:
    """Estimate difficulty for tricks without explicit difficulty scores.

    Uses physics complexity as a proxy:
    - More rotations = harder
    - More twists = harder
    - Off-axis > lateral > longitudinal
    - Non-standing entries = harder
    """
    for entry in unified:
        if entry["difficulty"] is not None:
            continue

        p = entry["physics"]
        score = 1.0

        # Rotation count (biggest factor)
        score += (p["rotation_count"] - 0.5) * 2.0

        # Twist count
        score += p["twist_count"] * 1.5

        # Axis complexity
        axis_bonus = {"lateral": 0, "sagittal": 0.5, "longitudinal": 0.5, "off_axis": 1.5}
        score += axis_bonus.get(p["rotation_axis"], 0)

        # Entry difficulty
        entry_bonus = {"standing": 0, "running": 0.3, "one_leg": 0.8, "wall": 0.5, "edge": 0.3}
        score += entry_bonus.get(p["entry"], 0)

        # Body shape (layout harder than tuck)
        shape_bonus = {"tuck": 0, "pike": 0.3, "layout": 0.5, "open": 0.2}
        score += shape_bonus.get(p["body_shape"], 0)

        # Non-flip families tend to be simpler
        family_mod = {"roll": -1.0, "vault": -0.5, "wall": 0, "bar": 0.5, "ground": -0.5, "kick": -0.3}
        score += family_mod.get(p["family"], 0)

        entry["difficulty"] = round(max(1.0, min(10.0, score)), 1)


def propagate_difficulty(unified: list[dict]) -> None:
    """Propagate difficulty constraints through the prerequisite graph.

    Rule: a trick must be harder than ALL its prerequisites.
    We iteratively raise difficulty until the graph is consistent.
    """
    by_canonical = {e["canonical_name"]: e for e in unified}

    # Forward pass: ensure trick >= max(prerequisites) + 0.5
    changed = True
    iterations = 0
    while changed and iterations < 20:
        changed = False
        iterations += 1
        for entry in unified:
            if entry["difficulty"] is None:
                continue
            prereqs = entry["related_tricks"]["prerequisites"]
            if not prereqs:
                continue

            max_prereq_diff = 0.0
            for pname in prereqs:
                prereq = by_canonical.get(pname)
                if prereq and prereq["difficulty"] is not None:
                    max_prereq_diff = max(max_prereq_diff, prereq["difficulty"])

            if max_prereq_diff > 0 and entry["difficulty"] <= max_prereq_diff:
                entry["difficulty"] = round(max_prereq_diff + 0.5, 1)
                entry["difficulty"] = min(entry["difficulty"], 10.0)
                changed = True

    # Backward pass: lower prerequisites that are harder than their subsequents
    changed = True
    iterations = 0
    while changed and iterations < 20:
        changed = False
        iterations += 1
        for entry in unified:
            if entry["difficulty"] is None:
                continue
            subsequents = entry["related_tricks"]["subsequents"]
            if not subsequents:
                continue

            min_subseq_diff = 10.0
            for sname in subsequents:
                subseq = by_canonical.get(sname)
                if subseq and subseq["difficulty"] is not None:
                    min_subseq_diff = min(min_subseq_diff, subseq["difficulty"])

            if min_subseq_diff < 10.0 and entry["difficulty"] >= min_subseq_diff:
                entry["difficulty"] = round(max(1.0, min_subseq_diff - 0.5), 1)
                changed = True


def print_stats(unified: list[dict]) -> None:
    """Print merge statistics."""
    print(f"\nTotal unified tricks: {len(unified)}")

    # Source coverage
    source_counts = Counter()
    multi_source = 0
    for entry in unified:
        for s in entry["sources"]:
            source_counts[s] += 1
        if len(entry["sources"]) > 1:
            multi_source += 1

    print(f"\nSource coverage:")
    for source, count in source_counts.most_common():
        print(f"  {source:20s}: {count}")
    print(f"  Multi-source:       {multi_source}")

    # Physics distribution
    physics_combos = set()
    for entry in unified:
        p = entry["physics"]
        combo = (p["rotation_axis"], p["direction"], p["rotation_count"],
                 p["twist_count"], p["body_shape"], p["entry"])
        physics_combos.add(combo)

    print(f"\nUnique physics combinations: {len(physics_combos)}")

    # Family distribution
    families = Counter(e["physics"]["family"] for e in unified)
    print(f"\nFamilies:")
    for fam, count in families.most_common():
        print(f"  {fam:15s}: {count}")

    # Difficulty distribution
    with_difficulty = [e for e in unified if e["difficulty"] is not None]
    explicit = [e for e in unified if e["difficulty"] is not None and any(
        s in e["sources"] for s in ("trickipedia", "tricking_bible"))]
    print(f"\nDifficulty: {len(with_difficulty)} total ({len(explicit)} from sources, "
          f"{len(with_difficulty) - len(explicit)} estimated)")

    # Relationship graph
    has_prereqs = sum(1 for e in unified if e["related_tricks"]["prerequisites"])
    has_subsequents = sum(1 for e in unified if e["related_tricks"]["subsequents"])
    print(f"\nRelationships: {has_prereqs} with prerequisites, {has_subsequents} with subsequents")


def main():
    parser = argparse.ArgumentParser(description="Build unified trick database")
    parser.add_argument("--data-dir", default="data", help="Data directory")
    parser.add_argument("--output", default="data/unified_tricks.json")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    # Import the existing name parser
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from parse_trick_names import parse_trick_name

    print("Loading sources...")
    pt, tp, lk, tb = load_sources(data_dir)
    print(f"  Parkourtheory: {len(pt)}")
    print(f"  Trickipedia:   {len(tp)}")
    print(f"  Loopkicks:     {len(lk)}")
    print(f"  Tricking Bible: {len(tb)}")

    print("\nBuilding name index (fuzzy matching)...")
    entries = build_name_index(pt, tp, lk, tb)
    print(f"  Unique tricks after dedup: {len(entries)}")

    print("\nBuilding unified entries...")
    unified = []
    for canonical, entry in entries.items():
        unified_entry = build_unified_entry(canonical, entry, parse_trick_name)
        unified.append(unified_entry)

    print("Resolving trickipedia prerequisites...")
    resolve_trickipedia_prerequisites(unified, tp)

    print("Estimating difficulty for tricks without scores...")
    estimate_difficulty_from_physics(unified)

    print("Propagating difficulty constraints through prerequisite graph...")
    propagate_difficulty(unified)

    # Sort by difficulty then name
    unified.sort(key=lambda e: (e["difficulty"] or 0, e["canonical_name"]))

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(unified, f, indent=2, ensure_ascii=False)

    print_stats(unified)
    print(f"\nSaved to {output_path}")

    # Show multi-source examples
    multi = [e for e in unified if len(e["sources"]) > 1]
    print(f"\n--- Multi-source examples (showing 15) ---")
    for entry in multi[:15]:
        p = entry["physics"]
        print(f"  {entry['name']:40s} sources={entry['sources']}  "
              f"axis={p['rotation_axis']:12s} rot={p['rotation_count']:.1f} "
              f"twist={p['twist_count']:.1f} diff={entry['difficulty']}")


if __name__ == "__main__":
    main()
