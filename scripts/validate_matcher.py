"""Validate the zero-shot matcher against the expanded trick database.

Tests:
1. Correct differentiation: backflip vs frontflip vs sideflip
2. Rotation counting: single vs double vs triple
3. Body shape: tuck vs pike vs layout
4. Entry type: standing vs wall vs gainer
5. Difficulty ordering: prerequisites should be easier than subsequents
6. Physics combination coverage

Usage:
    python scripts/validate_matcher.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from core.recognition.matcher import AXIS_VECTORS, Matcher3D
from ml.trick_physics import (
    TRICK_DEFINITIONS,
    BodyShape,
    Direction,
    EntryType,
    RotationAxis,
    TrickDefinition,
)
from core.models import TrickSignature3D


def make_signature(trick_def: TrickDefinition) -> TrickSignature3D:
    """Create a TrickSignature3D from a TrickDefinition for testing."""
    axis_vec = AXIS_VECTORS[trick_def.rotation_axis]
    return TrickSignature3D(
        total_flip_deg=trick_def.rotation_count * 360,
        total_twist_deg=trick_def.twist_count * 360,
        rotation_direction=trick_def.direction.value,
        body_shape=trick_def.body_shape.value,
        entry_type=trick_def.entry.value,
        peak_height_m=trick_def.typical_height_m,
        duration_s=trick_def.typical_duration_s,
        primary_rotation_axis=axis_vec,
    )


def test_differentiation(matcher: Matcher3D) -> int:
    """Test that the matcher can differentiate basic trick types."""
    print("\n=== Test 1: Basic Differentiation ===")
    passed = 0
    total = 0

    test_cases = [
        ("back_flip", "back_flip", "Backflip self-match"),
        ("front_flip", "front_flip", "Frontflip self-match"),
        ("side_flip", "side_flip", "Sideflip self-match"),
        ("cork", "cork", "Cork self-match"),
        ("gainer", "gainer", "Gainer self-match"),
        ("webster", "webster", "Webster self-match"),
        ("raiz", "raiz", "Raiz self-match"),
        ("double_full", "double_full", "Double full self-match"),
    ]

    for query_id, expected_id, desc in test_cases:
        if query_id not in TRICK_DEFINITIONS:
            print(f"  SKIP  {desc}: {query_id} not in definitions")
            continue

        sig = make_signature(TRICK_DEFINITIONS[query_id])

        matches = matcher.match(sig, top_k=5)
        top_id = matches[0].trick_id if matches else "NONE"
        total += 1
        if top_id == expected_id:
            print(f"  PASS  {desc}: top match = {top_id}")
            passed += 1
        else:
            top_names = [(m.trick_id, f"{m.confidence:.2f}") for m in matches[:3]]
            print(f"  FAIL  {desc}: expected {expected_id}, got {top_names}")

    print(f"\n  Result: {passed}/{total} passed")
    return total - passed


def test_rotation_ordering(matcher: Matcher3D) -> int:
    """Test that higher rotation counts produce harder matches."""
    print("\n=== Test 2: Rotation Ordering ===")
    passed = 0
    total = 0
    failures = 0

    # Groups of tricks ordered by rotation count
    groups = [
        [("back_flip", 1.0), ("double_back", 2.0)],
        [("back_full", 1.0), ("double_full", 2.0), ("triple_full", 3.0)],
        [("cork", 1.0), ("double_cork", 2.0), ("triple_cork", 3.0)],
        [("front_flip", 1.0), ("double_front_flip", 2.0)],
    ]

    for group in groups:
        valid_group = [(tid, rot) for tid, rot in group if tid in TRICK_DEFINITIONS]
        if len(valid_group) < 2:
            continue

        total += 1
        names = [f"{tid}({rot}x)" for tid, rot in valid_group]
        # Each trick should match itself, not the adjacent rotation
        all_correct = True
        for tid, rot in valid_group:
            sig = make_signature(TRICK_DEFINITIONS[tid])
            matches = matcher.match(sig, top_k=3)
            if matches and matches[0].trick_id != tid:
                all_correct = False
                print(f"  FAIL  {tid} matched as {matches[0].trick_id} (conf={matches[0].confidence:.2f})")
                failures += 1

        if all_correct:
            print(f"  PASS  Rotation group: {' < '.join(names)}")
            passed += 1

    print(f"\n  Result: {passed}/{total} groups correct")
    return failures


def test_body_shape(matcher: Matcher3D) -> int:
    """Test body shape differentiation."""
    print("\n=== Test 3: Body Shape Differentiation ===")

    shapes = ["tuck", "pike", "layout"]
    failures = 0

    for shape in shapes:
        sig = TrickSignature3D(
            total_flip_deg=360,
            total_twist_deg=0,
            rotation_direction="backward",
            body_shape=shape,
            entry_type="standing",
            peak_height_m=1.0,
            duration_s=0.8,
            primary_rotation_axis=AXIS_VECTORS[RotationAxis.LATERAL],
        )
        matches = matcher.match(sig, top_k=5)
        top = matches[0] if matches else None
        if top:
            print(f"  {shape:8s} -> {top.trick_id:25s} (conf={top.confidence:.2f}, shape_dist={top.shape_distance:.3f})")
            # Check that matching trick has correct shape or is close
            matched_def = TRICK_DEFINITIONS.get(top.trick_id)
            if matched_def and matched_def.body_shape.value != shape:
                if top.shape_distance > 0.3:
                    failures += 1
                    print(f"           WARNING: shape mismatch ({shape} vs {matched_def.body_shape.value})")

    return failures


def test_entry_types(matcher: Matcher3D) -> int:
    """Test entry type differentiation."""
    print("\n=== Test 4: Entry Type Differentiation ===")

    from core.models import TrickSignature3D

    entries = [
        ("standing", "back_flip"),
        ("running", "gainer"),
        ("one_leg", "webster"),
        ("wall", "palm_flip"),
    ]
    failures = 0

    for entry, expected_id in entries:
        if expected_id not in TRICK_DEFINITIONS:
            print(f"  SKIP  {entry}: {expected_id} not in definitions")
            continue

        sig = make_signature(TRICK_DEFINITIONS[expected_id])
        matches = matcher.match(sig, top_k=3)
        top = matches[0] if matches else None
        if top:
            matched_entry = TRICK_DEFINITIONS.get(top.trick_id, None)
            entry_val = matched_entry.entry.value if matched_entry else "?"
            status = "PASS" if top.trick_id == expected_id else "WARN"
            if top.trick_id != expected_id:
                failures += 1
            print(f"  {status}  {entry:10s} -> {top.trick_id:25s} (entry={entry_val}, conf={top.confidence:.2f})")

    return failures


def test_difficulty_ordering(unified_path: str = "data/unified_tricks.json") -> int:
    """Test that prerequisite tricks have lower difficulty than their subsequents."""
    print("\n=== Test 5: Difficulty Ordering (Prerequisites < Subsequents) ===")

    unified = json.loads(Path(unified_path).read_text())
    by_canonical = {e["canonical_name"]: e for e in unified}

    violations = 0
    checked = 0
    for entry in unified:
        if not entry["related_tricks"]["prerequisites"]:
            continue
        if entry["difficulty"] is None:
            continue

        for prereq_name in entry["related_tricks"]["prerequisites"]:
            prereq = by_canonical.get(prereq_name)
            if not prereq or prereq["difficulty"] is None:
                continue

            checked += 1
            if prereq["difficulty"] > entry["difficulty"]:
                violations += 1
                if violations <= 10:
                    print(f"  VIOLATION: {prereq['name']} (diff={prereq['difficulty']}) "
                          f"should be easier than {entry['name']} (diff={entry['difficulty']})")

    violation_rate = violations / checked * 100 if checked else 0
    print(f"\n  Checked: {checked} prerequisite relationships")
    print(f"  Violations: {violations} ({violation_rate:.1f}%)")
    return violations


def test_coverage() -> None:
    """Report physics combination coverage."""
    print("\n=== Test 6: Physics Coverage ===")

    parsed = json.loads(Path("data/parsed_tricks.json").read_text())

    combos = set()
    for t in parsed:
        combo = (t["rotation_axis"], t["direction"], t["rotation_count"],
                 t["twist_count"], t["body_shape"], t["entry"])
        combos.add(combo)

    # Check how many TRICK_DEFINITIONS cover unique combos
    def_combos = set()
    for td in TRICK_DEFINITIONS.values():
        combo = (td.rotation_axis.value, td.direction.value, td.rotation_count,
                 td.twist_count, td.body_shape.value, td.entry.value)
        def_combos.add(combo)

    overlap = combos & def_combos
    only_parsed = combos - def_combos
    only_defs = def_combos - combos

    print(f"  parsed_tricks.json combos:    {len(combos)}")
    print(f"  TRICK_DEFINITIONS combos:     {len(def_combos)}")
    print(f"  Overlap:                      {len(overlap)}")
    print(f"  Only in parsed (not in defs): {len(only_parsed)}")
    print(f"  Only in defs (not in parsed): {len(only_defs)}")


def main():
    print("PkVision — Zero-Shot Matcher Validation")
    print("=" * 50)
    print(f"\nTRICK_DEFINITIONS: {len(TRICK_DEFINITIONS)} tricks")

    # Load parsed tricks for the matcher
    parsed_path = Path("data/parsed_tricks.json")
    parsed = json.loads(parsed_path.read_text())
    print(f"parsed_tricks.json: {len(parsed)} tricks")

    # Initialize matcher with expanded TRICK_DEFINITIONS
    matcher = Matcher3D()

    total_failures = 0
    total_failures += test_differentiation(matcher)
    total_failures += test_rotation_ordering(matcher)
    total_failures += test_body_shape(matcher)
    total_failures += test_entry_types(matcher)
    test_difficulty_ordering()
    test_coverage()

    print(f"\n{'=' * 50}")
    print(f"Total matcher failures: {total_failures}")
    if total_failures == 0:
        print("All matcher tests passed!")
    else:
        print(f"Some tests had issues — review above for details")


if __name__ == "__main__":
    main()
