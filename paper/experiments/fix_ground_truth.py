#!/usr/bin/env python3
"""Apply agreed-upon FIG canonicalizations to ground_truth.csv.

One-shot cleanup:
  - Canonicalize aliases (Krok -> Kroc, Back Flip -> Backflip, etc.)
  - Apply the user's answers for ambiguous entries ("Double Full" -> Backflip 720, ...)
  - Drop rows with "NOT A TRICK" in fig_name
  - Auto-fill d_score, flip_count, twist_count, direction from fig_tricks_2025.json
  - Leave multi-trick POOL-A fig_name fields alone (they're comma-separated lists)
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CSV_PATH = REPO / "paper" / "experiments" / "ground_truth.csv"
FIG_JSON = REPO / "data" / "fig_tricks_2025.json"

FIELDNAMES = [
    "clip_path", "pool", "fig_name", "d_score",
    "flip_count", "twist_count", "direction", "takeoff", "notes",
]

# POOL-B single-clip canonicalization table.
# Matched case-insensitively on whole fig_name.
SINGLE_CANONICAL = {
    "back flip":          "Backflip",
    "back double full":   "Backflip 720",     # answered: Backflip 720
    "double full":        "Backflip 720",     # answered: Backflip 720
    "gainer full":        "Gainer 360",       # answered: Gainer 360
    "krok":               "Kroc",             # alias canonicalization
    "wall inward sideflip": "Wall Inward Frontflip",  # alias canonicalization
    "not a trick":        "",                 # drop signal
}

# POOL-A multi-trick substitutions.
# Applied token-by-token to each comma-separated entry.
POOL_A_SUBSTITUTIONS = {
    "gainer 360":               "Gainer 360",
    "gainer full":              "Gainer 360",
    "krok":                     "Kroc",
    "double cork":              "Double Cork",
    "double backflip 720":      "Double Backflip 720",
    "castaway":                 "Castaway Backflip",          # answered
    "aerial":                   "Aerial",
    "palm flip":                "Palm Backflip",              # answered
    "cork":                     "Cork",
    "backflip 1440":            "Backflip 1440",
    "sideflip":                 "Sideflip",
    "swing triple backflip":    "Swing Triple Gainer",        # answered
    "kong gainer":              "Kong Gainer",
    "swing full full":          "Swing Gainer 720",           # answered
    "gainer full":              "Gainer 360",
    "backflip 720":             "Backflip 720",
    "b-twist":                  "B-Twist",
    "b twist":                  "B-Twist",
    "inward wall sideflip":     "Wall Inward Frontflip",
}


def load_fig_lookup() -> dict[str, dict]:
    data = json.loads(FIG_JSON.read_text())
    out: dict[str, dict] = {}
    for cat_key, cat in data["categories"].items():
        for t in cat["tricks"]:
            entry = {
                "name": t["name"],
                "d_score": float(t.get("score", 0) or 0),
                "flip": t.get("flip", 0),
                "twist": t.get("twist", 0),
                "direction": t.get("direction") or "",
                "takeoff": t.get("takeoff") or "",
                "category": cat_key,
            }
            out[t["name"].lower()] = entry
            for alias in t.get("aliases", []) or []:
                out[alias.lower()] = entry
    return out


FIG = load_fig_lookup()


def canonicalize_multi(fig_name_field: str) -> str:
    """Canonicalize a comma-separated list of trick names."""
    parts = [p.strip() for p in fig_name_field.split(",") if p.strip()]
    fixed = []
    for p in parts:
        key = p.lower().strip()
        # Direct substitution first
        if key in POOL_A_SUBSTITUTIONS:
            fixed.append(POOL_A_SUBSTITUTIONS[key])
            continue
        # Otherwise try the FIG alias index (handles case / alias)
        if key in FIG:
            fixed.append(FIG[key]["name"])
            continue
        # Keep as-is (will show up as a warning later)
        fixed.append(p)
    return ", ".join(fixed)


def autofill_single(row: dict) -> dict:
    """Auto-fill D-score, flip, twist, direction from FIG for single-trick POOL-B."""
    name = row["fig_name"].strip()
    if not name:
        return row
    entry = FIG.get(name.lower())
    if not entry:
        return row
    # Use canonical FIG name (fixes any lingering case issues).
    row["fig_name"] = entry["name"]
    row["d_score"] = f"{entry['d_score']}"
    if not row["flip_count"].strip():
        row["flip_count"] = str(entry["flip"])
    if not row["twist_count"].strip():
        row["twist_count"] = str(entry["twist"])
    if not row["direction"].strip():
        row["direction"] = entry["direction"] or ""
    if not row["takeoff"].strip():
        row["takeoff"] = entry["takeoff"] or ""
    return row


def main() -> None:
    with CSV_PATH.open() as f:
        rows = list(csv.DictReader(f))

    dropped = 0
    kept: list[dict] = []

    for r in rows:
        name = r["fig_name"].strip()

        if r["pool"] == "POOL-A":
            r["fig_name"] = canonicalize_multi(name)
            kept.append(r)
            continue

        # POOL-B: apply single canonicalization.
        key = name.lower()
        if key in SINGLE_CANONICAL:
            r["fig_name"] = SINGLE_CANONICAL[key]
            name = r["fig_name"]

        if not name:
            dropped += 1
            continue  # drop "NOT A TRICK" or blank rows

        # Try FIG lookup for autofill.
        r = autofill_single(r)
        kept.append(r)

    with CSV_PATH.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        w.writeheader()
        w.writerows(kept)

    print(f"Wrote {len(kept)} rows to {CSV_PATH.relative_to(REPO)} (dropped {dropped}).")
    print()
    print("POOL-B single-clip entries after canonicalization:")
    for r in kept:
        if r["pool"] != "POOL-B":
            continue
        matched = "✓" if r["fig_name"].lower() in FIG else "✗"
        print(f"  {matched} {Path(r['clip_path']).name:35s} -> {r['fig_name']:28s} D={r['d_score']}")

    print()
    print("POOL-A multi-trick entries after canonicalization:")
    for r in kept:
        if r["pool"] != "POOL-A":
            continue
        names = [n.strip() for n in r["fig_name"].split(",") if n.strip()]
        unmatched = [n for n in names if n.lower() not in FIG]
        status = "all matched" if not unmatched else f"UNMATCHED: {unmatched}"
        print(f"  {Path(r['clip_path']).name:35s} ({len(names)} tricks) - {status}")
        for n in names:
            mark = "✓" if n.lower() in FIG else "✗"
            d = FIG.get(n.lower(), {}).get("d_score", "?")
            print(f"      {mark} {n:30s} D={d}")


if __name__ == "__main__":
    main()
