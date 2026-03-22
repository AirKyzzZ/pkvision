#!/usr/bin/env python3
"""Interactive CLI to label video clips with trick annotations.

Usage:
    python scripts/label.py --clips-dir data/clips/ --output data/clips/labels.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_trick_ids() -> list[str]:
    """Load available trick IDs from the catalog."""
    catalog_dir = Path(__file__).parent.parent / "data" / "tricks" / "catalog" / "en"
    trick_ids = []
    for path in sorted(catalog_dir.glob("*.json")):
        with open(path) as f:
            data = json.load(f)
        trick_ids.append(data["trick_id"])
    return trick_ids


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Label video clips with trick annotations")
    parser.add_argument("--clips-dir", default="data/clips", help="Directory containing video clips")
    parser.add_argument("--output", default="data/clips/labels.json", help="Output labels file")
    args = parser.parse_args()

    clips_dir = Path(args.clips_dir)
    output_path = Path(args.output)

    # Load existing labels
    existing_labels = []
    if output_path.exists():
        with open(output_path) as f:
            existing_labels = json.load(f)

    labeled_files = {entry["file"] for entry in existing_labels}

    # Find video files
    video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    video_files = sorted(
        p for p in clips_dir.iterdir()
        if p.suffix.lower() in video_extensions and p.name not in labeled_files
    )

    if not video_files:
        print("No unlabeled video files found in", clips_dir)
        if existing_labels:
            print(f"({len(existing_labels)} clips already labeled)")
        sys.exit(0)

    # Load trick IDs
    trick_ids = load_trick_ids()
    print("Available tricks:")
    for i, tid in enumerate(trick_ids, 1):
        print(f"  {i:2d}. {tid}")
    print()

    new_labels = []

    for video_file in video_files:
        print(f"\n--- {video_file.name} ---")

        # Get trick
        while True:
            trick_input = input(f"Trick ID or number (1-{len(trick_ids)}), 's' to skip, 'q' to quit: ").strip()

            if trick_input.lower() == "q":
                break
            if trick_input.lower() == "s":
                trick_input = None
                break

            if trick_input.isdigit():
                idx = int(trick_input) - 1
                if 0 <= idx < len(trick_ids):
                    trick_input = trick_ids[idx]
                    break
                print("  Invalid number.")
            elif trick_input in trick_ids:
                break
            else:
                # Allow custom trick IDs for new tricks
                confirm = input(f"  '{trick_input}' not in catalog. Use anyway? (y/n): ").strip()
                if confirm.lower() == "y":
                    break

        if trick_input is None:
            continue
        if trick_input.lower() == "q":
            break

        trick_id = trick_input

        # Get time range
        start_ms = input("Start time in ms (0 for beginning): ").strip()
        start_ms = int(start_ms) if start_ms else 0

        end_ms = input("End time in ms (empty for end of video): ").strip()
        end_ms = int(end_ms) if end_ms else None

        entry = {
            "file": video_file.name,
            "trick_id": trick_id,
            "start_ms": start_ms,
        }
        if end_ms is not None:
            entry["end_ms"] = end_ms

        new_labels.append(entry)
        print(f"  Labeled: {video_file.name} → {trick_id}")

    # Save
    all_labels = existing_labels + new_labels
    with open(output_path, "w") as f:
        json.dump(all_labels, f, indent=2)

    print(f"\nSaved {len(new_labels)} new labels ({len(all_labels)} total) to {output_path}")


if __name__ == "__main__":
    main()
