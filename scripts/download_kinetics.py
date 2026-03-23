#!/usr/bin/env python3
"""Download specific Kinetics-400 classes relevant to parkour.

Downloads only the parkour-relevant classes from Kinetics-400:
- somersaulting (~400 videos)
- gymnastics tumbling (~400 videos)
- parkour (~400 videos)
- cartwheeling (~400 videos)
- capoeira (~400 videos)
- bouncing on trampoline (~400 videos)
- vault (~400 videos)

Total: ~2,800 free labeled videos.

Uses yt-dlp to download from YouTube (Kinetics is YouTube-sourced).

Usage:
    python scripts/download_kinetics.py --classes somersaulting parkour
    python scripts/download_kinetics.py --all --max-per-class 50
    python scripts/download_kinetics.py --all --max-per-class 100 --output data/kinetics/
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from urllib.request import urlretrieve

sys.path.insert(0, str(Path(__file__).parent.parent))

KINETICS_CLASSES = [
    "somersaulting",
    "gymnastics tumbling",
    "parkour",
    "cartwheeling",
    "capoeira",
    "bouncing on trampoline",
    "vault",
]

# Kinetics-400 CSV URLs (train + validate splits)
KINETICS_CSV_URLS = {
    "train": "https://storage.googleapis.com/deepmind-media/Datasets/kinetics400/train.csv",
    "validate": "https://storage.googleapis.com/deepmind-media/Datasets/kinetics400/validate.csv",
}


def download_csv(split: str, cache_dir: Path) -> Path:
    """Download Kinetics CSV if not cached."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    csv_path = cache_dir / f"kinetics400_{split}.csv"
    if csv_path.exists():
        print(f"  Using cached {csv_path}")
        return csv_path
    url = KINETICS_CSV_URLS[split]
    print(f"  Downloading {split} CSV...")
    urlretrieve(url, csv_path)
    return csv_path


def parse_csv(csv_path: Path, target_classes: set[str]) -> list[dict]:
    """Parse Kinetics CSV and filter to target classes."""
    entries = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = row.get("label", "").strip()
            if label in target_classes:
                entries.append({
                    "youtube_id": row.get("youtube_id", "").strip(),
                    "start": int(row.get("time_start", 0)),
                    "end": int(row.get("time_end", 0)),
                    "label": label,
                })
    return entries


def download_clip(youtube_id: str, start: int, end: int, output_path: Path) -> bool:
    """Download and trim a YouTube clip."""
    if output_path.exists():
        return True

    url = f"https://www.youtube.com/watch?v={youtube_id}"
    duration = end - start

    try:
        result = subprocess.run(
            [
                "yt-dlp",
                "-f", "bestvideo[height<=480][ext=mp4]+bestaudio[ext=m4a]/best[height<=480][ext=mp4]/best",
                "--merge-output-format", "mp4",
                "--download-sections", f"*{start}-{end}",
                "-o", str(output_path),
                "--no-playlist",
                "--quiet",
                "--no-warnings",
                url,
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        return result.returncode == 0 and output_path.exists()
    except (subprocess.TimeoutExpired, Exception):
        return False


def main():
    parser = argparse.ArgumentParser(description="Download Kinetics-400 parkour classes")
    parser.add_argument("--classes", nargs="+", default=None,
                        help="Specific classes to download (default: all parkour-relevant)")
    parser.add_argument("--all", action="store_true", help="Download all parkour-relevant classes")
    parser.add_argument("--max-per-class", type=int, default=50,
                        help="Max videos per class (default 50, use -1 for all)")
    parser.add_argument("--output", default="data/kinetics", help="Output directory")
    parser.add_argument("--split", default="train", choices=["train", "validate", "both"])
    args = parser.parse_args()

    target_classes = set(args.classes) if args.classes else set(KINETICS_CLASSES)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / ".cache"

    print()
    print("  PkVision — Kinetics-400 Downloader")
    print("  " + "=" * 40)
    print(f"  Classes: {', '.join(sorted(target_classes))}")
    print(f"  Max per class: {args.max_per_class if args.max_per_class > 0 else 'all'}")
    print(f"  Output: {output_dir}")
    print()

    # Download CSVs
    splits = ["train", "validate"] if args.split == "both" else [args.split]
    all_entries: list[dict] = []

    for split in splits:
        csv_path = download_csv(split, cache_dir)
        entries = parse_csv(csv_path, target_classes)
        all_entries.extend(entries)

    # Count per class
    class_counts: dict[str, list[dict]] = {}
    for e in all_entries:
        class_counts.setdefault(e["label"], []).append(e)

    print(f"  Found in Kinetics CSV:")
    total_to_download = 0
    for cls in sorted(class_counts):
        available = len(class_counts[cls])
        to_dl = min(available, args.max_per_class) if args.max_per_class > 0 else available
        total_to_download += to_dl
        print(f"    {cls:30s} {available:4d} available, downloading {to_dl}")

    print(f"\n  Total to download: {total_to_download} clips")
    print()

    # Download
    labels = []
    total_success = 0
    total_fail = 0

    for cls, entries in class_counts.items():
        cls_dir = output_dir / cls.replace(" ", "_")
        cls_dir.mkdir(exist_ok=True)

        limit = args.max_per_class if args.max_per_class > 0 else len(entries)
        batch = entries[:limit]

        for i, entry in enumerate(batch):
            yt_id = entry["youtube_id"]
            filename = f"{yt_id}_{entry['start']}_{entry['end']}.mp4"
            output_path = cls_dir / filename

            print(f"  [{total_success + total_fail + 1}/{total_to_download}] "
                  f"{cls}: {yt_id}...", end=" ", flush=True)

            ok = download_clip(yt_id, entry["start"], entry["end"], output_path)

            if ok:
                labels.append({
                    "file": f"{cls.replace(' ', '_')}/{filename}",
                    "trick_id": cls.replace(" ", "_"),
                    "source": "kinetics400",
                    "youtube_id": yt_id,
                })
                total_success += 1
                print("OK")
            else:
                total_fail += 1
                print("FAIL")

    # Save labels
    labels_path = output_dir / "kinetics_labels.json"
    with open(labels_path, "w") as f:
        json.dump(labels, f, indent=2)

    print(f"\n  {'='*40}")
    print(f"  Downloaded: {total_success} clips")
    print(f"  Failed:     {total_fail} (YouTube videos removed/private)")
    print(f"  Labels:     {labels_path}")
    print(f"\n  Next: merge with your data and retrain")


if __name__ == "__main__":
    main()
