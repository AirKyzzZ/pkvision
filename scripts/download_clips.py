#!/usr/bin/env python3
"""Download and trim parkour clips from YouTube for training.

Usage:
    python scripts/download_clips.py --url "https://youtube.com/watch?v=..." --trick back_flip --start 2.0 --end 4.5
    python scripts/download_clips.py --batch clips_to_download.json

The batch JSON format:
[
    {"url": "https://...", "trick_id": "back_flip", "start": 2.0, "end": 4.5},
    {"url": "https://...", "trick_id": "front_flip", "start": 1.0, "end": 3.0}
]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path


def download_and_trim(
    url: str,
    trick_id: str,
    start: float,
    end: float,
    output_dir: Path,
    clip_index: int = 0,
) -> Path | None:
    """Download a YouTube video and trim to the specified time range."""
    duration = end - start
    output_name = f"{trick_id}_{clip_index:03d}.mp4"
    output_path = output_dir / output_name

    if output_path.exists():
        print(f"  SKIP: {output_name} already exists")
        return output_path

    with tempfile.TemporaryDirectory() as tmp:
        tmp_video = Path(tmp) / "full_video.mp4"

        # Download with yt-dlp
        print(f"  Downloading: {url}...")
        result = subprocess.run(
            [
                "yt-dlp",
                "-f", "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720][ext=mp4]/best",
                "--merge-output-format", "mp4",
                "-o", str(tmp_video),
                "--no-playlist",
                "--quiet",
                url,
            ],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            print(f"  ERROR downloading: {result.stderr[:200]}")
            return None

        # Find the actual downloaded file (yt-dlp might add extension)
        downloaded = list(Path(tmp).glob("full_video*"))
        if not downloaded:
            print("  ERROR: No file downloaded")
            return None

        actual_file = downloaded[0]

        # Trim with ffmpeg
        print(f"  Trimming: {start:.1f}s - {end:.1f}s ({duration:.1f}s)...")
        result = subprocess.run(
            [
                "ffmpeg",
                "-ss", str(start),
                "-i", str(actual_file),
                "-t", str(duration),
                "-c:v", "libx264",
                "-preset", "fast",
                "-an",  # no audio needed
                "-y",
                str(output_path),
            ],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            print(f"  ERROR trimming: {result.stderr[:200]}")
            return None

    print(f"  OK: {output_name} ({duration:.1f}s)")
    return output_path


def update_labels(labels_path: Path, clip_name: str, trick_id: str):
    """Add a new entry to labels.json."""
    labels = []
    if labels_path.exists():
        with open(labels_path) as f:
            labels = json.load(f)

    # Check for duplicates
    if any(e["file"] == clip_name for e in labels):
        return

    labels.append({
        "file": clip_name,
        "trick_id": trick_id,
        "start_ms": 0,
    })

    with open(labels_path, "w") as f:
        json.dump(labels, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Download and trim parkour clips")
    parser.add_argument("--url", help="YouTube URL")
    parser.add_argument("--trick", help="Trick ID (e.g., back_flip)")
    parser.add_argument("--start", type=float, help="Start time in seconds")
    parser.add_argument("--end", type=float, help="End time in seconds")
    parser.add_argument("--batch", help="Path to batch JSON file")
    parser.add_argument("--output", default="data/clips", help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    labels_path = output_dir / "labels.json"

    clips_to_process = []

    if args.batch:
        with open(args.batch) as f:
            clips_to_process = json.load(f)
    elif args.url and args.trick and args.start is not None and args.end is not None:
        clips_to_process = [
            {"url": args.url, "trick_id": args.trick, "start": args.start, "end": args.end}
        ]
    else:
        parser.print_help()
        sys.exit(1)

    # Count existing clips per trick for indexing
    existing_counts: dict[str, int] = {}
    if labels_path.exists():
        with open(labels_path) as f:
            for entry in json.load(f):
                tid = entry["trick_id"]
                existing_counts[tid] = existing_counts.get(tid, 0) + 1

    print(f"Processing {len(clips_to_process)} clip(s)...\n")

    success = 0
    for clip_info in clips_to_process:
        trick_id = clip_info["trick_id"]
        idx = existing_counts.get(trick_id, 0)

        result = download_and_trim(
            url=clip_info["url"],
            trick_id=trick_id,
            start=clip_info["start"],
            end=clip_info["end"],
            output_dir=output_dir,
            clip_index=idx,
        )

        if result:
            update_labels(labels_path, result.name, trick_id)
            existing_counts[trick_id] = idx + 1
            success += 1

    print(f"\nDone. {success}/{len(clips_to_process)} clips downloaded and trimmed.")
    print(f"Labels saved to: {labels_path}")
    print(f"\nNext steps:")
    print(f"  1. python scripts/extract_poses.py --clips-dir {output_dir} --labels {labels_path} --output {output_dir}/keypoints/")
    print(f"  2. python scripts/train.py")


if __name__ == "__main__":
    main()
