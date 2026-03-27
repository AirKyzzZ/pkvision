#!/usr/bin/env python3
"""Mass download all parkourtheory videos and build feature references.

Downloads all 1,627 parkourtheory trick videos, extracts YOLO keypoints,
converts to camera-invariant features, and saves as references for MLP training.

Resume-friendly: skips already-downloaded and already-extracted tricks.

Usage:
    python scripts/mass_download_and_extract.py
    python scripts/mass_download_and_extract.py --batch-size 50   # download in batches of 50
    python scripts/mass_download_and_extract.py --extract-only     # skip downloads, only extract
    python scripts/mass_download_and_extract.py --skip-existing    # skip tricks that already have refs
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))


def make_trick_id(name: str) -> str:
    """Convert display name to snake_case trick_id."""
    tid = name.lower().replace("-", "_").replace(" ", "_")
    tid = re.sub(r"[^a-z0-9_]", "", tid)
    tid = "_".join(part for part in tid.split("_") if part)
    return tid


def download_video(url: str, output_path: str, timeout: int = 90) -> bool:
    """Download a video from Cloudflare Stream. Returns True on success."""
    try:
        result = subprocess.run(
            ["curl", "-fsSL", "-o", output_path, "--max-time", str(timeout), url],
            capture_output=True, text=True, timeout=timeout + 15,
        )
        p = Path(output_path)
        return result.returncode == 0 and p.exists() and p.stat().st_size > 500
    except Exception:
        return False


def extract_yolo_keypoints(video_path: str) -> np.ndarray | None:
    """Extract YOLO 17-keypoint poses. Returns (3, T, 17) or None."""
    try:
        from ultralytics import YOLO
        model = YOLO("yolo11n-pose.pt")
        results = model(video_path, stream=True, verbose=False)

        frames = []
        for result in results:
            if result.keypoints is not None and len(result.keypoints) > 0:
                kp = result.keypoints[0]
                xy = kp.xyn[0].cpu().numpy()
                conf = kp.conf[0].cpu().numpy()
                frame = np.zeros((3, 17), dtype=np.float32)
                frame[0] = xy[:, 0]
                frame[1] = xy[:, 1]
                frame[2] = conf
                frames.append(frame)
            else:
                frames.append(np.zeros((3, 17), dtype=np.float32))

        if len(frames) < 4:
            return None

        arr = np.stack(frames, axis=0).transpose(1, 0, 2)
        return arr.astype(np.float32)
    except Exception as e:
        print(f"    YOLO error: {e}")
        return None


def keypoints_to_features(kp_array: np.ndarray) -> np.ndarray | None:
    """Convert (3, T, 17) keypoints to (T, features) feature vectors."""
    try:
        from scripts.build_references import keypoints_3tv_to_features
        return keypoints_3tv_to_features(kp_array, segment=True)
    except Exception as e:
        print(f"    Feature error: {e}")
        return None


def get_all_tricks_with_video(unified_path: str) -> list[dict]:
    """Get all tricks that have parkourtheory video URLs."""
    unified = json.loads(Path(unified_path).read_text())
    tricks = []
    for entry in unified:
        video_url = entry.get("metadata", {}).get("video_url", "")
        if not video_url:
            continue
        trick_id = make_trick_id(entry["name"])
        tricks.append({
            "trick_id": trick_id,
            "name": entry["name"],
            "video_url": video_url,
            "family": entry["physics"]["family"],
            "difficulty": entry.get("difficulty", 5),
        })
    return tricks


def main():
    parser = argparse.ArgumentParser(description="Mass download and extract parkourtheory videos")
    parser.add_argument("--unified", default="data/unified_tricks.json")
    parser.add_argument("--download-dir", default="data/parkourtheory_clips")
    parser.add_argument("--refs-dir", default="data/references")
    parser.add_argument("--batch-size", type=int, default=100,
                        help="Print progress every N tricks")
    parser.add_argument("--extract-only", action="store_true",
                        help="Skip downloads, only extract from already-downloaded")
    parser.add_argument("--download-only", action="store_true",
                        help="Only download, skip extraction")
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--download-threads", type=int, default=4,
                        help="Parallel download threads")
    args = parser.parse_args()

    download_dir = Path(args.download_dir)
    refs_dir = Path(args.refs_dir)
    download_dir.mkdir(parents=True, exist_ok=True)

    # Get all tricks with videos
    all_tricks = get_all_tricks_with_video(args.unified)
    print(f"\nPkVision — Mass Reference Builder")
    print(f"{'=' * 60}")
    print(f"Total tricks with video: {len(all_tricks)}")

    # Determine what needs downloading vs extracting
    needs_download = []
    needs_extract = []
    already_done = []

    for trick in all_tricks:
        tid = trick["trick_id"]
        video_path = download_dir / f"{tid}.mp4"
        ref_dir = refs_dir / tid
        has_video = video_path.exists() and video_path.stat().st_size > 500
        has_ref = ref_dir.exists() and list(ref_dir.glob("*.npy"))

        if has_ref:
            already_done.append(tid)
        elif has_video:
            needs_extract.append(trick)
        else:
            needs_download.append(trick)

    print(f"Already have references: {len(already_done)}")
    print(f"Need download + extract: {len(needs_download)}")
    print(f"Need extract only:       {len(needs_extract)}")

    # ── Phase 1: Download ──────────────────────────────────────────
    if not args.extract_only and needs_download:
        print(f"\n{'─' * 60}")
        print(f"PHASE 1: Downloading {len(needs_download)} videos ({len(needs_download) * 0.8 / 1024:.1f} GB est.)")
        print(f"{'─' * 60}")

        downloaded = 0
        failed_downloads = 0
        start_time = time.time()

        def _download_one(trick):
            tid = trick["trick_id"]
            video_path = download_dir / f"{tid}.mp4"
            return tid, download_video(trick["video_url"], str(video_path), timeout=args.timeout)

        with ThreadPoolExecutor(max_workers=args.download_threads) as pool:
            futures = {pool.submit(_download_one, t): t for t in needs_download}
            for i, future in enumerate(as_completed(futures), 1):
                tid, success = future.result()
                if success:
                    downloaded += 1
                    # Move to extract list
                    needs_extract.append(futures[future])
                else:
                    failed_downloads += 1

                if i % args.batch_size == 0 or i == len(needs_download):
                    elapsed = time.time() - start_time
                    rate = i / elapsed if elapsed > 0 else 0
                    eta = (len(needs_download) - i) / rate if rate > 0 else 0
                    print(f"  [{i:4d}/{len(needs_download)}] downloaded={downloaded} "
                          f"failed={failed_downloads} "
                          f"rate={rate:.1f}/s ETA={eta:.0f}s")

        print(f"\nDownload complete: {downloaded} succeeded, {failed_downloads} failed")

    if args.download_only:
        print("\nDownload-only mode. Exiting.")
        return

    # ── Phase 2: Extract keypoints + features ──────────────────────
    if needs_extract:
        print(f"\n{'─' * 60}")
        print(f"PHASE 2: Extracting features from {len(needs_extract)} videos")
        print(f"{'─' * 60}")

        extracted = 0
        failed_extract = 0
        start_time = time.time()

        for i, trick in enumerate(needs_extract, 1):
            tid = trick["trick_id"]
            video_path = download_dir / f"{tid}.mp4"

            if not video_path.exists():
                failed_extract += 1
                continue

            # Extract keypoints
            kp = extract_yolo_keypoints(str(video_path))
            if kp is None:
                failed_extract += 1
                if i % args.batch_size == 0:
                    print(f"  [{i:4d}/{len(needs_extract)}] {tid}: FAILED (no keypoints)")
                continue

            # Convert to features
            features = keypoints_to_features(kp)
            if features is None:
                failed_extract += 1
                continue

            # Save reference
            ref_dir = refs_dir / tid
            ref_dir.mkdir(parents=True, exist_ok=True)
            np.save(ref_dir / "parkourtheory.npy", features)
            extracted += 1

            if i % args.batch_size == 0 or i == len(needs_extract):
                elapsed = time.time() - start_time
                rate = i / elapsed if elapsed > 0 else 0
                eta = (len(needs_extract) - i) / rate if rate > 0 else 0
                print(f"  [{i:4d}/{len(needs_extract)}] extracted={extracted} "
                      f"failed={failed_extract} "
                      f"rate={rate:.1f}/s ETA={eta:.0f}s")

        print(f"\nExtraction complete: {extracted} succeeded, {failed_extract} failed")

    # ── Summary ────────────────────────────────────────────────────
    total_refs = len([d for d in refs_dir.iterdir()
                      if d.is_dir() and list(d.glob("*.npy"))])
    total_size_mb = sum(
        f.stat().st_size for f in download_dir.glob("*.mp4")
    ) / 1024 / 1024

    print(f"\n{'=' * 60}")
    print(f"DONE")
    print(f"  Total reference tricks: {total_refs}")
    print(f"  Downloaded clips: {total_size_mb:.0f} MB")
    print(f"\nNext: train MLP on all references:")
    print(f"  python -m ml.mlp.train --references-dir data/references --device auto")
    print(f"\nOr sync to windows-dev for GPU training:")
    print(f"  rsync -avz data/references/ windows-dev:pkvision/data/references/")
    print(f"  ssh windows-dev 'cd pkvision && python -m ml.mlp.train --references-dir data/references --device cuda --epochs 100'")


if __name__ == "__main__":
    main()
