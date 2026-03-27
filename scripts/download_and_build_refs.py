#!/usr/bin/env python3
"""Download parkourtheory videos and build reference features for training.

Pipeline per trick:
  1. Download MP4 from Cloudflare Stream URL
  2. Extract YOLO 17-keypoint poses
  3. Convert to (T, 60) camera-invariant features
  4. Save as reference in data/references/{trick_id}/

Usage:
    python scripts/download_and_build_refs.py
    python scripts/download_and_build_refs.py --max-tricks 20  # limit downloads
    python scripts/download_and_build_refs.py --skip-download   # only extract from already-downloaded
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.build_unified_tricks import normalize_name


def get_download_plan(
    unified_path: str,
    existing_refs_dir: str,
    max_tricks: int = 100,
) -> list[dict]:
    """Build a prioritized list of tricks to download.

    Prioritizes tricks that:
    1. Are in TRICK_DEFINITIONS (used by the classifier)
    2. Don't already have references
    3. Have parkourtheory video URLs
    4. Are from multiple sources (well-known)
    """
    from ml.trick_physics import TRICK_DEFINITIONS

    unified = json.loads(Path(unified_path).read_text())
    by_canonical = {e["canonical_name"]: e for e in unified}

    existing = set()
    refs_dir = Path(existing_refs_dir)
    if refs_dir.exists():
        for d in refs_dir.iterdir():
            if d.is_dir() and list(d.glob("*.npy")):
                existing.add(d.name)

    plan = []
    for trick_id, td in TRICK_DEFINITIONS.items():
        if trick_id in existing:
            continue

        canonical = normalize_name(td.name)
        entry = by_canonical.get(canonical)
        if not entry:
            continue

        video_url = entry.get("metadata", {}).get("video_url", "")
        if not video_url:
            continue

        plan.append({
            "trick_id": trick_id,
            "name": td.name,
            "canonical": canonical,
            "video_url": video_url,
            "sources": len(entry["sources"]),
            "difficulty": entry.get("difficulty", 5),
        })

    # Sort: more sources first, then easier tricks first
    plan.sort(key=lambda x: (-x["sources"], x["difficulty"]))
    return plan[:max_tricks]


def download_video(url: str, output_path: str, timeout: int = 60) -> bool:
    """Download a video from a Cloudflare Stream URL."""
    try:
        result = subprocess.run(
            ["curl", "-fsSL", "-o", output_path, "--max-time", str(timeout), url],
            capture_output=True, text=True, timeout=timeout + 10,
        )
        if result.returncode != 0:
            print(f"    curl failed: {result.stderr[:200]}")
            return False

        # Verify file exists and has content
        p = Path(output_path)
        if not p.exists() or p.stat().st_size < 1000:
            print(f"    Downloaded file too small ({p.stat().st_size} bytes)")
            return False

        return True
    except subprocess.TimeoutExpired:
        print(f"    Download timed out after {timeout}s")
        return False
    except Exception as e:
        print(f"    Download error: {e}")
        return False


def extract_yolo_keypoints(video_path: str, target_fps: int = 30) -> np.ndarray | None:
    """Extract YOLO 17-keypoint poses from a video.

    Returns: (3, T, 17) array — channels (x, y, confidence), frames, joints.
    Coordinates normalized to [0, 1].
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        print("    ERROR: ultralytics not installed. Run: pip install ultralytics")
        return None

    model = YOLO("yolo11n-pose.pt")

    results = model(video_path, stream=True, verbose=False)

    all_keypoints = []
    for result in results:
        if result.keypoints is not None and len(result.keypoints) > 0:
            # Take the person with highest confidence
            kp = result.keypoints[0]
            xy = kp.xyn[0].cpu().numpy()  # (17, 2) normalized
            conf = kp.conf[0].cpu().numpy()  # (17,)

            frame = np.zeros((3, 17), dtype=np.float32)
            frame[0] = xy[:, 0]  # x
            frame[1] = xy[:, 1]  # y
            frame[2] = conf       # confidence
            all_keypoints.append(frame)
        else:
            # No person detected — fill with zeros
            all_keypoints.append(np.zeros((3, 17), dtype=np.float32))

    if not all_keypoints:
        return None

    # Stack: (T, 3, 17) -> (3, T, 17)
    kp_array = np.stack(all_keypoints, axis=0)  # (T, 3, 17)
    kp_array = kp_array.transpose(1, 0, 2)      # (3, T, 17)

    return kp_array.astype(np.float32)


def keypoints_to_features(kp_array: np.ndarray) -> np.ndarray | None:
    """Convert (3, T, 17) keypoints to (T, 60) feature vectors."""
    from scripts.build_references import keypoints_3tv_to_features
    return keypoints_3tv_to_features(kp_array, segment=True)


def main():
    parser = argparse.ArgumentParser(description="Download parkourtheory videos and build references")
    parser.add_argument("--unified", default="data/unified_tricks.json")
    parser.add_argument("--refs-dir", default="data/references")
    parser.add_argument("--download-dir", default="data/parkourtheory_clips")
    parser.add_argument("--max-tricks", type=int, default=100)
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip downloading, only process already-downloaded videos")
    parser.add_argument("--timeout", type=int, default=120, help="Download timeout per video (seconds)")
    args = parser.parse_args()

    refs_dir = Path(args.refs_dir)
    download_dir = Path(args.download_dir)
    download_dir.mkdir(parents=True, exist_ok=True)

    # Get download plan
    plan = get_download_plan(args.unified, args.refs_dir, max_tricks=args.max_tricks)
    print(f"\nPkVision — Reference Builder")
    print(f"{'=' * 50}")
    print(f"Tricks to process: {plan and len(plan) or 0}")
    print(f"Existing references: {len(list(refs_dir.iterdir())) if refs_dir.exists() else 0}")
    print(f"Download dir: {download_dir}")

    if not plan:
        print("\nAll tricks already have references. Nothing to do.")
        # Check if we should process already-downloaded videos
        if args.skip_download:
            plan = _plan_from_downloads(download_dir, refs_dir)
            if not plan:
                return

    succeeded = 0
    failed = 0

    for i, trick in enumerate(plan):
        trick_id = trick["trick_id"]
        name = trick["name"]
        video_url = trick["video_url"]
        video_path = download_dir / f"{trick_id}.mp4"

        print(f"\n[{i+1}/{len(plan)}] {name} ({trick_id})")

        # Step 1: Download
        if not args.skip_download and not video_path.exists():
            print(f"  Downloading from Cloudflare Stream...")
            if not download_video(video_url, str(video_path), timeout=args.timeout):
                print(f"  FAILED to download")
                failed += 1
                continue
            size_mb = video_path.stat().st_size / 1024 / 1024
            print(f"  Downloaded: {size_mb:.1f} MB")
        elif video_path.exists():
            print(f"  Already downloaded")
        else:
            print(f"  No video file, skipping")
            failed += 1
            continue

        # Step 2: Extract keypoints
        print(f"  Extracting YOLO keypoints...")
        kp_array = extract_yolo_keypoints(str(video_path))
        if kp_array is None:
            print(f"  FAILED to extract keypoints")
            failed += 1
            continue
        print(f"  Keypoints: {kp_array.shape} ({kp_array.shape[1]} frames)")

        # Step 3: Convert to features
        print(f"  Converting to features...")
        features = keypoints_to_features(kp_array)
        if features is None:
            print(f"  FAILED to extract features (too few frames or bad data)")
            failed += 1
            continue
        print(f"  Features: {features.shape}")

        # Step 4: Save reference
        trick_ref_dir = refs_dir / trick_id
        trick_ref_dir.mkdir(parents=True, exist_ok=True)
        ref_path = trick_ref_dir / "parkourtheory.npy"
        np.save(ref_path, features)
        print(f"  Saved: {ref_path}")
        succeeded += 1

    print(f"\n{'=' * 50}")
    print(f"Results: {succeeded} succeeded, {failed} failed")
    print(f"Total references: {len([d for d in refs_dir.iterdir() if d.is_dir() and list(d.glob('*.npy'))])} tricks")

    if succeeded > 0:
        print(f"\nNext step — retrain the MLP:")
        print(f"  python -m ml.mlp.train --references-dir data/references --device auto")


def _plan_from_downloads(download_dir: Path, refs_dir: Path) -> list[dict]:
    """Build plan from already-downloaded videos that don't have refs yet."""
    from ml.trick_physics import TRICK_DEFINITIONS

    existing_refs = set()
    if refs_dir.exists():
        for d in refs_dir.iterdir():
            if d.is_dir() and list(d.glob("*.npy")):
                existing_refs.add(d.name)

    plan = []
    for video_path in download_dir.glob("*.mp4"):
        trick_id = video_path.stem
        if trick_id in existing_refs:
            continue
        td = TRICK_DEFINITIONS.get(trick_id)
        if td:
            plan.append({
                "trick_id": trick_id,
                "name": td.name,
                "video_url": "",  # Already downloaded
            })

    return plan


if __name__ == "__main__":
    main()
