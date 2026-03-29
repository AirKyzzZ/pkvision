"""Build Motion Signature Database from parkourtheory reference clips.

Downloads reference videos, extracts ViTPose 2D keypoints, and builds
a signature database for trick matching.

Usage:
    python scripts/build_signature_db.py --max-tricks 80 --output data/signature_db.pt
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import requests
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.recognition.motion_signature import SignatureDatabase


def download_video(url: str, output_path: str, timeout: int = 30) -> bool:
    """Download a video from URL."""
    try:
        resp = requests.get(url, timeout=timeout, stream=True)
        resp.raise_for_status()
        with open(output_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    except Exception as e:
        print(f"    Download failed: {e}")
        return False


def extract_vitpose_simple(video_path: str, yolo_model_path: str) -> np.ndarray | None:
    """Extract 2D keypoints using YOLO-pose (simpler than full ViTPose).

    Returns (T, 17, 3) array of [x, y, confidence] per frame.
    """
    from ultralytics import YOLO

    model = YOLO(yolo_model_path)
    cap = cv2.VideoCapture(video_path)
    T = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if T < 5:
        return None

    results = model(video_path, stream=True, verbose=False)
    all_kps = []
    last_good = np.zeros((17, 3), dtype=np.float32)

    for result in results:
        if (result.keypoints is not None and len(result.keypoints) > 0
                and result.keypoints.data.shape[-1] == 3
                and result.keypoints.data.shape[-2] == 17):
            kp_data = result.keypoints.data  # (N, 17, 3)
            avg_conf = kp_data[:, :, 2].mean(dim=1)
            best = avg_conf.argmax().item()
            kps = kp_data[best].cpu().numpy().astype(np.float32)  # (17, 3)
            if kps.shape == (17, 3):
                last_good = kps
                all_kps.append(kps)
            else:
                all_kps.append(last_good.copy())
        else:
            all_kps.append(last_good.copy())

    if len(all_kps) < 5:
        return None

    return np.stack(all_kps).astype(np.float32)


def map_fig_to_parkourtheory(fig_path: str, pt_path: str, map_path: str = "data/fig_to_parkourtheory_map.json") -> list[dict]:
    """Map FIG tricks to parkourtheory video URLs using manual mapping."""
    with open(fig_path, encoding="utf-8") as f:
        fig = json.load(f)
    with open(pt_path, encoding="utf-8") as f:
        pt = json.load(f)

    # Build parkourtheory index (case insensitive)
    pt_index = {}
    for t in pt:
        name = t.get("name", "").strip()
        if name and t.get("video_url"):
            pt_index[name.lower()] = t

    # Load manual mapping if available
    manual = {}
    if os.path.exists(map_path):
        with open(map_path) as f:
            manual_data = json.load(f)
        for cat_name, mappings in manual_data.items():
            if cat_name.startswith("_"):
                continue
            for fig_name, pt_name in mappings.items():
                if not fig_name.startswith("_") and pt_name != "NO MATCH":
                    manual[fig_name] = pt_name

    # Track used videos to avoid duplicates
    used_videos = set()
    mapped = []

    for cat_name, cat in fig["categories"].items():
        for trick in cat["tricks"]:
            fig_name = trick["name"]

            # Use manual mapping first, then fallback to auto
            pt_name = manual.get(fig_name)
            match = None

            if pt_name:
                match = pt_index.get(pt_name.lower())

            if not match:
                # Fallback: exact name match only (no partial)
                match = pt_index.get(fig_name.lower())

            if match:
                url = match["video_url"]
                if url in used_videos:
                    continue  # Skip duplicate videos
                used_videos.add(url)

                mapped.append({
                    "fig_name": fig_name,
                    "fig_id": fig_name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_"),
                    "category": cat_name,
                    "score": trick.get("score", 0),
                    "flip": trick.get("flip", 0),
                    "twist": trick.get("twist", 0),
                    "direction": trick.get("direction", ""),
                    "axis": trick.get("axis", ""),
                    "pt_name": match["name"],
                    "video_url": match["video_url"],
                })

    return mapped


def main():
    parser = argparse.ArgumentParser(description="Build Motion Signature Database")
    parser.add_argument("--max-tricks", type=int, default=80,
                        help="Max tricks to download and process")
    parser.add_argument("--output", default="data/signature_db.pt",
                        help="Output database path")
    parser.add_argument("--video-dir", default="data/reference_clips",
                        help="Directory to store downloaded clips")
    parser.add_argument("--yolo", default="yolo11n-pose.pt",
                        help="YOLO pose model path")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip downloading, use existing clips")
    args = parser.parse_args()

    print("\nPkVision — Building Motion Signature Database")
    print("=" * 60)

    # Map FIG tricks to parkourtheory videos
    mapped = map_fig_to_parkourtheory(
        "data/fig_tricks_2025.json",
        "data/parkourtheory_detailed.json",
    )
    print(f"Mapped {len(mapped)} FIG tricks to parkourtheory videos")

    # Limit
    mapped = mapped[:args.max_tricks]
    print(f"Processing {len(mapped)} tricks")

    # Create directories
    video_dir = Path(args.video_dir)
    video_dir.mkdir(parents=True, exist_ok=True)

    # Download and process each trick
    db = SignatureDatabase(norm_length=60)
    successful = 0
    failed = []

    for i, trick in enumerate(mapped):
        fig_name = trick["fig_name"]
        safe_name = trick["fig_id"]
        video_path = str(video_dir / f"{safe_name}.mp4")

        print(f"\n[{i+1}/{len(mapped)}] {fig_name} (D={trick['score']})")

        # Download
        if not args.skip_download or not os.path.exists(video_path):
            url = trick["video_url"]
            print(f"  Downloading from parkourtheory...")
            if not download_video(url, video_path):
                failed.append(fig_name)
                continue

        # Check video is valid
        cap = cv2.VideoCapture(video_path)
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()

        if frames < 10:
            print(f"  Skipping: too short ({frames} frames)")
            failed.append(fig_name)
            continue

        print(f"  Video: {frames} frames @ {fps:.0f}fps ({frames/fps:.1f}s)")

        # Extract keypoints
        print(f"  Extracting keypoints...")
        kps = extract_vitpose_simple(video_path, args.yolo)
        if kps is None:
            print(f"  Skipping: keypoint extraction failed")
            failed.append(fig_name)
            continue

        # Add to database
        try:
            db.add_reference(fig_name, kps, fps=fps)
            ref = db.references[-1]
            inv = "INV" if ref.went_inverted else "   "
            print(f"  Added: rot_speed={ref.rotation_speed:.1f} {inv} inversions={ref.num_inversions}")
            successful += 1
        except Exception as e:
            print(f"  Error adding reference: {e}")
            failed.append(fig_name)

    # Save database
    print(f"\n{'=' * 60}")
    print(f"Database built: {successful} tricks ({len(failed)} failed)")

    # Save as .pt
    db_data = {
        "references": [],
        "norm_length": db.norm_length,
    }
    for ref in db.references:
        db_data["references"].append({
            "name": ref.name,
            "trajectory": ref.trajectory,
            "duration_s": ref.duration_s,
            "max_height_change": ref.max_height_change,
            "rotation_speed": ref.rotation_speed,
            "went_inverted": ref.went_inverted,
            "num_inversions": ref.num_inversions,
        })

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(db_data, args.output)
    print(f"Saved to {args.output}")

    if failed:
        print(f"\nFailed tricks: {', '.join(failed[:20])}")


if __name__ == "__main__":
    main()
