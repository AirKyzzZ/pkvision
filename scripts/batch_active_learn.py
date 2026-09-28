#!/usr/bin/env python3
"""Batch process all downloaded videos for active learning.

Processes every video in data/active_learning/videos/ that hasn't been
processed yet. Then opens a labeling session for all pending tricks.

Usage:
    python scripts/batch_active_learn.py process   # segment + crop all videos
    python scripts/batch_active_learn.py label      # label all pending tricks
    python scripts/batch_active_learn.py stats      # show progress
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

VIDEOS_DIR = ROOT / "data" / "active_learning" / "videos"
PENDING_DIR = ROOT / "data" / "active_learning" / "pending"
CONFIRMED_DIR = ROOT / "data" / "active_learning" / "confirmed"


def get_processed_videos():
    """Get set of already-processed video names."""
    processed = set()
    if PENDING_DIR.exists():
        for session_dir in PENDING_DIR.iterdir():
            if session_dir.is_dir():
                manifest = session_dir / "manifest.json"
                if manifest.exists():
                    with open(manifest) as f:
                        m = json.load(f)
                    processed.add(m.get("video_name", ""))
    return processed


def batch_process():
    """Process all unprocessed videos."""
    from scripts.active_learn import process_video, suggest_labels

    if not VIDEOS_DIR.exists():
        print("  No videos directory found.")
        return

    video_exts = {".mp4", ".webm", ".mkv", ".mov"}
    all_videos = sorted([
        f for f in VIDEOS_DIR.iterdir()
        if f.suffix.lower() in video_exts and not f.name.startswith(".")
    ])

    processed = get_processed_videos()
    to_process = [v for v in all_videos if v.name not in processed]

    print(f"\n  Videos: {len(all_videos)} total, {len(processed)} already processed, {len(to_process)} new")

    if not to_process:
        print("  Nothing new to process.")
        return

    total_tricks = 0
    for i, video_path in enumerate(to_process):
        print(f"\n  [{i + 1}/{len(to_process)}] {video_path.name}")
        try:
            session_dir = process_video(str(video_path))
            if session_dir:
                suggest_labels(session_dir)
                manifest = session_dir / "manifest.json"
                with open(manifest) as f:
                    m = json.load(f)
                n_tricks = len(m.get("tricks", []))
                total_tricks += n_tricks
                print(f"  -> {n_tricks} tricks extracted")
        except Exception as e:
            print(f"  -> ERROR: {e}")
            continue

    print(f"\n  Done! {total_tricks} total tricks extracted from {len(to_process)} videos")
    print(f"  Run: python scripts/batch_active_learn.py label")


def batch_label():
    """Interactive labeling across all pending sessions."""
    if not PENDING_DIR.exists():
        print("  No pending sessions.")
        return

    # Collect all pending tricks across sessions
    all_pending = []
    for session_dir in sorted(PENDING_DIR.iterdir()):
        if not session_dir.is_dir():
            continue
        manifest_path = session_dir / "manifest.json"
        if not manifest_path.exists():
            continue
        with open(manifest_path) as f:
            manifest = json.load(f)
        for trick in manifest["tricks"]:
            if trick["status"] == "pending":
                all_pending.append((session_dir, manifest, trick))

    if not all_pending:
        print("  No pending tricks to label.")
        return

    print(f"\n  PkVision Active Learning — Labeling Session")
    print(f"  {'=' * 50}")
    print(f"  {len(all_pending)} tricks to label")
    print()
    print(f"  Commands:")
    print(f"    Y or Enter  = accept CLIP suggestion")
    print(f"    S           = skip (not a trick / bad crop)")
    print(f"    Q           = quit and save progress")
    print(f"    b,1,a       = backward, 1 flip, acrobatics")
    print(f"    f,2,w       = forward, 2+ flips, wall")
    print(f"    s,0,pk      = side, 0 flips, pk_basics")
    print(f"    n,1,sw      = none, 1 flip, swing")
    print()
    print(f"  Direction: b=backward, f=forward, s=side, n=none")
    print(f"  Flip:      0, 1, 2+")
    print(f"  Context:   a=acrobatics, w=wall, sw=swing, pk=pk_basics")
    print()

    confirmed_count = 0
    skipped_count = 0

    # Shorthand maps
    dir_map = {"b": "backward", "f": "forward", "s": "side", "n": "none",
               "backward": "backward", "forward": "forward", "side": "side", "none": "none"}
    flip_map = {"0": "0", "1": "1", "2": "2+", "2+": "2+", "3": "2+"}
    ctx_map = {"a": "acrobatics", "w": "wall", "sw": "swing", "pk": "pk_basics",
               "acrobatics": "acrobatics", "wall": "wall", "swing": "swing", "pk_basics": "pk_basics"}

    for i, (session_dir, manifest, trick) in enumerate(all_pending):
        suggestion = trick.get("suggested_label", {})
        s_dir = suggestion.get("direction", {}).get("value", "?")
        s_flip = suggestion.get("flip_count", {}).get("value", "?")
        s_ctx = suggestion.get("context", {}).get("value", "?")

        video_name = manifest.get("video_name", "?")[:40]
        preview_path = trick.get("preview", "")

        print(f"  [{i + 1}/{len(all_pending)}] {video_name} / {trick['name']}")
        print(f"    Time: [{trick['start_s']:.1f}s - {trick['end_s']:.1f}s]  Preview: {preview_path}")
        print(f"    CLIP: dir={s_dir}  flip={s_flip}  ctx={s_ctx}")

        try:
            response = input(f"    > ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n  Saving and exiting...")
            break

        if response in ("q", "quit"):
            break
        elif response in ("s", "skip"):
            trick["status"] = "skipped"
            skipped_count += 1
        elif response in ("y", "yes", ""):
            trick["confirmed_label"] = {
                "direction": s_dir,
                "flip_count": s_flip,
                "context": s_ctx,
            }
            trick["status"] = "confirmed"
            confirmed_count += 1
        elif "," in response:
            parts = [p.strip() for p in response.split(",")]
            if len(parts) >= 3:
                d = dir_map.get(parts[0], parts[0])
                f = flip_map.get(parts[1], parts[1])
                c = ctx_map.get(parts[2], parts[2])
                trick["confirmed_label"] = {
                    "direction": d,
                    "flip_count": f,
                    "context": c,
                }
                trick["status"] = "confirmed"
                confirmed_count += 1
                print(f"    -> {d}, {f}, {c}")
            else:
                print(f"    -> Need 3 values: dir,flip,context")
        else:
            print(f"    -> Unknown, skipping")

        # Save after each label (crash-safe)
        manifest_path = session_dir / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

    # Export confirmed
    _export_all_confirmed()

    print(f"\n  Session complete: {confirmed_count} confirmed, {skipped_count} skipped")
    print(f"  Run: python scripts/batch_active_learn.py stats")


def _export_all_confirmed():
    """Export all confirmed tricks to training format."""
    import shutil

    CONFIRMED_DIR.mkdir(parents=True, exist_ok=True)
    exported = []

    for session_dir in sorted(PENDING_DIR.iterdir()):
        if not session_dir.is_dir():
            continue
        manifest_path = session_dir / "manifest.json"
        if not manifest_path.exists():
            continue
        with open(manifest_path) as f:
            manifest = json.load(f)

        for trick in manifest["tricks"]:
            if trick["status"] != "confirmed":
                continue

            src = Path(trick["npy_path"])
            if not src.exists():
                continue

            label = trick["confirmed_label"]
            slug = f"{session_dir.name}_{trick['name']}"
            dst = CONFIRMED_DIR / f"{slug}.npy"

            if not dst.exists():
                shutil.copy2(src, dst)

            exported.append({
                "slug": slug,
                "npy_path": str(dst),
                "direction": label.get("direction", "none"),
                "flip_count": label.get("flip_count", "0"),
                "context": label.get("context", "acrobatics"),
                "source": "active_learning",
                "video": manifest.get("video_name", ""),
            })

    # Save manifest
    confirmed_manifest = CONFIRMED_DIR / "confirmed_clips.json"
    with open(confirmed_manifest, "w") as f:
        json.dump(exported, f, indent=2)

    if exported:
        print(f"\n  Exported {len(exported)} confirmed clips to {CONFIRMED_DIR}")


def show_stats():
    """Show active learning progress."""
    print(f"\n  PkVision Active Learning Stats")
    print(f"  {'=' * 50}")

    # Videos
    if VIDEOS_DIR.exists():
        videos = [f for f in VIDEOS_DIR.iterdir()
                  if f.suffix.lower() in {".mp4", ".webm", ".mkv", ".mov"}]
        print(f"  Downloaded videos: {len(videos)}")
    else:
        print(f"  Downloaded videos: 0")

    # Sessions
    if PENDING_DIR.exists():
        sessions = [d for d in PENDING_DIR.iterdir() if d.is_dir()]
    else:
        sessions = []

    total = confirmed = pending = skipped = 0
    for session_dir in sessions:
        manifest_path = session_dir / "manifest.json"
        if not manifest_path.exists():
            continue
        with open(manifest_path) as f:
            m = json.load(f)
        for t in m.get("tricks", []):
            total += 1
            s = t.get("status", "pending")
            if s == "confirmed":
                confirmed += 1
            elif s == "skipped":
                skipped += 1
            else:
                pending += 1

    print(f"  Processed sessions: {len(sessions)}")
    print(f"  Total tricks: {total}")
    print(f"    Confirmed: {confirmed}")
    print(f"    Pending:   {pending}")
    print(f"    Skipped:   {skipped}")

    # Confirmed clips
    confirmed_manifest = CONFIRMED_DIR / "confirmed_clips.json"
    if confirmed_manifest.exists():
        with open(confirmed_manifest) as f:
            clips = json.load(f)
        print(f"\n  Training clips from active learning: {len(clips)}")

        # Distribution
        from collections import Counter
        dirs = Counter(c["direction"] for c in clips)
        flips = Counter(c["flip_count"] for c in clips)
        ctxs = Counter(c["context"] for c in clips)
        if clips:
            print(f"    Direction: {dict(dirs)}")
            print(f"    Flip:      {dict(flips)}")
            print(f"    Context:   {dict(ctxs)}")

    print()


def main():
    parser = argparse.ArgumentParser(description="Batch active learning")
    parser.add_argument("command", choices=["process", "label", "stats"])
    args = parser.parse_args()

    if args.command == "process":
        batch_process()
    elif args.command == "label":
        batch_label()
    elif args.command == "stats":
        show_stats()


if __name__ == "__main__":
    main()
