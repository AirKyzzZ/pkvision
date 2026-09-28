#!/usr/bin/env python3
"""Full hybrid analysis — combines GVHMR physics + YOLO-pose DTW signatures.

Usage:
    python scripts/analyze_hybrid.py \
        --gvhmr-output data/gvhmr_outputs/IMG_5985/hmr4d_results.pt \
        --keypoints data/keypoints/IMG_5985_keypoints.json \
        --signature-db data/signature_db_v5.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.pose.rotation_tracker import smooth_rotations, track_rotation
from core.recognition.hybrid_matcher import HybridMatcher
from core.recognition.context_detector import detect_context, detect_context_from_translation
from scripts.analyze_3d import load_gvhmr_output, segment_tricks_3d
from ml.trick_physics import TrickContext


def main():
    parser = argparse.ArgumentParser(description="Hybrid pipeline analysis")
    parser.add_argument("--gvhmr-output", required=True)
    parser.add_argument("--keypoints", required=True, help="JSON from modal_yolo_keypoints.py")
    parser.add_argument("--signature-db", default="data/signature_db_v5.pt")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    print(f"\nPkVision — Hybrid Pipeline Analysis")
    print(f"{'=' * 70}")

    # ── Load GVHMR output ──
    print(f"\n[1] Loading GVHMR output...")
    global_orient, body_pose, transl = load_gvhmr_output(args.gvhmr_output)
    T_gvhmr = global_orient.shape[0]
    print(f"  GVHMR: {T_gvhmr} frames ({T_gvhmr/args.fps:.1f}s)")

    # ── Load 2D keypoints ──
    print(f"\n[2] Loading YOLO-pose keypoints...")
    with open(args.keypoints) as f:
        kp_data = json.load(f)
    keypoints = np.array(kp_data["keypoints"])  # (T, 17, 3)
    T_kp = keypoints.shape[0]
    kp_fps = kp_data.get("fps", args.fps)
    print(f"  Keypoints: {T_kp} frames, FPS: {kp_fps:.1f}")

    # ── Smooth & track rotation ──
    print(f"\n[3] Tracking 3D rotation...")
    global_orient = smooth_rotations(global_orient, sigma=1.5)
    tracking = track_rotation(global_orient)
    print(f"  Inversion crossings: {tracking['inversion_crossings']}")
    print(f"  Peak rotation rate: {tracking['rotation_rate'].max():.1f}°/frame")

    # ── Segment tricks ──
    print(f"\n[4] Segmenting tricks...")
    segments = segment_tricks_3d(tracking, fps=args.fps)
    print(f"  Found {len(segments)} segments")

    # ── Load hybrid matcher ──
    print(f"\n[5] Loading hybrid matcher (signature DB: {args.signature_db})...")
    matcher = HybridMatcher(signature_db_path=args.signature_db)
    n_refs = len(matcher.sig_db.references)
    print(f"  {n_refs} reference signatures loaded")

    # ── Match each segment ──
    print(f"\n[6] Matching segments (physics + DTW hybrid)")
    print(f"{'─' * 70}")

    for i, (start, end) in enumerate(segments):
        t_start = start / args.fps
        t_end = end / args.fps
        duration = t_end - t_start

        # Get corresponding keypoint segment (may differ in frame count)
        kp_ratio = T_kp / T_gvhmr
        kp_start = int(start * kp_ratio)
        kp_end = min(int(end * kp_ratio), T_kp - 1)
        kp_segment = keypoints[kp_start:kp_end + 1, :, :2]  # (seg_len, 17, 2)

        # Context detection from both 2D and 3D
        ctx_2d = detect_context(keypoints, kp_start, kp_end, kp_fps)
        ctx_3d = detect_context_from_translation(transl, start, end)
        # Use 3D context if it detects wall/bar, otherwise fall back to 2D
        context = ctx_3d if ctx_3d != TrickContext.GROUND else ctx_2d
        context_label = context.value

        # Hybrid matching
        matches, physics = matcher.match_segment(
            tracking, start, end,
            global_orient, body_pose, transl,
            kp_segment, fps=args.fps, top_k=args.top_k,
        )

        print(f"\n  Trick #{i+1} @ {t_start:.1f}s - {t_end:.1f}s ({duration:.1f}s) [context: {context_label}]")
        if physics:
            print(f"    Physics: {physics['flip_deg']:.0f}° flip ({physics['flip_count']:.1f}x) "
                  f"+ {physics['twist_deg']:.0f}° twist ({physics['twist_count']:.1f}x)")
            print(f"    Direction: {physics['direction']}  Shape: {physics['body_shape']}  "
                  f"Entry: {physics['entry']}  Axis: {physics['axis']}")

        if matches:
            print(f"    {'Rank':<5} {'Trick':35s} {'Hybrid':>7} {'Physics':>8} {'DTW':>8} {'D-Score':>8}")
            print(f"    {'─'*5} {'─'*35} {'─'*7} {'─'*8} {'─'*8} {'─'*8}")
            for rank, m in enumerate(matches):
                d_str = f"D={m.d_score:.1f}" if m.d_score > 0 else "—"
                marker = " ◄" if rank == 0 else ""
                print(f"    {rank+1:<5} {m.trick_name:35s} {m.hybrid_confidence:6.1%} "
                      f"{m.physics_confidence:7.1%} {m.signature_confidence:7.1%} "
                      f"{d_str:>8}{marker}")
        else:
            print(f"    No matches found")

    print(f"\n{'=' * 70}")
    print("DONE")


if __name__ == "__main__":
    main()
