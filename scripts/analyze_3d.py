#!/usr/bin/env python3
"""Analyze a parkour run using the full 3D pipeline.

Pipeline: Video → GVHMR (SMPL) → rotation_tracker → TrickSignature3D → Zero-shot matching

This script reads GVHMR output (hmr4d_results.pt) and runs the rotation tracking
+ matching pipeline. Handles both single-trick clips and full competition runs.

Usage:
    # Single trick clip:
    python scripts/analyze_3d.py --gvhmr-output data/gvhmr_outputs/backflip/hmr4d_results.pt

    # Full run with segmentation:
    python scripts/analyze_3d.py --gvhmr-output outputs/demo/run/hmr4d_results.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.models import TrickSignature3D
from core.pose.rotation_tracker import (
    extract_trick_physics,
    smooth_rotations,
    track_rotation,
)
from core.recognition.matcher import Matcher3D
from ml.trick_physics import TrickContext


def load_gvhmr_output(pt_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load GVHMR hmr4d_results.pt and return raw arrays.

    Returns:
        (global_orient, body_pose, transl) as numpy arrays.
        global_orient: (T, 3) axis-angle
        body_pose: (T, 23, 3) joint rotations (padded from 21 to 23)
        transl: (T, 3) world position
    """
    data = torch.load(pt_path, map_location="cpu", weights_only=False)
    params = data["smpl_params_global"]

    global_orient = params["global_orient"].numpy()  # (T, 3)
    body_pose = params["body_pose"].numpy()           # (T, 63) = 21 joints × 3
    transl = params["transl"].numpy()                 # (T, 3)

    T = global_orient.shape[0]

    # Reshape body_pose: (T, 63) → (T, 21, 3) → pad to (T, 23, 3)
    body_pose_21 = body_pose.reshape(T, 21, 3)
    body_pose_23 = np.zeros((T, 23, 3), dtype=np.float32)
    body_pose_23[:, :21, :] = body_pose_21

    return global_orient, body_pose_23, transl


def segment_tricks_3d(
    tracking: dict,
    fps: float = 30.0,
    min_rotation_deg: float = 120.0,
) -> list[tuple[int, int]]:
    """Segment a run into tricks using 3D rotation rate.

    Uses the rotation_rate from track_rotation() (total rotation speed per frame)
    to detect trick zones. Uses a two-pass approach:
    1. High threshold to find trick PEAKS (where the fast rotation happens)
    2. Low threshold to extend to the full trick boundaries (takeoff → landing)

    Args:
        tracking: Output from track_rotation()
        fps: Video frame rate
        min_rotation_deg: Minimum total rotation to count as a trick

    Returns:
        List of (start_frame, end_frame) tuples for each trick segment.
    """
    rotation_rate = tracking["rotation_rate"]
    tilt_angle = tracking["tilt_angle"]
    T = len(rotation_rate)
    if T < 10:
        return [(0, T - 1)]

    # Smooth rotation rate
    kernel_size = max(3, int(fps * 0.1))
    kernel = np.ones(kernel_size) / kernel_size
    smooth_rate = np.convolve(rotation_rate, kernel, mode="same")

    # Pass 1: Find trick PEAKS with a high threshold
    peak_threshold = max(np.mean(smooth_rate) + 1.0 * np.std(smooth_rate), 8.0)
    peak_active = smooth_rate > peak_threshold

    # Pass 2: Extend to full trick boundaries with a low threshold
    # A trick includes the takeoff and landing phases where rotation is building/decaying
    boundary_threshold = max(np.mean(smooth_rate) * 0.5, 3.0)
    combined_active = smooth_rate > boundary_threshold

    # Find peak regions first
    min_gap = int(0.6 * fps)
    min_duration = int(0.3 * fps)

    peak_regions = []
    in_seg = False
    start = 0
    for i in range(T):
        if peak_active[i] and not in_seg:
            start = i
            in_seg = True
        elif not peak_active[i] and in_seg:
            gap_end = min(i + min_gap, T)
            if np.any(peak_active[i:gap_end]):
                continue
            in_seg = False
            peak_regions.append((start, i - 1))
    if in_seg:
        peak_regions.append((start, T - 1))

    # Extend each peak region to full trick boundaries
    segments = []
    for ps, pe in peak_regions:
        # Extend backward while combined_active
        s = ps
        while s > 0 and combined_active[s - 1]:
            s -= 1
        # Extend forward while combined_active
        e = pe
        while e < T - 1 and combined_active[e + 1]:
            e += 1
        # Add padding for takeoff/landing context
        pad = int(0.3 * fps)
        s = max(0, s - pad)
        e = min(T - 1, e + pad)
        if e - s >= min_duration:
            segments.append((s, e))

    # Merge overlapping/close segments
    if len(segments) > 1:
        merged = [segments[0]]
        for s, e in segments[1:]:
            if s <= merged[-1][1] + min_gap:
                merged[-1] = (merged[-1][0], max(merged[-1][1], e))
            else:
                merged.append((s, e))
        segments = merged

    # Split long segments using two methods:
    # 1. Upright valleys (tilt < threshold)
    # 2. Rotation-rate valleys (low rotation speed = between tricks)
    split_segments = []
    min_split_gap = int(0.1 * fps)  # 100ms pause = potential trick boundary
    max_trick_duration = int(3.5 * fps)  # No single trick lasts > 3.5s

    for s, e in segments:
        seg_tilt = tilt_angle[s:e + 1]
        seg_rate = smooth_rate[s:e + 1]
        duration = e - s

        if duration < 1.5 * fps:
            # Short segment — don't split
            split_segments.append((s, e))
            continue

        split_points = []

        # Method 1: Tilt valleys (upright moments between tricks)
        # Use a higher threshold (35°) to catch more valleys on noisy data
        upright_thresh = 35 if duration > 3.0 * fps else 20
        upright = seg_tilt < upright_thresh
        valley_starts = []
        valley_ends = []
        in_valley = False
        for i in range(len(seg_tilt)):
            if upright[i] and not in_valley:
                in_valley = True
                valley_starts.append(i)
            elif not upright[i] and in_valley:
                in_valley = False
                valley_ends.append(i)
        if in_valley:
            valley_ends.append(len(seg_tilt) - 1)

        for vs, ve in zip(valley_starts, valley_ends):
            if ve - vs >= min_split_gap:
                split_points.append(s + (vs + ve) // 2)

        # Method 2: Rotation-rate valleys (low rotation = between tricks)
        # Only for segments > 3s where tilt splitting failed to produce enough splits
        if duration > 3.0 * fps:
            rate_threshold = max(np.mean(seg_rate) * 0.3, 2.0)
            low_rate = seg_rate < rate_threshold
            in_valley = False
            valley_starts_r = []
            valley_ends_r = []
            for i in range(len(seg_rate)):
                if low_rate[i] and not in_valley:
                    in_valley = True
                    valley_starts_r.append(i)
                elif not low_rate[i] and in_valley:
                    in_valley = False
                    valley_ends_r.append(i)
            if in_valley:
                valley_ends_r.append(len(seg_rate) - 1)

            for vs, ve in zip(valley_starts_r, valley_ends_r):
                if ve - vs >= min_split_gap:
                    mid = s + (vs + ve) // 2
                    # Only add if not too close to an existing split point
                    if all(abs(mid - sp) > int(0.5 * fps) for sp in split_points):
                        split_points.append(mid)

        split_points.sort()

        if not split_points:
            sub_segs = [(s, e)]
        else:
            boundaries = [s] + split_points + [e]
            sub_segs = []
            for i in range(len(boundaries) - 1):
                sub_s = boundaries[i]
                sub_e = boundaries[i + 1]
                if sub_e - sub_s >= min_duration:
                    sub_segs.append((sub_s, sub_e))

        # Force-split any sub-segment that still exceeds max trick duration
        final_segs = []
        for ss, se in sub_segs:
            sub_dur = se - ss
            if sub_dur <= max_trick_duration:
                final_segs.append((ss, se))
            else:
                # Split at the lowest rotation rate points
                sub_rate = smooth_rate[ss:se + 1]
                n_splits = int(sub_dur / max_trick_duration)
                chunk = sub_dur // (n_splits + 1)
                force_splits = []
                for k in range(1, n_splits + 1):
                    center = k * chunk
                    win = int(0.5 * fps)
                    search_s = max(0, center - win)
                    search_e = min(len(sub_rate), center + win)
                    if search_s < search_e:
                        best = search_s + int(np.argmin(sub_rate[search_s:search_e]))
                        force_splits.append(ss + best)
                force_splits.sort()
                fb = [ss] + force_splits + [se]
                for i in range(len(fb) - 1):
                    if fb[i + 1] - fb[i] >= min_duration:
                        final_segs.append((fb[i], fb[i + 1]))

        split_segments.extend(final_segs)

    segments = split_segments

    # Filter: only keep segments that look like real tricks
    tilt_cum = tracking["tilt_cumulative"]
    twist_cum = tracking["twist_cumulative"]
    filtered = []
    for s, e in segments:
        total_tilt = abs(tilt_cum[e] - tilt_cum[s])
        total_twist = abs(twist_cum[e] - twist_cum[s])
        total_rotation = total_tilt + total_twist * 0.7
        duration_s = (e - s) / fps

        # Must have enough total rotation
        if total_rotation < min_rotation_deg:
            continue

        # Must have significant peak rotation rate (not just slow movement)
        seg_rate = smooth_rate[s:e + 1]
        peak_rate = float(np.max(seg_rate))
        if peak_rate < 8.0:
            continue

        # Short segments (< 0.5s) or segments without inversion and low rotation are likely transitions
        max_tilt = float(np.max(tilt_angle[s:e + 1]))
        if duration_s < 0.5 and total_tilt < 200:
            continue
        if duration_s < 0.8 and max_tilt < 60 and total_tilt < 180:
            continue

        filtered.append((s, e))

    return filtered


def tracking_to_signature(
    physics: dict,
    start: int,
    end: int,
    fps: float = 30.0,
    transl: np.ndarray | None = None,
) -> TrickSignature3D:
    """Convert rotation_tracker physics dict to a TrickSignature3D for matching."""
    # Primary axis from axis classification
    axis = physics["axis"]
    if axis == "lateral":
        primary_axis = np.array([1.0, 0.0, 0.0])
    elif axis == "sagittal":
        primary_axis = np.array([0.0, 0.0, 1.0])
    elif axis == "longitudinal":
        primary_axis = np.array([0.0, 1.0, 0.0])
    else:  # off_axis
        angle_rad = np.radians(physics.get("off_axis_angle", 45))
        primary_axis = np.array([np.sin(angle_rad), np.cos(angle_rad), 0.0])

    # Sign for direction
    flip_sign = 1.0 if physics["direction"] == "backward" else -1.0

    # Peak height
    peak_height = 0.0
    if transl is not None:
        heights = transl[start:end + 1, 1]
        if len(heights) > 0:
            peak_height = float(np.max(heights) - np.min(heights))

    duration = (end - start) / fps

    return TrickSignature3D(
        primary_rotation_axis=primary_axis,
        total_flip_deg=physics["flip_count"] * 360.0 * flip_sign,
        total_twist_deg=physics["twist_count"] * 360.0,
        rotation_direction=physics["direction"],
        body_shape=physics["body_shape"],
        entry_type=physics["entry"],
        peak_height_m=peak_height,
        duration_s=duration,
        start_frame=start,
        end_frame=end,
        start_time_ms=start * (1000.0 / fps),
        end_time_ms=end * (1000.0 / fps),
    )


def main():
    parser = argparse.ArgumentParser(description="3D pipeline run analysis")
    parser.add_argument("--gvhmr-output", required=True, help="Path to hmr4d_results.pt")
    parser.add_argument("--fps", type=float, default=30.0, help="Video FPS")
    parser.add_argument("--top-k", type=int, default=5, help="Top-K matches per trick")
    parser.add_argument("--min-rotation", type=float, default=120.0,
                        help="Minimum rotation degrees to count as a trick")
    parser.add_argument("--no-smooth", action="store_true",
                        help="Skip rotation smoothing (use raw GVHMR output)")
    parser.add_argument("--context", choices=["ground", "wall", "bar", "all"],
                        default="ground",
                        help="Trick context filter: ground (acrobatics), wall, bar (swing), all")
    args = parser.parse_args()

    print(f"\nPkVision — 3D Pipeline Analysis")
    print(f"{'=' * 60}")

    # Step 1: Load GVHMR output
    print(f"\n[1] Loading GVHMR output...")
    global_orient, body_pose, transl = load_gvhmr_output(args.gvhmr_output)
    T = global_orient.shape[0]
    print(f"  Loaded {T} SMPL frames ({T/args.fps:.1f}s)")

    # Step 2: Smooth rotations (compensates GVHMR jitter)
    if not args.no_smooth:
        print(f"\n[2] Smoothing GVHMR rotations (sigma=1.5)...")
        global_orient = smooth_rotations(global_orient, sigma=1.5)
    else:
        print(f"\n[2] Skipping smoothing (raw GVHMR)")

    # Step 3: Track rotation
    print(f"\n[3] Tracking 3D rotation (gravity-referenced)...")
    tracking = track_rotation(global_orient)
    print(f"  Total tilt change: {tracking['tilt_cumulative'][-1]:.1f}°")
    print(f"  Raw twist: {abs(tracking['twist_cumulative'][-1]):.1f}°")
    print(f"  Inversion crossings: {tracking['inversion_crossings']}")
    print(f"  Peak rotation rate: {tracking['rotation_rate'].max():.1f}°/frame")

    # Step 4: Segment tricks
    print(f"\n[4] Segmenting tricks from 3D rotation data...")
    segments = segment_tricks_3d(tracking, fps=args.fps, min_rotation_deg=args.min_rotation)
    print(f"  Found {len(segments)} trick segments")

    # Step 5: Match each segment
    from ml.trick_physics import TRICK_DEFINITIONS
    matcher = Matcher3D()

    context_map = {
        "ground": TrickContext.GROUND,
        "wall": TrickContext.WALL,
        "bar": TrickContext.BAR_OR_RAIL,
        "all": None,
    }
    match_context = context_map[args.context]
    context_label = args.context if args.context != "all" else "all categories"

    # Count tricks in the selected context
    if match_context:
        n_tricks = sum(1 for td in TRICK_DEFINITIONS.values()
                       if hasattr(td, "context") and td.context == match_context)
    else:
        n_tricks = len(TRICK_DEFINITIONS)

    print(f"\n[5] Matching against {n_tricks} FIG tricks ({context_label})")
    print(f"{'─' * 60}")

    detections = []
    for i, (start, end) in enumerate(segments):
        t_start = start / args.fps
        t_end = end / args.fps
        duration = t_end - t_start

        # Extract physics with Berry phase correction
        physics = extract_trick_physics(tracking, start, end, global_orient, body_pose, transl)

        # Convert to TrickSignature3D for matching
        signature = tracking_to_signature(physics, start, end, args.fps, transl)

        # Match
        matches = matcher.match(signature, top_k=args.top_k, context=match_context)

        print(f"\n  Trick #{i+1} @ {t_start:.1f}s - {t_end:.1f}s ({duration:.1f}s)")
        print(f"    Measured: {physics['flip_deg']:.0f}° flip ({physics['flip_count']:.1f}x) "
              f"+ {physics['twist_deg']:.0f}° twist ({physics['twist_count']:.1f}x)")
        print(f"    Berry phase: {physics['geometric_phase_deg']:.0f}° "
              f"(raw twist was {physics['twist_deg_raw']:.0f}°)")
        print(f"    Direction: {physics['direction']}  Shape: {physics['body_shape']}  "
              f"Entry: {physics['entry']}  Axis: {physics['axis']}")

        if matches:
            print(f"    Top matches:")
            for rank, m in enumerate(matches):
                bar = "█" * int(m.confidence * 30)
                marker = " ◄── BEST" if rank == 0 else ""
                # Show FIG score if available
                fig_score = ""
                if m.trick_id in TRICK_DEFINITIONS:
                    td = TRICK_DEFINITIONS[m.trick_id]
                    if hasattr(td, "fig_score") and td.fig_score > 0:
                        fig_score = f" [D={td.fig_score:.1f}]"
                print(f"      {rank+1}. {m.trick_name:35s} {m.confidence:5.1%} {bar}{fig_score}{marker}")

            detections.append({
                "num": i + 1,
                "start_s": round(t_start, 2),
                "end_s": round(t_end, 2),
                "flip_count": physics["flip_count"],
                "twist_count": physics["twist_count"],
                "direction": physics["direction"],
                "body_shape": physics["body_shape"],
                "entry": physics["entry"],
                "top_match": matches[0].trick_name,
                "top_match_id": matches[0].trick_id,
                "confidence": round(matches[0].confidence, 3),
            })
        else:
            print(f"    No match found")

    # Summary
    print(f"\n{'=' * 60}")
    print(f"RUN SUMMARY — 3D Physics Analysis")
    print(f"{'=' * 60}")
    total_d_score = 0.0
    if detections:
        for det in detections:
            fig = ""
            if det["top_match_id"] in TRICK_DEFINITIONS:
                td = TRICK_DEFINITIONS[det["top_match_id"]]
                if hasattr(td, "fig_score") and td.fig_score > 0:
                    fig = f" D={td.fig_score:.1f}"
                    total_d_score += td.fig_score
            print(f"  {det['num']:2d}. {det['top_match']:35s} ({det['confidence']:.0%}){fig} "
                  f"flip={det['flip_count']:.1f}x twist={det['twist_count']:.1f}x "
                  f"@ {det['start_s']:.1f}s")
        if total_d_score > 0:
            print(f"\n  Total D-Score (base): {total_d_score:.1f}")
    else:
        print("  No tricks detected.")


if __name__ == "__main__":
    main()
