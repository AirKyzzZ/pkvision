#!/usr/bin/env python3
"""PkVision Full Pipeline — Video to 3D Render + Trick Detection + FIG Scoring.

End-to-end pipeline:
  1. Load GVHMR output (hmr4d_results.pt)
  2. Generate SMPL mesh vertices
  3. Render 3D mesh overlay video (in-camera) + global 3D view
  4. Track rotations with Berry phase correction
  5. Segment individual tricks
  6. Extract physics per trick (flips, twists, direction, shape, entry)
  7. Match each trick to FIG Code of Points 2025
  8. Select best 3 tricks -> compute final D-score
  9. Output: rendered videos + OBJ meshes + JSON scoring report

Usage:
    # Full pipeline with rendering + analysis:
    python scripts/pkvision_full.py \\
        --gvhmr-output C:/Users/pc/GVHMR/outputs/demo/backflip/hmr4d_results.pt \\
        --video C:/Users/pc/GVHMR/outputs/demo/backflip/0_input_video.mp4 \\
        --output-dir outputs/backflip

    # Analysis only (skip rendering):
    python scripts/pkvision_full.py \\
        --gvhmr-output C:/Users/pc/GVHMR/outputs/demo/IMG_4243/hmr4d_results.pt \\
        --no-render \\
        --output-dir outputs/IMG_4243

    # Export OBJ meshes for Blender:
    python scripts/pkvision_full.py \\
        --gvhmr-output C:/Users/pc/GVHMR/outputs/demo/backflip/hmr4d_results.pt \\
        --export-obj --obj-every-n 5 \\
        --output-dir outputs/backflip
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.pose.rotation_tracker import (
    extract_trick_physics,
    smooth_rotations,
    track_rotation,
)
from core.recognition.matcher import Matcher3D
from ml.trick_physics import TRICK_DEFINITIONS, TrickContext
from scripts.analyze_3d import load_gvhmr_output, segment_tricks_3d, tracking_to_signature
from core.scoring.competition import CompetitionScorer


def main():
    parser = argparse.ArgumentParser(
        description="PkVision Full Pipeline: Video -> 3D Render + Tricks + FIG Score",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Input
    parser.add_argument("--gvhmr-output", required=True,
                        help="Path to GVHMR hmr4d_results.pt")
    parser.add_argument("--video", default=None,
                        help="Original video for overlay (auto-detected if not set)")
    parser.add_argument("--fps", type=float, default=30.0)

    # Output
    parser.add_argument("--output-dir", default="outputs/pkvision",
                        help="Directory for all outputs")

    # Rendering
    parser.add_argument("--no-render", action="store_true",
                        help="Skip video rendering (analysis only)")
    parser.add_argument("--render-width", type=int, default=None,
                        help="Render width (auto from video if not set)")
    parser.add_argument("--render-height", type=int, default=None,
                        help="Render height (auto from video if not set)")
    parser.add_argument("--render-global", action="store_true",
                        help="Also render global 3rd-person view")

    # OBJ Export
    parser.add_argument("--export-obj", action="store_true",
                        help="Export mesh frames as .obj files")
    parser.add_argument("--obj-every-n", type=int, default=5,
                        help="Export every N-th frame as OBJ")

    # Analysis
    parser.add_argument("--context", choices=["ground", "wall", "bar", "all"],
                        default="all",
                        help="Trick context filter (default: all)")
    parser.add_argument("--min-rotation", type=float, default=120.0,
                        help="Minimum rotation degrees to count as a trick")
    parser.add_argument("--no-smooth", action="store_true",
                        help="Skip rotation smoothing")
    parser.add_argument("--top-k", type=int, default=5,
                        help="Top-K FIG matches per trick")

    args = parser.parse_args()

    t0 = time.time()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print()
    print("=" * 70)
    print("  PkVision Full Pipeline")
    print("  Video -> 3D Mesh + Trick Detection + FIG Competition Score")
    print("=" * 70)

    # ── Step 1: Load GVHMR output ──────────────────────────────────
    print(f"\n[1/8] Loading GVHMR output...")
    global_orient, body_pose, transl = load_gvhmr_output(args.gvhmr_output)
    T = global_orient.shape[0]
    duration = T / args.fps
    print(f"  {T} frames ({duration:.1f}s at {args.fps}fps)")

    # Auto-detect video path
    video_path = args.video
    if video_path is None:
        auto = Path(args.gvhmr_output).parent / "0_input_video.mp4"
        if auto.exists():
            video_path = str(auto)
            print(f"  Auto-detected video: {video_path}")

    # ── Step 2: Generate mesh vertices ─────────────────────────────
    render_enabled = not args.no_render or args.export_obj

    verts_incam = None
    verts_global = None
    faces = None
    K = None

    if render_enabled:
        print(f"\n[2/8] Generating SMPL mesh vertices...")
        from core.rendering.smpl_mesh import SMPLMeshGenerator

        gen = SMPLMeshGenerator()
        gen.load_gvhmr(args.gvhmr_output)
        K = gen.K

        if not args.no_render:
            print(f"  Generating in-camera vertices ({T} frames)...")
            verts_incam, faces = gen.generate(space="incam")
            print(f"  Vertices: {verts_incam.shape} ({verts_incam.nbytes / 1e6:.1f} MB)")

        if args.render_global or args.export_obj:
            print(f"  Generating global vertices ({T} frames)...")
            verts_global, faces = gen.generate(space="global")

        if faces is None:
            faces = gen.faces
    else:
        print(f"\n[2/8] Skipping mesh generation (--no-render)")

    # ── Step 3: Render videos ──────────────────────────────────────
    if not args.no_render and verts_incam is not None:
        print(f"\n[3/8] Rendering 3D mesh overlay...")
        import cv2
        from core.rendering.renderer import MeshRenderer

        # Determine resolution
        w, h = args.render_width, args.render_height
        if w is None or h is None:
            if video_path:
                cap = cv2.VideoCapture(video_path)
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap.release()
            else:
                w, h = 1920, 1080
        print(f"  Resolution: {w}x{h}")

        renderer = MeshRenderer(w, h)

        # In-camera overlay
        incam_path = str(out_dir / "render_incam.mp4")
        print(f"  Rendering in-camera overlay -> {incam_path}")
        renderer.render_video(
            verts_incam, faces, incam_path,
            K=K, video_path=video_path, mode="incam", fps=args.fps,
        )

        # Global view
        if args.render_global and verts_global is not None:
            global_path = str(out_dir / "render_global.mp4")
            print(f"  Rendering global 3D view -> {global_path}")
            renderer.render_video(
                verts_global, faces, global_path,
                mode="global", fps=args.fps,
            )
    else:
        print(f"\n[3/8] Skipping rendering")

    # ── Step 4: Export OBJ meshes ──────────────────────────────────
    if args.export_obj:
        verts_for_export = verts_global if verts_global is not None else verts_incam
        if verts_for_export is not None and faces is not None:
            print(f"\n[4/8] Exporting OBJ meshes (every {args.obj_every_n} frames)...")
            from core.rendering.smpl_mesh import SMPLMeshGenerator
            gen_export = SMPLMeshGenerator.__new__(SMPLMeshGenerator)
            obj_dir = str(out_dir / "meshes")
            # Reuse the actual gen if available
            if render_enabled:
                paths = gen.export_sequence(
                    verts_for_export, faces, obj_dir,
                    every_n=args.obj_every_n, fmt="obj",
                )
            else:
                # Manual export via trimesh
                import trimesh
                obj_dir_path = Path(obj_dir)
                obj_dir_path.mkdir(parents=True, exist_ok=True)
                paths = []
                for i in range(0, len(verts_for_export), args.obj_every_n):
                    p = str(obj_dir_path / f"frame_{i:05d}.obj")
                    m = trimesh.Trimesh(vertices=verts_for_export[i], faces=faces, process=False)
                    m.export(p)
                    paths.append(p)
            print(f"  Exported {len(paths)} OBJ files to {obj_dir}/")
        else:
            print(f"\n[4/8] No vertices for OBJ export")
    else:
        print(f"\n[4/8] Skipping OBJ export (use --export-obj)")

    # ── Step 5: Rotation tracking ──────────────────────────────────
    print(f"\n[5/8] Tracking 3D rotation (Berry phase corrected)...")
    if not args.no_smooth:
        # Adaptive smoothing: measure signal noise and adjust sigma
        deltas = np.linalg.norm(np.diff(global_orient, axis=0), axis=1)
        median_delta = float(np.median(deltas))
        if median_delta > 0.15:
            sigma = 3.0
            quality = "low"
        elif median_delta > 0.08:
            sigma = 2.0
            quality = "medium"
        else:
            sigma = 1.5
            quality = "high"
        print(f"  GVHMR signal quality: {quality} (jitter={median_delta:.3f}, sigma={sigma})")
        global_orient = smooth_rotations(global_orient, sigma=sigma)
    tracking = track_rotation(global_orient)
    print(f"  Total tilt: {tracking['tilt_cumulative'][-1]:.0f}deg")
    print(f"  Raw twist: {abs(tracking['twist_cumulative'][-1]):.0f}deg")
    print(f"  Peak rate: {tracking['rotation_rate'].max():.1f}deg/frame")

    # ── Step 6: Segment tricks ─────────────────────────────────────
    print(f"\n[6/8] Segmenting tricks...")
    segments = segment_tricks_3d(tracking, fps=args.fps, min_rotation_deg=args.min_rotation)
    print(f"  Found {len(segments)} trick segments")

    # ── Step 7: Match to FIG Code of Points ────────────────────────
    print(f"\n[7/8] Matching tricks to FIG Code of Points 2025...")
    matcher = Matcher3D()
    context_map = {
        "ground": TrickContext.GROUND,
        "wall": TrickContext.WALL,
        "bar": TrickContext.BAR_OR_RAIL,
        "all": None,
    }
    match_context = context_map[args.context]

    trick_results = []
    print(f"  {'-' * 64}")

    for i, (start, end) in enumerate(segments):
        t_start = start / args.fps
        t_end = end / args.fps

        # Extract physics
        physics = extract_trick_physics(tracking, start, end, global_orient, body_pose, transl)

        # Match
        signature = tracking_to_signature(physics, start, end, args.fps, transl)
        matches = matcher.match(signature, top_k=args.top_k, context=match_context)

        # Get FIG D-score for best match
        d_score = 0.0
        best_name = "Unknown"
        best_id = "unknown"
        confidence = 0.0

        if matches:
            best = matches[0]
            best_name = best.trick_name
            best_id = best.trick_id
            confidence = best.confidence
            td = TRICK_DEFINITIONS.get(best_id)
            if td and hasattr(td, "fig_score"):
                d_score = td.fig_score

        trick_result = {
            "num": i + 1,
            "trick_name": best_name,
            "trick_id": best_id,
            "d_score": d_score,
            "confidence": round(confidence, 3),
            "flip_count": physics["flip_count"],
            "twist_count": physics["twist_count"],
            "direction": physics["direction"],
            "body_shape": physics["body_shape"],
            "axis": physics["axis"],
            "entry": physics["entry"],
            "start_s": round(t_start, 2),
            "end_s": round(t_end, 2),
            "start_frame": start,
            "end_frame": end,
            "flip_deg": round(physics["flip_deg"], 1),
            "twist_deg": round(physics["twist_deg"], 1),
            "berry_phase_deg": round(physics["geometric_phase_deg"], 1),
            "max_tilt": round(physics["max_tilt"], 1),
            "went_inverted": physics["went_inverted"],
        }
        trick_results.append(trick_result)

        # Print per-trick info
        inv = "INV" if physics["went_inverted"] else "   "
        print(f"\n  Trick #{i+1} [{t_start:.1f}s - {t_end:.1f}s] {inv}")
        print(f"    Physics: {physics['flip_count']:.1f} flips + {physics['twist_count']:.1f} twists "
              f"({physics['direction']}, {physics['axis']}, {physics['body_shape']})")
        print(f"    Berry phase correction: {physics['geometric_phase_deg']:.0f}deg "
              f"(raw twist: {physics['twist_deg_raw']:.0f}deg)")

        if matches:
            for rank, m in enumerate(matches[:3]):
                td = TRICK_DEFINITIONS.get(m.trick_id)
                fig = f" D={td.fig_score:.1f}" if td and hasattr(td, "fig_score") and td.fig_score > 0 else ""
                bar = "#" * int(m.confidence * 25)
                marker = " <-- BEST" if rank == 0 else ""
                print(f"    {rank+1}. {m.trick_name:30s} {m.confidence:5.1%} {bar}{fig}{marker}")

    # ── Step 8: Competition scoring ────────────────────────────────
    print(f"\n[8/8] Computing competition D-score (best {3} unique tricks)...")
    scorer = CompetitionScorer()
    result = scorer.score_run(trick_results)

    print(f"\n{'=' * 70}")
    print(f"  COMPETITION RESULT")
    print(f"{'=' * 70}")
    print(f"\n  Tricks detected: {result.total_tricks_detected}")
    print(f"  Unique tricks:   {result.unique_tricks}")
    if result.repeated_tricks:
        print(f"  Repeated (excluded): {', '.join(result.repeated_tricks)}")

    print(f"\n  TOP {len(result.top3)} TRICKS:")
    print(f"  {'-' * 60}")
    for i, t in enumerate(result.top3):
        penalties = f" ({'; '.join(t.penalties)})" if t.penalties else ""
        print(f"  {i+1}. {t.trick_name:30s}  D={t.adjusted_d_score:.1f}  "
              f"({t.confidence:.0%} conf){penalties}")
        print(f"     {t.flip_count:.1f} flips + {t.twist_count:.1f} twists, "
              f"{t.direction}, {t.body_shape} @ {t.start_s:.1f}s")

    print(f"\n  +======================================+")
    print(f"  |  FINAL D-SCORE:  {result.final_d_score:5.1f}              |")
    print(f"  +======================================+")

    # ── Save JSON report ───────────────────────────────────────────
    report = {
        "pipeline": "PkVision v3 Full Pipeline",
        "input": {
            "gvhmr_output": args.gvhmr_output,
            "video": video_path,
            "frames": T,
            "duration_s": round(duration, 2),
            "fps": args.fps,
        },
        "scoring": {
            "final_d_score": result.final_d_score,
            "top3": [
                {
                    "rank": i + 1,
                    "trick_name": t.trick_name,
                    "trick_id": t.trick_id,
                    "base_d_score": t.d_score,
                    "adjusted_d_score": t.adjusted_d_score,
                    "confidence": t.confidence,
                    "penalties": t.penalties,
                    "timing": f"{t.start_s:.1f}s - {t.end_s:.1f}s",
                }
                for i, t in enumerate(result.top3)
            ],
            "total_tricks_detected": result.total_tricks_detected,
            "unique_tricks": result.unique_tricks,
            "repeated_excluded": result.repeated_tricks,
        },
        "tricks": trick_results,
        "outputs": {},
    }

    # Add output paths
    if not args.no_render and verts_incam is not None:
        report["outputs"]["render_incam"] = str(out_dir / "render_incam.mp4")
    if args.render_global:
        report["outputs"]["render_global"] = str(out_dir / "render_global.mp4")
    if args.export_obj:
        report["outputs"]["meshes_dir"] = str(out_dir / "meshes")

    report_path = str(out_dir / "report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Report saved: {report_path}")

    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed:.1f}s")
    print()


if __name__ == "__main__":
    main()
