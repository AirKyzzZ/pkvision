#!/usr/bin/env python3
"""VLM-based parkour trick judge — Phase 1 of PkVision.

End-to-end pipeline:
  Video → YOLO-pose → Inversion segmentation → Time-crop clips
  → Gemini (native video) → FIG match → D-score scorecard

Usage:
    # Analyze a full competition run
    python scripts/vlm_judge.py data/run_testing/IMG_5985.mov

    # Single trick clip (skip segmentation)
    python scripts/vlm_judge.py data/final_clips/backflip.mp4 --single-trick

    # Preview segmentation without calling VLM
    python scripts/vlm_judge.py data/run_testing/IMG_5985.mov --preview

    # Use pre-reviewed clips
    python scripts/vlm_judge.py --clips-dir /path/to/clips/

    # Switch model
    python scripts/vlm_judge.py video.mp4 --model gemini-2.5-flash
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

import numpy as np

from core.video import read_video, detect_and_track, smooth_boxes, save_clip
from core.segmentation import segment_tricks, TrickSegment
from core.vlm.base import VLMResponse
from core.vlm.fig_matcher import FIGMatcher


def load_yolo():
    """Load YOLO pose model."""
    from ultralytics import YOLO
    yolo_path = ROOT / "yolo11n-pose.pt"
    if yolo_path.exists():
        return YOLO(str(yolo_path))
    return YOLO("yolo11n-pose.pt")


def create_provider(provider: str, model: str):
    """Create the VLM provider."""
    if provider == "openrouter":
        from core.vlm.openrouter_provider import OpenRouterProvider
        return OpenRouterProvider(model=model)
    elif provider == "gemini":
        from core.vlm.gemini_provider import GeminiProvider
        return GeminiProvider(model=model)
    elif provider == "claude":
        from core.vlm.claude_provider import ClaudeProvider
        return ClaudeProvider(model=model)
    else:
        raise ValueError(f"Unknown provider: {provider}. Supported: openrouter, gemini, claude")


def segment_video(video_path: Path) -> tuple[list[np.ndarray], float, list[TrickSegment]]:
    """Read video, detect athlete, segment tricks."""
    print(f"Reading video: {video_path.name}")
    frames, fps = read_video(video_path)
    print(f"  {len(frames)} frames, {len(frames)/fps:.1f}s @ {fps:.0f}fps")

    print("Running YOLO-pose tracking...", end=" ", flush=True)
    yolo = load_yolo()
    tracking = detect_and_track(frames, yolo)
    detected = int(np.sum(~np.isnan(tracking.boxes[:, 0])))
    print(f"{detected}/{len(frames)} detections")

    tracking.boxes = smooth_boxes(tracking.boxes)

    print("Segmenting tricks...", end=" ", flush=True)
    segments = segment_tricks(
        tracking.head_y, tracking.hip_y,
        tracking.body_angle, fps,
    )
    print(f"{len(segments)} tricks found")

    return frames, fps, segments


def save_segments_as_clips(
    frames: list[np.ndarray],
    fps: float,
    segments: list[TrickSegment],
    output_dir: Path,
) -> list[Path]:
    """Save each trick segment as an MP4 clip."""
    output_dir.mkdir(parents=True, exist_ok=True)
    clip_paths = []
    for seg in segments:
        clip_frames = frames[seg.start_frame:seg.end_frame + 1]
        clip_path = output_dir / f"trick_{seg.index + 1:02d}.mp4"
        save_clip(clip_frames, fps, clip_path)
        clip_paths.append(clip_path)
    return clip_paths


def print_preview(segments: list[TrickSegment], fps: float, clip_paths: list[Path]):
    """Print segment timing info for user review."""
    print(f"\n{'='*60}")
    print(f"SEGMENTATION PREVIEW — {len(segments)} tricks detected")
    print(f"{'='*60}")
    for seg, path in zip(segments, clip_paths):
        start_s = seg.start_frame / fps
        end_s = seg.end_frame / fps
        print(f"  Trick {seg.index + 1}: {start_s:.1f}s → {end_s:.1f}s "
              f"(duration: {seg.duration:.1f}s, peak_inv: {seg.peak_inversion:.2f})")
        print(f"    Saved: {path}")
    print(f"\nReview clips, then re-run without --preview to send to VLM.")


def analyze_clips(
    clip_paths: list[Path],
    provider,
    matcher: FIGMatcher,
) -> list[dict]:
    """Send each clip to VLM and match to FIG tricks."""
    results = []
    total_in_tokens = 0
    total_out_tokens = 0

    for i, clip_path in enumerate(clip_paths):
        print(f"\nAnalyzing trick {i + 1}/{len(clip_paths)}: {clip_path.name}")
        print(f"  Uploading to {provider.model}...", end=" ", flush=True)

        response: VLMResponse = provider.analyze_trick(clip_path)
        total_in_tokens += response.input_tokens
        total_out_tokens += response.output_tokens
        print(f"done ({response.input_tokens} in, {response.output_tokens} out tokens)")

        for trick in response.tricks:
            fig_match = matcher.match(
                trick.trick_name,
                flip_count=trick.flip_count,
                twist_count=trick.twist_count,
                direction=trick.direction,
            )

            result = {
                "clip": clip_path.name,
                "vlm_name": trick.trick_name,
                "vlm_confidence": trick.confidence,
                "vlm_reasoning": trick.reasoning,
                "vlm_flips": trick.flip_count,
                "vlm_twists": trick.twist_count,
                "vlm_category": trick.category,
                "fig_name": fig_match.fig_name if fig_match else trick.trick_name,
                "d_score": fig_match.d_score if fig_match else 0.0,
                "match_level": fig_match.match_level if fig_match else "unmatched",
                "match_confidence": fig_match.confidence if fig_match else 0.0,
            }
            results.append(result)

            status = "✓" if fig_match else "✗ NO FIG MATCH"
            print(f"  → {result['fig_name']} (D={result['d_score']}) "
                  f"[{result['match_level']}] {status}")
            if trick.reasoning:
                print(f"    Reasoning: {trick.reasoning}")

    print(f"\nToken usage: {total_in_tokens} input + {total_out_tokens} output")
    return results


def print_scorecard(results: list[dict]):
    """Print the final D-score scorecard."""
    print(f"\n{'='*60}")
    print(f"D-SCORE SCORECARD")
    print(f"{'='*60}")
    print(f"{'#':<4} {'Trick':<30} {'D-Score':<10} {'Confidence':<12} {'Match':<15}")
    print(f"{'-'*4} {'-'*30} {'-'*10} {'-'*12} {'-'*15}")

    total_d = 0.0
    for i, r in enumerate(results):
        total_d += r["d_score"]
        print(f"{i+1:<4} {r['fig_name']:<30} {r['d_score']:<10.1f} "
              f"{r['vlm_confidence']:<12} {r['match_level']:<15}")

    print(f"{'-'*71}")
    print(f"{'TOTAL D-SCORE':<34} {total_d:<10.1f}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="VLM-based parkour trick judge")
    parser.add_argument("video", nargs="?", help="Path to video file")
    parser.add_argument("--single-trick", action="store_true",
                        help="Treat entire video as one trick (skip segmentation)")
    parser.add_argument("--preview", action="store_true",
                        help="Save segmented clips for review, don't call VLM")
    parser.add_argument("--clips-dir",
                        help="Use pre-saved clips instead of segmenting")
    parser.add_argument("--output-dir", default=None,
                        help="Where to save clips (default: temp dir or data/vlm_clips/)")
    parser.add_argument("--provider", default="openrouter",
                        choices=["openrouter", "gemini", "claude"],
                        help="VLM provider (default: openrouter)")
    parser.add_argument("--model", default=None,
                        help="Model name (default: auto per provider)")
    parser.add_argument("--consensus", nargs="*", default=None,
                        metavar="MODEL",
                        help="Use multi-model consensus (optional: specify models)")
    args = parser.parse_args()

    if not args.video and not args.clips_dir:
        parser.error("Either video path or --clips-dir is required")

    # Default model per provider
    if args.model is None:
        args.model = {
            "openrouter": "qwen/qwen3.5-omni",
            "claude": "claude-sonnet-4-20250514",
            "gemini": "gemini-2.5-pro",
        }[args.provider]

    matcher = FIGMatcher()

    # Mode 1: Use pre-reviewed clips
    if args.clips_dir:
        clips_dir = Path(args.clips_dir)
        clip_paths = sorted(clips_dir.glob("*.mp4")) + sorted(clips_dir.glob("*.mov"))
        if not clip_paths:
            print(f"No video clips found in {clips_dir}")
            return
        print(f"Using {len(clip_paths)} pre-reviewed clips from {clips_dir}")
        provider = create_provider(args.provider, args.model)
        results = analyze_clips(clip_paths, provider, matcher)
        print_scorecard(results)
        return

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Video not found: {video_path}")
        return

    # Mode 2: Single trick (whole video = one clip)
    if args.single_trick:
        print(f"Single-trick mode: {video_path.name}")
        if args.consensus is not None:
            from core.vlm.consensus import ConsensusJudge
            from core.vlm.openrouter_provider import DEFAULT_BENCHMARK_MODELS
            models = args.consensus if args.consensus else DEFAULT_BENCHMARK_MODELS
            judge = ConsensusJudge(models=models)
            result = judge.judge(video_path)
            print(f"\n  Consensus: {result.consensus_name} (D={result.d_score}) "
                  f"[{result.confidence}, {result.agreement:.0%} agreement]")
            print(f"  Votes: {[v.fig_name for v in result.votes]}")
            return
        provider = create_provider(args.provider, args.model)
        results = analyze_clips([video_path], provider, matcher)
        print_scorecard(results)
        return

    # Mode 3: Full run — segment, optionally preview, then analyze
    frames, fps, segments = segment_video(video_path)

    if not segments:
        print("No tricks detected in the video.")
        return

    # Save clips
    if args.output_dir:
        output_dir = Path(args.output_dir)
    elif args.preview:
        output_dir = ROOT / "data" / "vlm_clips" / video_path.stem
    else:
        output_dir = Path(tempfile.mkdtemp(prefix="pkvision_"))

    clip_paths = save_segments_as_clips(frames, fps, segments, output_dir)

    if args.preview:
        print_preview(segments, fps, clip_paths)
        return

    # Analyze with VLM
    provider = create_provider(args.provider, args.model)
    results = analyze_clips(clip_paths, provider, matcher)
    print_scorecard(results)


if __name__ == "__main__":
    main()
