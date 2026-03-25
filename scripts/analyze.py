#!/usr/bin/env python3
"""CLI tool to analyze a parkour video and display trick detections + scores.

Supports three detection strategies:
  - angle:    Angle threshold only (baseline, no training needed)
  - ensemble: DTW + MLP ensemble (few-shot, needs reference clips)
  - auto:     Ensemble if references/model exist, else angle threshold

Usage:
    # Basic (auto-detects best strategy)
    python scripts/analyze.py --input video.mp4

    # With ensemble (DTW + MLP)
    python scripts/analyze.py --input video.mp4 --strategy ensemble

    # French output with JSON export
    python scripts/analyze.py --input video.mp4 --lang fr --output results.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.explainability.trace import AuditTracer
from core.models import RunAnalysis, RunStatus, TrickConfig, TrickDetection
from core.pose.angles import frame_result_to_analysis, get_joint_velocities
from core.recognition.classifier import TrickClassifier
from core.recognition.ensemble import EnsembleStrategy
from core.scoring.engine import ScoringEngine


def _load_ensemble(project_root: Path, confidence: float):
    """Try to load DTW and/or MLP strategies for ensemble."""
    dtw_strategy = None
    mlp_strategy = None

    references_dir = project_root / "data" / "references"
    if references_dir.exists() and any(references_dir.iterdir()):
        from core.recognition.strategies.dtw import DTWStrategy
        dtw_strategy = DTWStrategy(
            references_dir=references_dir,
            min_confidence=confidence * 0.5,  # Lower threshold — ensemble combines later
        )
        print(f"       DTW: {len(dtw_strategy.loaded_tricks)} tricks with references")

    mlp_path = project_root / "data" / "models" / "mlp_v1.pt"
    if mlp_path.exists():
        try:
            from ml.mlp.inference import MLPStrategy
            mlp_strategy = MLPStrategy(
                checkpoint_path=mlp_path,
                min_confidence=confidence * 0.5,
            )
            if mlp_strategy.is_loaded():
                print(f"       MLP: {len(mlp_strategy.class_names)} classes loaded")
            else:
                mlp_strategy = None
        except ImportError:
            print("       MLP: torch not available, skipping")

    if dtw_strategy is None and mlp_strategy is None:
        return None

    # DTW-heavy weights: DTW generalizes better with few reference clips.
    # Shift toward MLP (0.6/0.4) once you have 5+ diverse clips per trick.
    return EnsembleStrategy(
        dtw=dtw_strategy,
        mlp=mlp_strategy,
        mlp_weight=0.3,
        dtw_weight=0.7,
        min_confidence=confidence,
    )


def _classify_with_ensemble(
    ensemble: EnsembleStrategy,
    tricks: list[TrickConfig],
    frame_analyses: list,
) -> list[TrickDetection]:
    """Classify frames using the ensemble, trying segmentation first."""
    detections: list[TrickDetection] = []

    # Try segmentation
    try:
        from core.recognition.segmentation import RunSegmenter
        segmenter = RunSegmenter()
        segments = segmenter.segment(frame_analyses)
        print(f"       Segmented into {len(segments)} active zone(s)")
    except ImportError:
        segments = None

    if segments:
        for seg in segments:
            seg_frames = frame_analyses[seg.start_frame:seg.end_frame + 1]
            if not seg_frames:
                continue
            for trick in tricks:
                det = ensemble.evaluate(trick, seg_frames)
                if det is not None:
                    detections.append(det)
    else:
        # Fallback: classify entire sequence
        for trick in tricks:
            det = ensemble.evaluate(trick, frame_analyses)
            if det is not None:
                detections.append(det)

    return detections


def main():
    parser = argparse.ArgumentParser(description="PkVision — Analyze a parkour video")
    parser.add_argument("--input", "-i", required=True, help="Path to video file")
    parser.add_argument("--lang", "-l", default="en", help="Language for trick names (en/fr)")
    parser.add_argument("--output", "-o", help="Save JSON results to file")
    parser.add_argument("--confidence", "-c", type=float, default=0.5,
                        help="Minimum confidence threshold")
    parser.add_argument("--model", "-m", help="Path to trained ST-GCN model (legacy)")
    parser.add_argument("--strategy", "-s", default="auto",
                        choices=["auto", "angle", "ensemble"],
                        help="Detection strategy: auto, angle, or ensemble")
    args = parser.parse_args()

    video_path = Path(args.input)
    if not video_path.exists():
        print(f"Error: Video not found: {video_path}")
        sys.exit(1)

    project_root = Path(__file__).parent.parent
    catalog_dir = project_root / "data" / "tricks" / "catalog"

    print("=" * 60)
    print("  PKVISION — Video Analysis")
    print("=" * 60)
    print(f"  Video:    {video_path.name}")
    print(f"  Language: {args.lang}")
    print(f"  Strategy: {args.strategy}")
    print("=" * 60)

    # Initialize
    tracer = AuditTracer()
    tracer.log_system(f"Analyzing {video_path.name}")

    # Step 1: Pose estimation
    print("\n[1/5] Loading pose detector...")
    from core.pose.detector import PoseDetector
    detector = PoseDetector()

    print("[2/5] Extracting poses from video...")
    frame_results = list(detector.process_video(video_path))
    print(f"       Extracted {len(frame_results)} frames with detected poses")

    if not frame_results:
        print("\nNo poses detected in the video. Ensure a person is visible.")
        sys.exit(0)

    # Step 2: Compute angles and velocities
    frame_analyses = [frame_result_to_analysis(fr) for fr in frame_results]
    angles_seq = [fa.angles for fa in frame_analyses]
    timestamps = [fa.timestamp_ms for fa in frame_analyses]
    velocities = get_joint_velocities(angles_seq, timestamps)
    for i, vel in enumerate(velocities):
        frame_analyses[i].velocities = vel

    # Step 3: Load trick catalog
    classifier = TrickClassifier(
        catalog_dir=catalog_dir,
        model_path=args.model,
        language=args.lang,
        confidence_threshold=args.confidence,
    )
    tracer.log_system(f"Loaded {len(classifier.tricks)} tricks from catalog")

    # Step 4: Detection
    print("[3/5] Detecting tricks...")
    detections: list[TrickDetection] = []
    use_ensemble = args.strategy == "ensemble" or args.strategy == "auto"

    if use_ensemble:
        ensemble = _load_ensemble(project_root, args.confidence)
        if ensemble is not None:
            tracer.log_system("Using DTW+MLP ensemble strategy")
            detections = _classify_with_ensemble(ensemble, classifier.tricks, frame_analyses)
        elif args.strategy == "ensemble":
            print("       WARNING: Ensemble requested but no references/model found.")
            print("       Falling back to angle threshold strategy.")

    if not detections:
        # Fallback to angle threshold
        tracer.log_system("Using angle threshold strategy")
        detections = classifier.classify(frame_analyses)

    for det in detections:
        tracer.log_detection(det)

    print(f"       Found {len(detections)} trick(s)")

    # Step 5: Scoring
    print("[4/5] Scoring...")
    trick_difficulties = {t.trick_id: t.difficulty for t in classifier.tricks}
    engine = ScoringEngine(confidence_threshold=args.confidence)
    score = engine.score(detections, trick_difficulties)
    tracer.log_scoring(score)

    # Display results
    print("\n[5/5] Results")
    print("=" * 60)
    print("  RESULTS")
    print("=" * 60)

    if not score.top3:
        print("\n  No tricks detected above confidence threshold.")
    else:
        print(f"\n  Top {len(score.top3)} Tricks:")
        print("  " + "-" * 56)
        for i, trick in enumerate(score.top3, 1):
            print(
                f"  #{i}  {trick.trick_name:25s} "
                f"diff={trick.difficulty:<4.1f}  "
                f"conf={trick.confidence:.0%}  "
                f"score={trick.weighted_score:.2f}"
            )
        print("  " + "-" * 56)
        print(f"  Total Score: {score.total_score:.2f} / {score.max_possible_score:.2f}")

    # All detections
    if detections:
        print(f"\n  All Detections ({len(detections)}):")
        for det in detections:
            t_start = det.start_time_ms / 1000.0
            t_end = det.end_time_ms / 1000.0
            print(
                f"    [{t_start:.1f}s - {t_end:.1f}s] {det.trick_name} "
                f"({det.confidence:.0%}) via {det.strategy_used}"
            )

    # Audit trail
    print("\n" + tracer.format_human_readable())

    # Save JSON output
    if args.output:
        result = RunAnalysis(
            video_path=str(video_path),
            status=RunStatus.COMPLETED,
            language=args.lang,
            detections=detections,
            score=score,
            audit_trail=tracer.entries,
            completed_at=datetime.utcnow(),
        )
        output_path = Path(args.output)
        with open(output_path, "w") as f:
            f.write(result.model_dump_json(indent=2))
        print(f"\nResults saved to: {output_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
