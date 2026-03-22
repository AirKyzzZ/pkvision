#!/usr/bin/env python3
"""CLI tool to analyze a parkour video and display trick detections + scores.

Usage:
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
from core.models import RunAnalysis, RunStatus
from core.pose.angles import frame_result_to_analysis, get_joint_velocities
from core.pose.detector import PoseDetector
from core.recognition.classifier import TrickClassifier
from core.scoring.engine import ScoringEngine


def main():
    parser = argparse.ArgumentParser(description="PkVision — Analyze a parkour video")
    parser.add_argument("--input", "-i", required=True, help="Path to video file")
    parser.add_argument("--lang", "-l", default="en", help="Language for trick names (en/fr)")
    parser.add_argument("--output", "-o", help="Save JSON results to file")
    parser.add_argument("--confidence", "-c", type=float, default=0.5, help="Minimum confidence threshold")
    parser.add_argument("--model", "-m", help="Path to trained ST-GCN model")
    args = parser.parse_args()

    video_path = Path(args.input)
    if not video_path.exists():
        print(f"Error: Video not found: {video_path}")
        sys.exit(1)

    catalog_dir = Path(__file__).parent.parent / "data" / "tricks" / "catalog"
    model_path = args.model or (Path(__file__).parent.parent / "data" / "models" / "stgcn_best.pt")

    print("=" * 60)
    print("  PKVISION — Video Analysis")
    print("=" * 60)
    print(f"  Video:    {video_path.name}")
    print(f"  Language: {args.lang}")
    print(f"  Model:    {'ST-GCN loaded' if Path(model_path).exists() else 'angle thresholds only'}")
    print("=" * 60)

    # Initialize
    tracer = AuditTracer()
    tracer.log_system(f"Analyzing {video_path.name}")

    print("\n[1/4] Loading pose detector...")
    detector = PoseDetector()

    print("[2/4] Extracting poses from video...")
    frame_results = list(detector.process_video(video_path))
    print(f"       Extracted {len(frame_results)} frames with detected poses")

    if not frame_results:
        print("\nNo poses detected in the video. Ensure a person is visible.")
        sys.exit(0)

    # Compute angles and velocities
    frame_analyses = [frame_result_to_analysis(fr) for fr in frame_results]
    angles_seq = [fa.angles for fa in frame_analyses]
    timestamps = [fa.timestamp_ms for fa in frame_analyses]
    velocities = get_joint_velocities(angles_seq, timestamps)
    for i, vel in enumerate(velocities):
        frame_analyses[i].velocities = vel

    print("[3/4] Detecting tricks...")
    model_p = Path(model_path) if Path(model_path).exists() else None
    classifier = TrickClassifier(
        catalog_dir=catalog_dir,
        model_path=model_p,
        language=args.lang,
        confidence_threshold=args.confidence,
    )
    detections = classifier.classify(frame_analyses)

    for det in detections:
        tracer.log_detection(det)

    print(f"       Found {len(detections)} trick(s)")

    print("[4/4] Scoring...")
    trick_difficulties = {t.trick_id: t.difficulty for t in classifier.tricks}
    engine = ScoringEngine(confidence_threshold=args.confidence)
    score = engine.score(detections, trick_difficulties)
    tracer.log_scoring(score)

    # Display results
    print("\n" + "=" * 60)
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
