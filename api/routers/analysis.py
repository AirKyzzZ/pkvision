"""Video analysis endpoints."""

from __future__ import annotations

import shutil
import tempfile
import uuid
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from api.deps import get_classifier, get_db
from core.explainability.trace import AuditTracer
from core.models import RunAnalysis, RunStatus
from core.pose.angles import frame_result_to_analysis, get_joint_velocities
from core.recognition.classifier import TrickClassifier
from core.scoring.engine import ScoringEngine
from db.models import AuditEntryRecord, Detection, Run

router = APIRouter()


@router.post("/analyze")
async def analyze_video(
    video: UploadFile = File(...),
    lang: str = "en",
    db: Session = Depends(get_db),
    classifier: TrickClassifier = Depends(get_classifier),
):
    """Upload a video for trick analysis. Returns run_id for polling results."""
    run_id = str(uuid.uuid4())

    # Save uploaded video to temp file
    suffix = Path(video.filename or "video.mp4").suffix
    tmp_dir = Path(tempfile.mkdtemp())
    video_path = tmp_dir / f"{run_id}{suffix}"

    with open(video_path, "wb") as f:
        shutil.copyfileobj(video.file, f)

    # Create run record
    run = Run(id=run_id, video_path=str(video_path), status="processing", language=lang)
    db.add(run)
    db.commit()

    try:
        # Run analysis (synchronous for PoC)
        result = _run_analysis(video_path, classifier, lang)

        # Save results
        run.status = "completed"
        run.completed_at = datetime.utcnow()
        run.result_json = result.model_dump_json()

        # Save detections
        for det in result.detections:
            db_det = Detection(
                id=det.detection_id,
                run_id=run_id,
                trick_id=det.trick_id,
                confidence=det.confidence,
                start_frame=det.start_frame,
                end_frame=det.end_frame,
                start_time_ms=det.start_time_ms,
                end_time_ms=det.end_time_ms,
                strategy_used=det.strategy_used,
                is_top3=any(s.trick_id == det.trick_id for s in (result.score.top3 if result.score else [])),
            )
            db.add(db_det)

        # Save audit entries
        for entry in result.audit_trail:
            db_entry = AuditEntryRecord(
                id=entry.entry_id,
                run_id=run_id,
                entry_type=entry.entry_type.value,
                timestamp_ms=entry.timestamp_ms,
                trick_id=entry.trick_id,
                confidence=entry.confidence,
                reasoning=entry.reasoning,
                created_by=entry.created_by,
            )
            db.add(db_entry)

        db.commit()

        return {"run_id": run_id, "status": "completed"}

    except Exception as e:
        run.status = "failed"
        db.commit()
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


def _run_analysis(
    video_path: Path,
    classifier: TrickClassifier,
    lang: str,
) -> RunAnalysis:
    """Execute the full analysis pipeline on a video."""
    from core.pose.detector import PoseDetector

    tracer = AuditTracer()
    tracer.log_system(f"Starting analysis of {video_path.name}")

    # Step 1: Pose detection
    detector = PoseDetector()
    frame_results = list(detector.process_video(video_path))
    tracer.log_system(f"Extracted poses from {len(frame_results)} frames")

    if not frame_results:
        return RunAnalysis(
            video_path=str(video_path),
            status=RunStatus.COMPLETED,
            language=lang,
            audit_trail=tracer.entries,
        )

    # Step 2: Compute angles and velocities
    frame_analyses = [frame_result_to_analysis(fr) for fr in frame_results]

    # Compute velocities
    angles_seq = [fa.angles for fa in frame_analyses]
    timestamps = [fa.timestamp_ms for fa in frame_analyses]
    velocities = get_joint_velocities(angles_seq, timestamps)

    for i, vel in enumerate(velocities):
        frame_analyses[i].velocities = vel

    tracer.log_system(f"Computed angles and velocities for {len(frame_analyses)} frames")

    # Step 3: Classify tricks
    detections = classifier.classify(frame_analyses)
    tracer.log_system(f"Detected {len(detections)} tricks")

    for det in detections:
        tracer.log_detection(det)

    # Step 4: Score
    trick_difficulties = {t.trick_id: t.difficulty for t in classifier.tricks}
    engine = ScoringEngine()
    score = engine.score(detections, trick_difficulties)
    tracer.log_scoring(score)

    return RunAnalysis(
        video_path=str(video_path),
        status=RunStatus.COMPLETED,
        language=lang,
        detections=detections,
        score=score,
        audit_trail=tracer.entries,
        completed_at=datetime.utcnow(),
    )
