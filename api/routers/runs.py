"""Run management and judge override endpoints."""

from __future__ import annotations

import json
import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from api.deps import get_db
from core.models import RunAnalysis
from db.models import AuditEntryRecord, Run

router = APIRouter()


class OverrideRequest(BaseModel):
    trick_id: str
    new_confidence: float
    judge_name: str
    reason: str


@router.get("/runs/{run_id}")
async def get_run(run_id: str, db: Session = Depends(get_db)):
    """Get full analysis results for a run."""
    run = db.query(Run).filter(Run.id == run_id).first()
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")

    result = None
    if run.result_json:
        result = json.loads(run.result_json)

    return {
        "run_id": run.id,
        "status": run.status,
        "video_path": run.video_path,
        "language": run.language,
        "created_at": run.created_at.isoformat() if run.created_at else None,
        "completed_at": run.completed_at.isoformat() if run.completed_at else None,
        "result": result,
    }


@router.post("/runs/{run_id}/override")
async def override_detection(
    run_id: str,
    override: OverrideRequest,
    db: Session = Depends(get_db),
):
    """Judge override — logs a new audit entry without modifying the original AI decision."""
    run = db.query(Run).filter(Run.id == run_id).first()
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")

    if run.status != "completed":
        raise HTTPException(status_code=400, detail="Can only override completed runs")

    # Find the original detection confidence
    original_confidence = None
    if run.result_json:
        result = json.loads(run.result_json)
        for det in result.get("detections", []):
            if det.get("trick_id") == override.trick_id:
                original_confidence = det.get("confidence", 0.0)
                break

    # Create audit entry for the override
    entry = AuditEntryRecord(
        id=str(uuid.uuid4()),
        run_id=run_id,
        entry_type="override",
        trick_id=override.trick_id,
        confidence=override.new_confidence,
        reasoning=(
            f"Judge override by {override.judge_name}: "
            f"{override.trick_id} confidence "
            f"{original_confidence:.1%} → {override.new_confidence:.1%} | "
            f"Reason: {override.reason}"
        ) if original_confidence is not None else (
            f"Judge override by {override.judge_name}: "
            f"{override.trick_id} set to {override.new_confidence:.1%} | "
            f"Reason: {override.reason}"
        ),
        created_by=override.judge_name,
    )

    db.add(entry)
    db.commit()

    return {
        "status": "override_recorded",
        "run_id": run_id,
        "entry_id": entry.id,
        "message": "Original AI decision preserved. Override logged as new audit entry.",
    }
