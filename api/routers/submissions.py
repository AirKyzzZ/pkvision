"""Community submission endpoints — clips and trick proposals."""

from __future__ import annotations

import uuid
from datetime import datetime

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

from api.deps import get_db
from db.models import SubmissionRecord

router = APIRouter()


class ClipSubmission(BaseModel):
    trick_name: str
    video_url: str
    submitter_name: str = ""
    camera_angle: str = "side"
    timestamp_in_video: str = ""
    description: str = ""


class TrickProposal(BaseModel):
    trick_name_en: str
    trick_name_fr: str = ""
    category: str
    difficulty_estimate: float
    description: str
    reference_video_url: str = ""
    tags: list[str] = []
    submitter_name: str = ""


@router.post("/submissions/clip")
async def submit_clip(
    submission: ClipSubmission,
    db: Session = Depends(get_db),
):
    """Submit a training clip for review."""
    record = SubmissionRecord(
        id=str(uuid.uuid4()),
        submission_type="clip",
        trick_name=submission.trick_name,
        video_url=submission.video_url,
        description=(
            f"Camera: {submission.camera_angle} | "
            f"Timestamp: {submission.timestamp_in_video} | "
            f"{submission.description}"
        ),
        submitter_name=submission.submitter_name,
    )

    db.add(record)
    db.commit()

    return {
        "status": "submitted",
        "submission_id": record.id,
        "message": "Thank you! Your clip will be reviewed by a maintainer.",
    }


@router.post("/submissions/trick")
async def submit_trick(
    proposal: TrickProposal,
    db: Session = Depends(get_db),
):
    """Propose a new trick for the catalog."""
    record = SubmissionRecord(
        id=str(uuid.uuid4()),
        submission_type="trick",
        trick_name=proposal.trick_name_en,
        video_url=proposal.reference_video_url or None,
        description=(
            f"FR: {proposal.trick_name_fr} | "
            f"Category: {proposal.category} | "
            f"Difficulty: {proposal.difficulty_estimate} | "
            f"Tags: {', '.join(proposal.tags)} | "
            f"{proposal.description}"
        ),
        submitter_name=proposal.submitter_name,
    )

    db.add(record)
    db.commit()

    return {
        "status": "submitted",
        "submission_id": record.id,
        "message": "Thank you! Your trick proposal will be reviewed by a maintainer.",
    }


@router.get("/submissions")
async def list_submissions(
    status: str = "pending",
    db: Session = Depends(get_db),
):
    """List submissions (for maintainer review)."""
    records = db.query(SubmissionRecord).filter(
        SubmissionRecord.status == status
    ).order_by(SubmissionRecord.created_at.desc()).all()

    return {
        "submissions": [
            {
                "id": r.id,
                "type": r.submission_type,
                "trick_name": r.trick_name,
                "video_url": r.video_url,
                "description": r.description,
                "submitter": r.submitter_name,
                "status": r.status,
                "created_at": r.created_at.isoformat() if r.created_at else None,
            }
            for r in records
        ]
    }
