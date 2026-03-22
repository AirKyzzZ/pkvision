"""Trick catalog endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from api.deps import get_classifier
from core.recognition.classifier import TrickClassifier

router = APIRouter()


@router.get("/tricks")
async def list_tricks(
    lang: str = "en",
    classifier: TrickClassifier = Depends(get_classifier),
):
    """List all tricks in the catalog."""
    return {"tricks": classifier.list_tricks(lang=lang)}


@router.get("/tricks/{trick_id}")
async def get_trick(
    trick_id: str,
    lang: str = "en",
    classifier: TrickClassifier = Depends(get_classifier),
):
    """Get a single trick by ID."""
    trick = classifier.get_trick_by_id(trick_id)
    if trick is None:
        raise HTTPException(status_code=404, detail=f"Trick not found: {trick_id}")

    return {
        "trick_id": trick.trick_id,
        "name": trick.get_name(lang),
        "category": trick.category,
        "difficulty": trick.difficulty,
        "detection_method": trick.detection_method.value,
        "tags": trick.tags,
        "phases": [
            {
                "name": p.name,
                "duration_range_ms": p.duration_range_ms,
                "angle_rules": [r.model_dump() for r in p.angle_rules],
            }
            for p in trick.phases
        ],
        "composable_with": trick.composable_with,
        "names": trick.names,
    }
