"""FastAPI dependencies — shared across routers."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from sqlalchemy.orm import Session

from core.recognition.classifier import TrickClassifier
from db.models import Base, get_engine, get_session_factory

PROJECT_ROOT = Path(__file__).parent.parent
CATALOG_DIR = PROJECT_ROOT / "data" / "tricks" / "catalog"
MODEL_PATH = PROJECT_ROOT / "data" / "models" / "stgcn_best.pt"


@lru_cache
def get_db_session_factory():
    db_url = os.getenv("DATABASE_URL", "sqlite:///./pkvision.db")
    return get_session_factory(db_url)


def get_db():
    """FastAPI dependency that yields a DB session."""
    factory = get_db_session_factory()
    session = factory()
    try:
        yield session
    finally:
        session.close()


@lru_cache
def get_classifier() -> TrickClassifier:
    """Singleton trick classifier with loaded catalog and model."""
    lang = os.getenv("DEFAULT_LANGUAGE", "en")
    model_path = Path(os.getenv("STGCN_MODEL_PATH", str(MODEL_PATH)))

    return TrickClassifier(
        catalog_dir=CATALOG_DIR,
        model_path=model_path if model_path.exists() else None,
        language=lang,
    )
