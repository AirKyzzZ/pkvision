"""SQLAlchemy models for PkVision persistence."""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import Boolean, Column, DateTime, Float, Integer, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker


class Base(DeclarativeBase):
    pass


class Run(Base):
    __tablename__ = "runs"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    video_path = Column(Text, nullable=False)
    status = Column(String, default="pending")  # pending/processing/completed/failed
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    language = Column(String, default="en")
    result_json = Column(Text, nullable=True)


class Detection(Base):
    __tablename__ = "detections"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    run_id = Column(String, nullable=False, index=True)
    trick_id = Column(String, nullable=False)
    confidence = Column(Float, nullable=False)
    start_frame = Column(Integer)
    end_frame = Column(Integer)
    start_time_ms = Column(Float)
    end_time_ms = Column(Float)
    angles_snapshot = Column(Text)  # JSON
    strategy_used = Column(String)  # "angle_threshold" or "temporal_model"
    is_top3 = Column(Boolean, default=False)


class AuditEntryRecord(Base):
    __tablename__ = "audit_entries"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    run_id = Column(String, nullable=False, index=True)
    entry_type = Column(String, nullable=False)  # detection/scoring/override
    timestamp_ms = Column(Float, nullable=True)
    trick_id = Column(String, nullable=True)
    confidence = Column(Float, nullable=True)
    reasoning = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    created_by = Column(String, default="ai")


class SubmissionRecord(Base):
    __tablename__ = "submissions"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    submission_type = Column(String, nullable=False)  # "clip" or "trick"
    trick_name = Column(String)
    video_url = Column(Text, nullable=True)
    description = Column(Text, default="")
    submitter_name = Column(String, default="")
    status = Column(String, default="pending")  # pending/approved/rejected
    created_at = Column(DateTime, default=datetime.utcnow)
    reviewed_at = Column(DateTime, nullable=True)
    reviewer_notes = Column(Text, default="")


def get_engine(database_url: str = "sqlite:///./pkvision.db"):
    return create_engine(database_url, echo=False)


def get_session_factory(database_url: str = "sqlite:///./pkvision.db"):
    engine = get_engine(database_url)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)
