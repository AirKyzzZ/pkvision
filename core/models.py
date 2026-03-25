"""Pydantic models for PkVision — shared across all modules."""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any

import numpy as np
from pydantic import BaseModel, Field


# ── Trick Catalog Models ──────────────────────────────────────────────


class AngleRule(BaseModel):
    joint: str
    min: float
    max: float
    description: str = ""


class VelocityRule(BaseModel):
    joint: str
    axis: str  # "vertical", "horizontal", "angular"
    direction: str  # "up", "down", "left", "right", "increasing", "decreasing"
    min_speed: float


class TrajectoryRule(BaseModel):
    rotation_axis: str  # "lateral" (side-to-side), "longitudinal" (twist), "sagittal" (front-back)
    rotation_direction: str  # "forward", "backward", "clockwise", "counterclockwise"
    min_rotation_degrees: float


class TrickPhase(BaseModel):
    name: str  # "approach", "execution", "landing"
    duration_range_ms: tuple[int, int]
    angle_rules: list[AngleRule] = Field(default_factory=list)
    velocity_rules: list[VelocityRule] = Field(default_factory=list)
    trajectory: TrajectoryRule | None = None


class DetectionMethod(str, Enum):
    ANGLE_THRESHOLD = "angle_threshold"
    TEMPORAL_MODEL = "temporal_model"


class TrickPrimitives(BaseModel):
    rotation_sagittal: str = "none"  # "forward", "backward", "none"
    rotation_sagittal_count: int = 0
    rotation_lateral: str = "none"  # "left", "right", "none"
    twist_count: int = 0
    takeoff: str = "two_feet"
    landing: str = "two_feet"
    obstacle_interaction: bool = False


class TrickConfig(BaseModel):
    trick_id: str
    category: str  # "flip", "vault", "twist", "combo"
    tags: list[str] = Field(default_factory=list)
    difficulty: float  # 0.0 - 10.0
    detection_method: DetectionMethod = DetectionMethod.ANGLE_THRESHOLD
    phases: list[TrickPhase]
    composable_with: list[str] = Field(default_factory=list)
    names: dict[str, str]  # {"en": "Front Flip", "fr": "Salto Avant"}
    primitives: TrickPrimitives | None = None

    def get_name(self, lang: str = "en") -> str:
        return self.names.get(lang, self.names.get("en", self.trick_id))


# ── Pose / Frame Models ──────────────────────────────────────────────


class FrameResult(BaseModel):
    """Raw output from pose detector for a single frame."""

    model_config = {"arbitrary_types_allowed": True}

    frame_idx: int
    timestamp_ms: float
    keypoints: Any  # np.ndarray (17, 2) or (17, 3)
    keypoint_confidences: Any  # np.ndarray (17,)
    frame_shape: tuple[int, int]  # (height, width)


class FrameAnalysis(BaseModel):
    """Frame with computed angles and velocities."""

    model_config = {"arbitrary_types_allowed": True}

    frame_idx: int
    timestamp_ms: float
    keypoints: Any  # np.ndarray
    keypoint_confidences: Any  # np.ndarray
    angles: dict[str, float]  # joint_name → angle in degrees (NaN if low confidence)
    velocities: dict[str, float] | None = None  # joint_name → angular velocity


# ── Detection Models ─────────────────────────────────────────────────


class AngleMatch(BaseModel):
    """A single angle rule match for explainability."""

    joint: str
    measured: float
    rule_min: float
    rule_max: float
    matched: bool


class TrickDetection(BaseModel):
    """A detected trick instance in a video run."""

    detection_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    trick_id: str
    trick_name: str
    confidence: float  # 0.0 - 1.0
    start_frame: int
    end_frame: int
    start_time_ms: float
    end_time_ms: float
    strategy_used: str  # "angle_threshold" or "temporal_model"
    angle_matches: list[AngleMatch] = Field(default_factory=list)
    phase_confidences: dict[str, float] = Field(default_factory=dict)


# ── Scoring Models ───────────────────────────────────────────────────


class ScoredTrick(BaseModel):
    trick_id: str
    trick_name: str
    difficulty: float
    confidence: float
    weighted_score: float  # difficulty * confidence
    detection: TrickDetection


class ScoreResult(BaseModel):
    top3: list[ScoredTrick]
    total_score: float
    max_possible_score: float  # sum of top3 difficulties (if all were 100% confidence)


# ── Audit Models ─────────────────────────────────────────────────────


class AuditEntryType(str, Enum):
    DETECTION = "detection"
    SCORING = "scoring"
    OVERRIDE = "override"
    SYSTEM = "system"


class AuditEntry(BaseModel):
    entry_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    entry_type: AuditEntryType
    timestamp_ms: float | None = None
    trick_id: str | None = None
    confidence: float | None = None
    reasoning: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    created_by: str = "ai"  # "ai" or judge identifier


# ── Run Models ───────────────────────────────────────────────────────


class RunStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class RunAnalysis(BaseModel):
    run_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    video_path: str
    status: RunStatus = RunStatus.PENDING
    language: str = "en"
    detections: list[TrickDetection] = Field(default_factory=list)
    score: ScoreResult | None = None
    audit_trail: list[AuditEntry] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: datetime | None = None


# ── Submission Models ────────────────────────────────────────────────


class SubmissionType(str, Enum):
    CLIP = "clip"
    TRICK = "trick"


class SubmissionStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


class Submission(BaseModel):
    submission_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    submission_type: SubmissionType
    trick_name: str
    video_url: str | None = None
    description: str = ""
    submitter_name: str = ""
    status: SubmissionStatus = SubmissionStatus.PENDING
    created_at: datetime = Field(default_factory=datetime.utcnow)
    reviewed_at: datetime | None = None
    reviewer_notes: str = ""


# ── 3D Biomechanics Models (v2 Pipeline) ─────────────────────────────


class SMPLFrame(BaseModel):
    """SMPL body mesh parameters for a single frame from GVHMR."""

    model_config = {"arbitrary_types_allowed": True}

    frame_idx: int
    timestamp_ms: float
    global_orient: Any  # np.ndarray (3,) axis-angle — body root rotation in world frame
    body_pose: Any      # np.ndarray (23, 3) axis-angle — joint rotations relative to parent
    transl: Any         # np.ndarray (3,) — world translation in meters
    betas: Any          # np.ndarray (10,) — body shape parameters
    joints_3d: Any = None  # np.ndarray (24, 3) — 3D joint positions in world frame (optional)


class BiomechanicalFrame(BaseModel):
    """Biomechanical features extracted from SMPL parameters for one frame."""

    model_config = {"arbitrary_types_allowed": True}

    frame_idx: int
    timestamp_ms: float

    # Rotation state (from global_orient swing-twist decomposition)
    rotation_axis: Any       # np.ndarray (3,) — instantaneous rotation axis in world frame
    accumulated_flip_deg: float = 0.0    # Total flip rotation (around lateral/sagittal axis)
    accumulated_twist_deg: float = 0.0   # Total twist rotation (around longitudinal axis)

    # Joint angles (from SMPL body_pose, in degrees)
    knee_angle: float = 180.0
    hip_angle: float = 180.0
    spine_angle: float = 180.0
    shoulder_angle: float = 0.0
    elbow_angle: float = 180.0

    # Center of mass (from SMPL transl)
    com_position: Any = None  # np.ndarray (3,) — x, y, z in meters
    com_velocity: Any = None  # np.ndarray (3,) — velocity in m/s
    com_height_m: float = 0.0
    is_airborne: bool = False

    # Body shape classification (soft scores)
    body_shape_scores: dict[str, float] = Field(
        default_factory=lambda: {"tuck": 0.0, "pike": 0.0, "layout": 0.0, "open": 0.0}
    )


class TrickSignature3D(BaseModel):
    """Complete 3D biomechanical signature of a trick segment.

    This is what gets matched against TrickDefinitions for zero-shot recognition.
    """

    model_config = {"arbitrary_types_allowed": True}

    # Rotation (the primary discriminators)
    primary_rotation_axis: Any  # np.ndarray (3,) — averaged rotation axis during aerial phase
    total_flip_deg: float       # Total flip rotation (0=none, 360=single, 720=double, etc.)
    total_twist_deg: float      # Total longitudinal twist (0=none, 360=full, 720=double full)
    rotation_direction: str     # "forward" or "backward" (sign of flip relative to facing direction)

    # Body shape (dominant during aerial phase)
    body_shape: str             # "tuck", "pike", "layout", or "open"
    avg_knee_angle: float = 180.0
    avg_hip_angle: float = 180.0

    # Entry and exit
    entry_type: str = "standing"  # "standing", "running", "one_leg", "wall", "edge"
    peak_height_m: float = 0.0
    duration_s: float = 0.0

    # Timing
    start_frame: int = 0
    end_frame: int = 0
    start_time_ms: float = 0.0
    end_time_ms: float = 0.0

    # Derived axis classification (for matching)
    @property
    def axis_classification(self) -> str:
        """Classify the primary rotation axis into a named category."""
        if self.primary_rotation_axis is None:
            return "lateral"
        ax = np.abs(self.primary_rotation_axis)
        if ax[0] > 0.6:  # Dominant x-component = lateral axis (flip)
            return "lateral"
        if ax[2] > 0.6:  # Dominant z-component = sagittal axis (side flip)
            return "sagittal"
        if ax[1] > 0.6:  # Dominant y-component = longitudinal (pure twist)
            return "longitudinal"
        return "off_axis"  # Mixed = cork

    @property
    def flip_count(self) -> float:
        """Number of complete flips (360° = 1.0)."""
        return round(abs(self.total_flip_deg) / 360.0, 1)

    @property
    def twist_count(self) -> float:
        """Number of complete twists (360° = 1.0)."""
        return round(abs(self.total_twist_deg) / 360.0, 1)
