# Architecture

This document describes the technical architecture of PkVision: the detection pipeline, the strategy pattern used for trick recognition, the data flow from video to score, and the database schema.

---

## Pipeline Overview

```
                          +-------------------+
                          |   Video Input     |
                          |  (file / API)     |
                          +---------+---------+
                                    |
                                    v
                     +------------------------------+
                     |     Pose Estimation           |
                     |  YOLO11n-pose (17 keypoints)  |
                     |  core/pose/detector.py         |
                     +-------------+----------------+
                                   |
                           list[FrameResult]
                                   |
                                   v
                     +------------------------------+
                     |     Angle Computation          |
                     |  Joint angles + velocities     |
                     |  core/pose/angles.py            |
                     +-------------+----------------+
                                   |
                          list[FrameAnalysis]
                                   |
                                   v
                     +------------------------------+
                     |     Sequence Analysis          |
                     |  Movement intensity windowing   |
                     |  core/recognition/sequence.py    |
                     +-------------+----------------+
                                   |
                          list[TrickWindow]
                                   |
                                   v
                     +------------------------------+
                     |     Trick Classification       |
                     |  Strategy dispatch per trick    |
                     |  core/recognition/classifier.py  |
                     |                                |
                     |  +----------+  +------------+ |
                     |  |  Angle   |  | ST-GCN     | |
                     |  | Strategy |  | Strategy   | |
                     |  +----------+  +------------+ |
                     +-------------+----------------+
                                   |
                        list[TrickDetection]
                                   |
                                   v
                     +------------------------------+
                     |     Confidence Refinement      |
                     |  Multi-factor scoring           |
                     |  core/recognition/confidence.py  |
                     +-------------+----------------+
                                   |
                        list[TrickDetection]
                                   |
                                   v
                     +------------------------------+
                     |     Scoring Engine             |
                     |  Top-3 selection by difficulty  |
                     |  Deduplication + weighting      |
                     |  core/scoring/engine.py          |
                     +-------------+----------------+
                                   |
                              ScoreResult
                                   |
                                   v
                     +------------------------------+
                     |     Audit Trail                |
                     |  Immutable decision log         |
                     |  core/explainability/trace.py    |
                     +------------------------------+
                                   |
                                   v
                          +-------------------+
                          |     Output        |
                          |  JSON / CLI / API |
                          +-------------------+
```

---

## Core Data Models

All shared data models are defined in `core/models.py` using Pydantic. The key models and their relationships:

```
TrickConfig                 # Trick definition from catalog
  ├── TrickPhase[]          # Phases of the trick
  │     ├── AngleRule[]     # Joint angle constraints
  │     ├── VelocityRule[]  # Angular velocity constraints
  │     └── TrajectoryRule  # Rotation constraints
  └── DetectionMethod       # angle_threshold | temporal_model

FrameResult                 # Raw pose detector output
  └── keypoints (17x2)     # COCO keypoint coordinates

FrameAnalysis               # Frame with computed angles
  ├── angles {}             # joint_name -> degrees
  └── velocities {}         # joint_name -> deg/sec

TrickDetection              # A detected trick instance
  ├── AngleMatch[]          # Which angle rules matched
  └── phase_confidences {}  # Per-phase confidence

ScoreResult                 # Final scoring output
  └── ScoredTrick[]         # Top-3 tricks with weighted scores

AuditEntry                  # Single audit log entry
  └── AuditEntryType        # detection | scoring | override | system

RunAnalysis                 # Complete analysis result
  ├── TrickDetection[]
  ├── ScoreResult
  └── AuditEntry[]
```

---

## Detection Strategy Pattern

Trick detection uses the **Strategy pattern**. Each detection method implements the same protocol, and the classifier dispatches to the appropriate strategy based on the trick's `detection_method` field.

### Protocol

Defined in `core/recognition/strategies/base.py`:

```python
class DetectionStrategy(Protocol):
    def evaluate(
        self,
        trick: TrickConfig,
        frames: list[FrameAnalysis],
    ) -> TrickDetection | None:
        ...
```

Every strategy receives:
- A `TrickConfig` describing the trick to look for
- A list of `FrameAnalysis` objects (frames with computed angles and velocities)

And returns either a `TrickDetection` (if the trick was found) or `None`.

### Implemented Strategies

#### 1. AngleThresholdStrategy (`core/recognition/strategies/angle.py`)

The baseline strategy. It works without any training data.

**How it works:**
1. For each trick, iterate through its phases sequentially.
2. For each phase, use a sliding window over the frame sequence.
3. Within each window, check if the angle rules are satisfied.
4. A rule is "matched" if any frame in the window has the specified joint angle within the defined range.
5. Phase confidence is the fraction of rules matched, weighted by how close angles are to the rule midpoint.
6. Overall confidence is the average of phase confidences.

**Strengths:** Interpretable, works with zero training data, every decision is fully explainable in terms of specific angle measurements.

**Limitations:** Relies on hand-crafted angle rules. Cannot learn from data. Struggles with tricks that are defined more by motion trajectory than static poses.

#### 2. TemporalModelStrategy (`core/recognition/strategies/temporal.py`)

The primary strategy for complex tricks. Uses a trained ST-GCN model.

**How it works:**
1. Load a trained ST-GCN checkpoint (model weights + trick class list).
2. Convert the frame sequence to ST-GCN input format: `(3, T, 17)` tensor (x, y, confidence per keypoint per frame).
3. Run inference to get class probabilities.
4. Check if the requested trick class has confidence above threshold.

**Strengths:** Learns from data, handles complex motions, improves with more training clips.

**Limitations:** Requires labeled training data. Less interpretable than the angle strategy (though phase confidences are still reported).

### Strategy Dispatch

The `TrickClassifier` in `core/recognition/classifier.py` selects the strategy based on each trick's `detection_method` field:

```
detection_method: "angle_threshold"  -->  AngleThresholdStrategy
detection_method: "temporal_model"   -->  TemporalModelStrategy (if model loaded)
                                     -->  AngleThresholdStrategy (fallback)
```

If no ST-GCN model is loaded, all tricks fall back to the angle threshold strategy.

### Adding a New Detection Strategy

To add a new detection strategy:

1. Create a new file in `core/recognition/strategies/` (e.g., `transformer.py`).
2. Implement a class with an `evaluate` method matching the `DetectionStrategy` protocol:
   ```python
   class TransformerStrategy:
       def evaluate(
           self,
           trick: TrickConfig,
           frames: list[FrameAnalysis],
       ) -> TrickDetection | None:
           # Your detection logic here
           ...
   ```
3. Add the new method to the `DetectionMethod` enum in `core/models.py`:
   ```python
   class DetectionMethod(str, Enum):
       ANGLE_THRESHOLD = "angle_threshold"
       TEMPORAL_MODEL = "temporal_model"
       TRANSFORMER = "transformer"  # new
   ```
4. Register it in `TrickClassifier._get_strategy()` in `core/recognition/classifier.py`.
5. Update the JSON schema in `data/tricks/schema.json` to allow the new method.
6. Tricks can now specify `"detection_method": "transformer"` in their JSON config.

---

## Data Flow: Video to Score

This section traces a complete analysis from a video file to a scored result.

### 1. Video Input

Entry points:
- CLI: `python scripts/analyze.py --input video.mp4`
- API: `POST /api/v1/analyze` with file upload

### 2. Pose Extraction

`PoseDetector.process_video()` iterates over every frame:

```
video.mp4
  --> cv2.VideoCapture reads frame by frame
  --> YOLO11n-pose extracts 17 keypoints per frame
  --> Selects the person with highest average keypoint confidence
  --> Yields FrameResult(frame_idx, timestamp_ms, keypoints, confidences)
```

Output: `list[FrameResult]`

### 3. Angle Computation

`frame_result_to_analysis()` and `get_joint_velocities()`:

```
FrameResult
  --> Compute 8 bilateral joint angles (left_knee, right_knee, left_hip, etc.)
  --> Compute 2 composite angles (spine, neck) from keypoint midpoints
  --> Compute averaged bilateral angles (knee, hip, elbow, shoulder)
  --> Compute angular velocities (degrees/second) between consecutive frames
  --> Output: FrameAnalysis(angles={}, velocities={})
```

Output: `list[FrameAnalysis]`

### 4. Sequence Windowing

`SequenceAnalyzer.find_trick_windows()`:

```
list[FrameAnalysis]
  --> Compute movement intensity per frame (average angular velocity)
  --> Find active regions (intensity > threshold)
  --> Merge nearby regions with padding
  --> Output: list[TrickWindow] (contiguous high-activity segments)
```

If the video is short or no high-activity regions are found, the entire sequence is treated as one window.

### 5. Trick Classification

`TrickClassifier.classify()`:

```
For each TrickWindow:
  For each TrickConfig in catalog:
    --> Select strategy (angle or temporal)
    --> Evaluate trick against window frames
    --> If detection: append TrickDetection

For each TrickDetection:
  --> Refine confidence via multi-factor scoring
  --> Filter by confidence threshold
```

Multi-factor confidence considers:
- Base detection confidence (40% weight)
- Keypoint quality in detection window (20% weight)
- Phase coverage (20% weight)
- Angle match quality (20% weight)

Output: `list[TrickDetection]`

### 6. Scoring

`ScoringEngine.score()`:

```
list[TrickDetection]
  --> Filter by confidence threshold
  --> Deduplicate overlapping detections (IoU > 50%)
  --> Sort by difficulty (descending), break ties by confidence
  --> Select top 3
  --> Compute weighted_score = difficulty * confidence for each
  --> Compute total_score = sum of weighted scores
  --> Output: ScoreResult(top3, total_score, max_possible_score)
```

### 7. Audit Trail

`AuditTracer` accumulates entries throughout the pipeline:

```
SYSTEM  | "Analyzing video.mp4"
DET     | "Detected Front Flip (front_flip) | Strategy: angle_threshold | Confidence: 85% | ..."
DET     | "Detected Back Flip (back_flip) | Strategy: temporal_model | Confidence: 72% | ..."
SCR     | "Top 3 tricks selected | Total score: 5.82 / 7.50 | #1 Front Flip ..."
OVR     | "Judge override by J.Smith: front_flip confidence 85% -> 90% | Reason: Clean landing"
```

Entries are immutable. Judge overrides create new `OVERRIDE` entries without modifying the original `DETECTION` entries.

---

## Database Schema

The persistence layer uses SQLAlchemy with SQLite (default). Defined in `db/models.py`.

### Tables

#### `runs`

Tracks each video analysis session.

| Column | Type | Description |
|--------|------|-------------|
| `id` | String (PK) | UUID |
| `video_path` | Text | Path to the analyzed video |
| `status` | String | pending / processing / completed / failed |
| `created_at` | DateTime | When the run was created |
| `completed_at` | DateTime | When analysis finished (nullable) |
| `language` | String | Language code (default: "en") |
| `result_json` | Text | Full RunAnalysis result as JSON (nullable) |

#### `detections`

Individual trick detections within a run.

| Column | Type | Description |
|--------|------|-------------|
| `id` | String (PK) | UUID |
| `run_id` | String (FK) | Reference to the parent run |
| `trick_id` | String | Trick identifier from catalog |
| `confidence` | Float | Detection confidence (0.0-1.0) |
| `start_frame` | Integer | First frame of detection |
| `end_frame` | Integer | Last frame of detection |
| `start_time_ms` | Float | Start timestamp in milliseconds |
| `end_time_ms` | Float | End timestamp in milliseconds |
| `angles_snapshot` | Text | JSON snapshot of angle measurements |
| `strategy_used` | String | "angle_threshold" or "temporal_model" |
| `is_top3` | Boolean | Whether this detection is in the top 3 |

#### `audit_entries`

Immutable audit log for all decisions.

| Column | Type | Description |
|--------|------|-------------|
| `id` | String (PK) | UUID |
| `run_id` | String (FK) | Reference to the parent run |
| `entry_type` | String | detection / scoring / override / system |
| `timestamp_ms` | Float | Video timestamp (nullable) |
| `trick_id` | String | Related trick (nullable) |
| `confidence` | Float | Confidence value (nullable) |
| `reasoning` | Text | Human-readable explanation |
| `created_at` | DateTime | When the entry was created |
| `created_by` | String | "ai" or judge identifier |

#### `submissions`

Community submissions (clips and trick proposals).

| Column | Type | Description |
|--------|------|-------------|
| `id` | String (PK) | UUID |
| `submission_type` | String | "clip" or "trick" |
| `trick_name` | String | Name of the trick |
| `video_url` | Text | Link to the video (nullable) |
| `description` | Text | Submission description |
| `submitter_name` | String | Who submitted it |
| `status` | String | pending / approved / rejected |
| `created_at` | DateTime | When submitted |
| `reviewed_at` | DateTime | When reviewed (nullable) |
| `reviewer_notes` | Text | Maintainer review notes |

### Entity Relationships

```
Run (1) -----> (*) Detection
Run (1) -----> (*) AuditEntry
Submission is standalone (not linked to runs)
```

### Database Configuration

Default: SQLite at `./pkvision.db`. Configurable via the `database_url` parameter:

```python
from db.models import get_session_factory

# SQLite (default)
Session = get_session_factory()

# PostgreSQL
Session = get_session_factory("postgresql://user:pass@localhost/pkvision")
```

Tables are auto-created on first connection via `Base.metadata.create_all()`.

---

## API Architecture

The API is built with FastAPI and organized into routers by domain:

```
api/
├── main.py          # App factory, middleware, router registration
├── deps.py          # Dependency injection (DB sessions, services)
└── routers/
    ├── analysis.py  # Video upload and analysis
    ├── runs.py      # Run management and retrieval
    ├── tricks.py    # Trick catalog browsing
    └── submissions.py  # Community submissions
```

All endpoints are prefixed with `/api/v1/`. CORS is open by default (configurable for production).

### Request Flow

```
Client
  --> FastAPI router
  --> Dependency injection (DB session, classifier, etc.)
  --> Core pipeline (pose -> recognition -> scoring -> audit)
  --> Response (JSON)
```

---

## ST-GCN Training Pipeline

The machine learning pipeline for training the temporal model:

```
data/clips/*.mp4           Raw video clips
        |
        v
scripts/label.py           Interactive labeling (trick_id + time range)
        |
        v
data/clips/labels.json     Label file [{file, trick_id, start_ms, end_ms}]
        |
        v
scripts/extract_poses.py   YOLO pose extraction per clip
        |
        v
data/clips/keypoints/*.npy Keypoint arrays (3, T, 17) per clip
        |
        v
scripts/train.py           Training entry point
        |
        v
ml/train.py                Training loop
  ├── ml/dataset.py        PyTorch Dataset (loads .npy + labels)
  ├── ml/augment.py        Data augmentation (noise, scaling, etc.)
  └── ml/stgcn/model.py    ST-GCN architecture
        |
        v
data/models/stgcn_best.pt  Saved checkpoint (weights + class list)
```

The trained model is loaded by `TemporalModelStrategy` at inference time.

---

## Worker (Future)

The `worker/tasks.py` file contains Celery task stubs for future asynchronous video processing. The current implementation processes videos synchronously. When async processing is needed:

1. Configure `CELERY_BROKER_URL` and `CELERY_RESULT_BACKEND` (Redis).
2. Implement the `analyze_video` task.
3. Run: `celery -A worker.tasks worker --loglevel=info`

This allows the API to accept video uploads and return immediately with a run ID, while processing happens in the background.
