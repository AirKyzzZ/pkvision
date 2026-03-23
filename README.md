<p align="center">
  <img src="assets/banner.png" alt="PkVision" width="500">
</p>

<p align="center">
  <strong>See Every Move. Score Every Trick.</strong><br>
  Open-source AI for automatic parkour trick detection and scoring.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License: MIT">
  <img src="https://img.shields.io/badge/Python-3.11%2B-blue.svg" alt="Python 3.11+">
  <img src="https://img.shields.io/badge/Contributions-Welcome-orange.svg" alt="Contributions Welcome">
</p>

---

## What is PkVision?

PkVision is an open-source artificial intelligence system that analyzes parkour videos, identifies tricks performed by athletes, scores the top 3 tricks by difficulty, and generates fully explainable audit trails for every decision. It is designed to bring standardized, transparent notation to parkour competition judging -- supporting the discipline's path toward Olympic inclusion through the FIG (Federation Internationale de Gymnastique). The project is backed by coaches with connections to international parkour federations and built by a community of developers, athletes, and judges.

### Why It Matters

Parkour is evolving from a street discipline into a competitive sport with structured rules. As the FIG works to standardize competition formats, automated and reproducible trick notation becomes essential. PkVision provides a nation-neutral, multilingual system where every score is auditable, every detection is explainable, and human judges always have the final say.

---

## How It Works

### Pipeline Overview

```mermaid
flowchart LR
    A["Video Input"] --> B["Pose Estimation\n(YOLO11n-pose)"]
    B --> C["Joint Angles\n& Velocities"]
    C --> D{"Detection\nStrategy"}
    D -->|Baseline| E["Angle Threshold\nMatching"]
    D -->|Primary| F["VideoMAE /\nST-GCN Model"]
    E --> G["Trick\nDetections"]
    F --> G
    G --> H["Scoring Engine\n(Top 3)"]
    H --> I["Audit Trail"]
    I --> J["Results\n+ Confidence"]

    style A fill:#1a1a2e,stroke:#4361ee,color:#fff
    style B fill:#1a1a2e,stroke:#4361ee,color:#fff
    style C fill:#1a1a2e,stroke:#4361ee,color:#fff
    style D fill:#0d1b2a,stroke:#f59e0b,color:#fff
    style E fill:#1a1a2e,stroke:#888,color:#ccc
    style F fill:#1a1a2e,stroke:#4361ee,color:#fff
    style G fill:#1a1a2e,stroke:#4361ee,color:#fff
    style H fill:#1a1a2e,stroke:#4361ee,color:#fff
    style I fill:#1a1a2e,stroke:#4361ee,color:#fff
    style J fill:#0d3b2a,stroke:#4ade80,color:#fff
```

### Step by Step

1. **Video input** -- A parkour clip is submitted (file upload or API).
2. **Pose estimation** -- YOLO11n-pose tracks 17 body keypoints frame by frame.
3. **Angle computation** -- Joint angles (knee, hip, elbow, shoulder, spine) and angular velocities are calculated from the keypoints.
4. **Trick detection** -- Movements are matched against a catalog of known tricks using one of two strategies:
   - **Angle threshold** (baseline) -- Matches angle rules across trick phases using a sliding window. Reliable, interpretable, works without training data.
   - **VideoMAE / ST-GCN** (primary) -- Pre-trained on Kinetics-400 (400 human actions), fine-tuned on parkour clips. Learns movement patterns from video frames or skeleton sequences.
5. **Scoring** -- The top 3 tricks are selected by difficulty, with each score weighted by detection confidence.
6. **Audit trail** -- Every detection includes full reasoning: which angles matched, which strategy was used, confidence levels, and phase-by-phase breakdowns.

Every decision is explainable. No black boxes.

### Detection Strategy Pattern

The detection layer is designed to be swappable. Different tricks can use different strategies simultaneously:

```mermaid
classDiagram
    class DetectionStrategy {
        <<protocol>>
        +evaluate(trick, frames) TrickDetection?
    }
    class AngleThresholdStrategy {
        +evaluate(trick, frames) TrickDetection?
        -match_phases()
        -evaluate_angle_rules()
    }
    class TemporalModelStrategy {
        +run_inference(trick, frames) TrickDetection?
        -load_model()
        -prepare_input()
    }
    class VideoMAEStrategy {
        +classify(video) dict
        -fine_tuned_model
        -kinetics_pretrained
    }

    DetectionStrategy <|.. AngleThresholdStrategy : implements
    DetectionStrategy <|.. TemporalModelStrategy : implements
    DetectionStrategy <|.. VideoMAEStrategy : implements

    class TrickClassifier {
        +classify(frames) List~TrickDetection~
        -get_strategy(trick) DetectionStrategy
        -tricks: List~TrickConfig~
    }

    TrickClassifier --> DetectionStrategy : dispatches to
```

### Training Pipeline

```mermaid
flowchart TB
    subgraph Data Collection
        A1["Your Video Clips\n(iPhone / Camera)"] --> B1["Auto-Label\n(VideoMAE Kinetics-400)"]
        B1 --> C1["Review & Correct\n(Labeler UI)"]
        A2["Community Submissions\n(GitHub Issues)"] --> C1
    end

    subgraph Training
        C1 --> D1["Prepare Training Data\n(merge labels + extract frames)"]
        D1 --> E1["Fine-tune VideoMAE\n(transfer learning)"]
        E1 --> F1["Trained Model\n(.pt checkpoint)"]
    end

    subgraph Inference
        G1["New Video"] --> H1["PkVision Pipeline"]
        F1 --> H1
        H1 --> I1["Trick: Back Flip\nConfidence: 87%\nDifficulty: 3.5"]
    end

    style A1 fill:#1a1a2e,stroke:#4361ee,color:#fff
    style A2 fill:#1a1a2e,stroke:#4361ee,color:#fff
    style E1 fill:#1a1a2e,stroke:#f59e0b,color:#fff
    style F1 fill:#0d3b2a,stroke:#4ade80,color:#fff
    style I1 fill:#0d3b2a,stroke:#4ade80,color:#fff
```

### Neural Network Architecture

PkVision uses two complementary neural network approaches:

```mermaid
flowchart TB
    subgraph VideoMAE ["VideoMAE (Primary — Transfer Learning)"]
        direction TB
        V1["Video Frames\n(16 frames sampled)"] --> V2["Patch Embedding\n(divide into 3D patches)"]
        V2 --> V3["Vision Transformer\n(12 self-attention layers)"]
        V3 --> V4["Pre-trained on\nKinetics-400\n(400 human actions)"]
        V4 --> V5["Fine-tuned\nClassifier Head"]
        V5 --> V6["Parkour Classes\n(flip, twist_cork,\nvault, tumbling...)"]
    end

    subgraph STGCN ["ST-GCN (Skeleton-Based)"]
        direction TB
        S1["Keypoint Sequence\n(x, y, confidence)\nper frame"] --> S2["Skeleton Graph\n(17 joints as nodes,\nbones as edges)"]
        S2 --> S3["Spatial Graph Conv\n(learn joint relationships)"]
        S3 --> S4["Temporal Conv\n(learn movement patterns\nacross frames)"]
        S4 --> S5["5 ST-GCN Blocks\n(64→64→128→128→256)"]
        S5 --> S6["Global Avg Pool\n+ Classifier"]
        S6 --> S7["Trick Class\n+ Confidence"]
    end

    style V4 fill:#1a1a2e,stroke:#f59e0b,color:#fff
    style V5 fill:#1a1a2e,stroke:#4361ee,color:#fff
    style S3 fill:#1a1a2e,stroke:#4361ee,color:#fff
    style S4 fill:#1a1a2e,stroke:#4361ee,color:#fff
```

### Scoring and Audit Flow

```mermaid
flowchart LR
    A["All Detections"] --> B["Filter by\nConfidence\n(>50%)"]
    B --> C["Deduplicate\n(overlapping\ntime ranges)"]
    C --> D["Rank by\nDifficulty"]
    D --> E["Select\nTop 3"]
    E --> F["Score =\nDifficulty x\nConfidence"]
    F --> G["Audit Trail\n(immutable log)"]
    G --> H{"Judge\nOverride?"}
    H -->|Yes| I["New Audit Entry\n(original preserved)"]
    H -->|No| J["Final Score"]
    I --> J

    style E fill:#1a1a2e,stroke:#f59e0b,color:#fff
    style G fill:#1a1a2e,stroke:#4361ee,color:#fff
    style I fill:#1a1a2e,stroke:#f55,color:#fff
    style J fill:#0d3b2a,stroke:#4ade80,color:#fff
```

---

## For Athletes and Coaches

Your clips are what make this system work. The more diverse the training data, the more accurate PkVision becomes for everyone.

### How you can help

- **Submit training clips** -- Open a [Clip Submission issue](../../issues/new?template=clip_submission.yml) on GitHub with a link to your video.
- **Propose new tricks** -- If a trick is missing from the catalog, open a [Trick Proposal issue](../../issues/new?template=trick_submission.yml).
- **Review detections** -- Try PkVision on your own clips and report inaccuracies.

### Filming guidelines

For best detection results, follow these guidelines when recording clips:

- Resolution: 720p or higher
- Frame rate: 30fps or higher
- Camera angle: side or diagonal preferred
- Framing: full body visible throughout the trick
- Timing: 1-2 seconds of buffer before and after the trick
- Clothing: contrasting colors against the background (avoid white-on-white or busy patterns)
- Stability: tripod or stable surface preferred
- Content: one trick per clip when possible

See [docs/CLIP_GUIDELINES.md](docs/CLIP_GUIDELINES.md) for the complete guide.

---

## For Judges and Federations

PkVision is built with competition integrity in mind.

### Scoring

- **Top 3 by difficulty** -- The system selects the 3 most difficult tricks detected, aligned with FIG competition scoring structures.
- **Weighted scores** -- Each trick's score is `difficulty x confidence`, rewarding both ambition and clean execution.

### Transparency

- **Full audit trail** -- Every detection includes reasoning: which angles matched which rules, which strategy was used, the confidence level, and per-phase breakdowns.
- **Human override** -- Judges always have the final say. Overrides create new audit entries; the original AI decisions are never deleted or modified.
- **Immutable history** -- The audit log is append-only. All entries (AI detections, scoring decisions, judge overrides) are preserved for review.

### Neutrality

- **Nation-neutral** -- No geographic bias in detection or scoring.
- **Multi-language** -- Trick names and output are available in English and French, with a straightforward path to add more languages.

### References

- [FIG Parkour Code of Points](https://www.gymnastics.sport/site/pages/disciplines/pres-PK.php)
- [International Parkour Federation -- Judges Criteria](https://internationalparkourfederation.org/judges-criteria/)

> **Note:** This project references FIG and Olympic standards for context but is not officially affiliated with the FIG, the IOC, or any national federation.

---

## For Developers

### Quick Start

```bash
git clone https://github.com/AirKyzzZ/pkvision.git
cd pkvision
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Run tests
pytest

# Start API server
uvicorn api.main:app --reload

# Analyze a video
python scripts/analyze.py --input video.mp4 --lang en
```

The API documentation is available at [http://localhost:8000/docs](http://localhost:8000/docs) when the server is running (Swagger UI) and at [http://localhost:8000/redoc](http://localhost:8000/redoc) (ReDoc).

### Architecture Overview

```
pkvision/
├── core/                          # Core detection pipeline
│   ├── models.py                  # Pydantic models shared across all modules
│   ├── pose/                      # YOLO11n-pose estimation
│   │   ├── detector.py            # Video/frame/webcam pose extraction
│   │   ├── angles.py              # Joint angle and velocity calculations
│   │   └── constants.py           # COCO keypoint indices and names
│   ├── recognition/               # Trick detection engine
│   │   ├── classifier.py          # Loads catalog, dispatches to strategies
│   │   ├── sequence.py            # Temporal windowing (movement intensity)
│   │   ├── confidence.py          # Multi-factor confidence refinement
│   │   └── strategies/            # Strategy pattern for detection
│   │       ├── base.py            # DetectionStrategy protocol
│   │       ├── angle.py           # Angle threshold strategy (baseline)
│   │       └── temporal.py        # ST-GCN inference strategy
│   ├── scoring/
│   │   └── engine.py              # Top-3 selection and scoring
│   └── explainability/
│       └── trace.py               # Audit trail generation
├── ml/                            # ST-GCN training pipeline
│   ├── train.py                   # Training loop (MPS/CUDA/CPU)
│   ├── dataset.py                 # PyTorch dataset for keypoint sequences
│   ├── augment.py                 # Data augmentation for pose sequences
│   ├── evaluate.py                # Model evaluation and metrics
│   └── stgcn/                     # ST-GCN model architecture
│       ├── model.py               # STGCN network definition
│       └── layers.py              # Graph convolution layers
├── api/                           # FastAPI REST API
│   ├── main.py                    # App setup, CORS, router registration
│   ├── deps.py                    # Dependency injection
│   └── routers/                   # API endpoints
│       ├── analysis.py            # POST /api/v1/analyze
│       ├── runs.py                # GET/POST /api/v1/runs
│       ├── tricks.py              # GET /api/v1/tricks
│       └── submissions.py         # POST /api/v1/submissions
├── db/
│   └── models.py                  # SQLAlchemy models (Run, Detection, AuditEntry, Submission)
├── data/
│   ├── tricks/
│   │   ├── schema.json            # JSON Schema for trick definitions
│   │   └── catalog/
│   │       ├── en/                # English trick configs (10 tricks)
│   │       └── fr/                # French trick configs (10 tricks)
│   ├── clips/                     # Training video clips and labels
│   └── models/                    # Trained model checkpoints
├── scripts/                       # CLI tools
│   ├── analyze.py                 # Full video analysis pipeline
│   ├── label.py                   # Interactive clip labeling
│   ├── extract_poses.py           # Bulk keypoint extraction
│   └── train.py                   # Model training entry point
├── worker/
│   └── tasks.py                   # Celery task stubs (async processing)
├── tests/                         # Test suite
│   ├── unit/
│   │   └── test_catalog.py
│   └── fixtures/
├── docs/                          # Documentation
└── requirements.txt
```

### How to Add a Trick

Adding a new trick to the catalog requires no code changes. Create a JSON file in `data/tricks/catalog/en/` following the schema defined in `data/tricks/schema.json`.

Example (`data/tricks/catalog/en/lazy_vault.json`):

```json
{
  "trick_id": "lazy_vault",
  "category": "vault",
  "tags": ["vault", "obstacle", "one_hand"],
  "difficulty": 1.5,
  "detection_method": "angle_threshold",
  "phases": [
    {
      "name": "approach",
      "duration_range_ms": [300, 800],
      "angle_rules": [
        {
          "joint": "knee",
          "min": 155,
          "max": 180,
          "description": "Legs nearly straight during approach"
        }
      ]
    },
    {
      "name": "execution",
      "duration_range_ms": [200, 500],
      "angle_rules": [
        {
          "joint": "hip",
          "min": 80,
          "max": 140,
          "description": "Hips swing sideways over obstacle"
        }
      ]
    },
    {
      "name": "landing",
      "duration_range_ms": [100, 400],
      "angle_rules": []
    }
  ],
  "composable_with": [],
  "names": {
    "en": "Lazy Vault",
    "fr": "Saut de Paresseux"
  }
}
```

For a French translation, create the same file in `data/tricks/catalog/fr/` with localized descriptions. The `names` field supports any number of language codes.

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/analyze` | Upload a video for analysis |
| `GET` | `/api/v1/runs` | List all analysis runs |
| `GET` | `/api/v1/runs/{id}` | Get a specific run result |
| `GET` | `/api/v1/tricks` | List all tricks in the catalog |
| `POST` | `/api/v1/submissions` | Submit a clip or trick proposal |
| `GET` | `/health` | Health check |

---

## Training the Model

PkVision uses transfer learning from a model pre-trained on Kinetics-400 (400 human actions including somersaulting, gymnastics tumbling, parkour). This means you don't need thousands of labeled clips to get started.

### Quick Start (Recommended)

```bash
# 1. Place your video clips in data/clips/
#    (any format: mp4, mov, avi, mkv)

# 2. Auto-label with pre-trained Kinetics-400 model
python scripts/auto_label.py

# 3. Review and correct labels (web UI at localhost:8501)
python scripts/labeler_ui.py

# 4. Prepare training data (merges auto + manual labels)
python scripts/prepare_training.py

# 5. Fine-tune the model
python scripts/finetune.py --epochs 20

# 6. Test on a new video
python scripts/test_model.py --input your_video.mp4
```

### Labeler UI

A web-based tool for fast clip labeling at `http://localhost:8501`:

- 40+ trick categories with category filters
- Multiple tricks per video (each with start/end timestamps)
- Add custom tricks directly from the UI
- Keyboard shortcuts: `1-0` select, `A` add tag, `D` done, `S` skip, `Space` play/pause
- Progress persists across sessions

### Training Approaches

| Approach | Data Needed | Accuracy | Use When |
|----------|-------------|----------|----------|
| Pre-trained only | 0 clips | Coarse (flip vs vault vs parkour) | Quick demo |
| Fine-tuned head | 50-200 clips | Good (back flip vs front flip) | Standard training |
| Full fine-tune | 500+ clips | Best (double cork vs b-twist) | Production model |

### Advanced: ST-GCN Skeleton Training

For skeleton-based training (complementary to VideoMAE):

```bash
python scripts/extract_poses.py \
  --clips-dir data/clips/ \
  --labels data/clips/labels.json \
  --output data/clips/keypoints/

python scripts/train.py --epochs 100 --batch-size 16
```

### Hardware support

- **Apple Silicon (MPS)** -- M1/M2/M3 acceleration via `torch.device("mps")`
- **NVIDIA GPU** -- CUDA support via standard PyTorch
- **CPU** -- Works everywhere, just slower

The training loop auto-detects the best available device (CUDA > MPS > CPU).

---

## Context: Parkour and the Olympics

Parkour (also known as freerunning in its acrobatic form) is a physical discipline focused on moving through environments by running, jumping, climbing, and vaulting. It has grown from a street practice into an internationally recognized sport.

The **FIG (Federation Internationale de Gymnastique)** has taken responsibility for standardizing parkour competition formats, developing a Code of Points, and organizing international events. Parkour has been discussed as a candidate discipline for future Olympic Games.

Automated, objective trick notation is a critical piece of this evolution. Competition judging in acrobatic disciplines relies on consistent identification of tricks and difficulty ratings. PkVision provides a tool that can assist judges with:

- Reliable trick identification at real-time or near-real-time speed
- Transparent, auditable scoring decisions
- A growing catalog of tricks with standardized difficulty ratings
- A system that improves as the community contributes more training data

> **Disclaimer:** This project references FIG and Olympic standards for context and alignment purposes. PkVision is an independent, community-driven open-source project and is not officially affiliated with the FIG, the IOC, or any national gymnastics or parkour federation.

---

## Contributing

We welcome contributions from developers, athletes, coaches, judges, and anyone interested in parkour and computer vision.

- **Code** -- Bug fixes, features, performance improvements
- **Trick definitions** -- Add new tricks to the catalog as JSON files
- **Training clips** -- Submit videos to improve detection accuracy
- **Translations** -- Add language codes to trick names

See [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md) for the full guide.

---

## References

- **Parkour Theory** -- [https://parkourtheory.com](https://parkourtheory.com)
- **FIG Parkour** -- [https://www.gymnastics.sport/site/pages/disciplines/pres-PK.php](https://www.gymnastics.sport/site/pages/disciplines/pres-PK.php)
- **International Parkour Federation -- Judges Criteria** -- [https://internationalparkourfederation.org/judges-criteria/](https://internationalparkourfederation.org/judges-criteria/)

---

## License

MIT License -- free for everyone, forever.

See [LICENSE](LICENSE) for the full text.
