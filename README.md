<p align="center">
  <img src="assets/banner.png" alt="PkVision" width="500">
</p>

<p align="center">
  <strong>See Every Move. Score Every Trick.</strong><br>
  Open-source AI for parkour trick detection and scoring using 3D biomechanics.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License: MIT">
  <img src="https://img.shields.io/badge/Python-3.11%2B-blue.svg" alt="Python 3.11+">
  <img src="https://img.shields.io/badge/CUDA-Required-orange.svg" alt="CUDA Required">
  <img src="https://img.shields.io/badge/Contributions-Welcome-brightgreen.svg" alt="Contributions Welcome">
</p>

---

## Table of Contents

- [What is Parkour?](#what-is-parkour)
- [What is PkVision?](#what-is-pkvision)
- [How It Works](#how-it-works)
- [The Science](#the-science)
- [For Athletes and Coaches](#for-athletes-and-coaches)
- [For Judges and Federations](#for-judges-and-federations)
- [Trick Knowledge Base](#trick-knowledge-base)
- [Getting Started](#getting-started)
- [Technology Stack](#technology-stack)
- [Research Context](#research-context)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)

---

## What is Parkour?

### Origins

Parkour was born in the suburbs of Lisses, France, in the late 1980s. Developed by David Belle -- inspired by the military obstacle course training (*parcours du combattant*) of his father Raymond Belle -- parkour is a discipline of movement. The goal: get from point A to point B as efficiently and fluidly as possible, using only the human body to overcome obstacles.

Over time, the discipline branched into distinct but overlapping practices. **Parkour** emphasizes efficiency and flow. **Freerunning**, popularized by Sebastien Foucan, adds acrobatic expression -- flips, twists, and aerial creativity. **Tricking** focuses purely on acrobatic combinations on flat ground. PkVision covers the competitive and acrobatic side of these disciplines, where trick identification and scoring matter.

### The Trick System

Parkour and freerunning tricks follow a **compositional naming convention** that directly encodes their physics. The name tells you exactly what the body does:

- **"Back Flip"** = backward direction + 1 flip + 0 twists
- **"Back Double Full"** = backward + 1 flip + 2 twists
- **"Double Cork"** = 2 flips + twists + off-axis rotation + one-foot takeoff

Tricks belong to families -- **flips** (rotation around the lateral axis), **twists** (rotation around the longitudinal axis), **vaults** (obstacle-based movements), and **transitions** (connecting movements). Within each family, tricks build on each other in **progression trees**:

```
Back Flip --> Back Full --> Back Double Full --> Back Triple Full
    |             |
  Gainer        Cork --> Double Cork --> Triple Cork
    |
  Webster
```

A gainer is a backflip with a one-foot running takeoff. A cork adds off-axis rotation. Each variation changes one physical property -- and changes the trick's name. This compositional structure is what makes automated identification possible.

### From the Streets to the Olympics

Parkour has grown from a street practice into an internationally recognized sport. The **FIG (Federation Internationale de Gymnastique)** has taken responsibility for standardizing competition formats, developing a Code of Points, and organizing international events. Parkour has been discussed as a candidate discipline for future Olympic Games.

As competition scales to the international level, **automated and reproducible trick notation becomes essential**. Human judges need to identify tricks precisely, consistently, and fairly across nations. In acrobatic disciplines like gymnastics, this is already supported by systems like Fujitsu's Judging Support System. But parkour has over 1,800 known tricks -- far more than any gymnastics discipline -- and no open-source system exists to identify them from video.

PkVision fills this gap.

---

## What is PkVision?

PkVision is an open-source system that identifies and scores parkour tricks from video using **3D body mesh reconstruction** and **zero-shot biomechanical matching**. It understands how the human body moves through 3D space -- rotation axes, flip counts, twist counts, body shape, takeoff type -- and matches these measurements against 1,800+ known parkour tricks without needing training videos for each one.

**The core idea:** A backflip is a 360 degree backward rotation around the lateral axis in a tucked position. If the system can measure those properties in 3D from video, it can name the trick -- even tricks it has never seen before.

<p align="center">
  <img src="docs/paper_figures/triptych_backflip.png" alt="From Video to 3D Biomechanics" width="700">
  <br>
  <em>Left: Input frame. Center: 2D joint locations (ViTPose). Right: 3D biomechanical analysis overlay.</em>
</p>

### Why 3D?

Previous approaches (including PkVision v1) used 2D pose estimation, which fundamentally cannot distinguish many tricks:
- A **cork** and a **back full** look identical from certain camera angles in 2D
- **Twists** are invisible when filmed from the side
- **Rotation counting** is imprecise from 2D projections

By reconstructing a full 3D body mesh from video, all ambiguity disappears. The rotation axis is a 3D vector, not a guess.

---

## How It Works

<p align="center">
  <img src="docs/paper_figures/pipeline_architecture.png" alt="Pipeline Architecture" width="800">
</p>

```
Video (smartphone, GoPro, competition camera)
  |
  +- Stage 1: 2D Pose Estimation (ViTPose -- 17 COCO keypoints)
  |  Detects the athlete and tracks 17 body joints per frame.
  |
  +- Stage 2: 3D Mesh Recovery (GVHMR -- SMPL body model)
  |  Reconstructs a full 3D body mesh in world coordinates,
  |  aligned with gravity. Outputs per-frame:
  |    - global_orient (3D) -- body orientation as axis-angle
  |    - body_pose (63D) -- 21 joint rotations
  |    - transl (3D) -- world position in meters
  |
  +- Stage 3: Biomechanical Feature Extraction
  |  From the SMPL parameters, computes:
  |    - Swing-twist decomposition --> flip degrees + twist degrees
  |    - Joint angles --> body shape (tuck / pike / layout)
  |    - Ankle analysis --> takeoff type (one-foot / two-foot)
  |    - COM trajectory --> jump height, aerial phase detection
  |
  +- Stage 4: Trick Segmentation
  |  Detects where each trick starts and ends in a full run
  |  using 3D angular velocity peaks + aerial phase detection.
  |
  +- Stage 5: Zero-Shot Trick Matching
  |  Compares the 3D biomechanical signature against 1,800+
  |  trick definitions. No training videos needed per trick.
  |
  +- Stage 6: Scoring (Top 3 by difficulty x confidence)
```

### Key Results

| Clip | Measured Flip | Measured Twist | Body Shape | Direction | Takeoff |
|------|:---:|:---:|:---:|:---:|:---:|
| Backflip | 349 deg (1.0 flip) | 20 deg (0 twist) | Tuck | Backward | Two-foot |
| Frontflip | 372 deg (1.0 flip) | 9 deg (0 twist) | Tuck | Forward | Two-foot |
| Back Double Full | 379 deg (1.1 flip) | 798 deg (2.2 twists) | Layout | Backward | Two-foot |
| Double Cork | 703 deg (2.0 flips) | 543 deg (1.5 twists) | Layout | Backward | One-foot |
| Gainer | 382 deg (1.1 flip) | 25 deg (0 twist) | Tuck | Backward | One-foot |

<p align="center">
  <img src="docs/paper_figures/pipeline_backflip.png" alt="Backflip Pipeline Analysis" width="800">
  <br>
  <em>Full pipeline analysis of a backflip: COM trajectory, angular velocity, cumulative rotation, joint angles, 3D trajectory, and biomechanical signature.</em>
</p>

---

## The Science

### Swing-Twist Decomposition

Every frame, the body's 3D rotation is decomposed into two components:

- **Swing (flip):** Rotation around the body's lateral axis (left-right). A backflip accumulates ~360 deg of swing.
- **Twist:** Rotation around the body's longitudinal axis (head-to-toe). A full twist accumulates ~360 deg.

This separation is the key insight. A **back full** = 360 deg swing + 360 deg twist. A **double cork** = 720 deg swing + 540 deg twist + off-axis entry. The system counts degrees, not patterns.

```python
from scipy.spatial.transform import Rotation

# For each consecutive frame pair:
R_delta = R_curr * R_prev.inv()           # Incremental rotation
body_y = R_prev.apply([0, 1, 0])          # Body's longitudinal axis in world
swing, twist = swing_twist_decompose(R_delta, body_y)
cumulative_flip += swing                   # Count flip degrees
cumulative_twist += twist                  # Count twist degrees
```

### Body Shape Classification

Joint angles from the SMPL body model directly reveal body shape during the aerial phase:

| Shape | Knee Angle | Hip Angle | Description |
|-------|:---:|:---:|---|
| **Tuck** | > 60 deg from rest | > 40 deg from rest | Knees to chest, tight ball |
| **Pike** | > 50 deg from rest | < 30 deg from rest | Straight legs, folded at hips |
| **Layout** | < 35 deg from rest | < 30 deg from rest | Fully extended body |
| **Open** | Layout + arms spread (shoulder > 100 deg) | -- | Extended with arms out |

### Takeoff Detection

The system analyzes pre-takeoff frames to determine entry type:

- **Two-foot:** Symmetric knee/hip angles at takeoff (backflip, front flip)
- **One-foot:** Asymmetric angles -- one leg kicks while the other plants (gainer, cork, webster)
- **Running:** Significant horizontal COM velocity at takeoff
- **Wall:** Horizontal velocity reversal near takeoff (wall push-off)

### Zero-Shot Matching

Each trick in the catalog is defined by its physics:

```python
"back_flip":   {"flips": 1.0, "twists": 0.0, "direction": "backward", "takeoff": "two_foot", "shape": "tuck"}
"double_cork": {"flips": 2.0, "twists": 2.0, "direction": "backward", "takeoff": "one_foot", "shape": "layout", "axis": "off_axis"}
"gainer":      {"flips": 1.0, "twists": 0.0, "direction": "backward", "takeoff": "one_foot", "shape": "tuck"}
```

The matcher computes a weighted distance between the measured 3D signature and each definition:

| Property | Weight | Why |
|---|:---:|---|
| Flip count | 30% | Strongest discriminator (single vs double vs triple) |
| Twist count | 20% | Distinguishes full from non-twist variants |
| Direction | 15% | Forward vs backward |
| Takeoff type | 15% | Backflip vs gainer, double full vs cork |
| Axis type | 15% | On-axis (flip) vs off-axis (cork) |
| Body shape | 5% | Tuck vs layout vs pike |

**No training videos needed.** Adding a new trick = adding one line to the definition table. The 1,837 known tricks map to **193 unique property combinations**. The matching table grows linearly -- adding trick #1,838 is adding one row.

---

## For Athletes and Coaches

Your clips are what make this system work. The more diverse the data, the more accurate PkVision becomes for everyone.

### How You Can Help

- **Submit training clips** -- Open a [Clip Submission issue](../../issues/new?template=clip_submission.yml) on GitHub with a link to your video.
- **Propose new tricks** -- If a trick is missing from the catalog, open a [Trick Proposal issue](../../issues/new?template=trick_submission.yml).
- **Review detections** -- Try PkVision on your own clips and report inaccuracies.

### Filming Guidelines

For best detection results:

- **Resolution:** 720p or higher
- **Frame rate:** 30fps or higher
- **Camera angle:** Side or diagonal preferred
- **Framing:** Full body visible throughout the trick
- **Timing:** 1-2 seconds of buffer before and after the trick
- **Clothing:** Contrasting colors against the background
- **Stability:** Tripod or stable surface preferred
- **Content:** One trick per clip when possible

See [docs/CLIP_GUIDELINES.md](docs/CLIP_GUIDELINES.md) for the complete guide.

---

## For Judges and Federations

PkVision is built with competition integrity in mind.

### Scoring

- **Top 3 by difficulty** -- The system selects the 3 most difficult tricks detected, aligned with FIG competition scoring structures.
- **Weighted scores** -- Each trick's score is `difficulty x confidence`, rewarding both ambition and clean execution.

### Transparency

- **Full audit trail** -- Every detection includes reasoning: which properties matched, the confidence level, and per-property breakdowns.
- **Human override** -- Judges always have the final say. Overrides create new audit entries; the original AI decisions are never deleted or modified.
- **Immutable history** -- The audit log is append-only. All entries (AI detections, scoring decisions, judge overrides) are preserved for review.

### Neutrality

- **Nation-neutral** -- No geographic bias in detection or scoring.
- **Multi-language** -- Trick names and output are available in English and French, with a straightforward path to add more languages.

### Multi-Camera Competition Mode

For competition-grade precision, PkVision supports multiple synchronized cameras:

```
Camera 1 (side)  --+
Camera 2 (front) --+-- Pose2Sim (3D triangulation) -- SMPL fitting -- Biomechanics -- Matching
Camera 3 (top)   --+
```

With 3-4 cameras, there is **zero ambiguity** in rotation axes, twist counts, and body shape. Single-camera mode already achieves good accuracy; multi-camera mode is for the highest precision.

**Tools:** [Pose2Sim](https://github.com/perfanalytics/pose2sim) for multi-view triangulation, compatible with any camera (GoPro, smartphone, webcam).

### References

- [FIG Parkour Code of Points](https://www.gymnastics.sport/site/pages/disciplines/pres-PK.php)
- [International Parkour Federation -- Judges Criteria](https://internationalparkourfederation.org/judges-criteria/)

> **Note:** This project references FIG and Olympic standards for context but is not officially affiliated with the FIG, the IOC, or any national federation.

---

## Trick Knowledge Base

PkVision draws from multiple sources to cover 1,800+ parkour tricks:

| Source | Tricks | Data |
|---|:---:|---|
| [Parkour Theory](https://parkourtheory.com) | 1,837 | Names, types, descriptions, prerequisites, subsequents |
| [Loopkicks Tricktionary](https://loopkickstricking.com/tricktionary) | 943 | Names, descriptions, categories (forward/backward/vertical) |
| [Tricking Bible](data/tricking_bible.pdf) | 45 | Difficulty classes (A-F), type, origin, prerequisites |
| Name parsing | 1,837 --> 193 | Auto-parsed into physics parameters from trick names |

All 1,837 trick names have been parsed into physics parameters (rotation axis, direction, count, twist, body shape, entry type) using the compositional naming convention of parkour. The name "Back Double Full" directly encodes: backward + 1 flip + 2 twists.

### Trick Progression Trees

Prerequisite/subsequent data gives us the progression graph, used for difficulty estimation and suggesting which tricks an athlete should learn next:

```
Back Flip --> Back Full --> Back Double Full --> Back Triple Full
    |             |
  Gainer        Cork --> Double Cork --> Triple Cork
    |
  Webster
```

---

## Getting Started

### Requirements

- Python 3.11+
- NVIDIA GPU with CUDA (for GVHMR inference)
- ~6GB VRAM minimum (RTX 2060 or better)

### Installation

```bash
git clone https://github.com/AirKyzzZ/pkvision.git
cd pkvision
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Install GVHMR (see docs/INSTALL_GVHMR.md for full guide)
git clone https://github.com/zju3dv/GVHMR.git
cd GVHMR && pip install -e . && cd ..

# Download model checkpoints (SMPL, GVHMR, ViTPose, HMR2)
# See docs/INSTALL_GVHMR.md for download links
```

### Analyze a Video

```bash
# Single trick clip
python scripts/analyze.py --input video.mp4

# Full competition run (auto-segments tricks)
python scripts/analyze.py --input full_run.mp4 --segment
```

### Run Tests

```bash
pytest tests/unit/ -q
# 250+ tests covering biomechanics, matching, segmentation, scoring
```

### Adding a New Trick

Adding a trick requires **no training, no video, no code changes**. Edit `data/parsed_tricks.json`:

```json
{
  "name": "My New Trick",
  "rotation_axis": "off_axis",
  "direction": "backward",
  "rotation_count": 2.0,
  "twist_count": 1.5,
  "body_shape": "layout",
  "entry": "one_leg",
  "family": "flip"
}
```

The matcher will immediately recognize this trick from any video that matches these properties.

---

## Technology Stack

### 3D Body Mesh Recovery: GVHMR

[GVHMR](https://github.com/zju3dv/GVHMR) (SIGGRAPH Asia 2024) reconstructs a world-grounded 3D body mesh from monocular video. It outputs [SMPL](https://smpl.is.tue.mpg.de/) body model parameters aligned with gravity.

**Why GVHMR over alternatives:**
- **Gravity-aligned** -- knows which direction is "up" regardless of camera angle
- **World coordinates** -- positions in meters, not pixels
- **Best accuracy** -- 19% better than WHAM on world-grounded trajectory metrics
- **Fast** -- 1.9 seconds for 8.6 seconds of video on RTX 2060

### 2D Pose Estimation: ViTPose

[ViTPose](https://github.com/ViTAE-Transformer/ViTPose) provides 17 COCO keypoints as input to GVHMR. YOLO handles person detection and tracking.

### Biomechanical Analysis: scipy + custom

The swing-twist decomposition and joint angle extraction use `scipy.spatial.transform.Rotation` with custom code in `core/pose/biomechanics.py`.

### Scoring & Segmentation: Custom

The scoring engine (`core/scoring/engine.py`) and run segmenter (`core/recognition/segmentation.py`) are custom implementations.

---

## Research Context

PkVision draws on recent advances in computer vision and sports biomechanics:

| Technology | Paper / Source | Role in PkVision |
|---|---|---|
| [GVHMR](https://github.com/zju3dv/GVHMR) | SIGGRAPH Asia 2024 | 3D body mesh recovery |
| [ViTPose](https://github.com/ViTAE-Transformer/ViTPose) | TPAMI 2023 | 2D pose estimation |
| [SMPL](https://smpl.is.tue.mpg.de/) | SIGGRAPH Asia 2015 | Parametric body model |
| [Fujitsu JSS](https://www.fujitsu.com/global/themes/data-driven/judging-support-system/) | Production system | Inspiration (gymnastics judging) |
| [AthletePose3D](https://github.com/calvinyeungck/AthletePose3D) | CVPR 2025 Workshop | Extreme pose fine-tuning data |
| [Pose2Sim](https://github.com/perfanalytics/pose2sim) | Open source | Multi-camera 3D reconstruction |
| Swing-twist decomposition | Classical mechanics | Rotation analysis |

---

## Roadmap

### Completed
- [x] 3D body mesh recovery from monocular video (GVHMR)
- [x] Swing-twist decomposition for flip/twist counting
- [x] Body shape classification (tuck/pike/layout)
- [x] Takeoff detection (one-foot vs two-foot)
- [x] Zero-shot trick matching against 1,800+ definitions
- [x] Run segmentation for full competition runs
- [x] 1,837 trick names parsed into physics parameters
- [x] Pipeline visualization figures for research paper

### In Progress
- [ ] Direction convention calibration (SMPL axis alignment)
- [ ] Cork vs double full differentiation (off-axis detection refinement)
- [ ] Multi-camera competition mode (Pose2Sim integration)

### Planned
- [ ] Landing quality scoring (joint angles at ground contact)
- [ ] Real-time processing pipeline
- [ ] Web interface for live competition judging
- [ ] Obstacle detection (wall, bar, rail) for vault/bar tricks
- [ ] AthletePose3D fine-tuning for better extreme pose accuracy

---

## Contributing

We welcome contributions from developers, athletes, coaches, judges, and anyone interested in parkour and computer vision.

- **Code** -- Bug fixes, features, pipeline improvements
- **Trick definitions** -- Add physics parameters for new tricks
- **Training clips** -- Submit videos of known tricks for validation
- **Biomechanics expertise** -- Help refine body shape / takeoff / landing detection
- **Translations** -- Add language codes to trick names

See [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md) for the full guide.

---

## License

MIT License -- free for everyone, forever.

See [LICENSE](LICENSE) for the full text.
