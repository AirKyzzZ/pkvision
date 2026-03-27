# PkVision — Next Session Setup

## What This Project Is

PkVision is a **3D biomechanics-based trick recognition system** for FIG Parkour competitions. It takes video input, reconstructs a full 3D body mesh using GVHMR, then identifies tricks from the FIG Code of Points 2025 using rotation physics.

## Current State

### What works
- **GVHMR inference** runs on windows-dev (RTX 2060) — produces SMPL parameters from video
- **Rotation tracking** with Berry phase correction — accurately counts flips/twists for lateral tricks
- **FIG trick matcher** — 148 tricks from the official 2025 Code of Points loaded from `data/fig_tricks_2025.json`
- **Run segmentation** — splits competition runs into individual tricks
- **Direction detection** — forward/backward via cross-product flip axis method
- **Off-axis detection** — corks detected from peak-frame tilt statistics

### What does NOT work and needs fixing
1. **No 3D mesh rendering** — pytorch3d won't compile with CUDA on Windows. The GVHMR demo videos show a full SMPL mesh (like Blender), but we can only see raw SMPL parameters. **This is the #1 priority** — we need to SEE the 3D model to verify tracking quality.
2. **GVHMR tracking quality on competition footage** — works great on close-up single-trick clips, fails on wide-angle competition videos. Flip counts drop to 0.5x instead of 1.0x.
3. **Old 2D detection code still in the codebase** — needs cleanup. The 2D approach (YOLO keypoints → angular velocity) is inferior to the 3D pipeline.
4. **No Layer 3 disambiguation** — backflip vs palm flip vs gainer all look identical from rotation physics. Need hand contact + takeoff detection from SMPL body_pose.

## Priority Tasks for Next Session

### 1. Get SMPL Mesh Rendering Working (CRITICAL)
The GVHMR demo renders a full 3D mesh like this: https://zju3dv.github.io/gvhmr/
We need to see this for our videos to verify GVHMR is tracking correctly.

**Options:**
- Fix pytorch3d CUDA compilation (the `nv/target` header issue)
- OR use pyrender (CPU/OpenGL) as an alternative renderer
- OR export SMPL mesh to .obj/.glb and view in Blender

**The rendering code is in:** `GVHMR/tools/demo/demo.py` function `render_incam()`
**It crashes at:** `pytorch3d.renderer.mesh.rasterize_meshes` — "Not compiled with GPU support"
**SMPL model file:** `GVHMR/inputs/checkpoints/body_models/smpl/SMPL_NEUTRAL.pkl` (247MB)

### 2. Fix GVHMR Quality on Competition Footage
- Consider YOLO pre-processing to crop/zoom the athlete before GVHMR
- Test with different video resolutions
- The test clip `data/run_testing/test_run_2.mp4` has 7 tricks but GVHMR only tracks ~50% accurately

### 3. Clean Up Old 2D Code
Remove or archive the 2D detection pipeline:
- `core/pose/features.py` (2D feature extraction)
- `scripts/test_run.py` (2D YOLO-based analysis)
- `scripts/test_run_physics.py` (2D physics-based analysis)
Keep only the 3D pipeline (`core/pose/rotation_tracker.py` + `scripts/analyze_3d.py`)

### 4. Layer 3 Disambiguation
Extract from SMPL `body_pose` joint angles:
- Hand contact detection (palm flip vs backflip)
- Takeoff type (webster vs frontflip = one-leg, gainer = running forward)
- Kick detection (frisbee vs aerial)

## Architecture

```
iPhone Video → GVHMR (RTX 2060) → SMPL Parameters → Rotation Tracker → FIG Matcher → D-Score
                                        ↓
                                   3D Mesh Render (for verification)
```

### Layer 1: Context Detection
Ground / Wall / Bar → determines FIG trick category

### Layer 2: Physics (BUILT)
Flip count, twist count, direction, axis, body shape → narrows to physics family

### Layer 3: Specific Trick (NEEDED)
Hand contact, takeoff type, kick, entry pattern → identifies exact FIG trick

### Layer 4: Scoring
D-score from FIG table + scaling modifiers (placement, form, entry, exit)

## Key Files

| File | Purpose |
|------|---------|
| `core/pose/rotation_tracker.py` | Berry phase-corrected rotation tracking (the core engine) |
| `core/recognition/matcher.py` | Zero-shot FIG trick matcher with context filtering |
| `ml/trick_physics.py` | FIG trick definitions loader from JSON |
| `data/fig_tricks_2025.json` | Official FIG Code of Points 2025 (148 tricks) |
| `scripts/analyze_3d.py` | End-to-end analysis pipeline |

## GVHMR Setup on Windows-Dev

- **Location:** `C:\Users\pc\GVHMR`
- **Python:** `C:\Users\pc\gvhmr_env\Scripts\python.exe` (Python 3.11 + PyTorch 2.5.1 CUDA 12.1)
- **CUDA:** 12.1 toolkit at `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1`
- **Run inference:** `cd C:\Users\pc\GVHMR && C:\Users\pc\gvhmr_env\Scripts\python.exe tools/demo/demo.py --video INPUT.mp4 -s`
- **Output:** `outputs/demo/{NAME}/hmr4d_results.pt`
- **Rendering crashes** at pytorch3d rasterizer — needs CUDA-compiled pytorch3d or alternative renderer

## Test Data

| File | Content | GVHMR Output |
|------|---------|--------------|
| `data/gvhmr_outputs/backflip/` | Single backflip (close-up) | ✓ Works well |
| `data/gvhmr_outputs/frontflip/` | Single frontflip | ✓ Works well |
| `data/gvhmr_outputs/back_double_full/` | Back double full | ✓ Works (twist=2.0 after Berry phase) |
| `data/gvhmr_outputs/double_cork/` | Double cork | ~OK (off-axis detected, values approximate) |
| `data/gvhmr_outputs/gainer/` | Single gainer | ✓ Works well |
| `data/gvhmr_outputs/IMG_4243/` | Competition run (6 tricks) | ✓ 8 tricks detected |
| `data/gvhmr_outputs/test_run_2/` | FISE-style run (7 tricks) | ✗ Poor tracking quality |
| `data/run_testing/test_run_2.mp4` | Raw video for test_run_2 | |
| `data/run_testing/elo_fise.mp4` | Eloan Hitz FISE 2025 | ✗ Camera too far |

## FIG Scoring System

- **D-score** (Difficulty): Base trick value from table + scaling
- **E-score** (Execution): Safety (9) + Landing Quality (3) + Flow (6) + Flow Quality (1)
- 4 trick categories: Swing (bar), Wall, Acrobatics (ground), PK Basics
- ~130 official tricks scored 0.1 to 7.7
- Source: `data/fig_tricks_2025.json`
