# 10: Tooling stack for "count, don't learn" V1 (state as of 2026-10-03)

Survey date: 2026-10-03. Desk research only. Nothing installed, no project data touched, $0 spent.
Versions and dates come from PyPI JSON (`https://pypi.org/pypi/<pkg>/json`), the GitHub API (`gh api`), Hugging Face
model APIs and the vendors' docs, all fetched on 2026-10-03.

Scope: which **tools** to run now, per layer, on an RTX 2060 6 GB (Windows, ssh), a Mac M3 Pro (MPS/MLX, ~12 GB free
disk) and an occasional rented 4090. This file does not repeat the science in `../2026-09-28/03-pose-hmr-acrobatics.md`
(trampoline fine-tune, RePoGen, SAM 3D Body accuracy, twist geometry) or `../2026-09-28/02-video-vlms.md` (VLM
benchmarks, API prices). It cites them where it relies on them.

Grading, same as 03: **A** = measured in-domain (parkour). **B** = measured on an analog. **C** = vendor claim, config
fact, derivation, or my estimate. Speed/VRAM numbers for the 2060 and the Mac are **C (my estimate)** unless a source
is given; no vendor publishes 2060 numbers.

---

## 0. Bottom line

1. **No new 2D pose model fixes inversion out of the box.** Nothing released in 2025–2026 reports accuracy on inverted
   or acrobatic bodies except the trampoline paper already in 03. RTMPose has no successor (no RTMPose2/RTMW2 found;
   MMPose's last release is 1.3.2 from 2024-07-12). rtmlib is alive (0.0.16, 2026-08-03) and stays the right runtime.
2. **The cheapest real lever is a config fact, not a new model.** RTMPose-x was trained with bbox rotation drawn from a
   truncated normal in **[-90°, +90°], applied with p = 0.6** (`RandomBBoxTransform(rotate_factor=90)` in the
   body8-halpe26 config; mmpose `_truncnorm(-1, 1)`), and Ultralytics trains with `degrees: 0.0` by default. A body
   past 90° of in-plane rotation is outside what these models saw. BlazePose (MediaPipe) solves exactly this by
   rotating the crop so the mid-hip to mid-shoulder line is vertical, using the previous frame's keypoints. Doing the
   same before RTMPose (rotate crop upright, infer, rotate back) costs no training and no money. Expected gain is
   **unmeasured** (C); it is the first thing to probe.
3. **Switch RTMPose-x checkpoint to the Halpe-26 variant** (same 17.3 GFLOPs, same rtmlib API, AP 80.0 on Body8 vs
   78.8 COCO for the 17-kp model per rtmlib's table). It adds head-top, neck, hip-centre, big toe, small toe and heel,
   which are the off-plane points the twist-direction rule needs (03 §5).
4. **Sapiens2 is now the strongest open 2D candidate and is easy to run**: in `transformers` 5.18.0 (2026-09-30) with
   pose preprocessing; 0.4B pose model is 0.398B params, 1.26 TFLOPs per 1024×768 crop, 308 keypoints. On Meta's own
   11K in-the-wild 308-kp test it gets 76.9 mAP (0.4B) vs 70.2 for RTMW-X retrained on the same keypoints (B). No
   COCO-17 AP, no acrobatic number, no VRAM figure published. Licence permits commercial use but bans "biometric
   processing" and health inference, which needs a legal read before a product.
5. **Licence hygiene is the main thing to change for the product track.** `core/pose/detector.py` loads
   `yolo26n-pose.pt` (Ultralytics, AGPL-3.0). BoxMOT is AGPL-3.0. DEIMv2 and EdgeCrafter/ECPose are explicitly
   non-commercial (2026 licence text). Apache-2.0 replacements exist for every layer: YOLOX/RTMPose (rtmlib), RF-DETR
   N–L, D-FINE, RT-DETRv4, roboflow `trackers`, SAM 2.1.
6. **Video I/O:** the repo decodes with `cv2.VideoCapture` and trusts `CAP_PROP_FPS`. iPhone footage is often
   variable-frame-rate and slow-mo files carry retiming metadata (C). Use PyAV 19 / torchcodec 0.17 for real
   timestamps. `decord` is dead (last release 2021, wheels for cp36–cp38 only).
7. **The 2060 is still supported** by CUDA 13.x (only pre-Turing removed), TensorRT 11.3 (SM 7.5 minimum) and PyTorch
   2.14 Windows wheels (cu128–cu134 arch list starts at 7.5). Limits: no bf16, no FlashAttention-2/3 (C), 6 GB.

---

## 1. What the repo uses today (read from the code, 2026-10-03)

| Layer | Current | Source in repo | Note |
|---|---|---|---|
| Detector + 2D pose (P1 extraction) | `rtmlib.Body(mode="performance", backend="onnxruntime")` = **YOLOX-x (HumanArt+COCO, person AP 61.3) + RTMPose-x body7 384×288 (COCO AP 78.8)** | `scripts/extract_pose.py`, `scripts/extract_pose_modal.py` | YOLOX-x is ~282 GFLOPs at 640 vs 17 GFLOPs for RTMPose-x (C, YOLOX paper figures), so the detector dominates compute when run every frame |
| Runtime | `onnxruntime-gpu==1.19.2` pinned on Modal | `extract_pose_modal.py` | Current is 1.30.0 (2026-09-10), CUDA 13 by default |
| Alt pose | `yolo26n-pose.pt` via `ultralytics>=8.3.0` | `core/pose/detector.py`, `requirements.txt` | AGPL-3.0 |
| Decoding | `cv2.VideoCapture`, `CAP_PROP_FPS` | all three files above | Nominal fps only, no PTS |
| 3D | MotionBERT script | `scripts/motionbert_3d_judge.py` | Apache-2.0 code, H3.6M-trained weights (dataset licence is research-only, C) |
| Tracking | rtmlib `PoseTracker` greedy IoU (if used) | rtmlib `pose_tracker.py` | Fine for one athlete |
| Experiment tracking | none found in code | grep for mlflow/wandb/trackio | |

---

## 2. Hardware facts that constrain choices

| Fact | Value | Source | Grade |
|---|---|---|---|
| CUDA 13.x arch support | "Removed support for Maxwell, Pascal, and Volta GPUs"; Turing kept. Current CUDA 13.4 Update 1; CUDA 13 needs driver ≥ 580 | NVIDIA CUDA Toolkit release notes | C (vendor) |
| TensorRT | 11.3.0 (PyPI 2026-09-09, proprietary licence): "supports NVIDIA hardware with compute capability SM 7.5 or higher" | TensorRT support matrix | C |
| PyTorch | 2.14.1 (2026-09-30). Windows wheel arch list for cu128/129/130/132/134 = `7.5;8.0;8.6;9.0;10.0;12.0` | `pytorch/.ci/wheel/windows/build_env_setup.py` | C (build config) |
| ONNX Runtime GPU | 1.30.0 (2026-09-10); from 1.27 PyPI GPU builds use CUDA 13.0; separate CUDA 12.8 builds exist | onnxruntime.ai CUDA EP docs | C |
| Turing limits | no native bf16, FlashAttention-2/3 need Ampere+; use fp16 or fp32. Models trained in bf16 (Sapiens2, SAM 3) can overflow in fp16 | general knowledge | C |
| Mac | MPS via torch, CoreML EP via onnxruntime (rtmlib maps `device="mps"` to `CoreMLExecutionProvider`), MLX 0.32.3 (2026-09-29) | rtmlib `tools/base.py`; PyPI | C |

---

## 3. Person detection and tracking

### 3.1 Detectors (COCO val2017, all classes unless noted)

| Model | Version / date | COCO AP50:95 | T4 TRT FP16 latency | Params | Licence | 2060 / Mac | Grade |
|---|---|---|---|---|---|---|---|
| YOLOX-x HumanArt (rtmlib default "performance") | weights 2023 | **61.3 person AP** on HumanArt+COCO | n/a | 99M (C) | Apache-2.0 code; HumanArt data terms not verified | fits | C |
| YOLOX-m HumanArt ("balanced") | 2023 | 59.1 person AP | n/a | 25M (C) | same | fits | C |
| **YOLO26** n/s/m/l/x | ultralytics 8.4.0, **2026-01-14**; latest 8.4.172 (2026-10-03) | 40.3 / 47.7 / 52.5 / 54.1 / 56.9 (SAB-measured by Roboflow; Ultralytics claims 40.9–57.5) | 1.7–9.6 ms | 2.6–56.9M | **AGPL-3.0** / Enterprise | fits | B |
| YOLO27 | "final R&D… no set launch date" | n/a | n/a | n/a | AGPL | n/a | C |
| **RF-DETR** N/S/M/L | rfdetr 1.11.1 (2026-09-30), ICLR 2026 (arXiv 2511.09554) | 48.4 / 53.0 / **54.7** / **56.5** | 2.3 / 3.5 / 4.4 / 6.8 ms | 30–34M | **Apache-2.0** | fits; also callable from rtmlib 0.0.16 low-level API | B |
| RF-DETR XL / 2XL | same | 58.6 / 60.1 | 11.5 / 17.2 ms | 126M | **PML 1.0** (`rfdetr_plus`) | fits | B |
| D-FINE N–X | repo pushed 2026-08-19 | 42.7–59.3 (SAB) | 2.1–11.5 ms | 3.8–62M | Apache-2.0 | fits | B |
| RT-DETRv4 S–X | ECCV 2026, arXiv 2510.25257, repo pushed 2026-07-06 | 49.8 / 53.7 / 55.4 / 57.0 (self-reported) | 3.7–12.9 ms | n/a | Apache-2.0 | fits | C |
| DEIMv2 Atto–X | 2025-09-26 release, arXiv 2509.20787 | 23.8–57.8 (self-reported) | 1.1–13.8 ms | 0.5–50M | **DEIMv2 License: non-commercial only** (2026 text) | fits | C |
| EdgeCrafter ECDet | TMLR 2026, repo 2026-08 | 51.7–59.9 | 5.4–12.7 ms | 10–49M | **EdgeCrafter License: non-commercial only** | fits | C |
| SAM 3 (detector+tracker) | 2025-11-19; SAM 3.1 2026-03-27 | RF100-VL 61.6 AP50:95 fine-tuned (paper) | 30 ms/img on H200 (Ultralytics docs) | ~850M per paper; Ultralytics page says 473.6M (conflict) | SAM License, gated HF | 4090 only (C) | C |

Detection is not PkVision's bottleneck for one athlete per clip. The relevant risk is recall on inverted, blurred
bodies, and **no detector reports that** (not found). Practical choice: keep rtmlib's YOLOX (Apache) and drop from
-x to -m, or swap to RF-DETR-M via rtmlib 0.0.16 (`RFDETR` class added in PR #72), and run detection only to
(re)initialise while keypoint-derived boxes track the athlete in between (rtmlib `pose_to_bbox`, 1.25 expansion).
Caveat (C): a box derived from the previous frame lags during a fast rotation, so re-detect every frame inside the
flight window.

### 3.2 Trackers

| Tool | Version / date | Algorithms | Benchmarks (HOTA, default params) | Licence | Grade |
|---|---|---|---|---|---|
| **roboflow `trackers`** | 2.6.1 (2026-09-25) | SORT, ByteTrack, OC-SORT, BoT-SORT, C-BIoU, McByte | SportsMOT: ByteTrack 73.0, OC-SORT 71.7, BoT-SORT 73.8, McByte 76.5. DanceTrack: 53.3 / 54.1 / 57.8 / 67.2 | **Apache-2.0** | B |
| Ultralytics `model.track` | 8.4.172 | TrackTrack (default), BoT-SORT, ByteTrack, OC-SORT, Deep OC-SORT, FastTracker | not restated | AGPL-3.0 | C |
| BoxMOT | 25.0.0 (2026-09-09) | many incl. OccluBoost, native C++ backend | in repo | **AGPL-3.0** | C |
| rtmlib `PoseTracker` | 0.0.16 | greedy IoU on keypoint boxes, `det_frequency` | none | Apache-2.0 | C |
| **SAM 2.1** | repo pushed 2026-05-30 (PyPI `sam2` stale at 1.1.0, 2024-12-21; install from git) | promptable video masklets | n/a | Apache-2.0 | C |
| **SAMURAI** | last push 2025-03-18 | SAM 2.1 + Kalman motion-aware memory, zero-shot | LaSOT/NfS SOTA (README) | Apache-2.0 | B |
| SAM 3 / 3.1 video | 2026-03-27 (3.1 "Object Multiplex", ~7× faster at 128 objects on H100) | text-prompted detection + tracking | SA-Co/VEval | SAM License, gated; needs Py 3.12, torch ≥ 2.7, CUDA ≥ 12.6 | C |

For one athlete, MOT is overkill. A mask track (SAM 2.1 or SAMURAI, prompt with the first detection) is the most
robust way to keep the right person through blur and self-occlusion, and the mask also gives a clean crop for pose and
for VLM frames. SAM 2.1 small/base+ should fit 6 GB in fp16 (C). SAM 3 is a 4090 job.

### 3.3 Motion blur

Only capture-side fixes are measured (03 §3.2: short shutter, 120/240 fps). No detector or pose model reports
blur-robustness on sports footage beyond SDPose's COCO-OOD corruption split (03) and TAR-ViTPose's PoseTrack gain
(§5). No deblurring pre-step validated for pose was found (not found).

---

## 4. 2D pose

### 4.1 Candidates

COCO AP is COCO val2017 keypoint AP unless noted. FLOPs per person crop.

| Model | Version / date | AP | Keypoints | Params / FLOPs | Speed | 2060 6 GB | Mac | Licence | Grade |
|---|---|---|---|---|---|---|---|---|---|
| **RTMPose-x body7** (current) | weights 2023-06-29; rtmlib 0.0.16 (2026-08-03) | **78.8** COCO (384×288) | COCO-17 | 49.4M / 17.2 G | not published for 2060; ran on Modal T4 | fits (C) | CoreML EP / CPU | Apache-2.0 (data mix not verified) | C |
| **RTMPose-x Halpe26** | 2023-06-06 | **80.0** Body8 AP | Halpe-26 (COCO-17 + head-top, neck, hip, 3 points per foot) | 50.0M / 17.3 G | same | fits | same | Apache-2.0 | C |
| RTMO-l (one-stage) | 2023-12 | 74.8 | COCO-17 | n/a | n/a | fits | same | Apache-2.0 | C |
| RTMW-x | 2023-11 | 70.2 COCO-WholeBody | 133 | 29.3 G | n/a | fits | same | Apache-2.0 | C |
| DWPose-l | 2023-07 | 66.5 COCO-WholeBody | 133 | n/a | n/a | fits | same | Apache-2.0 | C |
| RTMW3D-x (in rtmlib) | 2024-06 weights | 68.0 COCO-WholeBody | 133, 3D per frame | n/a | n/a | fits (C) | same | Apache-2.0 code; HF weights `Soykaf/RTMW3D-x` licence not verified | C |
| ViTPose++-s/b/l (rtmlib ONNX) | 2023 | 75.8 / 77.0 / 78.6 | COCO-17 (+25, +133 variants) | n/a | n/a | fits | same | Apache-2.0 | C |
| ViTPose / ViTPose++ in `transformers` | `usyd-community/vitpose-*`, 2025-01 | as above | COCO-17 | b/l/h | n/a | b/l fit | MPS | Apache-2.0 | C |
| **Sapiens2-0.4B pose** | 2026-04-24 release; `transformers` 5.18.0 (2026-09-30) adds pose preprocessing | **COCO not reported**. 76.9 mAP on Meta 11K 308-kp in-the-wild test (flip test, GT boxes) vs RTMW-X 70.2, DWPose-L 66.5 retrained on same keypoints | 308 (Sociopticon); COCO-17 subset needs mapping | 0.398B / **1.26 T** at 1024×768 | not published; my estimate 5–10 fps on a 2060 in fp16 | likely fits (0.8 GB fp16 weights), fp16 overflow risk | MPS via transformers, slow (C) | **Sapiens2 License**: commercial allowed, bans biometric processing, surveillance, health inference; audit clause | B (Meta test) / C (fit) |
| Sapiens2-0.8B / 1B / 5B pose | same | 79.4 / 80.4 / 82.3 | 308 | 0.82 / 1.46 / 5.07B | n/a | 0.8B maybe, 1B+ no (C) | 0.8B maybe | same | B |
| **YOLO26-pose** n/s/m/l/x | 8.4.0, 2026-01-14 | 57.2 / 63.0 / 68.8 / 70.4 / **71.6** (Ultralytics, e2e); Roboflow SAB measures x at 71.0 | COCO-17, RLE head | 2.9–57.6M / 7.6–202 G (whole image) | T4 TRT 1.8–12.2 ms | fits | MPS | **AGPL-3.0** | B |
| **RF-DETR Keypoint (preview)** | rfdetr 1.10–1.11 (Sep 2026) | **71.8** | COCO-17 | 40.7M | 9.7 ms T4 | fits | MPS, CoreML export | **Apache-2.0** | B (Roboflow SAB) |
| ECPose S/M/L (EdgeCrafter) | 2026-03, TMLR 2026 | 68.9–74.5 | COCO-17 | 10–34M | 5.5–11.8 ms | fits | n/a | **non-commercial** | C |
| SDPose-OOD | MIT, repo pushed 2026-10-02; training scripts not released | beats Sapiens-2B on COCO-OOD (03) | COCO-17 / wholebody | SD U-Net backbone, heavy | n/a | unlikely (C) | n/a | MIT (SD weights terms apply, not verified) | C |
| **MediaPipe Pose Landmarker (BlazePose GHUM)** | mediapipe 1.0.1 (2026-08-14) | no COCO AP; paper: PCK@0.2 84.1 (Full) on yoga set vs OpenPose 83.4 | 33 | 3.5M (Full, 2020 paper) | 30+ fps mobile CPU | CPU | CPU | Apache-2.0 | B (2020) |
| TAR-ViTPose (video) | 2026-03, Apache-2.0, weights released | +2.3 mAP over ViTPose on PoseTrack17 | COCO/PoseTrack | ViT | n/a | likely | n/a | Apache-2.0 | B |

Not found: any 2025–2026 2D model that claims robustness to in-plane rotation or inversion with a measured number,
other than the trampoline fine-tune and RePoGen already in 03. FlyPose (WACV 2026, arXiv 2601.05747) targets steep
aerial views, not inversion.

### 4.2 Why inverted athletes fail, at the config level (C, verified in configs)

- RTMPose-x halpe26 config: `RandomBBoxTransform(scale_factor=[0.5, 1.5], rotate_factor=90)`; mmpose samples rotation
  as `truncnorm(-1, 1) × rotate_factor` with `rotate_prob=0.6`. So training never rotates beyond ±90°, and a
  head-down athlete is only covered by whatever handstands exist in the 7–8 datasets.
- Ultralytics `cfg/default.yaml`: `degrees: 0.0`, `flipud: 0.0`. If the released YOLO26-pose weights used defaults
  (not confirmed), they never saw synthetic rotation at all.
- BlazePose (arXiv 2006.10204 §2.6): "We estimate rotation as the line L between mid-hip and mid-shoulder points and
  rotate the image so L is parallel to the y-axis", based on the detector or the previous frame's keypoints. Pose2Sim's
  README says BlazePose "handles upside-down postures" and RTMPose "does not currently work well for acrobatic
  movements where the person is upside down" (developer statements, C).

**Implication:** an upright-normalised crop turns most inverted frames into in-distribution frames for RTMPose
without training. Implementation in rtmlib's low-level API is ~40 lines: rotate the full frame about the previous
hip centre by −θ (θ = torso angle from vertical), crop, run `RTMPose`, rotate the keypoints back. On the first flight
frame or after a tracking loss, fall back to the 4-way rotation TTA from 03 §10 and keep the most confident pass.
This also removes most left/right confusions that come from the model "seeing" an upside-down body as a mirrored
upright one (C, mechanism not measured).

### 4.3 Left/right swaps

- **Off-the-shelf fix: not found.** Pose2Sim's `handle_LR_swap` parameter is documented as "Not implemented yet" and
  its roadmap lists "Handling left/right swaps" for v0.13; Sports2D's `flip_left_right` is "DEPRECATED. This parameter
  is ignored." No 2025–2026 monocular L/R correction tool or paper was found.
- What to build (C): per frame, for each limb chain (shoulder-elbow-wrist, hip-knee-ankle-foot) compare the cost of
  "as labelled" vs "swapped" against a constant-velocity prediction from the last two good frames; swap the whole
  chain if the swapped cost is lower by a margin; enforce that shoulders and hips swap together unless the torso
  chirality sign(u × v) has a confident, persistent flip (that flip is the twist signal, so the rule must only fire on
  single-frame flickers, with ≥3-frame hysteresis). Log every swap so the twist counter can be audited.
- Learned alternatives: SmoothNet (ECCV 2022, Apache-2.0, `cure-lab/SmoothNet`) and TAR-ViTPose (2026) are trained on
  everyday motion and will likely damp real fast rotation (the GVHMR failure in 03, C). Use only after the rule-based
  layer, and check counts before/after.

---

## 5. Temporal stabilisation and 2D-to-3D lifting

| Tool | Version / date | What it does | Licence | Fit | Grade |
|---|---|---|---|---|---|
| **Sports2D / Pose2Sim filters** | sports2d 0.8.34, pose2sim 0.10.49 (both 2026-07-10) | Hampel outlier rejection, gap interpolation (`interp_gap_smaller_than`, default 10 frames), Butterworth (default 6 Hz, order 4), Kalman, One-Euro, GCV spline, LOESS, median, acc-minimising | BSD-3-Clause | CPU, both machines | C |
| SmoothNet | 2022 (ECCV) | plug-and-play learned temporal refiner for 2D/3D | Apache-2.0 | small | B (paper) |
| TAR-ViTPose | 2026-03 (arXiv 2603.05929) | video 2D pose, joint-centric temporal aggregation; +2.3 mAP PoseTrack17; inference script for video | Apache-2.0 | likely 2060 | B |
| NCSTR (2603.20323), Again-Pose (2606.29230), VEPE (2509.01095) | 2025–2026 | video pose under blur / degraded frames | **code not found** | n/a | n/a |
| MotionBERT | ICCV 2023; repo push 2026-03-14 | 2D→3D lift, mesh, action; already in repo | Apache-2.0 code, H3.6M-derived weights | 2060 | B |
| PoseFormerV2 | CVPR 2023 | frequency-domain lifter, robust to noisy 2D | MIT | 2060 | B |
| MotionAGFormer | WACV 2024 | lifter used by AthletePose3D (03 F11) | Apache-2.0 | 2060 | B |
| Rotation-augmented lifting | arXiv 2601.13913 (Jan 2026) | random in-plane rotation at training restores 64 mm MPJPE under rotated input (03 F10) | code not found | n/a | B |

Filter caution (C): any low-pass filter applied to x/y keypoints attenuates exactly the high-frequency signal that
encodes a fast twist. Filter **after** the swap fix, prefer velocity-adaptive One-Euro, and for counting use unwrapped
angles (pelvis→thorax angle, hip-vector angle) rather than smoothed coordinates. Pre-register a check that
filtering does not change counts on the 12 reference clips.

---

## 6. 3D / HMR worth a probe (deltas since 03)

| Model | Status at 2026-10-03 | VRAM / speed | Licence | Verdict |
|---|---|---|---|---|
| **SAM 3D Body** | checkpoints 2025-11-19 (DINOv3-H+ 840M, ViT-H 631M); last repo push 2026-02-19; inference only, gated HF, sanctioned jurisdictions blocked | No official figure. Issue #136 (2026): "For most images, 24 GB might be sufficient… seen certain inputs push memory usage above 24 GB" (anecdotal). Not a 2060 model | SAM License (commercial with restrictions) | Probe on a 4090 only, as in 03 §10 |
| **Fast SAM 3D Body** | ECCV 2026, `yangtiming/Fast-SAM-3D-Body`, pushed 2026-09-29 | up to 10.9× end-to-end speed-up, ~65 ms/frame on RTX 5090 in their teleop demo; optional TensorRT | **MIT code** (SAM 3D Body weights still SAM License) | Use this wrapper for any bulk run |
| SAM-Body4D | v0.2.0, 2026-03-21 | SAM 3 masklets → SAM 3D Body per frame + smoothing | MIT code | Video wrapper; same VRAM class |
| MHR body model | v1.0.1 2025-11-19, repo pushed 2026-09-11 | n/a | Apache-2.0 | Needed by SAM 3D Body; unlike SMPL it is Apache |
| Multi-HMR 2 (NAVER) | arXiv 2606.14841, code `naver/multi-hmr2` (2026-08) | n/a | **NAVER Non-Commercial License** | Skip for product |
| PromptHMR | code 2026-01 | n/a | **Non-commercial scientific research only** | Skip for product |
| GEM/GENMO | 2026-06 push | n/a | **NVIDIA OneWay Noncommercial** | Skip for product |
| DuoMo | v1 2026-05-27 | H200-class | XRCIA Noncommercial (03) | Skip |
| RTMW3D-x (rtmlib) | 2024-06 weights | small, fits 2060 | Apache-2.0 code | Free per-frame 3D probe for twist direction; untested on inversion (C) |

Any 2026 acrobatics-capable HMR: **not found.** Searches for 2026 HMR on flips, gymnastics, parkour or breakdance
returned DanceHMR (dance only), FactorizedHMR, MoPO, Multi-HMR 2 and Again-Pose; none evaluates acrobatic rotation.

---

## 7. Utilities

### 7.1 Video decoding

| Tool | Version / date | Platforms | GPU decode | Licence | Notes | Grade |
|---|---|---|---|---|---|---|
| **torchcodec** | 0.17.0 (2026-09-30), torch ≥ 2.11, Py ≥ 3.10 | Linux, macOS arm64, Windows | CUDA (NVDEC) on Linux by default; Windows needs `--index-url .../whl/cu130` | BSD-3 | New low-level beta API (demux/decode/convert), frame-accurate seeking, HEIC/AVIF/WebP image decode (0.16) | C |
| **PyAV** | 19.0.1 (2026-10-03), **Py ≥ 3.12** | all incl. win_amd64, macOS arm64 | no (CPU FFmpeg) | BSD-3 | Exact PTS/time_base per frame, best for VFR audit and audio | C |
| decord | 0.6.0 (2021-06-14), wheels cp36–cp38 | stale | was CUDA | Apache-2.0 | **Do not use** | C |
| decord2 (fork) | 3.4.0 (2026-05-30), cp310–cp314 | Linux, macOS arm64, Windows | claims GPU | Apache-2.0 | small project (65 stars) | C |
| OpenCV | opencv-python 5.0.0.93 (2026-07-02); 4.14.0.94 (2026-07-28) | all | no | Apache-2.0 | `CAP_PROP_FPS` is nominal; no PTS-accurate seek | C |

iPhone footage (C): HEVC, often VFR; slow-mo files are captured at 120/240 fps and carry retiming metadata, so the
"fps" OpenCV reports can be the playback rate. Before counting rotations per second, read per-frame PTS with PyAV and
normalise to CFR with FFmpeg when needed. Label Studio also asks for CFR video (~30 fps) to avoid frame-count drift.
Turing NVDEC decodes HEVC 4:2:0 10-bit (C, NVIDIA matrix not re-checked).

### 7.2 Annotation / verification

Constraint from memory: the owner will not label at scale; the tool is for verifying weak labels on a small queue.

| Tool | Version / date | Video features relevant here | Install | Licence | Grade |
|---|---|---|---|---|---|
| **Label Studio** | 1.23.2 (2026-09-29, security: sandboxed HTML/SVG uploads); 1.23.0 (2026-03-13); SDK 2.1.2 | `Video` tag (frameRate, playback speed, timelineHeight), **`TimelineLabels`** (click/drag frame ranges, `value.ranges[{start,end}]`), `VideoRectangle`, `VideoVector`. No skeleton/keypoints on video in docs | `pip install label-studio` (Py 3.10+) | Apache-2.0 | C |
| **CVAT** | 2.77.0 (2026-09-28); cvat-sdk 2.77.0 | skeleton tracks with interpolation on video, AI tools tracking, SAM2 agent via Docker compose (2.59.0, 2026-03-06), `TrackingFunction` in SDK auto-annotation (2.69.0) | Docker compose (self-host) | MIT | C |
| IMPose | arXiv 2606.04480 (2026-06) | propagates one-frame keypoint corrections across a video | code not found | n/a | C |

Recommendation: Label Studio with `TimelineLabels` + `Choices` for trick / flip / twist / direction verification and
take-off/landing ranges. Bring in CVAT only if keypoint correction on video becomes necessary.

### 7.3 Experiment tracking (free)

| Tool | Version / date | Model | Licence / terms | Grade |
|---|---|---|---|---|
| **Trackio** (Hugging Face) | 0.40.0 (2026-09-30) | local-first SQLite (Parquet export), `import trackio as wandb` drop-in, CLI + SQL queries for agents, optional free HF Space dashboard | MIT | C |
| **MLflow** | 3.16.1 (2026-09-16) | local file/SQL backend, UI, model registry | Apache-2.0 | C |
| W&B (now **CoreWeave Forge**) | wandb 0.30.0 (2026-09-09) | hosted. Free plan: 5 GB/mo storage, up to 5 seats, "personal development". Academic licence: 200 GB, academic email, research "unrelated to a for-profit entity" | MIT client; hosted terms | C |
| Aim | 3.29.1 (2025-05-08), no release since | local | Apache-2.0 | stale, avoid (C) |

### 7.4 Transcription

| Tool | Version / date | Where | Licence | Notes |
|---|---|---|---|---|
| **whisper-large-v3-turbo** weights | 2024-10-01 | both | MIT | still the default open multilingual model |
| **mlx-whisper** | 0.4.3 (2025-08-29) | Mac | MIT | `mlx_whisper.transcribe(..., word_timestamps=True)`; `mlx-community/whisper-large-v3-turbo` |
| whisper.cpp | v1.9.4 (2026-09-11) | Mac (Metal, Core ML encoder on ANE), CUDA | MIT | VAD built in; `make large-v3-turbo` |
| faster-whisper | 1.2.1 (2025-10-31) | 2060 (CTranslate2) | MIT | int8/fp16 on Turing (C) |
| WhisperX | 3.8.6 (2026-05-25) | 2060 | BSD-2 | word alignment + diarisation |
| Qwen3-ASR-1.7B + Qwen3-ForcedAligner-0.6B | 2026-01-28 | 2060 tight (2B params, C), vLLM is Linux-only | Apache-2.0 | French supported; LibriSpeech-other 3.38 vs Whisper-large-v3 3.97 (vendor) |
| NVIDIA Parakeet-TDT-0.6B-v3 / Canary-1B-v2 | 2025-08 (updated 2026-08) | 2060 | CC-BY-4.0 | European languages incl. French |

### 7.5 Open video VLMs runnable locally (adds sizes to 02 §1.2; accuracy evidence stays in 02)

| Model | HF date | Params (safetensors) | Licence | 2060 6 GB | Mac (MLX, disk ≤ 12 GB) |
|---|---|---|---|---|---|
| Qwen3.5-0.8B / 2B | 2026-02-28 | 0.8B / 2B | Apache-2.0 | yes (fp16) | yes |
| **Qwen3.5-4B** | 2026-02-27 | 4.66B | Apache-2.0 | 4-bit only (~3 GB); video path in llama.cpp not verified | `mlx-community/Qwen3.5-4B-4bit` = 3.06 GB |
| **Qwen3.5-9B** | 2026-02-27 | 9.65B | Apache-2.0 | no | `mlx-community/Qwen3.5-9B-4bit` = 5.98 GB |
| Qwen3.6-27B / 35B-A3B | 2026-04-21 / 04-15 | 27.8B / 36.0B | Apache-2.0 | no | 35B-A3B 4-bit is ~20 GB, exceeds free disk |
| Qwen3.8-27B | 2026-08-05 | 27.8B | Apache-2.0 | no | no (disk) |
| **Molmo2-4B / 8B** | 2025-12-14 | 4.85B / 8.66B | Apache-2.0 | 4B at 4-bit (C) | `mlx-community/Molmo2-8B-4bit` exists |
| Gemma 4 E4B | 2026-03-02 | 8.0B total | Apache-2.0 | 4-bit (C) | yes |
| InternVL3.5 1B–241B | 2025-08/09 (no newer OpenGVLab VLM found) | | MIT code | small sizes | yes |

Runners: `mlx-vlm` 0.7.4 (2026-09-28, MIT) on the Mac; `llama-cpp-python` 0.3.36 (2026-10-01) or `transformers` 5.18.0
on the 2060; `vllm` 0.30.0 ships Linux wheels only (WSL2 on Windows). 02 already showed open ≤12B models near chance on
physical motion (MotionBlind), so local VLMs are a context/vault reader, not a twist counter.

---

## 8. Recommended stack

| Layer | Recommendation | Install | Licence | Why it beats the current setup |
|---|---|---|---|---|
| Runtime (2060) | torch 2.14.1 cu130 + onnxruntime-gpu 1.30.0 (or its CUDA 12.8 build), driver ≥ 580; fp16 | `uv pip install torch --index-url https://download.pytorch.org/whl/cu130`; `uv pip install onnxruntime-gpu` | BSD / MIT | Unpins the 2024 ORT 1.19.2; Turing still in every arch list |
| Data I/O | **PyAV 19** for PTS/VFR audit and audio; **torchcodec 0.17** for batch decode to tensors (CUDA on 2060); FFmpeg CLI for CFR | `uv pip install av torchcodec` | BSD-3 | `cv2.CAP_PROP_FPS` is nominal; iPhone VFR/slow-mo breaks rotations-per-second maths. Drop decord |
| Detector | rtmlib YOLOX-**m** HumanArt (or RF-DETR-M via rtmlib 0.0.16) | `uv pip install -U rtmlib` (+ `rfdetr`) | Apache-2.0 | ~4× fewer detector FLOPs than YOLOX-x for one athlete (C); removes AGPL YOLO26 from the product path |
| Tracker | One athlete: detection + keypoint-box propagation, re-detect each flight frame. Hard clips: **SAM 2.1 / SAMURAI** masklet from the first box. Multi-person scenes: roboflow `trackers` (BoT-SORT/OC-SORT) | `uv pip install trackers`; SAM 2.1 from git | Apache-2.0 | Apache instead of AGPL BoxMOT/Ultralytics; masks survive blur better than IoU (C) |
| 2D pose (default) | **RTMPose-x Halpe26 in rtmlib + upright-normalised crops** (rotate by previous torso angle), 4-way TTA fallback | rtmlib `Custom`/`RTMPose` with the halpe26 ONNX URL | Apache-2.0 | Brings inverted frames inside RTMPose's ±90° training range; adds feet/head-top for chirality. $0 |
| 2D pose (probe arms) | **Sapiens2-0.4B** (transformers 5.18, fp16, flight frames only); **MediaPipe Pose Landmarker** 1.0.1 (rotation-aligned ROI, CPU) | `uv pip install transformers mediapipe` | Sapiens2 License / Apache-2.0 | Strongest open in-the-wild pose model; BlazePose is the only shipped model designed around rotation alignment |
| Fine-tune path (if probe says CONDITIONAL) | ViTPose via `transformers` or RF-DETR Keypoint (both Apache, maintained); avoid MMPose training (no release since 2024-07, mmcv 2.2.0 from 2024-04) | pip | Apache-2.0 | MMPose/mmcv lag torch 2.14 (C) |
| Temporal | Own L/R-swap fixer (§4.3) → Hampel + gap interpolation → One-Euro, counting on unwrapped angles; Sports2D/Pose2Sim filter code as reference | `uv pip install sports2d` or copy functions | BSD-3 | No maintained swap fixer exists; generic smoothers damp twists |
| 3D (optional) | SAM 3D Body through **Fast SAM 3D Body** on a rented 4090; RTMW3D-x as a free 2060 probe | git | SAM License + MIT / Apache-2.0 | Only HMR with measured inverted-pose gains (03 F9); 10× cheaper wrapper |
| VLM | Frontier API as in 02; local Qwen3.5-9B-4bit (Mac, mlx-vlm) only for non-motion questions | `uv pip install mlx-vlm` | Apache-2.0 | Fits 12 GB disk; open models are near chance on motion (02) |
| Labelling | **Label Studio 1.23.2** (TimelineLabels + Choices); CVAT 2.77 only for keypoint fixes | `uv pip install label-studio` | Apache-2.0 / MIT | pip install, no Docker; frame-range labels match take-off/landing needs |
| Tracking runs | **Trackio 0.40** (local SQLite, wandb API) or MLflow 3.16 | `uv pip install trackio` | MIT / Apache-2.0 | Free, offline, agent-queryable; W&B free tier now capped at 5 GB/mo |
| Transcription | mlx-whisper 0.4.3 + large-v3-turbo on the Mac; faster-whisper 1.2.1 on the 2060 | pip | MIT | Already the standard; Qwen3-ASR only if French accuracy disappoints |

### Upgrades from the current stack and expected gain

| Change | Expected gain | Evidence |
|---|---|---|
| Upright-normalised crops before RTMPose | Fewer failures and L/R flips on inverted flight frames; size of gain **unknown** | C (training-config range + BlazePose design); measure on the 12-clip probe |
| RTMPose-x body7 → Halpe26 checkpoint | +foot and head-top points for twist direction; Body8 AP 80.0 | C |
| YOLOX-x → YOLOX-m or RF-DETR-M, re-detect only in flight | Faster extraction, no accuracy change expected for one athlete | C |
| cv2 → PyAV/torchcodec timestamps | Correct time base for rotation rate and phase timing | C |
| Ultralytics/BoxMOT → rtmlib/RF-DETR/`trackers` | Removes AGPL from the product path | licence texts |
| Add Sapiens2-0.4B as probe arm | Possibly better keypoints on hard frames; +6.7 mAP over RTMW-X on Meta's in-the-wild test | B, not acrobatic |
| Add explicit L/R swap fixer | Fewer spurious facing flips in the twist counter | C |
| onnxruntime-gpu 1.19.2 → 1.30.0 | Current CUDA 13 / cuDNN 9 support | C |

---

## 9. Licence risk register (paper + possible product)

| Item | Licence | Risk |
|---|---|---|
| Ultralytics YOLO26 / YOLO11 weights and code (`core/pose/detector.py`) | AGPL-3.0 or paid Enterprise | **High for a closed product**; fine for the paper |
| BoxMOT | AGPL-3.0 | High for product |
| DEIMv2, EdgeCrafter (ECDet/ECPose) | Intellindust licences: "No rights are granted for Commercial Use" | **Non-commercial** |
| NS-AQA reference code | non-commercial (03) | Reimplement the rule |
| PromptHMR | non-commercial scientific research | **Non-commercial** |
| GENMO/GEM | NVIDIA OneWay Noncommercial | **Non-commercial** |
| DuoMo | XRCIA Noncommercial (03) | **Non-commercial** |
| Multi-HMR 2 | NAVER Non-Commercial | **Non-commercial** |
| AthletePose3D, Fit3D, H3.6M-derived weights (MotionBERT) | research-only datasets | **Non-commercial** weights risk |
| RePoGen | GPL-3.0 + SMPL-X non-commercial (03) | Copyleft + NC body model |
| SMPL / SMPL-X | non-commercial | Prefer MHR (Apache-2.0) for synthetic renders |
| BBoxMaskPose v2 | GPL-3.0 | Copyleft |
| RF-DETR XL / 2XL | PML 1.0 | Check terms; N–L are Apache-2.0 |
| Sapiens2 | Sapiens2 License: commercial OK, bans "biometric processing", surveillance, health inference without consent, acknowledgement in publications, Meta audit right | **Legal read needed** before a product |
| SAM 2.1 | Apache-2.0 | Low |
| SAM 3 / 3.1, SAM 3D Body | SAM License, gated HF, sanctions block | Medium (custom terms) |
| Fast SAM 3D Body, SAM-Body4D | MIT code, wraps SAM-License weights | Weights terms dominate |
| Qwen3.5/3.6/3.8-27B, Molmo2, Gemma 4 | Apache-2.0 | Low |
| Parakeet / Canary | CC-BY-4.0 | Attribution |
| TensorRT | proprietary NVIDIA licence | Redistribution terms |
| W&B free tier | "personal development" | Use academic licence or Trackio/MLflow |
| rtmlib default weights (YOLOX HumanArt, RTMPose body7) | Apache-2.0 code; training-data terms (HumanArt, AIC, etc.) **not verified** | Unknown |

---

## 10. Gaps and "not found"

- Not found: any 2D pose model (2025–2026) with a measured number on inverted/acrobatic bodies beyond the trampoline
  fine-tune in 03; any RTMPose/RTMW/DWPose successor.
- Not found: a maintained monocular left/right swap corrector (Pose2Sim: "Not implemented yet").
- Not found: a measured study of rotation-aligned (BlazePose-style) crops applied to a heatmap/SimCC top-down model on
  inverted people.
- Not found: Sapiens2 COCO-17 AP, Sapiens2 VRAM or speed, fp16 stability on Turing.
- Not found: official SAM 3D Body VRAM (only an anecdote that 24 GB can be exceeded); official SAM 3 parameter count is
  inconsistent between sources (~850M in paper per Roboflow vs 473.6M on Ultralytics' page).
- Not found: whether released YOLO26-pose weights were trained with rotation augmentation (default config says
  `degrees: 0.0`).
- Not found: detector recall on inverted or motion-blurred people for any detector.
- Not found: code for NCSTR, Again-Pose, IMPose, VEPE, or the rotation-equivariance lifter (2601.13913).
- Not found: any 2026 HMR evaluated on flips/twists.
- Unverified: torchcodec CUDA decoding on Windows with a Turing NVDEC; llama.cpp video input for Qwen3.5; HumanArt and
  body7 training-data licence inheritance; YOLO27 date ("no set launch date").
- Mac RAM size is not stated in the brief; MLX sizes above are disk sizes.

---

## 11. Sources (accessed 2026-10-03)

Package indexes and repos (version, date, licence as listed in the tables)
- PyPI JSON API: https://pypi.org/pypi/{ultralytics,rtmlib,mmpose,mmcv,onnxruntime,onnxruntime-gpu,torch,torchvision,torchcodec,av,decord,decord2,opencv-python,boxmot,supervision,rfdetr,trackers,sam2,label-studio,label-studio-sdk,cvat-sdk,mlflow,wandb,aim,trackio,openai-whisper,faster-whisper,mlx-whisper,whisperx,mlx-vlm,mlx,transformers,vllm,llama-cpp-python,mediapipe,sports2d,pose2sim,tensorrt,onnx}/json
- Ultralytics: https://github.com/ultralytics/ultralytics (AGPL-3.0; v8.4.0 release note "Ultralytics YOLO26 has arrived", 2026-01-14); https://docs.ultralytics.com/models/ ; https://docs.ultralytics.com/models/yolo26/ ; https://docs.ultralytics.com/tasks/pose/ ; https://docs.ultralytics.com/modes/track/ ; https://docs.ultralytics.com/models/sam-3/ ; `ultralytics/cfg/default.yaml`
- rtmlib: https://github.com/Tau-J/rtmlib (Apache-2.0; 0.0.16 2026-08-03; README model zoo; `tools/solution/body.py`, `tools/solution/pose_tracker.py`, `tools/base.py`)
- MMPose: https://github.com/open-mmlab/mmpose (Apache-2.0; v1.3.2 2024-07-12); `projects/rtmpose/README.md`; `projects/rtmpose/rtmpose/body_2d_keypoint/rtmpose-x_8xb256-700e_body8-halpe26-384x288.py`; `mmpose/datasets/transforms/common_transforms.py`
- Sapiens2: https://github.com/facebookresearch/sapiens2 (Sapiens2 License; release 2026-04-24); https://huggingface.co/facebook/sapiens2-pose-0.4b ; Khirodkar et al., "Sapiens2", ICLR 2026, arXiv 2604.21681 (Tables 1, 3, 8)
- transformers: https://github.com/huggingface/transformers (Apache-2.0; v5.18.0 2026-09-30, "Add pose estimation keypoint preprocessing to Sapiens2ImageProcessor #47199"); `src/transformers/models/{sapiens2,vitpose,sam3_tracker_video,rf_detr,d_fine,deimv2}`
- ViTPose in transformers: https://huggingface.co/usyd-community (vitpose-*, 2025-01)
- SAM 3D Body: https://github.com/facebookresearch/sam-3d-body (SAM License; checkpoints 2025-11-19; INSTALL.md; issue #136); arXiv 2602.15989
- Fast SAM 3D Body: https://github.com/yangtiming/Fast-SAM-3D-Body (MIT; ECCV 2026); arXiv 2603.15603
- SAM-Body4D: https://github.com/gaomingqi/sam-body4d (MIT; v0.2.0 2026-03-21)
- MHR: https://github.com/facebookresearch/MHR (Apache-2.0)
- SAM 3: https://github.com/facebookresearch/sam3 (SAM License; RELEASE_SAM3p1.md, 2026-03-27); arXiv 2511.16719
- SAM 2: https://github.com/facebookresearch/sam2 (Apache-2.0)
- SAMURAI: https://github.com/yangchris11/samurai (Apache-2.0)
- RF-DETR: https://github.com/roboflow/rf-detr (Apache-2.0 / PML 1.0; 1.11.1 2026-09-30; README benchmark tables incl. keypoint preview; release notes 1.11.0/1.11.1); arXiv 2511.09554 (ICLR 2026)
- D-FINE: https://github.com/Peterande/D-FINE (Apache-2.0)
- RT-DETRv4: https://github.com/RT-DETRs/RT-DETRv4 (Apache-2.0; ECCV 2026); arXiv 2510.25257
- DEIMv2: https://github.com/Intellindust-AI-Lab/DEIMv2 (DEIMv2 License, non-commercial); arXiv 2509.20787
- EdgeCrafter: https://github.com/Intellindust-AI-Lab/EdgeCrafter (EdgeCrafter License, non-commercial; TMLR 2026)
- trackers: https://github.com/roboflow/trackers (Apache-2.0; 2.6.1 2026-09-25)
- BoxMOT: https://github.com/mikel-brostrom/boxmot (AGPL-3.0; v25.0.0 2026-09-09)
- MediaPipe Pose Landmarker: https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker ; Bazarevsky et al., "BlazePose", arXiv 2006.10204 (§2.2, §2.6, Table 1)
- Pose2Sim: https://github.com/perfanalytics/pose2sim (BSD-3; v0.10.49) README (`handle_LR_swap`, acrobatics note, roadmap)
- Sports2D: https://github.com/davidpagnon/Sports2D (BSD-3; v0.8.34) README (filter options, `flip_left_right` deprecated)
- SDPose-OOD: https://github.com/T-S-Liang/SDPose-OOD (MIT); arXiv 2509.24980
- TAR-ViTPose: https://github.com/zgspose/TARViTPose (Apache-2.0); arXiv 2603.05929
- SmoothNet: https://github.com/cure-lab/SmoothNet (Apache-2.0); arXiv 2112.13715
- MotionBERT: https://github.com/Walter0807/MotionBERT (Apache-2.0); arXiv 2210.06551
- PoseFormerV2: https://github.com/QitaoZhao/PoseFormerV2 (MIT); arXiv 2303.17472
- MotionAGFormer: https://github.com/TaatiTeam/MotionAGFormer (Apache-2.0)
- Multi-HMR 2: https://github.com/naver/multi-hmr2 (NAVER Non-Commercial); arXiv 2606.14841
- PromptHMR licence: https://github.com/yufu-wang/PromptHMR/blob/main/LICENSE ; GENMO licence: https://github.com/NVlabs/GENMO
- Video pose papers without code found: NCSTR arXiv 2603.20323; Again-Pose arXiv 2606.29230; VEPE arXiv 2509.01095; IMPose arXiv 2606.04480; FlyPose arXiv 2601.05747 (WACV 2026)
- NVIDIA: CUDA Toolkit release notes https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html ; TensorRT support matrix https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/support-matrix.html
- PyTorch release notes https://github.com/pytorch/pytorch/releases (v2.14.0 2026-09-02, v2.14.1 2026-09-30); `.ci/wheel/windows/build_env_setup.py` (TORCH_CUDA_ARCH_LIST_TABLE)
- ONNX Runtime CUDA EP: https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html
- torchcodec: https://github.com/meta-pytorch/torchcodec (BSD-3; v0.17.0 2026-09-30, v0.16.0 2026-08-13 release notes)
- PyAV: https://github.com/PyAV-Org/PyAV (BSD-3; v19.0.1 2026-10-03)
- decord: https://github.com/dmlc/decord ; decord2: https://github.com/johnnynunez/decord2
- Label Studio: https://github.com/HumanSignal/label-studio (Apache-2.0; 1.23.0–1.23.2 release notes); https://labelstud.io/tags/video ; https://labelstud.io/tags/timelinelabels
- CVAT: https://github.com/cvat-ai/cvat (MIT; CHANGELOG.md 2.59.0–2.77.0)
- Trackio: https://github.com/gradio-app/trackio (MIT; 0.40.0)
- MLflow: https://github.com/mlflow/mlflow (Apache-2.0; v3.16.1)
- W&B / CoreWeave Forge pricing: https://coreweave.com/forge-pricing (redirect from wandb.ai/site/pricing)
- Aim: https://github.com/aimhubio/aim (Apache-2.0; v3.29.1 2025-05-08)
- whisper.cpp: https://github.com/ggml-org/whisper.cpp (MIT; v1.9.4 2026-09-11); mlx-whisper: https://github.com/ml-explore/mlx-examples/tree/main/whisper ; faster-whisper: https://github.com/SYSTRAN/faster-whisper ; WhisperX: https://github.com/m-bain/whisperX
- Hugging Face model API: openai/whisper-large-v3-turbo; Qwen/Qwen3-ASR-1.7B (model card WER table); nvidia/parakeet-tdt-0.6b-v3; nvidia/canary-1b-v2; Qwen/Qwen3.5-{0.8B,2B,4B,9B}; Qwen/Qwen3.6-{27B,35B-A3B}; Qwen/Qwen3.8-27B; allenai/Molmo2-{4B,8B}; allenai/MolmoMotion-4B-H1-F32 (trajectory forecasting, not relevant); google/gemma-4-E4B-it; OpenGVLab model list; mlx-community/{Qwen3.5-4B-4bit,Qwen3.5-9B-4bit,Molmo2-8B-4bit,Qwen3.6-35B-A3B-4bit}
- mlx-vlm: https://github.com/Blaizzy/mlx-vlm (MIT; v0.7.4)

In-repo: `scripts/extract_pose.py`, `scripts/extract_pose_modal.py`, `core/pose/detector.py`, `requirements.txt`,
`scripts/motionbert_3d_judge.py`. Prior survey: `../2026-09-28/02-video-vlms.md`, `../2026-09-28/03-pose-hmr-acrobatics.md`.
