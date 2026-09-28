# 03: Pose and mesh recovery on acrobatic, inverted, fast-rotating motion

Survey date: 2026-09-28. Desk research only. No project data touched, no code run on project data, $0 spent.
Track: monocular 3D HMR and world-grounded motion recovery, 2D pose robustness to inversion and blur, direct
orientation/rotation estimation, skeleton SSL on fine-grained sports, and fine-tuning feasibility.
Companion file: `01-twist-analog-sports.md` covers twist counting in analog sports in depth. This file focuses on
the pose/HMR layer that such counters depend on, and re-tests claims (a) to (d).

Grading: **A** = measured in-domain (parkour/tricking). **B** = measured on an analog (other sport, other
dataset, inverted poses in general). **C** = claim, vendor statement, derivation, or my own estimate.

Tools used: alphaXiv discovery and full-text, arXiv abstracts, WebSearch/WebFetch on GitHub READMEs and licences.
Two PDFs were downloaded to the session scratchpad (FineGym supplementary class list, RePoGen, OpenPose) to verify
numbers; nothing was written into the repo except this file.

**The only A-grade evidence is PkVision's own:** RTMPose-x mean keypoint confidence 0.638 over 1,618 clips with
drops on corks and ~0.22 to 0.29 on one hang-castaway double back; GVHMR reading flips at ~0.5x on competition
footage; skeleton SSL +0.023 in-house. No external paper measures pose, HMR or rotation accuracy on parkour.

---

## 0. Bottom line

| Claim | Verdict | One-line reason |
|---|---|---|
| (a) Monocular 3D HMR is dead for acrobatics | **WEAKENED** | Still no paper reports flip/twist rotation accuracy for any HMR, and every 2025–26 video/world-grounded model is still trained on AMASS/BEDLAM-style data. But SAM 3D Body (Nov 2025) measurably fixed single-frame inverted and very-hard poses (2D aPCK on inverted bodies 78 vs 46 for CameraHMR), so "dead" is now "untested, with one credible candidate". |
| (b) Stock 2D pose fails upside-down | **STILL TRUE for stock COCO models, cheaply fixable** | Stock ViTPose-S/B lose 8 to 18 AP on real trampoline footage and confuse left/right. Fine-tuning with rotation augmentation to ±180° plus a few thousand synthetic inverted renders recovers COCO-level accuracy in two independent analogs. |
| (c) Skeleton SSL is unvalidated on acrobatics | **STILL TRUE** | No 2025–26 skeleton SSL (AMR, GFP, S2I, SOfA) reports FineGym, FineDiving or skating numbers. The only analog with a rotation sport is VIFSS, which is domain-specific view-invariant contrastive pretraining, not generic SSL. |
| (d) Twist ≥ 1.5 is geometrically unrecoverable from one camera | **OVERTURNED as a geometry claim; UNKNOWN in-domain** | Single-camera systems count 1.5 to 3.5 twists from 2D keypoints alone: a training-free hip-vector rule gets 93.3% twist-count accuracy on diving; skeleton-only classifiers reach ≥85.9% on Diving48 and 94.2% on FineGym (which has 1.5/2/2.5/3-twist classes). The geometry argues magnitude is observable and direction needs chirality cues. Parkour accuracy is unmeasured. |

The practical consequence: the twist bottleneck is **2D perception quality under inversion and blur (left/right
labels, face points, dropped frames)**, not projective geometry. That bottleneck has measured, cheap fixes in
analog domains.

---

## 1. Findings table

| # | Finding | Source | Grade |
|---|---|---|---|
| F1 | Training-free twist counter on 2D keypoints: right-to-left **hip vector "petals"** (each petal = 0.5 twist) gives **93.27%** twist-count accuracy; pelvis-to-thorax vector rotation gives **97.31%** somersault-count accuracy. Pose from HRNet, single broadcast view, MTL-AQA competition dives (platform-focused system). Failures: diver half underwater, strong motion blur. | Okamoto & Parmar, "Hierarchical NeuroSymbolic Approach for Comprehensive and Explainable AQA", CVPRW 2024 (CVSports best paper), arXiv 2403.13798. Code github.com/laurenok24/NSAQA, non-commercial use only. | B |
| F2 | Diving48 (48 classes that encode twist 0 to 3+ in half steps, e.g. `Back_15som_05Twis_FREE` vs `Back_15som_15Twis_FREE`): **skeleton-only ST-GCN++ 85.9% top-1**, PoseC3D heatmaps 82.8%, with fully automatic D-FINE detection + ViTPose-L. A correct class implies a correct twist count, so micro twist-attribute accuracy is ≥ 85.9%. | Hassan et al., "FineX", arXiv 2608.13458 (Aug 2026). Code release not stated. | B |
| F3 | FineGym Gym99 contains twist-count classes (FX salto bwd stretched with 1.5/2/2.5/3 twist, salto fwd stretched with 1/1.5/2 twist, double salto bwd tucked with 1/2 twist; VT with 1/1.5/2/2.5 twist). **Skeleton-only ST-GCN++ 94.2% top-1 / 91.9% mean-class**, PoseC3D 94.4 / 92.0 (FineX); PYSKL PoseC3D two-stream 94.1% mean-class. Caveat: FineGym skeletons were extracted with **ground-truth athlete boxes**. No per-class twist breakdown. | FineGym supplementary class list (sdolivia.github.io/FineGym/resources/supp.pdf); PYSKL README (github.com/kennymckormick/pyskl, Apache-2.0); FineX 2608.13458 | B |
| F4 | FineGym 2020: ST-GCN on 2020-era pose reached only 25.2% mean-class / 36.4% top-1 on Gym99 elements because "detections and pose estimations of the gymnast are missed in multiple frames, especially in frames with intense motion". The jump to ~94% came from better 2D pose (HRNet on good crops), not better geometry. | Shao et al., FineGym, CVPR 2020, arXiv 2004.06704 | B |
| F5 | Figure skating (rotation about the long axis, 1 to 4 revolutions, broadcast single view, 25 fps): VIFSS element-level (jump type × rotation count) **F1@50 92.56**, raw 2D pose 78.78, **lifted 3D pose (MotionAGFormer) 76.57, worse than 2D**. The 3D lifter "fails to capture the rotational motion" of a quad toe loop absent from its 3D training set. | Tanaka, Suzuki, Fujii, "VIFSS", arXiv 2508.10281 (Aug 2025). Code github.com/ryota-skating/VIFSS, Apache-2.0. | B |
| F6 | Trampoline, real multi-view test set (8 cameras, 120 fps): stock COCO **ViTPose-S AP 55.8, ViTPose-B 65.6** vs ViTPose-S 73.8 on COCO val. Fine-tuned on LSP (10k real sports) + 2,520 synthetic trampoline renders: **73.1 (S) / 75.6 (B)**. Synthetic alone: only 60.1. Best model cuts triangulated 3D MPJPE by 46.1 mm (42.7%). Qualitatively fixes left/right wrist confusion. | Drolet-Roy et al., "Human Pose Estimation in Trampoline Gymnastics", arXiv 2604.01322 (Apr 2026). Code/data: github.com/VisionICLab/trampoline_syn_data says **"Code will be available soon"** (not released at survey date). | B |
| F7 | Rare views: rotation augmentation to ±180° vs the standard ±40° gives **AP 59.1 vs 45.9** on RePo bottom views and **89.0 vs 72.3** on the sequence set. 500 synthetic RePoGen images already help; gains saturate near 3,000. Anatomical plausibility of the sampled poses was not required. | Purkrabek & Matas, RePoGen, FG 2024, arXiv 2307.06737. Code, data, weights: github.com/MiraPurkrabek/RePoGen, GPL-3.0 + SMPL-X non-commercial terms. | B |
| F8 | OpenPose paper lists "non typical poses and upside-down examples" as main failure cases; more rotation augmentation "visually seems to partially solve" them at a ~5% global COCO cost. Pose2Sim README (RTMPose default): "does not currently work well for acrobatic movements where the person is upside down", suggests BlazePose. | Cao et al., OpenPose, arXiv 1812.08008; github.com/perfanalytics/pose2sim (BSD-3) | C (developer statements) |
| F9 | **SAM 3D Body** (single image, promptable, MHR body model): SA1B-Hard 2D aPCK on "Pose – Inverted body" **78.18 vs CameraHMR 46.12 vs PromptHMR 39.83**; "Contortion or bending" 65.2 vs 47.1. 3D (synthetic + >100-camera multi-view eval): pose_3d:very_hard PVE **114 vs 186 to 214 mm**; aux:orient_ambiguous PVE **42 vs 84**. Gains attributed to VLM-driven mining of hard poses. | Yang et al. (Meta), arXiv 2602.15989 (checkpoints 2025-11-19). Code + weights: github.com/facebookresearch/sam-3d-body, SAM License (commercial use allowed with restrictions). **No training code.** | B |
| F10 | 2D-to-3D lifters without rotation augmentation degrade from **62.9 to 209.6 mm MPJPE** on in-plane rotated inputs; training with random in-plane rotation restores **64.3 mm** and beats equivariant-by-design nets. Qualitative only on SportsCap gymnastics. | Melnyk et al. (Linköping), arXiv 2601.13913 (Jan 2026) | B |
| F11 | Athletic motion: lifters trained on Human3.6M reach ~214 mm MPJPE on high-speed athletic motions; fine-tuning on AthletePose3D drops it to **65 mm** (v3 abstract). Joint angles correlate well (r 0.82 to 0.90), velocities poorly. Motions: running, throws, 6 skating jumps. No flips. | Yeung et al., AthletePose3D, arXiv 2503.07499 (CVPRW 2025). github.com/calvinyeungck/AthletePose3D, non-commercial research only. | B |
| F12 | Fit3D benchmark of 19 HMR methods on extreme fitness poses: inverted "mule kick" among the hardest (155.5 mm mean). Adding Fit3D to HMR2.0 training cut hardest-pose MPJPE by 12.3 mm and joint-angle error 16.1° to 10.4°. | Fieraru, J. Imaging (Jul 2026), PMC13413046. Fit3D non-commercial licence. | B |
| F13 | Online pose on gymnastics: up to **69.6% of frames dropped during rapid motion**; FineTec's completion module keeps Gym288-skeleton at 78.1% top-1 under 75% frame loss. | Shao et al., FineTec, AAAI 2026, arXiv 2512.25067. Gym288-skeleton on HF (Lozumi/Gym288-skeleton). | B |
| F14 | 2025–26 world-grounded video HMR: **DuoMo** (CVPR 2026) trains on AMASS, BEDLAM, 3DPW, Goliath; **GEM/GENMO** (ICCV 2025) and **GEM-X** ("NVIDIA-owned data", composition not stated); **PromptHMR** (CVPR 2025, no training code); **TRAM** (MIT, training code). None reports acrobatic data or any flip/twist rotation metric. | 2603.03265; 2505.01425; github.com/NVlabs/GEM-X; 2504.06397; github.com/yufu-wang/tram | B (for what they train on) / absence of evidence |
| F15 | AnyLift learns a 2D motion diffusion prior from **in-domain internet 2D keypoints** (plus GVHMR-reprojected local poses) and beats GVHMR/WHAM on internet **gymnastics** and martial-arts videos on 2D reprojection and FID. No 3D ground truth for gymnastics. | Li et al., AnyLift, CVPR 2026, arXiv 2604.17818; code github.com/awfuact/anylift-release (licence not stated). Predecessor MVLift, CVPR 2025, github.com/lijiaman/mvlift_release. | B (proxy metrics) |
| F16 | Skeleton SSL 2025–26 (AMR 2606.11450, GFP 2509.03609, S2I 2603.05963, SOfA 2609.07078) evaluates on NTU/PKU-MMD/3D sensor sets only. VIFSS pretraining (contrastive on random 2D projections of 3D mocap) keeps **>60% element F1@50 with 1% of labels** where the no-pretraining model collapses to ~0; at 100% labels the gain is +2.9 F1@50. | Abstracts of the four SSL papers; VIFSS 2508.10281 Fig. 9 | C (absence) / B (VIFSS) |
| F17 | Retraining cost reference: GVHMR converges in **13 h on 2× RTX 4090** (26 GPU-h). | GVHMR, SIGGRAPH Asia 2024, arXiv 2409.06662; github.com/zju3dv/GVHMR | B |
| F18 | SAM 3D Body latency is "several seconds per image"; Fast SAM 3D Body is a training-free 10.9× end-to-end speed-up with on-par accuracy. Third-party reports put SAM-3D-family inference under ~10 GB VRAM (unofficial). | arXiv 2603.15603 (Mar 2026); VRAM from web snippets | B (latency) / C (VRAM) |

---

## 2. Monocular 3D HMR and world-grounded recovery, 2025–2026

### 2.1 Models

| Model | Input | Training data (as stated) | Acrobatic evaluation | Code / licence | Fit |
|---|---|---|---|---|---|
| **SAM 3D Body** (Meta, arXiv 2602.15989; release 2025-11-19) | Single image, optional 2D keypoint / mask prompts | ~7M images: licensed stock photos mined by a VLM for hard poses, multi-view captures, synthetic | Inverted-body and very-hard-pose categories (F9). No flips, no video, no rotation-over-time metric. | Inference code + ViT-H (631M) and DINOv3-H+ (840M) weights, SAM License (commercial OK with restrictions). No training code. | Rented 4090. Not on 2060 6 GB (C). |
| **SAM-Body4D** (arXiv 2512.08406) | Video; SAM 3 masklets → SAM 3D Body per frame, first-frame shape reuse, Kalman smoothing of pose params | Training-free | Qualitative only | github.com/gaomingqi/sam-body4d (licence not verified) | Rented GPU |
| **Fast SAM 3D Body** (arXiv 2603.15603) | Same, accelerated | Training-free | LSPET on par or better | Code not verified | Makes bulk runs affordable |
| **DuoMo** (Meta, CVPR 2026, arXiv 2603.03265) | Video, camera-space diffusion then world-space diffusion, mesh vertices | AMASS, BEDLAM, 3DPW, Goliath | None | github.com/facebookresearch/DuoMo, XRCIA Noncommercial Research License. Inference + training scripts, no preprocessed labels. Needs GVHMR + PromptHMR third-party installs. | Rented GPU. 20 s video in 37.5 s on H200. |
| **GEM / GENMO** (NVIDIA, ICCV 2025, arXiv 2505.01425) | Video, generalist estimation + generation | Mocap + video (standard sets) | None found | github.com/NVlabs/GENMO, NVIDIA OneWay Noncommercial | Rented GPU |
| **GEM-X** (NVIDIA) | Video, world-grounded, 77-joint SOMA body | "NVIDIA-owned data only" | None found | github.com/NVlabs/GEM-X, Apache-2.0 code, NVIDIA Open Model License weights | Rented GPU |
| **PromptHMR** (CVPR 2025, arXiv 2504.06397) | Image/video + prompts, TRAM SLAM for world | Standard | None | github.com/yufu-wang/PromptHMR, no training code planned | Rented GPU |
| **TRAM** (ECCV 2024) | Video, SLAM + VIMO | Standard | None | github.com/yufu-wang/tram, MIT, training code | Rented GPU |
| **GVHMR** (SIGGRAPH Asia 2024, TPAMI 2026) | Video, gravity-view coords, ViTPose 2D + HMR2 features | AMASS, BEDLAM, H36M, 3DPW | PkVision: flips read ~0.5x (A) | github.com/zju3dv/GVHMR | Already tried |
| **AnyLift / MVLift** (CVPR 2026 / 2025) | 2D keypoint sequences → multi-view 2D diffusion → world 3D | In-domain internet 2D keypoints + GVHMR-reprojected local poses | Internet gymnastics beats GVHMR/WHAM on 2D metrics (F15) | anylift-release (licence not stated); mvlift_release (licence not stated, AIST data only provided) | Unknown compute; research bet |
| **DanceHMR** (ByteDance, arXiv 2605.18102) | Video, hand-aware whole body | Mocap + synthetic, two-stage curriculum | Dance only | Not checked | n/a |
| **Sapiens2** (Meta, ICLR 2026, arXiv 2604.21681) | Image, 0.4B to 5B, 1K res; pose (308 kp), seg, normals, pointmap | 1B human images pretraining | None on acrobatics | github.com/facebookresearch/sapiens2, Sapiens2 License (commercial allowed with restrictions); pose weights 0.4B/0.8B/1B/5B | 0.4B likely fits 2060 in fp16 (C, unverified) |

**Rotation-estimation accuracy on flips/twists:** not found for any HMR or world-grounded model. No paper reports
root-orientation error, somersault count or twist count from an HMR output on acrobatic video. The closest measured
statements are negative: VIFSS's 3D lifter fails on unseen quads (F5), lifters without rotation augmentation
triple their error under in-plane rotation (F10), and PkVision's GVHMR under-counts flips (A).

A plausible mechanism for GVHMR's 0.5x (C, my inference): its temporal regressor and priors are trained on AMASS,
where full-speed somersaults barely exist, so the learned dynamics damp fast root rotation; on top of that, its
ViTPose 2D inputs degrade when inverted (F6, F8). A per-frame model (SAM 3D Body) avoids the first failure but
not a new one: frame-to-frame 180° front/back flips in the root orientation, which would corrupt unwrapping.

### 2.2 Acrobatic motion and pose datasets (for fine-tuning or evaluation)

| Dataset | Content | 3D GT | Access / licence | Use for PkVision |
|---|---|---|---|---|
| **TramPoseFit / SynTramPose** (2604.01322) | 10 trampoline sequences, 3 elite athletes, 18-cam Vicon 200 fps fitted to SMPL; 2,520 synthetic multi-view renders | Yes (SMPL) | "Code will be available soon"; not released | Best-matched acrobatic SMPL motion if released |
| **AthletePose3D** (2503.07499) | 1.3M frames, running, throws, 6 skating jumps | Yes | Non-commercial research | Fast-rotation lifting, no flips |
| **FS-Jump3D** (via VIFSS) | Figure-skating jumps incl. triples, 12-cam markerless | Yes | Public (per paper) | View-invariant pretraining |
| **Fit3D** | Extreme fitness incl. inverted | Yes | Non-commercial | Inverted-pose HMR fine-tune |
| **BEDLAM2.0** (arXiv 2511.14394) | Synthetic humans + moving cameras | Yes | Research licence | Base for synthetic renders; no acrobatics claimed |
| **Text-to-skeleton acrobatic set** (arXiv 2603.08028) | 2,000 Blender videos of flips, cartwheels, stunts from **Mixamo** motions | Rendered | Mixamo terms for ML training not verified | Cheap acrobatic motion source, licence risk |
| **KungFuAthlete** (arXiv 2602.13656) | Martial-arts jump subset, from training videos | Video-derived (not mocap) | Not checked | Aerials, but GT quality inherits HMR errors |
| **FineGym / Gym288-skeleton** | Broadcast gymnastics, 2D HRNet skeletons | No | FineGym terms; HF Lozumi/Gym288-skeleton | 2D temporal pretraining with twist classes |
| **Diving48, MTL-AQA, FineDiving** | Broadcast diving, twist/somersault codes | No | Research | 2D twist-counter validation |

No open, SMPL-format, flip-and-twist mocap dataset released in 2025–2026 was found. AMASS still lacks acrobatics.

---

## 3. 2D pose robustness to inversion and fast motion

### 3.1 What is measured

- Stock COCO models lose 8 to 18 AP on trampoline footage (F6) and mislabel left/right sides. Pose2Sim's
  RTMPose default is flagged by its developer as not working well upside down (F8). PkVision's own RTMPose-x
  confidence drops on corks and inverted hangs (A).
- Newer foundation-scale estimators are better on inverted bodies: SAM 3D Body's 2D reprojection wins every
  SA1B-Hard category, most on inverted bodies (F9). Sapiens2 (+4 mAP over Sapiens) and SDPose (Stable Diffusion
  U-Net features; beats Sapiens-2B on the COCO-OOD corruption split incl. blur, 54.3 vs 52.8 whole-body AP;
  arXiv 2509.24980, MIT, github.com/T-S-Liang/SDPose-OOD) claim OOD robustness but report nothing on acrobatics (C for
  this use).
- PMPose / BBoxMaskPose v2 (arXiv 2601.15200) improves crowded scenes; irrelevant for single athletes.
- RTMPose successors in MMPose (RTMO, RTMW, DWPose) add crowd, whole-body and speed. No inversion benchmark found.

### 3.2 Test-time and training-time tricks

| Trick | Evidence | Grade | Cost |
|---|---|---|---|
| **Rotation augmentation to ±180° during fine-tuning** | RePoGen +13 AP on bottom views (F7); OpenPose partial fix (F8); for lifters, 3x error removed (F10) | B | Fine-tune, ≤$10 |
| **Synthetic inverted renders (few hundred to 3k) mixed with real sports images** | RePoGen (F7); trampoline +17 AP only when mixed with real LSP data (F6) | B | Blender renders ($0 on Mac/2060) + fine-tune |
| **Test-time rotation (rotate frame 0/90/180/270°, keep the most confident pass, rotate keypoints back)** | No measured study on human pose found. FOCAL (arXiv 2507.10375) shows test-time canonicalization over discrete 2D rotations helps classification/segmentation zero-shot, but did not test keypoints. | C | Free, 4x inference |
| **Temporal 2D pose (video) to ride through blur** | TAR-ViTPose +2.3 mAP on PoseTrack17 over ViTPose (arXiv 2603.05929, github.com/zgspose/TARViTPose); FineTec completion under 75% frame loss (F13) | B | Inference-time |
| **Capture-side: shorter shutter / higher fps** | Connolly et al. used 1/120 s shutter at 30 fps to cut blur (arXiv 1709.03399); NS-AQA fails under strong blur (F1) | B | iPhone 120/240 fps slow-mo is free |
| **Promptable HMR to correct 2D** | SAM 3D Body accepts 2D keypoint prompts; accurate prompts improve 2D and 3D (its Table 7) | B | Needs good prompts |

---

## 4. Estimating body orientation or rotation directly

| Approach | What it gives | Evidence on acrobatics | Grade |
|---|---|---|---|
| **Hand-crafted 2D keypoint vectors** (NS-AQA hip-vector petals for twist; pelvis-to-thorax rotation for somersault; Connolly's shoulder separation as |twist angle|) | Half-twist and half-somersault counts, training-free | 93.3% twist / 97.3% somersault on diving (F1); trampoline skill ID 80.7% with an unsigned shoulder-separation twist feature, but twisting somersaults (full front, full back, rudi) were excluded for lack of examples; only twist jumps and half-twist drops were classified | B |
| **SAM 3D Body per-frame root orientation** | Full SO(3) per frame; flips and twists by swing–twist decomposition and unwrapping | Orientation-ambiguous category PVE 42 vs 84 for CameraHMR (F9); never tested on flips | B (static) / C (flips) |
| **MEBOW** (CVPR 2020, arXiv 2011.13688), **PedRecNet** (arXiv 2204.11548) | Body azimuth for upright pedestrians | Not designed for inverted or twisting bodies | C |
| **Orient Anything V2** (arXiv 2601.05573, Jan 2026) | Object orientation and relative rotation between image pairs | Trained on synthesized 3D assets, never evaluated on humans in flight | C |
| **2D-to-3D uplifting with joint rotations** (Ludwig et al., arXiv 2504.09953) | Root and joint rotations in one pass | Evaluated on Fit3D only | C for acrobatics |
| **Optical-flow spin-rate estimation of a human** | — | **Not found** | — |
| **Angular-momentum / ballistic-flight constraints for pose in the air** | — | **Not found** (only ground-contact dynamics work, e.g. arXiv 2606.08133) | — |

---

## 5. Claim (d): the twist geometry, worked through

This is a derivation (grade C by itself); the measured support is F1 to F5.

Set up a near-orthographic camera looking along z. Let the body's long axis be u (pelvis to thorax) and its
lateral axis v (right hip or shoulder to left). A twist is a rotation about u.

1. **Twist about an axis in the image plane** (side-view flips, upright twisting jumps). Projected shoulder/hip width
   scales with |cos θ|: it collapses to zero at each quarter turn and grows again, so the magnitude of twist is
   visible as the NS-AQA "petals" (one per half twist). With labeled left/right keypoints, the 2D cross product
   sign(u × v) is invariant to in-plane rotation (it does not care that the athlete is upside down) and flips each
   time the athlete's front/back facing flips. Counting facing changes gives half-twists.
2. **Direction (left vs right twist)** is lost in joint positions alone under orthographic projection: the depth
   sign of the rotation is ambiguous (the classic tilt/Necker ambiguity for the near-planar torso quadrilateral).
   It is recoverable from an off-plane labeled point (nose, ears, feet when the body is extended) or from
   occlusion order. These cues are small (nose offset ~10 cm vs torso ~40 cm) and noisy (the head turns on the
   neck), so direction is fragile but not impossible.
3. **Twist about an axis pointing at the camera** (long axis toward the lens). This is an in-plane rotation of the
   shoulder line, which is fully visible in principle. The May 2026 memo's "twist is invisible when the spin axis
   is near the camera axis" has the geometry backwards. The real problem in this configuration is perception:
   extreme foreshortening makes shoulders, hips and head overlap, and 2D estimators fail.
4. **Front-on filming of flips** couples the two rotations: a somersault about an axis parallel to the image
   plane also flips front/back facing twice per flip. Facing changes then mix flip and twist; separating them needs
   the flip count from the long-axis foreshortening cycle, or a full per-frame 3D orientation (SAM 3D Body).

So twist count is recoverable from one camera whenever the 2D estimator gets left/right labels and face points
right in the flight frames. The failure modes that remain are perception failures (F4, F6, F8, F13, A-grade
confidence drops), not an information-theoretic wall. That is why F1 to F3 exist: diving and gymnastics broadcasts
are side-on and well lit, and 2D pose on good crops is now good enough.

**Caveats that keep the in-domain verdict at UNKNOWN:** (i) Diving48 and FineGym are closed label sets where twist
correlates with other attributes, so classifiers can partly guess twist from context; NS-AQA is rule-based and
does not have that crutch, but works on clean, side-on, high-contrast diving; (ii) no source gives per-class
accuracy for the ≥1.5-twist classes; (iii) corks and off-axis rotations are not covered by any of these; (iv)
parkour clips are handheld, variable angle, often front-on.

---

## 6. Skeleton SSL and pretraining with fine-grained sports validation

- **Generic masked/contrastive skeleton SSL, 2025–26:** AMR (arXiv 2606.11450, NTU-60/120, PKU-MMD), GFP
  (arXiv 2509.03609, ICCV 2025), S2I skeleton-to-image with vision-pretrained backbones (arXiv 2603.05963, NTU,
  PKU-MMD), SOfA cross-sensor foundation model (arXiv 2609.07078, ten 3D skeleton sets). **None reports FineGym,
  FineDiving, Diving48 or skating.** The 2024 SSL benchmark (arXiv 2406.02978) has no fine-grained sports either.
- **Sports-validated pretraining:** only VIFSS (F5, F16): contrastive view-invariant pretraining on random virtual
  2D projections of 3D mocap, then supervised fine-tuning. It matters most at low label counts (1% labels: >60%
  element F1@50 vs ~0). It is not SSL on unlabeled in-domain video; it needs 3D acrobatic motion to project.
- **Supervised fine-grained skeleton models:** FineTec (AAAI 2026) and FineX (Aug 2026) on Gym99/Gym288/Diving48
  show 2D skeletons carry the signal when supervised (F2, F3, F13). Semi-supervised fine-grained work (SeFAR
  arXiv 2501.01245, FinePseudo arXiv 2409.01448) is RGB, not skeleton.
- **In-house (A):** SSL gave +0.023.

Verdict (c): **STILL TRUE.** If PkVision pretrains a skeleton encoder, the evidence favours VIFSS-style
view-invariant pretraining on projected 3D acrobatic motion over generic masked reconstruction, but the 3D
acrobatic motion is the missing ingredient (Section 2.2).

---

## 7. Fine-tuning feasibility and GPU hours

Rented 4090 at ~$0.3–0.7/h. Estimates marked (C) are mine.

| Job | Evidence for the recipe | Compute | Cost tier | Blockers |
|---|---|---|---|---|
| Rotation-TTA on existing RTMPose-x | None measured for keypoints (C) | 4x inference; Mac/2060 | Free | None |
| Rule-based twist/flip counters (NS-AQA style) on existing keypoints | F1 | CPU | Free | None |
| Fine-tune a 2D estimator (ViTPose-B or RTMPose-m) on COCO subset + LSP-style sports images + 500 to 3,000 synthetic inverted renders, rotation aug ±180° | F6, F7 | Render 1–3k images with Blender EEVEE on Mac/2060 (~1–3 h, C). Fine-tune ~2–10 h on a 4090 (C). | ≤$20 ($1–7) | Motion source for renders: RePoGen shows random joint sampling + full SO(3) root works without acrobatic mocap; TramPoseFit not released; Mixamo ML terms unverified. SMPL/SMPL-X licences are non-commercial. |
| SAM 3D Body inference, 12-clip probe (~1–2k frames) | F9, F18 | ~0.5–2.5 h on a 4090 incl. setup | ≤$3 | 2060 likely too small (C) |
| SAM 3D Body over all 1,618 clips as a weak orientation teacher (~100k flight frames) | F18 | Vanilla: several s/frame → ~80–140 h. Fast 3DB (10.9x): ~8–15 h (C) | ≤$150 vanilla, ≤$20 with Fast 3DB | Only worth it if the probe shows it counts twists |
| Fine-tune SAM 3D Body | — | — | Not feasible | No training code released |
| Retrain GVHMR with added acrobatic motion | F17 | 26 GPU-h from scratch (13 h on 2x4090), less to fine-tune | ≤$20–50 | No acrobatic SMPL motion; its 2D inputs still fail inverted |
| Fine-tune HMR2.0 / TokenHMR / CameraHMR on synthetic acrobatic renders | Fit3D showed the principle (F12) | GPU-hours not reported by any source; my estimate 10–40 4090-h plus 10–50k renders | ≤$50 | Renders, licences, and still no temporal rotation model |
| Fine-tune a lifter (AthletePose3D recipe) | F11 | Small (60 epochs on 81-frame clips; hours) | ≤$20 | Needs 3D acrobatic GT |
| Train AnyLift/MVLift 2D-diffusion prior on PkVision's 1,618 clips' 2D keypoints | F15 | Not reported | Unknown | Garbage-in if 2D keypoints fail inverted; licences unstated |

---

## 8. Verdicts with evidence

**(a) Monocular 3D HMR is dead for acrobatics: WEAKENED.**
For: no 2025–26 HMR reports flip/twist rotation accuracy (absence across F14); video/world-grounded models train on
AMASS/BEDLAM-type data (F14); 3D lifting fails on unseen fast rotations and underperforms 2D features for rotation
recognition (F5); lifters collapse under in-plane rotation without augmentation (F10); in-house GVHMR 0.5x (A).
Against: SAM 3D Body's hard-pose data engine measurably fixed single-frame inverted/very-hard poses (F9); sport
fine-tunes cut lifter error by ~70% (F11) and HMR error on inverted fitness poses (F12); AnyLift beats GVHMR on
internet gymnastics by proxy metrics (F15). Net: HMR is no longer "dead", it is "unmeasured on flips, with one
per-frame candidate worth a $3 probe". Using HMR to count twists over time is still unsupported by any evidence.

**(b) Stock 2D pose fails upside-down: STILL TRUE for stock COCO models; not a blocker.**
Stock degradation is measured (F6: −8 to −18 AP; F8; A-grade confidence drops), including left/right confusion,
which is exactly what a twist counter needs. Two analogs close the gap with rotation augmentation and a few
thousand synthetic renders mixed with real sports images (F6, F7), and newer foundation estimators already do much
better on inverted bodies (F9). The claim should be restated as "stock COCO-trained RTMPose/ViTPose degrade upside
down; the fix is cheap and measured".

**(c) Skeleton SSL is unvalidated on acrobatics: STILL TRUE.** No 2025–26 skeleton SSL reports any acrobatic or
fine-grained-sport number (F16). The one sports-validated pretraining (VIFSS) is view-invariant contrastive
learning on projected 3D mocap and needs acrobatic 3D motion PkVision does not have.

**(d) Twist ≥ 1.5 is geometrically unrecoverable from one camera: OVERTURNED (geometry); UNKNOWN (parkour accuracy).**
Single-view 2D keypoints count dive twists (examples shown up to 3 twists; the sibling survey 01 reports the range as 0 to 3.5) at 93.3% with a training-free rule (F1); skeleton-only
classifiers reach ≥85.9% twist-attribute accuracy on Diving48 (F2) and 94.2% on FineGym, which contains
1.5/2/2.5/3-twist classes (F3); figure-skating rotation levels 1 to 4 at 92.6 F1@50 (F5). The geometry (Section 5)
says twist magnitude is observable from foreshortening and facing changes, and direction needs chirality cues.
What remains is perception quality on inverted, blurred, front-on parkour frames and corks, for which no
measurement exists.

---

## 9. Best pose/HMR candidates for twist and flip

| Rank | Candidate | Code / licence | Compute fit | Cost | Main risk |
|---|---|---|---|---|---|
| 1 | **Best available 2D keypoints + explicit rule-based counters** (NS-AQA hip-vector petals + chirality sign(u × v) for twist; pelvis-to-thorax unwrap for flips). Start with existing RTMPose-x + 4-way rotation TTA; swap in Sapiens2-pose-0.4B if it wins the probe; fine-tune with the RePoGen/trampoline recipe only if needed. | RTMPose/MMPose Apache-2.0; rtmlib; Sapiens2 License; NS-AQA reference code non-commercial (reimplement the rule, it is a few lines); RePoGen GPL-3.0 + SMPL-X non-commercial | Mac and 2060 for inference and counters; 4090 only for the optional fine-tune | Free to probe; ≤$20 with fine-tune | Left/right swaps and hallucinated face points on inverted/back-facing frames flip the facing sign and break the count; front-on camera angles mix flip and twist. |
| 2 | **SAM 3D Body per-frame root orientation** (optionally SAM-Body4D masklets for tracking), twist by swing–twist decomposition about the long axis, flips by unwrapping; its reprojected 2D keypoints also feed candidate 1 | github.com/facebookresearch/sam-3d-body, SAM License; no training code | Rented 4090 (2060 likely too small, unverified) | Probe ≤$3; bulk ≤$20 with Fast 3DB, up to ~$100 vanilla | Per-frame 180° front/back flips corrupt unwrapping; smoothing (Kalman in SAM-Body4D) may re-introduce GVHMR-style damping; never tested on flips; cannot be fine-tuned. |
| 3 | **Domain-trained 2D-diffusion lifting (AnyLift)** trained on PkVision's own 1,618-clip 2D keypoints, as a research bet for world-grounded flips | anylift-release, licence not stated | Unknown; rented GPU | Unknown | Depends on candidate 1's 2D quality; evaluated only by 2D proxies; heavy engineering. |

Downstream of any of these, VIFSS-style view-invariant embeddings (Apache-2.0) are the best-supported skeleton
representation, if 3D acrobatic motion becomes available.

---

## 10. Probe design (≤ 12 clips, ≤ $3, exploratory)

**Clips (12).** The 12 July reference clips with corrected labels, checked to include: ≥3 twist ≥1.5 (incl. at least
one double full), ≥2 corks/off-axis, ≥2 bad angles (one front-on), and the hang-castaway double back
(RTMPose-x conf 0.22–0.29). If the July set lacks a category, swap in one of the ~13 own iPhone clips, ideally one
shot at 120/240 fps. Use only the corrected references, not the gold-99 FIG-join labels (~35% wrong on twist).

**Arms (same 12 clips each).**
- A0: RTMPose-x as already extracted (or re-run locally), $0.
- A1: RTMPose-x with 4-way frame rotation (0/90/180/270°), per frame keep the pass with the highest mean flight-keypoint confidence, rotate keypoints back. $0, Mac/2060.
- A2: Sapiens2-pose-0.4B, COCO-17 subset of its 308 keypoints. $0 if it fits the 2060, else ~$0.3 on a 4090.
- A3: SAM 3D Body (ViT-H): (i) its 2D reprojections fed to the same counters; (ii) its per-frame root orientation. ~$1–3 on a 4090.

**Counters (written and frozen before any output is looked at).**
- Flip count: unwrap the image-plane angle of pelvis→thorax over the flight segment, count half-rotations (NS-AQA rule). For A3(ii), unwrap the swing of the long axis in 3D.
- Twist count: (a) NS-AQA hip-vector petals with fixed inner/outer thresholds normalized by torso length; (b) number of sign changes of sign(u × v) with a 3-frame hysteresis; (c) for A3(ii), twist angle about the long axis relative to takeoff, unwrapped.
- Direction: sign of the flip-angle derivative vs travel direction.
- Abstain rule: if <60% of flight frames have mean keypoint conf ≥0.3, output "abstain".

**Metrics.** Per attribute exact match vs corrected references; twist MAE in half-twists; abstention rate.
Label-free diagnostics per arm: mean flight-frame confidence, share of frames where shoulder-chirality and
hip-chirality disagree (a left/right swap proxy), and 1-frame facing flickers.

**Decision rules (pre-registered).**
- GO for monocular 2D twist route: best arm twist exact ≥9/12 including ≥2 of the ≥1.5-twist clips, and flip exact ≥10/12.
- CONDITIONAL: 7–8/12 → run the ≤$20 fine-tune (rotation aug ±180° + synthetic inverted renders) and repeat the same 12 clips once.
- NO-GO for monocular pose-based twist: every arm ≤6/12 → twist needs a second view or a different signal (VLM track).
- If A3(ii) beats the best 2D arm by ≥2 clips on twist, SAM 3D Body becomes the twist weak-labeler candidate (bulk ≤$20 with Fast 3DB).

**Honesty note.** n = 12 is tiny: 9/12 has a Wilson 95% interval of roughly 0.47 to 0.91. This probe can kill an
arm or rank arms; it cannot confirm the 85% twist-precision threshold from the framing doc.

---

## 11. Gaps and "not found"

- **Not found:** any measurement of HMR root-orientation error, flip count or twist count on acrobatic video (2025–26).
- **Not found:** per-class accuracy for ≥1.5-twist classes on FineGym or Diving48 (only aggregates).
- **Not found:** any evaluation on corks/off-axis rotations or on handheld phone footage.
- **Not found:** a measured study of test-time image rotation for human keypoints on inverted people.
- **Not found:** optical-flow-based human spin-rate estimation; physics (angular-momentum) constrained pose in flight.
- **Not found:** skeleton SSL (2025–26) validated on any acrobatic or fine-grained sports dataset.
- **Not found:** an open SMPL-format flip-and-twist mocap dataset. TramPoseFit/SynTramPose code and data are announced but not released.
- **Unverified:** SAM 3D Body and Sapiens2 VRAM needs (no official numbers); SAM-Body4D and AnyLift/MVLift licences; Mixamo terms for ML training; GPU-hours for fine-tuning HMR2.0/TokenHMR/CameraHMR (never reported).
- **Unverified here:** DuoMo's rotation behaviour on fast motion (trained on AMASS/BEDLAM, so expect the GVHMR failure, C).

---

## 12. Sources

HMR and world-grounded motion
- Yang et al. SAM 3D Body: Robust Full-Body Human Mesh Recovery. Meta, arXiv 2602.15989 (Feb 2026; checkpoints 2025-11-19). github.com/facebookresearch/sam-3d-body, SAM License, inference only.
- Gao, Miao, Han. SAM-Body4D: Training-Free 4D Human Body Mesh Recovery from Videos. arXiv 2512.08406 (Dec 2025). github.com/gaomingqi/sam-body4d, licence not verified.
- Yang et al. Fast SAM 3D Body. arXiv 2603.15603 (Mar 2026). Code not verified.
- Wang et al. DuoMo: Dual Motion Diffusion for World-Space Human Reconstruction. CVPR 2026, arXiv 2603.03265. github.com/facebookresearch/DuoMo, XRCIA Noncommercial Research License.
- Li et al. GENMO/GEM: A Generalist Model for Human Motion. ICCV 2025, arXiv 2505.01425. github.com/NVlabs/GENMO, NVIDIA OneWay Noncommercial.
- NVIDIA GEM-X. github.com/NVlabs/GEM-X, Apache-2.0 code, NVIDIA Open Model License weights.
- Wang et al. PromptHMR. CVPR 2025, arXiv 2504.06397. github.com/yufu-wang/PromptHMR, no training code.
- Wang et al. TRAM. ECCV 2024. github.com/yufu-wang/tram, MIT.
- Shen et al. GVHMR: World-Grounded Human Motion Recovery via Gravity-View Coordinates. SIGGRAPH Asia 2024 / TPAMI 2026, arXiv 2409.06662. github.com/zju3dv/GVHMR.
- Li et al. AnyLift: Scaling Motion Reconstruction from Internet Videos via 2D Diffusion. CVPR 2026, arXiv 2604.17818. Project awfuact.github.io/anylift, code github.com/awfuact/anylift-release (licence not stated).
- Li, Liu, Wu. MVLift: Lifting Motion to the 3D World via 2D Diffusion. CVPR 2025, arXiv 2411.18808. github.com/lijiaman/mvlift_release (licence not stated).
- DanceHMR. ByteDance, arXiv 2605.18102 (May 2026). Not checked for code.
- Tesch et al. BEDLAM2.0: Synthetic Humans and Cameras in Motion. arXiv 2511.14394 (Nov 2025). Research licence.
- Fieraru. Hitting the Gym with Fit3D: Benchmarking and Improving Monocular 3D Human Reconstruction on Extreme Fitness Motions. J. Imaging (Jul 2026), https://pmc.ncbi.nlm.nih.gov/articles/PMC13413046/. Fit3D non-commercial.
- Ludwig et al. Efficient 2D to Full 3D Human Pose Uplifting including Joint Rotations. arXiv 2504.09953 (Apr 2025).
- Melnyk et al. On the Role of Rotation Equivariance in Monocular 2D-to-3D Human Pose Lifting. arXiv 2601.13913 (Jan 2026).
- Yeung et al. AthletePose3D. CVPRW 2025, arXiv 2503.07499. github.com/calvinyeungck/AthletePose3D, non-commercial research.
- Peng et al. SBF: skeleton augmentation (predicted 3D skeleton 63.9% vs 2D 93.6% on NTU). arXiv 2604.03590 (Apr 2026).

2D pose
- Drolet-Roy et al. Human Pose Estimation in Trampoline Gymnastics: How to Improve Performance on Extreme Poses. arXiv 2604.01322 (Apr 2026). github.com/VisionICLab/trampoline_syn_data ("code will be available soon").
- Purkrabek & Matas. RePoGen: Improving 2D Human Pose Estimation in Rare Camera Views with Synthetic Data. FG 2024, arXiv 2307.06737. github.com/MiraPurkrabek/RePoGen, GPL-3.0 (+ SMPL-X non-commercial).
- Purkrabek et al. BBoxMaskPose v2 / PMPose. arXiv 2601.15200 (Jan 2026). Code on project page.
- Khirodkar et al. Sapiens2. ICLR 2026, arXiv 2604.21681. github.com/facebookresearch/sapiens2, Sapiens2 License.
- Liang et al. SDPose: Exploiting Diffusion Priors for Out-of-Domain and Robust Pose Estimation. arXiv 2509.24980 (Sep 2025). github.com/T-S-Liang/SDPose-OOD, MIT.
- Fang et al. TAR-ViTPose: Temporal Aggregate-and-Restore ViT. arXiv 2603.05929 (Mar 2026). github.com/zgspose/TARViTPose.
- Jiang et al. RTMW. arXiv 2407.08634. MMPose (Apache-2.0); rtmlib.
- Cao et al. OpenPose. TPAMI 2019, arXiv 1812.08008.
- Pose2Sim README. github.com/perfanalytics/pose2sim, BSD-3-Clause.
- Singhal et al. FOCAL: Test-Time Canonicalization by Foundation Models for Robust Perception. arXiv 2507.10375 (Jul 2025).

Orientation
- Wu et al. MEBOW: Monocular Estimation of Body Orientation In the Wild. CVPR 2020, arXiv 2011.13688.
- Burgermeister & Curio. PedRecNet. arXiv 2204.11548.
- Wang et al. Orient Anything V2. arXiv 2601.05573 (Jan 2026).

Twist/rotation recognition analogs
- Okamoto & Parmar. Hierarchical NeuroSymbolic Approach for Comprehensive and Explainable AQA (NS-AQA). CVPRW 2024, arXiv 2403.13798. github.com/laurenok24/NSAQA, non-commercial.
- Hassan et al. FineX: Fine-Grained Action Recognition with Cross-Attentive Latent Sparse Experts. arXiv 2608.13458 (Aug 2026).
- Shao et al. FineGym. CVPR 2020, arXiv 2004.06704; supplementary class list sdolivia.github.io/FineGym/resources/supp.pdf.
- Duan et al. PoseC3D: Revisiting Skeleton-based Action Recognition. CVPR 2022, arXiv 2104.13586. PYSKL github.com/kennymckormick/pyskl (Apache-2.0).
- Shao et al. FineTec. AAAI 2026, arXiv 2512.25067. Gym288-skeleton on Hugging Face (Lozumi/Gym288-skeleton).
- Li, Li, Vasconcelos. Diving48 (RESOUND). ECCV 2018. svcl.ucsd.edu/projects/resound (unreachable at survey time; vocabulary examples from secondary sources).
- Tanaka, Suzuki, Fujii. VIFSS. arXiv 2508.10281 (Aug 2025). github.com/ryota-skating/VIFSS, Apache-2.0.
- Connolly, Silvestre, Bleakley. Automated Identification of Trampoline Skills Using Computer Vision Extracted Pose Estimation. arXiv 1709.03399 (2017).
- Yang et al. (Fujitsu). Enhancing Multi-Camera Gymnast Tracking Through Domain Knowledge Integration. arXiv 2511.16532 (Nov 2025). Multi-camera, used at Gymnastics World Championships.

Skeleton SSL
- Sun et al. AMR: Adaptive Masked Reconstruction. arXiv 2606.11450 (Jun 2026).
- GFP. ICCV 2025, arXiv 2509.03609.
- Yang et al. S2I: Skeleton-to-Image Encoding. arXiv 2603.05963 (Mar 2026).
- Do, Chen, Kim. SOfA: Generalist Foundation Model for Cross-Sensor Skeleton Representation Learning. arXiv 2609.07078 (Sep 2026).
- Self-Supervised Skeleton-Based Action Representation Learning: A Benchmark and Beyond. arXiv 2406.02978.
- SeFAR. AAAI 2025, arXiv 2501.01245. FinePseudo, arXiv 2409.01448.

Acrobatic data sources
- Taghipour et al. Controllable Complex Human Motion Video Generation via Text-to-Skeleton Cascades (Blender + Mixamo acrobatics set). arXiv 2603.08028 (Mar 2026).
- Lei et al. KungFuAthlete. arXiv 2602.13656 (Feb 2026).
