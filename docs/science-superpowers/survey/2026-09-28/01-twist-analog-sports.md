# 01: Twist and rotation counting in analog sports

Survey date: 2026-09-28. Desk research only: no project data touched, no code run on project data, $0 spent.
Track: how diving, gymnastics, trampoline, figure skating, freestyle ski/snowboard/skate and general
repetition counting recognise twist (longitudinal-axis) and somersault counts from video, and how well.

Grading: **A** = measured on parkour / in-domain. **B** = measured on an analog sport. **C** = claim,
vendor statement or anecdote.

**No grade-A evidence exists.** I found no paper, dataset or product that measures twist or flip counting on
parkour or tricking video, handheld or otherwise.

---

## 0. Bottom line

1. **The claim "twist ≥1.5 is geometrically unrecoverable from one camera" is OVERTURNED as stated.**
   Several single-camera systems count multi-half-twist longitudinal rotations at roughly 86 to 97% aggregate
   accuracy (B): diving broadcast (Diving48, MTL-AQA) and figure-skating broadcast (1 to 4 revolutions).
   One of them, NS-AQA, is **training-free**: it uses hand-written rules on 2D keypoints and gets 93.3%
   twist-count accuracy on dives with 0 to 3.5 twists.
   What is still unknown: whether this holds on handheld iPhone parkour. There is no A-grade evidence either
   way. Per-class accuracy for the ≥1.5-twist classes is not reported anywhere (see Gaps).
2. **What does break, consistently:** (a) monocular 3D lifting and HMR under-count fast rotations that are
   absent from training data. This is the same failure as PkVision's GVHMR 0.5x. (b) Errors concentrate on
   adjacent counts, e.g. 1.5 vs 2.5. (c) Frontier VLMs are weak at counting in general.
3. **Most transferable method:** an NS-AQA-style geometric counter on the 2D hip and shoulder vectors that
   PkVision already has (RTMPose-x). It costs $0, needs no labels and no training, and fits a 12-clip probe
   directly.

---

## 1. Findings by sport

### 1.1 Diving

| # | Finding | Method / input | Twist-specific number | Dataset | Grade | Transfer to handheld parkour |
|---|---|---|---|---|---|---|
| D1 | **NS-AQA** (Okamoto & Parmar, CVPR 2024 Workshops, CVsports best paper; arXiv 2403.13798) | Rule-based "microprograms" on per-frame **2D pose**. Twist = count of "petals" traced by the right-to-left **hip vector** (1 petal = 0.5 twist). Somersault = rotation of the pelvis-to-thorax vector. No training for recognition. Single broadcast view. | **#Twists acc 93.27%**, #Somersaults 97.31%, rotation type 99.37%, position 97.28% | MTL-AQA (1,412 dives, 0 to 3.5 twists, 0 to 4.5 somersaults) | B | **High for the method** (same 2D keypoints we have, no labels). Parkour differences: shorter flight, handheld camera, varied views, corks. |
| D2 | **C3D-AVG-MTL** (Parmar & Morris, CVPR 2019; arXiv 1904.04346) | Supervised 3D-CNN multitask on 96 frames at 112×112, single broadcast view | **#Twists 93.20%**. Baselines: Nibali et al. 79.89%, MSCADC 82.72% | MTL-AQA | B | Low: needs labelled data. |
| D3 | **Diving48 SOTA 2026: FineX** (Hassan et al., arXiv 2608.13458, Aug 2026) | RGB R(2+1)D + PoseC3D heatmaps + ST-GCN++ skeleton fusion. Pose from D-FINE + ViTPose-L. | 48-way top-1 **92.9%**. Pose-only **ST-GCN++ 85.9%**, PoseC3D 82.8%, PeVL 92.5%, AIM ViT-L 90.6%. Classes encode twist 0/0.5/1/1.5/2/2.5/3/3.5; 13 of 48 classes have ≥1.5 twists. A correct class implies a correct twist count, so these lower-bound the twist accuracy on average. | Diving48 v2 (16k train / 2k test, single-view competition video) | B | Medium: shows 2D skeleton alone carries the twist signal, but it is supervised on 16k clean labels. |
| D4 | **TQN** (Zhang, Gupta, Zisserman, CVPR 2021; arXiv 2104.09496) | Query per attribute (take-off, somersault, twist, flight pose), S3D RGB, dense sampling | Diving48 81.8% per-video / 74.5% per-class. **"The main errors come from counting the number of turns and twists, especially those with similar counts."** 3.5-twist is 0.3% of training data. | Diving48, FineGym | B | Qualitative: twist is the hardest attribute, and errors are off by one half-twist. |
| D5 | Recent Diving48 backbones: MamBOA 86.24% (arXiv 2606.15275, Jun 2026); CAN 88.4% (arXiv 2512.18750, Dec 2025) | RGB video backbones | Aggregate only | Diving48 | B | Low. |
| D6 | **VLM zero-shot diving AQA** (Velesaca et al., arXiv 2609.19354, Sep 2026) | Qwen2.5-VL 3/7/32B, LLaVA-NeXT-Video, MiniCPM-V 2.6, InternVL3 prompted with judging criteria | **Twist or dive-type recognition not evaluated.** Score Spearman ≤0.32 standalone, 0.67 with a trained regressor. | AQA-7 10 m platform | B | Shows open VLMs do not judge dives zero-shot. Says nothing about twist counting. |
| D7 | Baidu AI diving assistant for the Chinese team (Paris 2024) | Ernie-based, video | **No numbers** | n/a | C | n/a |
| D8 | Twist kinematics (Walker, Sinclair, Cobley, ISBS 2015; 5253B dive) | IMU + 120 fps side camera | Peak twist angular velocity **−1435 ± 28 °/s**, plateau 1347 °/s for 0.18 s | 1 diver, 11 trials | B | Sets the fps requirement, see §3. |

Other diving datasets: FineDiving (CVPR 2022) and FineDiving-HM / FineParser (CVPR 2024) target procedure-aware
AQA and segmentation. I found no twist-count accuracy reported on them. NS-AQA gets AIoU@0.5 of 93.9% on
FineDiving temporal segmentation with rules.

**NS-AQA implementation details (read from the repo, `rule_based_programs/microprograms/dive_recognition_functions.py`,
github.com/laurenok24/NSAQA, non-commercial licence):**
- Keypoints are MPII-16 (pelvis index 6, thorax 7, r_hip 2, l_hip 3). Scale = median thorax–pelvis distance.
- Hip vector h = r_hip − l_hip, y-axis flipped. Thresholds are scale-normalised: `valid = scale/1.5`
  (longer vectors are rejected as outliers), `outer = scale/3.2`, `inner = scale/3.4`.
- A petal is counted when |h| rises above `outer`, with hysteresis: it must drop below `inner` before the
  next petal counts. There is an extra increment when the segment h(t−1)→h(t) passes within 0.5 px of the
  origin while |h| stays > `outer`, meaning the vector flipped between two frames.
- Frames are only used when the diver is off the board and the body is open (`position_tightness > 80`).
  Tuck and pike frames are skipped.
- Implication for PkVision: an **L/R keypoint swap** looks exactly like that fast flip, so it would add a
  spurious half-twist. The magnitude-only part is swap-invariant.

### 1.2 Gymnastics (artistic)

| # | Finding | Method / input | Number | Grade | Transfer |
|---|---|---|---|---|---|
| G1 | **Fujitsu Judging Support System (JSS)**, FIG. All 10 apparatus since the 2023 Worlds (Antwerp). | **4 to 8 fixed HD cameras per apparatus**, markerless multi-view 3D skeleton, element recognition by pretrained models trained on about 8,000 routines. Earlier (2019 Stuttgart, 4 apparatus) it used **MEMS lidar**, "2M+ pulses/s, 15 m range". Used only for inquiries and blocked scores, decided by the Superior Jury. | "About 2,000 elements, **about 90% accuracy vs a human**" (Fujitsu spokesperson, MIT Tech Review, 2024-01-16). "**95% accuracy**, results in 5 s to <1 min" (Fujitsu marketing page). No per-element or twist-count accuracy published. | C | What it needs that iPhones lack: calibrated, synchronised, fixed multi-camera rigs (4+ views), venue installation and a large labelled routine corpus. Fujitsu also markets a smartphone "Human Motion Analytics" product, with no twist numbers. |
| G2 | Fujitsu Research tracking paper (Yang et al., arXiv 2511.16532, Nov 2025) | **4 calibrated RGB cameras, 30 fps, 1080×1920**, YOLOX + HRNet 2D, triangulation, 3D pose, "frame-wise code of points". The authors state single-camera 3D trajectory recovery is "an ill-posed problem". | Tracking only: ID switches, ADE. No judging accuracy. | B (setup) | Confirms JSS runs at only **30 fps**, so high fps is not what makes it work. |
| G3 | JSS vs judges on a balance-beam leap (Ishikawa et al., Frontiers in Sports and Active Living 2026, doi 10.3389/fspor.2026.1798778) | 4 JSS cameras (Baumer VCXG-23C, **30 Hz**) | On 46 borderline trials: both accept 5, both reject 9, JSS-only 3, judges-only 1, mixed 28. **Judges' Fleiss κ = 0.44.** | B | Human agreement on borderline calls is itself moderate, so a "gold" reference will be noisy. |
| G4 | **FineGym** (Shao et al., CVPR 2020; arXiv 2004.06704) | Broadcast 720p/1080p. Element classes include "salto backward stretched with 2 twist" etc. | 2020 failure modes explicitly include "**degree of rotation**" and "**counting the times of saltos**". ST-GCN (skeleton) failed then because pose failed on saltos (Gym99 36.4%). | B | Historical baseline. |
| G5 | **FineGym SOTA 2026** (FineX, arXiv 2608.13458) | Same fusion as D3 | Gym99 97.1%, Gym288 94.3% top-1 / 76.2% mean-class. **Pose-only ST-GCN++ Gym99 94.2%, Gym288 85.9%.** No per-twist breakdown. | B | Since 2020, 2D pose on acrobatics has stopped being the blocker (ST-GCN 36% to ST-GCN++ 94% on Gym99). Supervised, though. |
| G6 | TQN on FineGym (arXiv 2104.09496) | "twist" query with attributes 0.5/1/1.5/2/2.5/3; "turn" query 0.5 to 3 | Confusion matrices only (App. I). No per-query accuracy in text. | B | Same off-by-one pattern. |

I found no 2025–2026 paper on twist counting specifically for vault or floor from monocular video.

### 1.3 Trampoline

| # | Finding | Method / input | Number | Grade | Transfer |
|---|---|---|---|---|---|
| T1 | Connolly, Silvestre, Bleakley (arXiv 1709.03399, 2017) | **Single side-on consumer camera, 1080p30, 1/120 s shutter**, Stacked Hourglass 2D pose + MonoCap filter. Twist proxy = normalised **shoulder separation** (0 to 180°). 1-NN classification on angle trajectories. | **80.7%** over 20 skills. Twist content only up to full-twist jump (F2F) and half-twist variants. **Twisting somersaults (Rudi, Full Front, Full Back) were excluded (<10 examples).** | B | The shoulder-width twist proxy is the same idea as NS-AQA. Twisting somersaults remain unmeasured. |
| T2 | Woltmann et al., German J. Exercise & Sport Research 2022 | IMU | 96.4% (DNN) / 96.1% (CNN) on 10 jump types | B | Sensors are forbidden in our setting. |
| T3 | Helten et al., Sports Engineering 2011 | 10-IMU suit, DTW | 84.7% over 14 skills | B | Same. |
| T4 | Drolet-Roy et al. (arXiv 2604.01322, Apr 2026) | 2D pose fine-tuned on synthetic trampoline SMPL renders (from 18-camera Vicon at 200 fps), evaluated on **8 OptiTrack colour cameras at 120 fps** | 2D AP 55.8 to 73.1 (ViTPose-S). 3D MPJPE −19.6%. With **3 cameras only 51% of joints triangulate validly** (31% for the baseline). No skill or twist recognition. | B | Warning for the 2-phone plan: few-view triangulation of inverted acrobatic poses loses many joints. Synthetic acrobatic renders fix 2D pose, which is a cheap PkVision lever. |

I found no automatic trampoline difficulty (twist/somersault count) system from video in 2025–2026.
FIG trampoline measures time of flight and horizontal displacement electronically; difficulty is still
counted by judges.

### 1.4 Figure skating (rotation about the vertical axis = twist-axis analog)

| # | Finding | Method / input | Number | Grade | Transfer |
|---|---|---|---|---|---|
| F1 | **VIFSS** (Tanaka, Suzuki, Fujii; arXiv 2508.10281, Aug 2025) | **Broadcast single view, 25 fps**. 2D pose into a view-invariant contrastive pose embedding (pretrained on FS-Jump3D + H36M etc.), fine-tuned on SkatingVerse, then FACT temporal segmentation. Element level = 23 labels of **jump type × rotations (1 to 4)**. | Element-level **F1@50 92.56%, frame acc 85.82%**. Raw 2D pose 78.78 F1@50. **3D pose (MotionAGFormer) 76.57, worse than 2D.** It failed on a quad toe loop because quads were absent from its 3D training set. | B | **Directly analogous to GVHMR 0.5x:** 3D lifting collapses on unseen fast rotations, while 2D-based features do not. Supervised with about 20k labelled clips. |
| F2 | **SkatingVerse challenge** 1st place (Sun et al., arXiv 2404.14032; workshop report 2405.17188) | RGB ensemble (Unmasked Teacher + UniformerV2 + InfoGCN skeleton), DINO crop | **95.73% top-1** on 28 classes (23 = jump type × single/double/triple/quad) | B | Supervised (19,993 train clips). Rotation counts are confounded with jump-type and air-time priors. |
| F3 | FS-Jump3D (Tanaka et al., arXiv 2408.16638, 2024) | **12 hardware-synced cameras**, Theia3D markerless mocap, 253 jumps up to triples | Dataset. The authors note that 2D-to-3D lifting "smoothed out" fast rotations. | B | Same lesson as F1. |
| F4 | YourSkatingCoach (arXiv 2410.20427) | 2D AlphaPose + Transformer-CRF, 30 fps, single view | Air-time detection 96.3% frame acc, mean duration error 25%. No rotation count. | B | Flight segmentation from 2D is doable. |
| F5 | **OOFSkate** (Jerry Lu, ex-MIT Sports Lab; NBC coverage at Milan-Cortina 2026) | Phone or tablet **single-camera** video. Outputs jump height, rotation speed, rotation count, landing. | **No accuracy published.** Anecdote: flagged a **quarter-revolution** short landing on a quad toe loop. A. Hosoi (MIT): "how many times did they go around ... none of those rely on depth." | C | Closest deployed analog to our setting (phone, single view). Unvalidated. |
| F6 | ISU AI initiative (Reuters via Yahoo Sports, Feb 2026, and CGTN 2026-02-11) | "**Six high-resolution cameras**" tested for 2 seasons. Targets rotations and edges first. Data-support only from 2026-27. | None | C | Even ISU chose multi-camera for official calls. |
| F7 | China "Figure Skating AI-Assisted Scoring System 1.0" (2022) | 8 keypoints, CV + DL | None | C | n/a |

### 1.5 Freestyle ski / snowboard / skate / tricking

| # | Finding | Method / input | Number | Grade | Transfer |
|---|---|---|---|---|---|
| S1 | **Merz et al., "Is a cork a legal shortcut?"**, Sports Biomechanics 24(7), 2024, doi 10.1080/14763141.2024.2399255 | IMU-measured amount of rotation vs named rotation, 149 tricks | Deviation from the named value: **corks median −25° (min −89°)**, flips −28° (min −94°), flatspins +21° | B | Named counts are **conventions**. Continuous measured rotation must be rounded with about ±90° tolerance, which matters for corks. |
| S2 | IMU vs markerless video for snowboard rotation (Sensors / ScienceDirect 2025, S2665917425000662) | Board IMU vs markerless multi-camera video, 8 elite riders, 88 trampoline bounce-board tricks | Abstract: "high validity" for rotation amount. Numbers not retrievable (HTTP 403). | B | Multi-camera markerless tracks rotation amount well. |
| S3 | **Google Cloud × U.S. Ski & Snowboard tool** (Google Cloud blog 2026-02-20) | "Any camera": Gemini for detection, DeepMind 3D skeleton, then quaternion integration of torso-frame rotation ("Rotational Degrees = Σα_i") | **No validation.** Example: Shaun White Cab Double Cork 1440 measured at **1,122°** (−318°) and attributed to "rotational efficiency". | C | **Inference (mine):** −318° is 3.5× Merz's worst cork shortcut (−89°), so it more likely shows monocular 3D under-counting, like GVHMR 0.5x, than real efficiency. |
| S4 | Snowboard halfpipe airtime by IMU + 1D U-Net (Sensors 2024, PMC11548732) | IMU | Airtime events | B | n/a |
| S5 | Skateboard trick classification (Abdullah et al. PMC8384043; SkateboardAI arXiv 2311.11467) | Small datasets, CNN / transfer learning | 90 to 100% on ≤5 to 15 tricks (e.g., FS 180 vs kickflip). **No spin counting beyond 180.** | B | Negligible. |
| S6 | "92.7% freestyle aerials trick recognition with IMU + multi-view video" (Springer chapter, 2025, via search snippet) | IMU + multi-view | 92.7% | C | Could not verify. |
| S7 | Parkour / tricking | No video recognition work on flips or twists. The only parkour video dataset found (Li et al., arXiv 2111.01591) covers kong vault, safety vault, pull-up and moving-up, with no flips. | none | n/a | **No A-grade evidence exists.** |

### 1.6 General: counting, VLMs, 3D HMR, fps

| # | Finding | Number | Grade | Relevance |
|---|---|---|---|---|
| V1 | **PushupBench** (arXiv 2604.23407, Apr 2026): repetition counting, 446 clips | Exact count: **Gemini 3 Flash 42.1%**, Gemini 3 Pro 39.8%, GPT-5 10.9%, **Claude Sonnet 4.5 9.5%, Claude Opus 4.5 4.9%**, Qwen3-VL ≤11.8%, supervised TransRAC 6.7%. **fps ablation: 1 fps 17.4% → 5 fps 42.1%** (a Nyquist effect). In RL rollouts, genuine counting accuracy collapses above about 10 repetitions (31% for GT 1–5). | B | VLM counting is weak and **frame-rate bound**. Twist counts are small (≤6 half-twists), which is the easier regime, but the frames must cover the flight densely. |
| V2 | **ActionAtlas** (NeurIPS 2024 D&B; arXiv 2410.05774): 580 fine-grained sports moves, multiple choice | GPT-4o best **45.5%** (chance 20.9%, non-expert humans 61.6%). More frames help. | B | Fine-grained motion is still weak for VLMs as of the 2024–25 models. |
| V3 | PkVision internal, July 2026: Claude on 24-frame grids | Twist 10/11 on clean clips vs corrected refs | A (internal, n=11) | The only in-domain number. It is small-n and on clean clips. |
| H1 | **AthletePose3D** (arXiv 2503.07499, 2025) | Monocular 3D on athletic motion: **MPJPE 214 mm** off the shelf, 65 mm after fine-tuning. Joint angles correlate, but **velocity estimation is limited**. | B | Integrating rotation from monocular 3D is unreliable unless fine-tuned in-domain. This corroborates the GVHMR failure. |
| H2 | VIFSS / FS-Jump3D (F1, F3) | 3D lifting fails on unseen quads | B | Same. |
| R1 | Classic class-agnostic repetition counters (RepNet, TransRAC, PoseRAC; cited in V1) | TransRAC 6.7% exact on PushupBench | B | Not a fit for 1 to 3 rotations in <1 s. |

---

## 2. Verdict on "twist ≥1.5 is geometrically unrecoverable from a single camera"

**OVERTURNED (as a geometric claim). In-domain feasibility on handheld iPhone parkour: UNKNOWN.**

Evidence against the claim (all B):
- **NS-AQA** counts twists from **one broadcast camera's 2D keypoints with no training**: 93.27% on MTL-AQA,
  where twists range 0 to 3.5. A supervised 3D-CNN reaches 93.20% on the same data.
- **Diving48**: 92.9% on 48-way single-view classification, including 13 classes with 1.5 to 3.5 twists.
  **2D skeleton alone reaches 85.9%.**
- **Figure skating**: 1 to 4 vertical-axis revolutions (2 to 8 half-turns) recognised from 25 fps broadcast
  at F1@50 92.6% (VIFSS) and 95.7% top-1 (SkatingVerse).
- Geometry: the image projection of a body-fixed lateral axis (hips, shoulders) either oscillates in length
  (period 180°) or rotates in-plane as the body twists, depending on view. Both signals are observable in 2D.
  What 2D cannot fix is the **sign** (twist direction) under depth reversal, and possibly the split of a
  **cork** into flip and twist components. The count itself is recoverable.

Evidence that keeps the claim's spirit ("hard", not "impossible"):
- Twist is the **worst attribute** everywhere it is broken out (TQN; MTL-AQA 93.2% twists vs 96.9% somersaults).
  Errors fall on **adjacent counts**.
- Per-class accuracy for the ≥1.5-twist classes is **not reported** in any source I found. Rare classes
  (3.5 twists = 0.3% of Diving48 training data) are likely much worse than the aggregate.
- **Monocular 3D lifting and HMR do fail** on fast unseen rotations (VIFSS/MotionAGFormer, FS-Jump3D,
  AthletePose3D velocities, and likely the Google 1440 → 1,122° case). So the original claim was probably
  induced by the GVHMR route, which the analog evidence says to abandon in favour of 2D-native signals.
- All positives come from **fixed, level, side-on broadcast cameras** with long flights (diving about
  1.5–2 s) and clean backgrounds. Handheld parkour differs on every one of those axes.

## 3. Requirements the analog evidence implies (fps, view)

- **Angular rate:** twist peaks at about **1,400 °/s** in elite dives (Walker 2015). At **30 fps** that is
  about 47°/frame, or **3.8 frames per half-twist**, which is enough for a 180°-period signal (Nyquist needs
  <90°/frame). At 60 fps it is 7.5 frames. JSS and VIFSS both work at 25–30 fps, so **high fps is not
  required; keypoint reliability during blur is**.
- **VLM frame grids:** a 24-frame grid spread over a 1.5 s clip samples about 16 fps. If flight is about
  0.6 s, that leaves about 10 frames for a double full (4 half-twists), 2.5 frames per half-twist, which is
  borderline. **Crop grids to the flight phase at ≥30 fps.** This is the PushupBench Nyquist lesson.
- **View:** NS-AQA thresholds are tuned for a side view, where the hip vector is short when the athlete is
  untwisted. From a front or back view the untwisted hip vector is long, which shifts the petal count by one.
  A robust counter must count **both** length lobes and in-plane angle turns, or normalise by view.
- **Shutter:** Connolly used 1/120 s at 30 fps to limit blur. iPhones in daylight usually do better.
  Indoor gyms are the risk.

## 4. Most transferable methods (ranked)

### M1. Training-free 2D geometric twist counter (NS-AQA style), extended
- **What:** from existing RTMPose-x COCO-17 keypoints, compute hip vector h = r_hip − l_hip and shoulder vector
  s = r_sh − l_sh, normalised by torso length (mid-shoulder to mid-hip). Only use flight frames with an open
  body. Count half-twists as hysteresis lobes of |h| and |s| (NS-AQA thresholds: outer = torso/3.2,
  inner = torso/3.4, reject > torso/1.5) plus unwrapped in-plane angle turns when |h| stays large (front or
  back views). Fuse h and s, and flag disagreement as low-confidence.
- **Expected accuracy:** 93% in the diving analog (B). **Handheld parkour: unknown.** My prior is about
  65–85% exact half-twist, lower on ≥1.5 twists and corks. That is a guess, not a measurement.
- **Needs:** 1 view (side or 3/4 preferred), ≥30 fps, flight segmentation, keypoints on inverted frames.
- **Cost:** $0 (CPU on existing keypoints). No labels, which fits the no-manual-labels constraint.
- **Main risk:** **L/R keypoint swaps** on inverted bodies. NS-AQA's origin-crossing rule turns each swap into
  a spurious half-twist. Also dropouts in blur, handheld roll/zoom (length-based counting is invariant to them,
  angle-based counting is not), and cork decomposition.

### M2. Tool-augmented VLM judge (Claude/Gemini) on flight-cropped dense frames + keypoint trace
- **What:** give the model 20–30 frames sampled at ≥30 fps **inside the flight only**, plus a plotted |h(t)|,
  |s(t)| trace from M1. Ask for half-twist count with reasoning. Use it as a tie-breaker when M1 is
  low-confidence.
- **Expected accuracy:** internal 10/11 twist on clean clips (A, n=11). General VLM counting is weak
  (PushupBench: Claude Opus 4.5 4.9% exact, Gemini 3 Flash 42%). Twist counts are small, and 5 fps vs 1 fps
  alone moved Gemini from 17% to 42%.
- **Needs:** 1 view. $0 on the Max subscription, roughly <$1 per 12 clips via API.
- **Main risk:** hallucinated counts and anchoring on trick names or captions (PushupBench found models reading
  on-screen counters). Poor frontier-model calibration on counting.

### M3. Two-phone triangulated twist angle ("Fujitsu-lite")
- **What:** 2 iPhones on tripods (audio sync already solved), calibrated from a checkerboard or from the
  athlete's own 2D keypoints. Triangulate hips and shoulders, then integrate the 3D lateral-axis rotation about
  the body long axis.
- **Expected accuracy:** JSS reaches about 90% element agreement with 4–8 fixed cameras (C). Markerless
  multi-camera tracks snowboard rotation amount with "high validity" (B).
- **Needs:** 2 fixed phones at ≥30 fps and a calibration step.
- **Cost:** $0 compute, plus setup effort.
- **Main risk:** with few views, joints on inverted poses often fail to triangulate (3 cameras = 51% valid
  joints in trampoline, B). Handheld phones break extrinsics. Only worth it if M1 fails because of view
  ambiguity rather than keypoint failure.

Not recommended: supervised skeleton/RGB classifiers (Diving48/FineGym/VIFSS style). They need 16–20k clean
labels, conflict with the no-manual-labels constraint, and PkVision's own 2D-skeleton → attribute transformer
already stalled at macro-F1 0.39. Possible later use: train on **M1 pseudo-labels**. Also not recommended:
monocular HMR/3D-lift integration, which is refuted by the analogs above.

## 5. Proposed probe (≤12 clips, ≤$5): design only, not run

**Question:** does an NS-AQA-style 2D counter recover half-twist counts on PkVision clips, including ≥1.5?

1. **Clips (12):** use the corrected references from the July judge calibration, not the raw gold manifest
   (≥5/12 known label errors). Stratify: 3 × 0 twist (back, front, side flip), 3 × full, 3 × double full or
   more, 2 × odd ≥1.5 (e.g., 1.5 or 2.5), 1 × cork. Record per-clip fps and view (side / 3/4 / front).
2. **Pre-register before looking at outputs:** flight segmentation rule (ankle-height or velocity threshold);
   torso normalisation; NS-AQA thresholds (outer torso/3.2, inner torso/3.4, valid torso/1.5); open-body gate;
   fusion rule for h/s; primary metric = exact half-twist accuracy; secondary = off-by-one rate and accuracy on
   the ≥1.5 subset.
3. **Run:** numpy on existing `data/keypoints/parkourtheory_pose/`. Save per-frame |h|/torso, |s|/torso,
   in-plane angle, keypoint confidence and detected L/R-swap events for every clip.
4. **Arm B (optional, $0 on Max):** the same 12 flights to Claude as a flight-cropped ≥30 fps grid, with and
   without the |h|,|s| trace, to test whether the trace helps.
5. **Decision (pre-registered):** ≥10/12 exact and ≥2/3 on the odd ≥1.5 + cork subset → scale M1 to all
   1,618 clips as automatic twist weak labels. 6–9/12 → inspect traces to separate keypoint failure (swaps,
   dropouts) from view ambiguity, then choose between swap-repair and M3. ≤5/12 → the 2D signal is not usable
   on this footage; M3 or M2 only.
6. **Cost:** $0 compute (CPU). API fallback for arm B <$1. Wall time: a few hours.

## 6. Gaps (searched, not found)

- **Per-class twist accuracy for ≥1.5 twists** in any paper (NS-AQA, MTL-AQA, TQN, Diving48 SOTA give
  aggregates or confusion figures only).
- **MTL-AQA twist class distribution**, so the majority-class baseline behind 93% is unknown. This matters
  because 0-twist dives likely dominate.
- **Fujitsu JSS twist-count accuracy** per element: not published. Only "about 90%" and "95%" overall claims.
- **Trampoline twisting-somersault recognition from video**: none since 2017, and the 2017 work excluded them.
- **Freestyle ski/snowboard spin counting from video with numbers**: none validated. The Google tool is
  unvalidated; the 92.7% aerials claim could not be verified.
- **Any twist or rotation counting measured on phone or handheld footage**: none (OOFSkate has no numbers).
- **VLM zero-shot accuracy on the Diving48 or FineGym twist attribute**: none found.
- **Parkour or tricking flip/twist recognition from video**: none (zero A-grade sources).
- **ISU camera system accuracy**: not published.
- **Controlled study of fps vs twist-count accuracy**: none. Only the PushupBench repetition ablation and the
  FineGym frame-sampling study.
- **2025–2026 arXiv work on twist counting**: arXiv search for somersault/twisting + video since 2025-01-01
  returned nothing relevant.
- The snowboard IMU-vs-video paper (S2) full numbers: paywalled (403).

## 7. Sources

Diving
- Okamoto, Parmar. Hierarchical NeuroSymbolic Approach for Comprehensive and Explainable Action Quality Assessment. CVPR 2024 Workshops (CVsports). arXiv 2403.13798. Code: https://github.com/laurenok24/NSAQA
- Parmar, Morris. What and How Well You Performed? A Multitask Learning Approach to AQA. CVPR 2019. arXiv 1904.04346
- Hassan et al. FineX: Fine-Grained Action Recognition with Cross-Attentive Latent Sparse Experts. arXiv 2608.13458 (Aug 2026)
- Zhang, Gupta, Zisserman. Temporal Query Networks for Fine-grained Video Understanding. CVPR 2021. arXiv 2104.09496
- Çelik. MamBOA: State-Space Architecture for Video Recognition. arXiv 2606.15275 (Jun 2026)
- Li et al. Context-Aware Network ... Multi-scale Spatio-temporal Attention. arXiv 2512.18750 (Dec 2025)
- Velesaca et al. Can Vision-Language Models Judge Olympic Diving? arXiv 2609.19354 (Sep 2026)
- Kanojia et al. Attentive Spatio-Temporal Representation Learning for Diving Classification. CVPRW 2019. arXiv 1905.00050 (no per-attribute numbers)
- Walker, Sinclair, Cobley. A kinematic analysis of the backward 2.5 somersaults with 1.5 twists dive (5253B). ISBS 2015. https://ojs.ub.uni-konstanz.de/cpa/article/download/6503/5870
- Chinese AI making smart moves at Paris Olympics (Baidu diving AI). China Daily, 2024-08-07. https://www.chinadaily.com.cn/a/202408/07/WS66b2afe5a3104e74fddb8c2e.html

Gymnastics
- Shao et al. FineGym. CVPR 2020. arXiv 2004.06704
- Yang et al. (Fujitsu Research). Enhancing Multi-Camera Gymnast Tracking Through Domain Knowledge Integration. arXiv 2511.16532 (Nov 2025)
- Ishikawa et al. Improving the approval of jumping techniques on the balance beam ... Frontiers in Sports and Active Living 2026. https://www.frontiersin.org/journals/sports-and-active-living/articles/10.3389/fspor.2026.1798778/full
- How AI is changing gymnastics judging. MIT Technology Review, 2024-01-16. https://www.technologyreview.com/2024/01/16/1086498/ai-gymnastics-judging-jss-world-championships-antwerp-paris-olympics/
- Fujitsu press release, JSS for all 10 apparatus, 2023-10-05. https://info.archives.global.fujitsu/global/about/resources/news/press-releases/2023/1005-02.html
- Fujitsu press release, first official use at the 2019 Worlds. https://www.fujitsu.com/global/about/resources/news/press-releases/2019/1002-01.html
- Synced Review, Meet Fujitsu's AI Gymnastics Judges (lidar specs), 2019. https://syncedreview.com/2019/01/26/meet-fujitsus-ai-gymnastics-judges/
- Fujitsu Human Motion Analytics page (95% claim, smartphone support). https://mkt-europe.global.fujitsu.com/human-motion-analytics-en_reg

Trampoline
- Connolly, Silvestre, Bleakley. Automated Identification of Trampoline Skills Using Computer Vision Extracted Pose Estimation. arXiv 1709.03399 (2017)
- Woltmann et al. Sensor-based jump detection and classification with ML in trampoline gymnastics. German J. Exercise & Sport Research, 2022. https://fis.tu-dresden.de/portal/en/publications/sensorbased-jump-detection-and-classification-with-machine-learning-in-trampoline-gymnastics(59b13dd6-2475-40ce-84f4-3f784a9accdb).html
- Helten et al. Classification of trampoline jumps using inertial sensors. Sports Engineering 14(2), 2011 (as cited in 1709.03399)
- Drolet-Roy et al. Human Pose Estimation in Trampoline Gymnastics: Improving Performance Using a New Synthetic Dataset. arXiv 2604.01322 (Apr 2026)

Figure skating
- Tanaka, Suzuki, Fujii. VIFSS. arXiv 2508.10281 (Aug 2025)
- Tanaka, Suzuki, Fujii. 3D Pose-Based Temporal Action Segmentation for Figure Skating (FS-Jump3D). arXiv 2408.16638 (2024)
- Sun et al. 1st Place Solution to the 1st SkatingVerse Challenge. arXiv 2404.14032. Zhao et al. The SkatingVerse Workshop & Challenge. arXiv 2405.17188
- Gan et al. SkatingVerse. IET Computer Vision 2024. https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/cvi2.12287
- Chen et al. YourSkatingCoach. arXiv 2410.20427
- OOFSkate: MIT MechE "3 Questions: Using AI to help Olympic skaters land a quint", 2026-02-09. https://meche.mit.edu/news-media/3-questions-using-ai-help-olympic-skaters-land-quint ; The AI Insider, 2026-02-10. https://theaiinsider.tech/2026/02/10/mit-researchers-are-using-ai-to-help-olympic-figure-skaters-and-viewers/
- ISU weighs AI's role in judging. CGTN, 2026-02-11. https://news.cgtn.com/news/2026-02-11/International-Skating-Union-weighs-AI-s-role-in-judging--1KFKl1sSgLK/p.html ; Reuters via Yahoo Sports, "Less human, more AI" (six-camera claim from search snippet; URL 404 at fetch time). https://ca.sports.yahoo.com/news/less-human-more-ai-figure-095515853.html

Ski / snowboard / skate / parkour
- Merz et al. Is a cork a legal shortcut? Sports Biomechanics 24(7), 2024. doi 10.1080/14763141.2024.2399255
- Comparative analysis of IMUs and markerless video motion capture for rotational parameters in snowboard freestyle. 2025. https://www.sciencedirect.com/science/article/pii/S2665917425000662
- Google Cloud blog. Using Google Cloud AI to measure the physics of U.S. freestyle snowboarding and skiing, 2026-02-20. https://cloud.google.com/blog/products/ai-machine-learning/measure-physics-of-freestyle-snowboarding-and-skiing/
- IMU Airtime Detection in Snowboard Halfpipe (U-Net). Sensors 2024. https://pmc.ncbi.nlm.nih.gov/articles/PMC11548732
- The classification of skateboarding tricks via transfer learning pipelines. PMC8384043. SkateboardAI, arXiv 2311.11467
- Li et al. Estimating 3D Motion and Forces of Human-Object Interactions from Internet Videos (parkour actions without flips). arXiv 2111.01591

General / VLM / 3D
- PushupBench: Your VLM is not good at counting pushups. arXiv 2604.23407 (Apr 2026)
- Salehi et al. ActionAtlas. NeurIPS 2024 D&B. arXiv 2410.05774
- Yeung et al. AthletePose3D. arXiv 2503.07499 (2025)
