# 04: Consumer multi-view capture (2+ iPhones) for fast acrobatic motion

Survey date: 2026-09-28. Desk research only: no project data touched, no code run on project data, $0 spent.
Track: smartphone multi-view markerless capture tools, 1-view vs 2-view rotation estimation, iPhone capture
specifics (fps, shutter, rolling shutter, framing), 2-phone practicality for one trick vs a full run, and the
monocular-geometry question behind "twist ≥1.5 is unrecoverable from one camera".

Grading: **A** = measured in-domain (parkour/tricking, including this project's own July probe).
**B** = measured on an analog task or sport. **C** = claim, vendor or developer statement, or my own
derivation (derivations are marked "C, derived" and show their arithmetic).

Source numbers in square brackets refer to the Sources list at the end.

---

## 0. Bottom line

1. **"Twist ≥1.5 is geometrically unrecoverable from one camera" is OVERTURNED as a geometric claim.**
   The geometry does not forbid it. Silhouettes are ambiguous, but labelled keypoints plus appearance cues
   (face, chest vs back, occlusion order) carry the twist angle. Measured counterexamples:
   - Single-view diving broadcast: Diving48 has 17 twisting classes with 0.5 to 3.5 twists. SOTA reaches
     92.9% top-1 over all 48 classes (B) [26], and a pose-only STGCN++ reaches about 85.9% (B) [26].
   - Figure-skating broadcast at 25 fps: 1 to 4 revolutions, element-level F1@50 92.6% (B) [21].
   - This project's own single-view probe: twist 10/11 and 8/10, including a triple full (A, tiny n,
     curated clips) [34].

   What survives is a weaker, practical claim. Monocular twist reading depends on viewpoint and pixel
   count, and its errors cluster on adjacent counts (±0.5) (B) [24].
2. **"A second synced iPhone makes twist recoverable" is WEAKENED: not demonstrated, and the automated
   evidence points the other way.**
   - The geometry is sufficient. Two views with *hand-digitised* landmarks measured a 1½-twist somersault
     to 2.1° mean orientation error in 1990 (B) [18].
   - Sync is solved (B) [9][1][12].
   - But stock 2-view pipelines fail on inverted bodies: OpenCap (≥2 iPhones) has 40.1° RMSE on
     handstands and cartwheels vs 11.6° on normal tasks, and the pelvis and shoulders are worst, which are
     exactly the twist segments (B) [4].
   - On real trampoline footage, COCO-trained ViTPose gets AP 55.8, and only 31% of joints triangulate
     with 3 cameras (B) [13].
   - With 2 cameras there is no redundancy to reject a bad 2D detection.
   - No study measured twist-count accuracy from 2 phones vs 1. The second phone is *necessary* for
     triangulated 3D, but *not sufficient* without an acrobatics-robust 2D detector.
3. **Frame rate is not the twist bottleneck; exposure time, pixels on the athlete and viewpoint are.**
   - At 60 fps, a 3 to 4 rev/s twist advances only 18 to 24° per frame (C, derived).
   - Skating quads (above 6 rev/s) are counted from 25 fps broadcast (B) [21].
   - What 240 fps buys is a forced short exposure (≤1/240 s), at the cost of 1080p and no manual control
     in the native app. A 60 fps capture with manual 1/1000 s shutter gives less blur at 4K. Blackmagic
     Camera and Apple's Final Cut Camera both expose manual shutter for free (C) [28][29].
4. **For a full 20 to 45 s run on a ≥40 m field, static triangulation from 2 phones is impractical.**
   - A static 24 mm iPhone covering 40 m puts the athlete at about 84 px (1080p) to 168 px (4K) tall
     (C, derived).
   - Panned cameras break static calibration.
   - The realistic product path is 1 to 2 *panned* phones read per view by a monocular learned or VLM
     reader, fused late at trick level, with abstention. Triangulation stays a V1 research option.

---

## 1. Smartphone and consumer multi-view capture tools (as of Sep 2026)

| Tool | Views | Calibration | Sync | Accuracy evidence | Fast/aerial/inverted | Grade |
|---|---|---|---|---|---|---|
| **OpenCap** (Stanford) [1] | ≥2 iOS devices (2 = default) | Intrinsics pre-computed per iPhone model. Extrinsics from **one image** of a checkerboard (an A4 print works). | Cross-correlation of keypoint velocities | MAE 4.5° (1.7 to 10.3°) over 18 DOF on walk/squat/sit-to-stand/drop-jump, recorded at 720×1280 @ 60 Hz. 3 cameras did not improve; 5 cameras improved "mildly". Cameras at ±45° for jumps. | See next rows | B |
| OpenCap on gymnastics [4] | OpenCap default | checkerboard | as above | RMSE **40.14° ± 14.48°** on handstand/cartwheel/handstand-walk/hop vs 11.57° ± 6.66° on non-gymnastics tasks. Greatest discrepancies at **pelvis and shoulder**. | Inverted bodies: fails | B |
| OpenCap on dynamic tasks [5] | OpenCap default | checkerboard | as above | N=41. Sagittal RMSE 7.0 to 13.4°. Out-of-plane NRMSE 29 to 136%, r −0.09 to 0.80. Worse when the subject "translate[s] rapidly across the cameras' depth of field"; authors cite sync and distal-tracking issues. | Running, cutting: degraded | B |
| OpenCap scoping review [6] | mostly 2 | checkerboard | as above | As summarised: sport tasks lose trials (cricket bowling 56% usable, volleyball ~30% excluded, ~10% of trials discarded for sync/processing failures in one study). *I did not re-verify these in the primary papers.* | degraded | B/C |
| **OpenCap Monocular** [7] (Mar 2026) | 1 iPhone | none | n/a | Built on WHAM. 4.8° MAE on walking/squat/sit-to-stand. | Not tested on fast/aerial motion; WHAM priors are AMASS-like | B (its tasks only) |
| **Pose2Sim** [8] | ≥2 (`min_cameras_for_triangulation = 2`) | Checkerboard intrinsics (target <0.5 px). Extrinsics by clicking **scene points** or a static board (target <1 cm, acceptable to 2.5 cm). Keypoint ("human-as-calibration") extrinsics: *"coming soon"*. | Correlation of vertical keypoint speed. Needs a clear vertical movement and ≥5 s. | README: 2 to 6° joint-angle error (its own validation claim). Recommends front + 45° at hip height with 2 cameras [10]. "Top views right above the subjects do not yield good results". | README: **"does not currently work well for acrobatic movements where the person is upside down"** (suggests BlazePose). `handle_LR_swap`: *"Not implemented yet"*. | C (developer statement) |
| **FreeMoCap** [30] | Recommends 3–4 "good", 5–6 "better", 7+ "best" for complex movements | ChArUco board waved in view (Anipose) | `skelly_synchronize`: audio cross-correlation, or a brightness flash | No quantitative accuracy in the docs. "60 FPS for fast movements". | none reported | C |
| **EasyMocap** (ZJU) [31] | multi-view (demo datasets use 21–23 synced cameras); also monocular and mirror modes | Requires calibrated input | Requires synchronised input | none on acrobatics | none reported | C |
| **Argus** [9] | 2+ consumer cameras (GoPro) | Wand + checkerboard, sparse bundle adjustment; RMS reprojection ~0.9–1.0 px | **Audio cross-correlation, sub-frame** | "Soft" sync raised reprojection error by **mean 0.14 px (9%), max 0.54 px (35%)**. This figure comes from simulating ±0.5-frame slips at 120 fps against hardware-synced reference cameras. | Bird flight (fast) | B |
| **Kineo** [11] (Oct 2025) | sparse consumer cameras | **Calibration-free**: intrinsics, distortion and extrinsics solved from 2D keypoints ("human-as-calibration") | Audio cross-correlation. Derives the camera-distance limit for sub-frame audio sync (e.g. a 20 m path difference at 60 Hz ≈ 4 frames). | 83 to 91% lower W-MPJPE than prior calibration-free methods on EgoHumans/H36M | Not tested on acrobatics | B (its tasks) |
| **CalTennis protocol** [12] (Caltech, Jun 2026) | 2–6 **iPhone 14+**, main camera, 1080p60, on ~$40 MagSafe tripods | Extrinsics from **known scene geometry** (court-line intersections) plus intrinsics from iPhone metadata | **Pose-based**: interpolate per-view poses, then grid-search a ±1000 ms offset that minimises cross-view disagreement. iPhones stamp wall-clock time only to the nearest second. | 11 M frames, 40 players; players ran it themselves with minimal instruction | Tennis: fast but upright | B |
| **Fujitsu JSS tracking** [14] (reference, not consumer) | **4 calibrated RGB cameras**, 1080×1920 | fixed install | hardware | With only **2 opposing views**, lines of sight are near-parallel and triangulation amplifies 2D bias. Fixed with a domain prior (gymnast stays in a vertical plane). Fine-tuned HRNet 2D. | Gymnastics (deployed at FIG events) | B |
| Trampoline HPE [13] (Apr 2026) | 8 synced cameras at 120 fps (3–8 tested) | ChArUco, RMSE 1.1 px | hardware | COCO ViTPose-S AP **55.8** (ViTPose-B 65.6) on real multi-view trampoline, vs 73.8 on COCO val. Fine-tuned on LSP + synthetic trampoline: 73.1 / 75.6. With 3 cameras at a 15 px threshold, **31% of joints triangulate (baseline) vs 51% (fine-tuned)**. Fine-tuned 3-camera MPJPE of 54.7 mm beats baseline 8-camera 56.3 mm. Their Pose2Sim triangulation used ≥3 cameras. | Twisting somersaults: stock 2D fails, fine-tuning helps | B (closest analog) |

**Number of views, synthesis:**
- Upright, slow tasks: 2 phones suffice (OpenCap) [1].
- Acrobatics: every measured system uses 3–12 views. JSS uses 4 [14], the trampoline study 3–8 [13],
  FS-Jump3D 12 hardware-synced cameras [21].
- On acrobatics, even 3 views need an acrobatics-fine-tuned 2D model to triangulate most joints [13].
- With 2 views, a single wrong 2D detection cannot be voted out. Pose2Sim drops cameras only while more than
  `min_cameras` remain [8].

**Two-camera geometry (B, slow tasks) [10][15][1]:**
- ~45° separation was best with MediaPipe at 15 fps: MAE 9.3° static, 12.9° dynamic.
- ~135° separation was worst: 25.3° / 30.6°.
- Front configurations (~69° separation) beat same-quadrant pairs (~28°). Hip rotation RMSE was 5.4° vs
  13.0°, and transverse-plane CMC fell below 0.6 for narrow pairs.
- Diagonal pairs (~161°) did not help consistently.
- Near-opposing pairs are ill-conditioned for triangulation [14].

**Calibration methods available:**
- Checkerboard or ChArUco [1][8][30]. OpenCap needs one image; an A4 print is fine at a few metres. It is
  too small to detect at 20+ m (C, derived).
- Wand [9].
- Known scene points: tape-measured obstacle corners or floor marks [8][12].
- Human keypoints / "human-as-calibration" [11]; Pose2Sim's version is still "coming soon" [8].
- Research-grade SfM or feed-forward calibration for *moving* cameras [35][36][37]. None of these is
  validated on fast inverted motion.

---

## 2. Sync error at 60 / 120 / 240 fps

**Measured.** Audio cross-correlation gives a sub-frame offset estimate, because audio runs at 44.1–48 kHz.
The residual cost at 120 fps was +0.14 px mean reprojection error (B) [9]. OpenCap and Pose2Sim instead
sync from keypoint motion [1][8]. CalTennis syncs by minimising cross-view pose disagreement after
interpolation [12].

**Derived impact of an uncorrected residual of ±0.5 frame (integer-frame alignment) (C, derived):**

| fps | ½ frame | Orientation mismatch at 3 rev/s (1080°/s) | at 4 rev/s | Position mismatch at 9 m/s (feet in a somersault) |
|---|---|---|---|---|
| 60 | 8.3 ms | 9.0° | 12.0° | 7.5 cm |
| 120 | 4.2 ms | 4.5° | 6.0° | 3.8 cm |
| 240 | 2.1 ms | 2.3° | 3.0° | 1.9 cm |

Counting half-twists tolerates tens of degrees, so even integer-frame sync at 60 fps is fine for counting.
Sub-frame interpolation matters only for precise angles.

**Traps that are larger than fps:**
- **Sound travel**
  - Sound travel adds 2.9 ms per metre of path difference (C, derived; Kineo gives the same effect [11]).
  - A clap 10 m closer to one phone shifts sync by 29 ms: 1.75 frames at 60 fps, 7 frames at 240 fps.
  - Mitigations:
    - Clap equidistant from the phones.
    - Correct for the measured distances.
    - Refine with pose-based sync [1][8][12].
- **Frame timing**
  - The native Camera app records **variable frame rate** and can auto-drop to 24 fps in low light (C) [32].
  - Final Cut Camera advertises constant frame rate only for 24–60 fps (C) [29].
  - So sync on per-frame presentation timestamps, never on frame indices (C, derived).
- **Wall-clock timestamps**
  - iPhone wall-clock timestamps resolve only to 1 s, so they cannot be used for sync (B) [12].
- **Rolling shutter** (C, derived)
  - It adds a row-dependent time offset within each frame.
  - Main camera readout is under 3 ms per frame (B) [27]. Over an athlete spanning 40% of the frame height,
    that is about 1.2 ms, or about 1 cm / 1.7° at the rates above.
  - Negligible for counting at any fps; comparable to the sync residual only at 240 fps.

**Clock drift:** FreeMoCap advises sessions under 30 minutes to limit drift (C) [30]. A 20–45 s run is far
inside that.

---

## 3. Rotation estimation: 2 wide-baseline views vs 1 view

**Two views (geometry sufficient, automation unproven):**
- **Yeadon 1990** (J Biomech) [18].
  - Setup: two camera views, joint centres digitised **by hand**, cameras synced from the data.
  - Result: a forward somersault with 1½ twists, 17 orientation angles, **average error 2.1°**.
  - Grade B (classic; manual 2D).
- **Yeadon 1989** [19].
  - Setup: **two pan-and-tilt cameras**. Each frame's camera orientation comes from 2 digitised reference
    markers, and the cameras are synced from digitised displacement data (no hardware sync).
  - Result: 1° orientation error and 0.05 m centre-of-mass error in ski jumping.
  - Precedent that *panned* views can still be triangulated if each frame sees known reference points. B.
- **Snowboard freestyle, 2025** [20].
  - Setup: 8 elite riders, 88 tricks with 540–900° rotations. The multi-camera markerless video system was
    the criterion; the board IMU was compared against it.
  - Result: rotation amount CCC 0.998, SDD ±8.18°, bias 1.80° ± 16.02° LoA.
  - The criterion system's camera count is **not found**. It measures total board rotation, not body twist
    count. B.
- **OpenCap gymnastics** [4]: pelvis and shoulder errors dominate on inverted tasks (B).
  Implication: the automated 2-view pipeline fails exactly on the segments that define twist.
- **Trampoline** [13] (B):
  - Stock 2D is the bottleneck; fine-tuning on acrobatic data recovers more joints than adding cameras does.
  - The authors report qualitatively "better side disambiguation (left/right wrists)" after fine-tuning,
    i.e. left/right swaps are a known failure mode.
- **AthletePose3D** [3]: models trained on conventional data "perform poorly on athletic motions".
  Fine-tuning cut SOTA monocular MPJPE from 214 mm to 65 mm, so the domain gap is large but trainable
  (B).
- **Direct 1-phone vs 2-phone twist-count comparison: not found** (see Gaps).

**One view (measured counterexamples to "unrecoverable"):**
- **Diving48** [22][24][25][26]: single fixed broadcast view, 48 classes = takeoff × somersaults × twists ×
  position.
  - Twist values present: 0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5.
  - 17 of 48 classes twist; 12 of them have ≥1.5 twists. Tallied from the label map [23].
  - Top-1 accuracy: TQN 81.8% on v2 (2021) [24]; AIM ViT-L 90.6% [25]; FineX **92.9%**, with a pose-only
    STGCN++ baseline 7.0 points lower (≈85.9%) [26].
  - Because any twist error makes the class wrong, twist-attribute accuracy is ≥ class accuracy over the
    whole test set (C, derived). Per-class accuracy on the twisting subset is **not reported**.
  - TQN's confusion analysis: "The main errors come from counting the number of turns and twists,
    especially those with similar counts" [24].
  - Grade B.
- **FineGym** [17]: element classes include salto with 1, 1.5, 2, 2.5, 3 twists. Listed failure modes are
  "degree of rotation" and "counting the times of saltos". The skeleton model (ST-GCN) struggled because
  poses were missed in frames with intense motion (B).
- **VIFSS** [21]: figure-skating broadcast, 25 fps.
  - Element level (jump type × 1–4 rotations): F1@50 92.6%, frame accuracy 85.8%.
  - It uses view-invariant pose embeddings. Explicit 3D lifting did *worse*, and failed on a quad toe loop
    absent from 3D training data.
  - Caveat: rotation count correlates with jump type and athlete priors. B.
- **MEBOW** [33]: single-image 360° body-orientation (yaw) estimation, MAE 8.4°, 93.9% within 22.5° on
  COCO. Evidence that front/back/side yaw is learnable from appearance, but only on *upright* people. B.
- **This project, July 2026** [34]: Claude frame-grid judge on single-view parkourtheory clips.
  - Twist exact **10/11** and **8/10** against corrected references, including a triple full (twist 3).
  - On 99 clips, 2 independent reads agreed with each other on twist 93% of the time. That is agreement,
    not accuracy.
  - A, but tiny n and curated, well-framed clips.
- Sibling note 01 reports NS-AQA, a training-free 2D-keypoint rule counter, at 93.3% twist accuracy on
  0–3.5-twist dives. Not re-verified here.

---

## 4. iPhone capture specifics

**Resolution and fps (C, vendor):**
- iPhone 17 Pro: 4K up to 120 fps Dolby Vision; slo-mo 1080p up to 240 fps [38][32].
- iPhone 18 Pro was announced 2026-09-09 and keeps 4K120 [39]. I did not verify its slo-mo spec.
- Native slo-mo gives no manual shutter or ISO. Whether the 240 fps mode crops the field of view is
  **not found**.

**Apps with manual shutter (C, vendor):**
- **Blackmagic Camera (iOS, free)** [28].
  - Frame rates up to 120 fps. 240 fps is Android-only.
  - Shutter 1/24 to 1/8000 s, or angle 1.1–360°.
  - Timecode; "Sync Record Across Cameras" (record start only, not frame sync).
  - Genlock only via ProDock on iPhone 17 Pro.
- **Apple Final Cut Camera (free)** [29].
  - Manual shutter, ISO, white balance and focus.
  - "Frame rates up to 240 fps are available on certain devices and at certain frame sizes". Whether manual
    shutter also works *at* 240 fps is not explicitly documented.
  - Constant frame rate at 24–60 fps.
  - Live Multicam with up to 4 devices into Final Cut Pro for iPad.
  - v2.0 adds genlock and ProRes RAW on iPhone 17 Pro.

**Rolling shutter** (B, lab-measured on iPhone 17 Pro via the Blackmagic app, ProRes RAW) [27]:
- 24 mm main: **<3 ms**. Another reading: 2.3 ms at 2160p 24–60 fps.
- 13 mm ultra-wide: 5.6 ms (17:9), 6.0 ms (16:9), 7.4 ms (open gate).
- 100 mm tele: 5.6–7.5 ms.
- Readout at 120/240 fps: **not found**.

Derived skew over the athlete (C, derived):
- Main camera: ~1 cm and 1.7°.
- Ultra-wide: ~2 cm and 3.5°.
- Irrelevant for counting; minor for triangulation.

**Motion blur (C, derived).** Blur ≈ speed × exposure.

| Exposure | Feet in a somersault (9 m/s) | Shoulders in a twist plus some translation (4 m/s) |
|---|---|---|
| 1/60 s | 15 cm | 6.7 cm |
| 1/240 s | 3.8 cm | 1.7 cm |
| 1/1000 s | 0.9 cm | 0.4 cm |

- Indoors, auto exposure tends toward 1/fps, so a 60 fps default indoors gives heavy blur. Bright daylight
  lets auto exposure choose short shutters. That tendency is a general claim, not measured here (C).
- Setting a fast shutter manually trades blur for noise.
- FineGym and the trampoline study both name intense motion and blur as pose-failure causes (B) [17][13].

**Twist rate per frame (C, derived).**
- At 3–4 rev/s (1080–1440°/s):
  - 30 fps: 36–48° per frame.
  - 60 fps: 18–24° per frame.
  - 240 fps: 4.5–6° per frame.
- Counting half-twists aliases only near 180° per frame.
- Skating quads above 6 rev/s (a claim) [40] are counted from 25 fps broadcast, where they spin ~86° per
  frame (B) [21].

**Framing a moving athlete (C, derived).**
- Main-camera horizontal FOV is about 74° (24 mm equivalent).
- Covering 40 m needs a distance of about 27 m.
- That gives 96 px/m at 4K and 48 px/m at 1080p: a standing athlete is 168 or 84 px, a tucked one 86 or
  43 px.
- A single-trick zone about 8 m wide gives 480 px/m at 4K, so the athlete is about 840 px tall.
- The ultra-wide (about 108° horizontal) covers 40 m from about 14.5 m at the same px/m, with more
  distortion and slower readout.
- Pose models take 192–288 px-wide crops, so a 43–168 px athlete loses the face and back cues that
  monocular twist reading needs.

**Tripod vs handheld vs pan:**
- **Static tripod** is required for fixed-calibration triangulation (OpenCap, Pose2Sim) [1][8].
- **Pan/tilt on a tripod** keeps the camera centre fixed. It is triangulable only if each frame's rotation
  is recovered from reference points or scene features [19].
- **Handheld or gimbal** makes calibration a per-frame SLAM problem. Research methods exist [35][36][37],
  but none is validated on acrobatics.
- Monocular learned readers (VIFSS-style, VLM judges) tolerate pan and zoom: broadcast skating is panned [21].

**iPhone-specific capture hygiene (C, derived):**
- Turn stabilisation off, or keep it constant. EIS crops and warps frames, which breaks fixed intrinsics.
- Lock the lens at 1×. The native app auto-switches lenses.
- Lock AE/AF; focus breathing changes the focal length.
- Disable auto low-light FPS.
- Use CFR where possible.
- **OpenMMLab model servers shut down on 2026-06-06.** Pose2Sim shipped a fix. It is a practical risk for
  RTMPose model downloads (C) [8].

---

## 5. Practicality: one trick vs a whole run

**Single trick (PoC):**
- Placement:
  - Phone A side-on, perpendicular to the direction of travel, at hip height, 5–8 m from takeoff.
  - Phone B at 45–70° from A on the front-diagonal (from [1][10][15]).
  - Avoid B nearly opposite A (~180°) [14] and avoid same-quadrant pairs (<30°) [15].
- Calibration:
  - A4/A3 checkerboard held at the takeoff spot (OpenCap single image [1]), or 6+ tape-measured scene
    points (Pose2Sim "scene") [8].
  - About 5–15 min per placement (C, estimate). Intrinsics are done once per phone, or from metadata [12].
- Sync:
  - Clap at the midpoint between the phones, plus pose-based refinement [1][8][12].
- Failure modes (graded):
  - 2D failure and L/R swaps on inverted or twisting frames (B) [4][13].
  - With 2 cameras there is no outlier rejection (C) [8].
  - The athlete leaves the overlap volume or translates in depth (B) [5][15].
  - Blur under auto exposure indoors (C, derived).
  - VFR or low-light fps drop (C) [32].
  - Stabilisation or lens switching changes intrinsics (C).
  - Obstacles occlude one view (C).

**Whole run (20–45 s; field set by the TD; ≥40×10 m field of play per FIG Technical Regulations as
reported in search, not verified in full text) [41][42]:**
- 2 static phones give a 43–168 px athlete (§4). That is too small for reliable twist cues and too sparse
  for triangulation.
- Static zone coverage (2 phones per trick zone) needs about 4–6 phones and tripods for a 40 m field.
  - Calibrate each pair on scene points.
  - Sound-delay-aware sync across up to 40 m: up to 117 ms, 7 frames at 60 fps (C, derived).
  - Setup is about 45–90 min (C, estimate).
- Panned phones need per-frame extrinsics:
  - known reference markers visible in each frame [19], or
  - SfM/SLAM methods not validated on this use [35][36].
- Late fusion of per-view monocular reads needs only *segment-level* time alignment. Audio sync, even
  with 100 ms of sound delay, is sufficient. No calibration is needed (C, derived).

**Does one well-placed 240 fps camera beat two 60 fps cameras for twist? UNKNOWN (no measurement found).
The derived argument favours "60 fps + manual 1/1000 s shutter + 4K" over "240 fps native slo-mo":**
- Temporal sampling at 60 fps is already sufficient (§4).
- 240 fps costs resolution (1080p), manual control (native app) and light.
- Its only unique benefit, short exposure, is available at 60 fps through manual shutter apps [28][29].
- Whether 2×60 beats 1×60 depends entirely on whether the 2D stage survives inverted frames (§3).
- The filming test in §9 settles this for $0–5.

---

## 6. Monocular geometry: is longitudinal twist observable from one view?

**Theory (C, derived, with cited building blocks).**
- Let φ(t) be the body's yaw about its longitudinal axis relative to the camera.
- **Silhouette only.**
  - Under near-orthographic viewing (athlete far away, long lens), a silhouette is identical for φ and its
    depth-mirror. Rotation *direction* is bistable: this is the "spinning dancer" silhouette illusion,
    whose perceived direction is set by assumed viewing elevation [43].
  - Silhouette width minima (profile views) still occur twice per revolution, so the *number of profile
    passes* is observable.
  - Twist-and-return is indistinguishable from continuous twist.
- **Labelled 2D keypoints.**
  - The signed shoulder or hip width, x_R − x_L, is proportional to cos φ.
  - Every half-twist produces one sign change, so the **count of half-twists is observable, provided the
    left/right labels are correct**.
  - Under orthography the depth-mirror keeps the same L/R labels and flips front/back. So direction is still
    ambiguous from points alone: this is the classic monocular "forward/backward flipping" ambiguity
    [44].
  - Face visibility, chest vs back and nose direction in profile give sin φ, so appearance resolves the
    direction and the full φ.
- **Consequence.**
  - Monocular twist counting is *possible*.
  - It relies on exactly the cues a stock 2D detector gets wrong on inverted or back-facing bodies:
    left/right identity (B) [13] and the front/back decision.
  - Learned appearance models can supply those cues (MEBOW 360° yaw [33]; Diving48 and skating results in
    §3).
  - "Unrecoverable" is wrong; "detector-limited and view-limited" is right.

**Camera angles where twist becomes ambiguous or hard (C, derived, plus B where cited):**
1. **Longitudinal axis roughly perpendicular to the optical axis** (side-on to a vertical twisting body):
   the *best* case for counting. Direction needs face/back cues. This is the default side-on view of most
   parkour flips.
2. **Longitudinal axis roughly parallel to the optical axis** (camera overhead or underneath, or looking
   along a laid-out body):
   - Twist becomes in-plane rotation, which is geometrically easy.
   - But the body is maximally foreshortened and self-occluded, and detectors fail on these rare views (B)
     [16][8].
   - A somersault sweeps the body axis through this orientation twice per rotation when filmed from the
     front or back.
3. **Far or small athlete, heavy blur, tucked body:** the appearance cues (face, chest, hip ordering) vanish
   and the silhouette-only ambiguities return. This is the main risk for full-run static wide shots.
4. **Off-axis corks:** splitting the rotation into somersault vs twist uses Euler angles (Yeadon's
   somersault/tilt/twist). Those angles are singular near 90° tilt, so the twist/flip split is
   convention-sensitive even in perfect 3D (C, derived; angle set from [18]). A second view improves the
   3D, but does not remove this *labelling* ambiguity.

**Evidence summary:**
- Monocular counting of 1.5–3.5 twists works at a class-level ~86–93% in fixed-view diving (B) [26].
- It works for 1–4 revolutions in skating (B) [21] and in this project's small probe (A) [34].
- Errors concentrate on adjacent counts (B) [24].
- Where monocular 3D *lifting* is used, fast unseen rotations are under-counted (B) [21].

---

## 7. Verdicts

| Claim | Verdict | Key evidence |
|---|---|---|
| "Twist ≥1.5 is unrecoverable from one camera" | **OVERTURNED** (as a geometric claim). A practical residue survives: view- and resolution-limited, with ±0.5 confusions. | Theory §6 (C). Diving48 twisting classes up to 3.5 with SOTA 92.9% / pose-only ≈85.9% (B) [26]. TQN errors on neighbouring counts (B) [24]. VIFSS 1–4 revolutions, F1 92.6% at 25 fps (B) [21]. July probe 10/11 including a triple full (A, n=11) [34]. |
| "A second synced iPhone makes twist recoverable" | **WEAKENED** (not demonstrated; geometrically sufficient; the automated stock pipeline is measured to fail on inverted bodies) | Yeadon 2-view manual: 2.1° (B) [18]. Audio sync +0.14 px (B) [9]. OpenCap gymnastics 40° RMSE, pelvis/shoulder worst (B) [4]. Trampoline AP 55.8, 31% of joints triangulated with 3 cameras (B) [13]. Pose2Sim upside-down caveat, no L/R-swap handling (C) [8]. Opposing-view ill-conditioning (B) [14]. No 1-vs-2-phone twist study (gap). |
| (sub-claim) "Hardware genlock is needed" | **OVERTURNED** (already superseded) | Audio sub-frame [9], pose-based sync [1][12] (B) |
| (sub-claim) "240 fps single camera beats 2×60 fps" | **UNKNOWN** | No measurement. The derived argument (§5) favours 60 fps with a fast manual shutter. |

---

## 8. Recommended capture setups

**(i) V1 single-trick PoC.**
- Hardware: 2 iPhones (own + borrowed) and 2 phone tripods, e.g. ~$40 MagSafe tripods as in CalTennis
  [12], or improvised supports for $0.
- Placement: side-on plus front-diagonal at 45–70°, hip height, 5–8 m away, athlete at ≥40% of frame
  height.
- App: Final Cut Camera or Blackmagic Camera. 4K60 (or 1080p60) CFR, shutter 1/1000 s in daylight
  (≥1/500 indoors), stabilisation off, lens locked at 1×, AE/AF locked.
- Sync and calibration:
  - Clap at the midpoint, plus pose-based refinement.
  - A4 checkerboard or tape-measured scene points.
  - Pose2Sim (free) with RTMPose, and BlazePose as the upside-down fallback [8].
- Local RTX 2060 for processing.
- **Cost:** $0–80. **Effort:** ~15 min setup per spot; processing in hours.
- Treat the second phone as an *experiment* (§9), not a dependency.

**(ii) Full-run product.**
- **Primary (cheapest, most robust):**
  - 1 operator pans one iPhone on a tripod fluid head or cheap gimbal. The phone sits at mid-long-side,
    15–25 m out, keeping the athlete large. 4K60 with fast shutter.
  - Optionally a 2nd static wide phone covers the whole field for segmentation and context, or a 2nd
    panned phone from a course end gives a second opinion on big tricks.
  - Per-view monocular readers (learned view-invariant pose features or the VLM judge) are fused at
    trick level with abstention.
  - Audio gives segment-level alignment. No calibration.
  - **Cost:** $0–150 (fluid-head tripod ~$50–100, optional gimbal). **Effort:** 1–2 camera operators per
    run, ~5 min setup (C, estimates).
- **Only if the §9 test shows triangulation clearly beats per-view fusion:**
  - Static zone rig: 2 phones per trick zone at 45–70°, 4–6 phones total, scene-point calibration per
    pair, pose-based sync.
  - An acrobatics-fine-tuned 2D model is mandatory [13].
  - **Cost:** ≤$150–250 with borrowed phones. **Effort:** 45–90 min setup; fragile to occlusion and to the
    athlete leaving zones.

---

## 9. Cheap real-conditions filming test (≤1 session, ≤$5): one phone vs two for twist

**Pre-register before filming:**
- Primary metric: exact twist count per trick against the performer's spoken trick name. Naming a trick
  you just performed is allowed free ground truth.
- Report separately on (a) twist ≥1.5 and (b) corks/off-axis.
- **Decision rule:**
  - Adopt the 2-phone path only if a 2-view method (triangulation or late fusion) beats the *best single
    view* by ≥2 clips on the hard subset, **and** triangulation completeness (shoulders and hips valid at
    ≤15 px reprojection during flight) is ≥70%.
  - Otherwise adopt 1 phone plus abstention.
  - Adopt 240 fps only if the 240 fps block beats the 60 fps fast-shutter block on the same tricks.

**Setup ($0–5: A4 checkerboard print, optional phone clamp):**
- Outdoors in daylight, flat ground, one takeoff spot.
- P1 side-on at 6–8 m. P2 front-diagonal ~60° from P1, same distance, hip height.
- If a 3rd phone is available: P3 in line with the run direction (the predicted "hard" view). Otherwise
  move P2 there for one block.
- Block A: all phones at 60 fps CFR, 1/1000 s, stabilisation off, 1× locked (Final Cut Camera or
  Blackmagic).
- Block B: P1 switches to native 240 fps slo-mo.
- Hold the checkerboard at takeoff for 3 s at the start. Clap at the midpoint at the start and end (drift
  check). Tape-measure 6 scene points as backup.

**Trick list:** each trick ×3 reps, named aloud before each. Pick from the performer's repertoire across
twist levels 0 / 1 / 1.5 / 2 plus corks, e.g. back tuck, back full, double full, b-twist or 540, cork,
gainer full. Expect about 20–40 clips in ~2 h.

**Analysis ($0, local or Max quota):**
- (a) Per-view monocular:
  - the existing Claude frame-grid judge [34];
  - an NS-AQA-style signed shoulder/hip-width sign-change counter on RTMPose 2D (see note 01).
- (b) Pose2Sim triangulation, then unwrapped pelvis/shoulder yaw, then a half-twist count.
- (c) Late fusion of the per-view reads.
- Also log the sync residual (audio vs pose-based) and the per-view 2D failure rate on inverted frames.

**Why it settles the question:** it measures, on the target domain and hardware, the exact thing no paper
has measured:
- 1 view vs 2 views for twist ≥1.5;
- whether stock 2D survives inverted frames well enough to triangulate;
- whether 240 fps adds anything over a fast shutter.

It is exploratory, n≈20–40, with no manual labelling.

---

## 10. Gaps (not found)

- No study measures **twist-count accuracy from 2 smartphones vs 1** on any acrobatic sport.
- No per-class accuracy for **≥1.5-twist classes** on Diving48, FineGym or FineDiving; only aggregates and
  qualitative confusion statements.
- No inverted-body benchmark for 2026 2D models (RTMPose, ViTPose++, Sapiens2 [2], BlazePose). Only
  trampoline AP [13] and qualitative reports.
- iPhone **rolling-shutter readout at 120/240 fps**, and whether 240 fps slo-mo crops the field of view.
- Whether **Final Cut Camera allows manual shutter at 240 fps**; not explicitly documented.
- The camera count and setup of the markerless criterion system in the snowboard study [20].
- Measured **twist rates in parkour** (rev/s): not found; 2–4 rev/s is assumed.
- Validation of calibration-free or moving-camera multi-view methods [11][35][36][37] on fast inverted
  motion.
- iPhone 18 Pro slo-mo specifications (not verified).

---

## Sources

1. Uhlrich S.D. et al. "OpenCap: Human movement dynamics from smartphone videos." *PLOS Computational Biology*, Oct 2023. doi:10.1371/journal.pcbi.1011462. PDF: https://mobl.mech.utah.edu/wp-content/uploads/2023/12/opencap.pdf
2. "Sapiens2." arXiv:2604.21681, Apr 2026 (listed only as an existing 2026 2D/3D human foundation model; no inverted-body evaluation found).
3. Yeung C., Suzuki T., Tanaka R., Yin Z., Fujii K. "AthletePose3D: A Benchmark Dataset for 3D Human Pose Estimation and Kinematic Validation in Athletic Movements." arXiv:2503.07499, Mar 2025.
4. Buchanan L., He L., Zavatsky A. "Evaluation of a smartphone-based markerless motion capture system for assessing the kinematics of gymnastics." *Journal of Biomechanics*, 2026. doi:10.1016/j.jbiomech.2026.113477. https://pubmed.ncbi.nlm.nih.gov/42485837/
5. Liang H., Grant C.J., Bolgarskaya Y. "Validity of using a smartphone-based markerless motion capture system for quantitative analysis of human dynamic movements." *Sports Biomechanics*, 2026. doi:10.1080/14763141.2026.2689518
6. "Validations and applications of markerless motion capture using OpenCap: a scoping review." *Frontiers in Digital Health*, 2026. doi:10.3389/fdgth.2026.1882536. https://www.frontiersin.org/journals/digital-health/articles/10.3389/fdgth.2026.1882536/full
7. "OpenCap Monocular: 3D Human Kinematics and Musculoskeletal Dynamics from a Single Smartphone Video." arXiv:2603.24733, Mar 2026. https://utahmobl.github.io/OpenCap-monocular-project-page/
8. Pose2Sim README and Demo `Config.toml`, perfanalytics/pose2sim (accessed 2026-09-28). https://github.com/perfanalytics/pose2sim ; release note on OpenMMLab server shutdown: https://pypi.org/project/pose2sim/
9. Jackson B.E., Evangelista D.J., Ray D.D., Hedrick T.L. "3D for the people: multi-camera motion capture in the field with consumer-grade cameras and open source software." *Biology Open* 5:1334–1342, 2016. https://journals.biologists.com/bio/article/5/9/1334/1215/
10. Samani, Razeghi, Abdoli-Eramaki, Choobineh. "Optimizing camera placement in multi-view markerless motion capture: a validation study against gold-standard kinematics for static and dynamic tasks." *BMC Research Notes* 19:303, 2026. doi:10.1186/s13104-026-07886-4. https://pmc.ncbi.nlm.nih.gov/articles/PMC13378133/
11. Javerliat C., Raimbaud P., Lavoué G. "Kineo: Calibration-Free Metric Motion Capture From Sparse RGB Cameras." arXiv:2510.24464, Oct 2025.
12. Demler I., Xie X., Werner B., Szczuka A., Perona P. "CalTennis: Large Multi-View Tennis Video Dataset and Benchmark of Monocular-to-3D Pose Estimation." arXiv:2606.20542, Jun 2026.
13. Drolet-Roy L., Nogues V., Gaudet S., Charbonneau E., Begon M., Séoud L. "Human Pose Estimation in Trampoline Gymnastics: Improving Performance Using a New Synthetic Dataset." arXiv:2604.01322, Apr 2026.
14. Yang F., Odashima S., Masui S., Kusajima I., Yamao S., Jiang S. (Fujitsu Research). "Enhancing Multi-Camera Gymnast Tracking Through Domain Knowledge Integration." arXiv:2511.16532, Nov 2025.
15. Zhao X., Zhang Y., Graham R.B. "Influence of camera geometry on 3D joint angle estimation in markerless motion capture." *Frontiers in Sports and Active Living*, 2026. doi:10.3389/fspor.2026.1850018. https://pmc.ncbi.nlm.nih.gov/articles/PMC13402369/
16. Purkrabek M., Matas J. "Improving 2D Human Pose Estimation in Rare Camera Views with Synthetic Data (RePoGen)." arXiv:2307.06737, 2023/2024.
17. Shao D., Zhao Y., Dai B., Lin D. "FineGym: A Hierarchical Video Dataset for Fine-grained Action Understanding." CVPR 2020. arXiv:2004.06704.
18. Yeadon M.R. "The simulation of aerial movement—I. The determination of orientation angles from film data." *Journal of Biomechanics* 23(1):59–66, 1990. doi:10.1016/0021-9290(90)90369-E
19. Yeadon M.R. "A method for obtaining three-dimensional data on ski jumping using pan and tilt cameras." *International Journal of Sport Biomechanics* 5(2):238, 1989. doi:10.1123/IJSB.5.2.238
20. "Comparative analysis of inertial measurement units and markerless video motion capture systems for assessing rotational parameters in snowboard freestyle." ScienceDirect (Elsevier), Mar 2025. https://www.sciencedirect.com/science/article/pii/S2665917425000662
21. Tanaka R., Suzuki T., Fujii K. "VIFSS: View-Invariant and Figure Skating-Specific Pose Representation Learning for Temporal Action Segmentation." arXiv:2508.10281, Aug 2025 (includes FS-Jump3D, 12 synced Qualisys Miqus Video cameras).
22. Li Y., Li Y., Vasconcelos N. "RESOUND: Towards Action Recognition without Representation Bias" (Diving48). ECCV 2018. http://www.svcl.ucsd.edu/projects/resound/dataset.html
23. Diving48 label map (mmaction2 copy): https://github.com/farewellthree/STAN/blob/main/tools/data/diving48/label_map.txt (twist values tallied 2026-09-28).
24. Zhang C., Gupta A., Zisserman A. "Temporal Query Networks for Fine-grained Video Understanding." CVPR 2021. arXiv:2104.09496.
25. Yang T. et al. "AIM: Adapting Image Models for Efficient Video Action Recognition." ICLR 2023. arXiv:2302.03024.
26. Hassan I.U., Ahmad T., Bessis N., Behera A. "Fine-Grained Action Recognition with Cross-Attentive Latent Sparse Experts (FineX)." arXiv:2608.13458, Aug 2026 (Diving48 92.9% top-1, +7.0 over STGCN++; verified in PDF text). Also Frame2Freq, arXiv:2602.18977, Feb 2026 (92.2% per search snippet; not verified in PDF).
27. CineD. "Lab Test of the iPhone 17 Pro – Rolling Shutter, Dynamic Range Trials and Exposure Challenges." https://www.cined.com/lab-test-of-the-iphone-17-pro-rolling-shutter-dynamic-range-trials-and-exposure-challenges/
28. Blackmagic Design. "Blackmagic Camera – Tech Specs." https://www.blackmagicdesign.com/products/blackmagiccamera/techspecs
29. Apple. "Final Cut Camera at a glance" (Apple Support) https://support.apple.com/guide/final-cut-camera/dev154067693/ios ; "Apple announces Final Cut Camera 2.0," Apple Newsroom, Sep 2025 https://www.apple.com/newsroom/2025/09/apple-announces-final-cut-camera-2-0/ ; Final Cut Camera release notes https://support.apple.com/en-us/120849
30. FreeMoCap documentation: camera setup https://docs.freemocap.org/freemocap/docs/guides/camera-setup/ ; multi-camera calibration https://docs.freemocap.org/documentation/multi-camera-calibration.html ; skelly_synchronize https://github.com/freemocap/skelly_synchronize
31. EasyMocap, zju3dv. https://github.com/zju3dv/EasyMocap
32. Apple Support. "Change video recording settings on iPhone" (auto low-light FPS; slo-mo 1080p240). https://support.apple.com/guide/iphone/change-video-recording-settings-iphc1827d32f/ios ; Apple Community thread on variable frame rate: https://discussions.apple.com/thread/255018388
33. Wu C. et al. "MEBOW: Monocular Estimation of Body Orientation In the Wild." CVPR 2020. arXiv:2011.13688.
34. PkVision internal: July 2026 Claude-as-judge probe and gold-99 audit (`data/labeling/claude_probe/gold99_report.md`, project memory `project_claude_judge_probe`).
35. Sun S. et al. "Dense Dynamic Scene Reconstruction and Camera Pose Estimation from Multi-View Videos." arXiv:2603.12064, Mar 2026.
36. Liu J. et al. "TROPHIES: Temporal Reconstruction of Places, Humans, and Cameras from Multi-view Videos." arXiv:2606.02350, Jun 2026.
37. Qin X. et al. "Unconstrained Multi-view Human Pose Estimation with Algebraic Priors." arXiv:2604.24312, Apr 2026.
38. DIYPhotography. "iPhone 17 Pro Camera Specs: Complete Technical Breakdown." https://www.diyphotography.net/iphone-17-pro-camera-specs-complete-technical-breakdown/
39. Apple Newsroom. "Apple debuts iPhone 18 Pro and iPhone 18 Pro Max," Sep 2026. https://www.apple.com/newsroom/2026/09/apple-debuts-iphone-18-pro-and-iphone-18-pro-max/
40. ACSM. "The Science of Figure Skating: Jumps." https://acsm.org/the-science-of-figure-skating-jumps/ ; NC State. "The Physics of Figure Skating Jumps," Feb 2026. https://news.ncsu.edu/2026/02/the-physics-of-figure-skating-jumps/
41. FIG. Parkour Code of Points 2025–2028 (freestyle 20–45 s; field of play set by TD). https://www.gymnastics.sport/publicdir/rules/files/en_1.1%20-%20PK%20Code%20of%20Points%202025-2028.pdf
42. FIG. Technical Regulations 2025 (parkour field of play "minimum 40 x 10 m", per search snippet; not verified in full text). https://www.gymnastics.sport/publicdir/rules/files/en_1.1%20-%20Technical%20Regulations%202025.pdf
43. Troje N.F., McAdam M. "The Viewing-from-Above Bias and the Silhouette Illusion." *i-Perception* 1:143–148, 2010. doi:10.1068/i0408
44. Sminchisescu C., Triggs B. "Kinematic Jump Processes for Monocular 3D Human Tracking." CVPR 2003.
45. Yeadon M.R. "The physics of twisting somersaults" (2000). https://www.lboro.ac.uk/microsites/ssehs/biomechanics/papers/twistphysics00.pdf (background on twist mechanics; not used for numbers).
