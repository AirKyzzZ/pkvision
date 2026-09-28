# PkVision feasibility: global research result (2026-09-28)

This answers the question set in `docs/science-superpowers/questions/2026-09-28-pkvision-feasibility.md`. It rests on 7 parallel desk-research tracks (`01`–`07` in this folder, about 2,850 lines, every source cited). No project data was touched and $0 was spent. The four load-bearing papers were re-checked against arXiv: NS-AQA 2403.13798, FineX 2608.13458, VIFSS 2508.10281 and trampoline pose 2604.01322.

Evidence grades: A = measured on parkour, B = measured on an analog sport, C = claim, vendor statement or estimate.

## Bottom line

**CONDITIONAL GO. Keep working on it, but do not code the product yet.**

The blocker that killed the project in 2026 was the claim that twist can't be read from one camera. That claim no longer holds as a geometric argument. Analog sports read twist count from a single fixed camera's 2D keypoints:

| Analog | Result | Training |
|---|---|---|
| NS-AQA, diving | 93.3% twist count, 97.3% somersault count | none, rule-based |
| Diving48 | ~86% for a skeleton-only model | supervised |
| Figure skating (VIFSS) | 92.6 F1@50 for jump type × rotation count | supervised |

All of this is **B-grade**. Nothing has been measured on parkour, handheld iPhone footage or corks, except the July n=11 Claude probe. Measuring it takes under ~$15 of compute, one filming session and about 1–2 hours of your verification time.

## What changed in how we see the problem

1. **Count twists and flips with rules instead of learning them.** Every earlier approach tried to *learn* twist from labels that turned out to be ~35% wrong. NS-AQA shows twist and somersault can be *counted* with hand-written rules on 2D keypoints and **no training**:
   - half-twists are counted as "petals" traced by the hip vector;
   - somersaults come from the rotation of the pelvis-to-thorax vector.

   This removes the label wall for the crux attribute. Labels are then needed only to *evaluate* (a small trusted set) and for the learned attributes (context, entry).
2. **The bottleneck moved from geometry to 2D keypoint quality on upside-down bodies.** Left/right swaps, blur and front-on angles are now what break twist counting. These problems have measured fixes:
   - rotation augmentation to ±180° (bottom-view AP 45.9 → 59.1);
   - fine-tuning on real sports images plus ~2.5k synthetic renders (trampoline AP 55.8 → 73.1);
   - 4-way rotation at test time.

   There is also a new candidate model: SAM 3D Body scores 78.2 vs 39.8–46.1 aPCK on inverted bodies.
3. **3D lifting is the wrong direction.** In figure skating, lifted 3D did worse than raw 2D (76.6 vs 78.8) and failed on quads it hadn't seen in training. That matches the GVHMR 0.5× flip failure here.
4. **The end product needs less than we thought.** FIG D-score is the sum of the three highest scaled trick values, and runs last 20–45 s. The judge-assist mainly needs to catch high-value aerial tricks. PK basics, where the decoder's cue-degenerate ceiling was, almost never reach the top three at elite level.

## Verdicts per sub-question

| # | Sub-question | Verdict | Condition or reason |
|---|---|---|---|
| 1 | Twist | **CONDITIONAL** | Geometry is fine (B); in-domain accuracy UNKNOWN. Depends on 2D keypoints surviving inverted frames. **Corks: no evidence anywhere.** |
| 2 | Other attributes | Flip / direction: **likely GO**. Axis (cork) and context/entry: **UNKNOWN** | Flip: 97.3% somersault in diving (B), July 11/11 (A, n small). Direction: 11/11, but twist direction needs L/R and face points. Context/entry need a learned or VLM head. |
| 3 | Composition | **CONDITIONAL** | Compounding is normal. The best video models land within about ±1 point of accuracy(A)×accuracy(B), and joint modelling adds ≤4 points. The fix: joint decoding with feasibility masks, abstention and a coarse fallback name. Evaluation must hold out attribute combinations. |
| 4 | Labels without manual labelling | **CONDITIONAL** | Synthetic data alone: no. Rule counters, name parser and blinded multi-vote VLM labels, anchored by 50–100 owner-verified clips: yes. Proving ≥85% twist precision needs roughly 50–184 accepted verified clips. |
| 5 | Real conditions (iPhone) | **UNKNOWN** | No phone or handheld measurement exists in any sport. Capture guidance exists (below). |
| 6 | Prior art | **GO, the gap is real** | No parkour recognition or judging system exists. The closest is Owl AI (camera-only action sports, unvalidated). FIG says video "is NOT intended to replace" judges, so an assist fits and a replacement doesn't. |

## The six 2026 claims, re-tested

| Claim | Now | Key evidence |
|---|---|---|
| VLMs can't count flips/twists | Flips **WEAKENED**, near-overturned on clean clips. Twists **WEAKENED / UNKNOWN** | July 11/11 flip (95% CI 72–100%) and 10/11 twist (CI 59–100%). A 24-frame grid under-samples multi-twists, so the triple full was probably *recognised*, not counted. No external flip/twist VLM benchmark exists. |
| Monocular 3D HMR is dead for acrobatics | **WEAKENED** | Still no flip/twist measurement and no acrobatic training data. SAM 3D Body handles inverted single frames much better; it's worth one probe arm, not a pipeline. |
| Twist ≥1.5 unrecoverable from one camera | **OVERTURNED** as geometry. In-domain UNKNOWN | Diving, figure skating and Diving48 (see Bottom line). The May note had the geometry backwards: rotation about the camera axis is in-plane and fully visible. Errors concentrate on ±0.5 twist. |
| Stock 2D pose fails upside-down | **STILL TRUE**, but fixable | Trampoline ViTPose-S AP 55.8 vs 73.8 on COCO, with L/R confusion. Fixes measured (see point 2 above). |
| Skeleton SSL unvalidated on acrobatics | **STILL TRUE** | Only VIFSS-style view-invariant pretraining helped, and it needs 3D acrobatic motion we don't have. |
| Twist needs hardware genlock | **Superseded** (already known) | Audio sync works. New catch: sound travels ~2.9 ms per metre, so a clap 10 m closer to one phone costs 7 frames at 240 fps. Clap at the midpoint, then refine sync from pose. |

## Corrections to earlier project notes

- **Gemini 3.5 Pro has not been released.** The newest Pro model is Gemini 3.1 Pro Preview (paid); the newest Flash models are 3.6–3.8.
- **"GPT-5.6-video" does not exist.** OpenAI's API accepts images only.
- **The July note misread its source.** It said "Gemini 3.1 Pro fails flip/twist counting", but arXiv 2604.08294 is about execution-quality scoring, not counting.
- **FIG freestyle runs last 20–45 s, not 60–70 s.** Difficulty stops counting after 45 s.
- **FIG rules changed in 2026:**
  - at most 2 of the top 3 tricks may come from the same category;
  - an unintended off-axis tilt costs −0.3 to −0.5;
  - the table is reissued yearly, and a 2027 Table of Tricks is expected around October 2026.
- **The trampoline pose paper's code is already released.** Track 03 said it was "coming soon"; the arXiv abstract links it (`VisionICLab/trampoline_syn_data`).

## Candidate architectures, ranked

### 1. "Count, don't learn" (recommended for V1)

- **Pipeline:**
  1. **Keypoints:** best 2D keypoints available. RTMPose-x with 4-way test-time rotation; switch to Sapiens2 if it wins; fine-tune with rotation augmentation (≤$20) only if needed.
  2. **Rule counters for flip, twist and direction:** NS-AQA hip-vector petals plus a facing sign for twist, torso-angle unwrapping for flips. They abstain when keypoint confidence drops during the airborne phase.
  3. **Learned or VLM head for context, entry and axis.**
  4. **Joint decoding over the ontology:** feasibility masks, a top-k or conformal set, and a coarse fallback such as "backflip, twist ∈ {1, 1.5}".
  5. **A human confirms.**
- **Cost:** $0–20. Runs on the Mac or the 2060.
- **Evidence:** B (diving 93–97%).
- **Main risk:** left/right swaps on inverted bodies, and corks.
- **Killed if:** every pose arm scores ≤6/12 on twist against trusted truth.

### 2. "VLM reader + skeleton student" (second opinion, and the label source for learned heads)

- **Pipeline:**
  1. Blinded, cross-model VLM votes on dense frames from the airborne phase: Gemini 3.1 Pro on native video plus Claude on frame grids. Trick names and on-screen text are stripped.
  2. Weak labels are noise-corrected against 50–100 verified clips.
  3. The existing CueModel is trained on them as a cheap student.
- **Cost:** Gemini Flash free tier to ≤$50 for single votes; about $60–95 for a cross-model ensemble.
- **Evidence:** A, but n is small.
- **Main risk:** correlated guessing from trick priors.
- **Killed if:** the reversed-clip control fails, i.e. predicted direction doesn't flip when the clip is played backwards, or twist is below 9/12.

### 3. "Two-phone triangulation" (only if #1 fails because of viewpoint, not keypoints)

- **Evidence:** the geometry is proven (hand-digitised 2-view, 2.1° error). But automated pipelines fail on inverted bodies (OpenCap 40° RMSE on handstands, only 31–51% of joints triangulate), and a static rig doesn't scale to a 40 m course.
- **Cost:** $0–80.

### Not pursued

- Video HMR as the primary method (kept only as a probe arm).
- Metric learning.
- Text-to-motion synthesis: it can't control rotation counts, and HY-Motion's licence excludes the EU and forbids training on its outputs.
- Physics RL.
- Synthetic stick-figure data is kept for one $0 diagnostic: twist recoverability per camera angle, with injected L/R swaps.

## Validation before coding (the real-conditions check)

The exact decision rules get fixed in the analysis design and pre-registration step. This is the shape:

0. **Trusted yardstick, the prerequisite.**
   - Restore the clips from the Modal volume (free).
   - The cleanest ground truth is **your own filmed session, with each trick named as it's performed**.
   - The 12 July reference clips and the 26-clip dispute queue need your verify-only pass.
   - Watch out for anchoring bias: verifying against a VLM's proposal biases any later VLM evaluation.
1. **Rule-counter probe ($0–3).**
   - Pose arms: RTMPose-x, RTMPose-x with test-time rotation, Sapiens2, and SAM 3D Body.
   - Clips: 12 trusted clips plus your own footage, stratified to include ≥1.5 twists, corks, bad angles and the inverted clip.
2. **VLM blind probe (≤$5).**
   - Gemini 3.1 Pro on native video; Gemini 3.8 Flash free tier, grid vs native (this settles grid vs native); Claude Opus 5.5 on a dense grid of the airborne phase.
   - The reversed-clip control applies to all three.
3. **Filming test (one session, ≤$5).**
   - Two to three phones: side-on, 45–70° apart, plus one inline. 60 fps with a 1/1000 shutter, and a 240 fps block.
   - Tricks at twist 0 / 1 / 1.5 / 2 plus corks, 3 reps each, named aloud.
   - Settles one phone vs two, and 60 vs 240 fps.
4. **Label-source audit (≤$3.4).**
   - Twist lexicon from the name parser and VLM votes, on 60 clips, then your verify-only pass (30–45 min).
   - This gates any bulk labelling spend.

The ≤12-clip probes can only rank approaches or kill them (9/12 has a 95% CI of 0.47–0.91). Confirming the 85% twist bar needs a bigger trusted set.

## Proposed threshold revisions (need your approval)

From track 06:

- **Split the evaluation three ways:** held-out clips of known tricks, held-out *attribute combinations* (the only real open-vocabulary test), and your own footage.
  - A random clip split overstates generalisation: −15 points on Something-Else.
- **Make the compositional bar selective, i.e. allowed to abstain:** ≥85% exact at ≥50% coverage on known tricks. Target 40–50% on held-out combinations; the best published result is 44%.
- **Normalise top-3 against the oracle ceiling.** The decoder given perfect attributes (oracle) reaches 85.1% top-3, so an 80% bar would demand ~94% of the maximum.
- **Treat the ~13 own clips as a smoke test only.** 7/13 has a CI of [29%, 77%].
- **Twist 85% can't be measured until the evaluation labels are fixed.** With 35% label noise, a truly 90%-accurate model would score around 60%.

## End-product implications (FIG full-run judge-assist)

- **Outputs a judge would need:**
  1. A run clock with 20/40/45 s flags.
  2. A trick timeline with high recall on high-value aerial tricks.
  3. Table-of-Tricks identity and base value with top-k alternatives.
  4. Scaling cues with evidence (placement, form, entry and exit, tilt).
  5. Validity flags (failed, repeat, category cap).
  6. Proposed D with alternatives.
  7. Slow-motion evidence clips.
  8. Low-confidence tricks routed to the judge.
- **Capture:** one static phone on a 40 m course shows the athlete at only 84–168 px, too small for twist. Use 1–2 panned phones and read each view with a single-camera reader.
- **Segmentation:** plausible (skating reaches 92.6 F1@50 from a moving broadcast camera), but it's supervised there and unmeasured on phones or parkour. A $0 check is to run a detector for airborne phases on public full runs and count misses in one viewing pass.
- **Adoption path:** Fujitsu's system took 2017 → 2019 for FIG approval, and supervisors will review its output from October 2026. An assist for review, education and athlete self-check is the realistic entry point.

## Largest open risks

1. No in-domain measurement of any twist method on parkour or iPhone footage.
2. Corks: no source anywhere counts twists on off-axis rotations.
3. Left/right keypoint swaps on inverted bodies, which break the facing sign the twist counter relies on.
4. Trusted labels: every number since May was scored against partly wrong gold.
5. No 1-phone vs 2-phone twist study exists; only our own filming test can settle it.
