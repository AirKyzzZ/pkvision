# 05: Training and eval data for acrobatic attribute recognition without manual labels

Survey track 05 for the Sep-2026 feasibility question (`docs/science-superpowers/questions/2026-09-28-pkvision-feasibility.md`), sub-question 4 ("Labels without manual labelling").
Desk research only. No code was run on project data and no money was spent. The only files downloaded were public papers (to `pdftotext` two PDFs), public repo metadata via the GitHub API, and public class lists.
Date: 2026-09-28.

**Grading.**
- A = in-domain (parkour, or this project) measured.
- A-doc = a fact read directly in a primary document (licence text, price list, dataset page). It is not a measurement.
- B = analog domain measured (gymnastics, diving, figure skating, trampoline, generic action recognition or noisy-label benchmarks).
- C = claim, vendor statement, or my own unmeasured design reasoning.
- calc = arithmetic I did (confidence intervals, cost estimates). Assumptions are stated next to each one.

**Tools used.** alphaXiv discovery and full text, arXiv search and abstracts, WebSearch and WebFetch, the GitHub API (licences, repo trees), and `pdftotext` on the PoseConv3D and FineGym PDFs. "Not found" means I searched and found nothing. It does not mean the thing does not exist.

Cross-references: track 01 (twist analog sports), 02 (video VLMs), 06 (compositional recognition: NS-AQA rule-based somersault and twist counting on diving pose, TQN per-attribute heads on Diving48), 07 (prior-art judging). I don't repeat them here.

---

## TL;DR

1. **The label wall is a trust-and-volume problem for twist, and verify-only loops can fix the trust part.** Every analog that solves rotation or twist counting from 2D skeletons has thousands of *clean* labels. Even then, rotation count is the hardest attribute:
   - FineGym99 with PoseConv3D reaches 94.3% mean-class accuracy. Gym99 includes "salto backward stretched with 1.5 / 2 / 2.5 / 3 twist" classes. [B]
   - On MMFS figure skating, the temporal / rotation labels (TL22) reach 76.7–78.4%, against 92.2–95.4% for the spatial labels (SL24), with the same HRNet 2D skeletons. [B]
2. **Noise correction needs a small trusted set, and that trusted set is exactly what verify-only produces.** Gold Loss Correction (GLC) with 5% trusted data takes CIFAR-10 "flip" noise from 53.3 to 6.6 (area under the error curve). With 1% trusted data it takes 60%-wrong weak-classifier labels to 26.9% test error. [B] At PkVision scale, 1–5% of ~1,519 clips is 15–76 clips (calc).
3. **Tens of verified clips can steer training but cannot certify a twist threshold.** Certifying twist precision ≥ 85% (lower Wilson 95% bound) needs 48/50 or 92/100 correct. The July "10/11 twist" probe has a Wilson 95% CI of 0.62–0.98. [calc]
4. **Synthetic twist labels are perfect by construction, but no measured sim-to-real result exists for twist counting from 2D keypoints (not found).** Analog evidence says synthetic-only training lags real training by a lot, and gains appear when it is mixed with real labels:
   - SURREACT, synthetic only: 63.0% vs 86.9% for real training, same view. [B]
   - Trampoline pose: synthetic-only fine-tuning gives +4.3 AP, real sports images + synthetic gives +17.3 AP. [B]
   - VIFSS, random 2D projections of 3D poses as pretraining, then real fine-tuning: +14.5 frame accuracy at figure-skating "type + rotation count" level. [B]
5. **Text-to-motion and video generation are not usable as twist-count label sources in Sep 2026.**
   - Text-to-motion models cannot reliably produce a requested count ("four punches … rarely return four separable strokes"). [B]
   - HY-Motion 1.0, the model with an explicit "Gymnastics & Acrobatics" category, is licensed *excluding the EU*. Its licence also forbids using its outputs to improve any other AI model. [A-doc]
6. **VLM weak labels are the fastest route, but agreement is not accuracy.**
   - In-domain: relabelling 65 clips (4%) with Claude frame grids raised held-out flip F1 from 0.407 to 0.492. [A, from the brief]
   - LLM judges can be 92–97% self-consistent while correlating only 0.28–0.34 with humans. [B]
   - The best frontier VLM gets 42.1% exact on repetition counting. [B]
   - Bulk labelling of 1,519 clips on the Claude API with the Batch discount costs about $10–17 per vote with Sonnet 5 and about $19–35 per vote with Opus 5.5. [calc on A-doc prices]
7. **Name supervision is how every analog dataset got its labels, and it has in-domain evidence.**
   - Diving48 labels were transcribed from on-screen information boards. [A-doc]
   - FineGym labels came from official documents. [A-doc]
   - VideoNet (CVPR 2026) mined parkour training clips from YouTube titles and transcripts. Fine-tuning Molmo2-4B on them moved parkour multiple-choice accuracy 46.9 → 56.9 and binary accuracy 56.9 → 74.4. [A]
   - Extending the project's parser to explicit twist tokens is the cheapest twist label source. Its precision on twist is unmeasured.
8. **Best ≤ $5 pilot: a 60-clip twist label-source audit.** Three sources (parser twist lexicon, blinded 3-vote Sonnet 5 batch, 1-vote Opus 5.5 batch) go into a verify-only queue, with preregistered go/no-go thresholds. Estimated spend $1.9–3.4. Details in section 8.

---

## 1. Synthetic acrobatic data

### 1.1 Text-to-motion and motion generation (2025–2026)

| # | Finding | Grade |
|---|---|---|
| S1 | **HY-Motion 1.0** (Tencent Hunyuan, arXiv 2512.23464, Dec 2025) is a 1B-parameter DiT flow-matching model trained on >3,000 h of motion, with "Gymnastics & Acrobatics" among 200+ categories. It needs 24–26 GB VRAM at minimum, so it will not run on the RTX 2060 (6 GB). | A-doc (README) |
| S2 | The **HY-Motion licence** says: "THIS LICENSE AGREEMENT DOES NOT APPLY IN THE EUROPEAN UNION, UNITED KINGDOM AND SOUTH KOREA". Section 5(b) adds: "You must not use the Tencent HY-MOTION 1.0 Works or any Output … to improve any other AI model". So it is unusable for a France-based project that wants synthetic training data. | A-doc (License.txt read via GitHub API) |
| S3 | No 2025–2026 text-to-motion paper reports controllable **flip or twist counts** (not found). The closest evidence goes against it: "Text-to-motion models are competent at the action a prompt names but unreliable at when each stroke lands: four punches alternating left and right rarely return four separable strokes." That is the motivation of *Per-Stroke Temporal Control for Text-to-Motion* (arXiv 2607.15717, Jul 2026). | B |
| S4 | **Kimodo** (NVIDIA, arXiv 2603.15546, Mar 2026) is a kinematic diffusion model trained on 700 h of optical mocap. It is controllable through text *and* kinematic constraints: full-body keyframes and sparse joint positions or rotations. Root-orientation keyframes could in principle impose exact flip and twist totals, with the model filling in plausible body motion. That is untested for acrobatics, and whether its training data contains any flips is not found. Code is Apache-2.0. Weights are under the NVIDIA Open Model License or NVIDIA R&D licence depending on the checkpoint, and the SMPL-X checkpoint is not redistributed. | C (use), A-doc (licence per web summary, not verified in the licence file) |
| S5 | **Go to Zero / MotionMillion** (arXiv 2507.07095, ICCV 2025) is million-scale motion recovered from web video. Monocular recovery of acrobatics is the known failure mode (project memory: monocular HMR on flips is a dead end), so its flip labels and kinematics are suspect. Not evaluated for twists (not found). | C |
| S6 | **Text-to-skeleton cascades for flips and cartwheels** (Taghipour et al., arXiv 2603.08028, Mar 2026) built a Blender dataset of 2,000 synthetic acrobatic and stunt videos from **Mixamo** FBX motions, captioned from the Mixamo motion names ("backflip" becomes "a person performs a backflip"). The authors state that closed-source generators (Sora, Kling) are "unreliable" for such motions. No count control. Dataset release status: only samples on the project page (full release not found). Mixamo's terms for ML-training use were not verified. | B (dataset exists), C (Sora/Kling statement) |
| S7 | **Video generators.** OpenAI's Sora 2 announcement (30 Sep 2025) claims "Olympic gymnastics routines" and "triple axels". It makes no claim of count controllability. No independent benchmark of rotation-count fidelity in generated video was found. | C |
| S8 | **Generative skeleton augmentation for action recognition** does help in *low-data, common-action* settings: Dong et al., arXiv 2604.14933, Apr 2026, on HumanAct12 / NTU-VIBE. GenPrior (arXiv 2608.02236, Aug 2026) uses text-to-motion priors for zero-shot skeleton action recognition on NTU/PKU-MMD. Neither involves acrobatics or counts. | B |

### 1.2 Physics-based character control

| # | Finding | Grade |
|---|---|---|
| S9 | **DeepMimic** (Peng et al., SIGGRAPH 2018, arXiv 1804.02717, code MIT) learns backflip, frontflip, cartwheel and spin-kick by *imitating mocap references*. Physics RL makes a reference physically plausible. It does not invent twist counts without a reference or heavy reward engineering. | B |
| S10 | **PARC** (Xu, Shi, Yin, Peng, SIGGRAPH 2025, arXiv 2505.04002) runs a generate → physics-correct → retrain loop for *parkour terrain traversal* (wall climbs, gap jumps). No flips or twists. Relevant only to the "context" attribute (wall/vault), not to twist. Code repo `mshoe/PARC`: no licence asserted. | B |
| S11 | **MaskedMimic / ProtoMotions** (NVIDIA, arXiv 2409.14393, code Apache-2.0) and **InstantMimic** (arXiv 2609.09821, Sep 2026: "training time for diverse physics-based skills to a few seconds") make physics imitation cheap. They still need reference motions. No twisting-somersault RL result from 2024–2026 was found. | B (speed claim), C (applicability) |
| S12 | **Optimal control of twisting somersaults.** Begon's group (Montreal) generates physically optimal twisting somersaults with **bioptim** (MIT licence):<br>• Charbonneau et al., *Optimal control as a tool for innovation in aerial twisting on a trampoline*, Applied Sciences 2020, doi 10.3390/app10238363.<br>• *Optimal forward twisting pike somersault without self-collision*, Sports Biomechanics 2022.<br>• *Including visual criteria into predictive simulation of acrobatics…*, Sports Biomechanics 25(5), 2025, doi 10.1080/14763141.2025.2577924.<br>Twist and somersault counts are exact by construction. The output is in biorbd multibody format and needs retargeting to a keypoint skeleton (effort medium-high). | B (method exists), C (use for PkVision) |
| S13 | **Analytic twisting-somersault models** (Tong & Dullin, arXiv 1510.08046 and 1612.06455) give closed-form rigid-body-plus-arm kinematics for twisting somersaults and even designed a new dive. They are a cheap physically consistent prior for procedural generation, for example how the twist rate depends on arm position. | B/C |

### 1.3 Real acrobatic mocap to seed synthesis

| # | Finding | Grade |
|---|---|---|
| S14 | **CMU mocap** (free for any use) contains only a handful of acrobatic takes: subject 85 "JumpTwist" and "BreakSequencewithFlips", 87 "Backflip" and "cartwheels", 88 "backflip", "cartwheel into backflip", 89 "flips", 90 "cartwheel". **No twisting somersaults** (from the index file). AMASS repackages these as SMPL under a non-commercial research licence. | A-doc |
| S15 | **TramPoseFit** (10 trampoline sequences, 3 elite athletes, 1–5 acrobatic jumps each, 18 Vicon cameras at 200 fps, fitted to SMPL). This is the only 2025–2026 SMPL source of *real* twisting somersaults I found (arXiv 2604.01322, Apr 2026). The paper says the toolchain is open-source, but no repo or dataset release was found. | B (exists), not found (release) |
| S16 | **FS-Jump3D** (253 figure-skating jumps, 4 skaters, 12 views, Theia3D markerless; `github.com/ryota-skating/FS-Jump3D`) is licensed **CC BY-NC-SA 4.0**. Figure-skating rotations are twists about the longitudinal axis without a somersault. **AthletePose3D** (CVPRW 2025, arXiv 2503.07499; running, track and field, figure skating; ~1.3 M frames) is non-commercial research only, behind a licence agreement. | A-doc |

### 1.4 Rendering or projecting to 2D keypoints and video: measured sim-to-real

| # | Finding | Grade |
|---|---|---|
| S17 | **VIFSS** (Tanaka, Suzuki, Fujii, arXiv 2508.10281, Aug 2025, code Apache-2.0 at `ryota-skating/VIFSS`) is the closest analog to "synthetic views for a twist-like count".<br>• Method: random 2D perspective projections of 3D poses (azimuth ±180°, elevation ±30°) from FS-Jump3D, Human3.6M, MPI-INF-3DHP and AIST++, with jitter and 1% joint masking, used for view-invariant contrastive pretraining. It then fine-tunes on SkatingVerse and on their broadcast annotations.<br>• Element-level results (jump type **and** rotation count, for example "3 Axel" vs "4 Salchow"): frame accuracy 71.34 → **85.82** and F1@50 78.78 → **92.56** vs raw 2D pose.<br>• Failure mode: a temporal 3D lifter trained on doubles and triples failed on a quad not in training ("discrepancies in the number of rotations … between training and inference"). | B |
| S18 | **Trampoline pose, synthetic** (Drolet-Roy et al., arXiv 2604.01322, Apr 2026). 2,520–3,624 Blender images from TramPoseFit, tested on real multi-view trampoline (MRT):<br>• ViTPose-S: COCO baseline 55.8 AP; +synthetic only 60.1; +LSP (real sports images) 66.0; **+LSP+synthetic 73.1**.<br>• 3D MPJPE −19.6% (−12.5 mm).<br>• Qualitatively, "better side disambiguation (e.g., distinguishing left and right wrists)", which is exactly the RTMPose failure that breaks twist counting.<br>• Takeaway: synthetic alone is insufficient; mixed with real data it helps. | B |
| S19 | **SURREACT** (Varol et al., IJCV 2021, arXiv 1912.04070; NTU CVS protocol, RGB):<br>• Synthetic only reaches 63.0% on real 0° test.<br>• Real-only reaches 86.9% at the seen view and 53.6% at the unseen 90° view.<br>• Synth + real at 90°: **69.0%** (+15.4).<br>• Raw motion parameters as input perform "significantly worse" than rendered videos at unseen views.<br>Code repo has no licence asserted. | B |
| S20 | **Generic synthetic human pipelines** can produce 2D and 3D keypoint GT from arbitrary cameras, but none include acrobatics:<br>• UnrealPose-1M (arXiv 2601.00991, Jan 2026)<br>• Avatar4D / Syn2Sport (arXiv 2512.16199, Dec 2025; baseball and ice hockey)<br>• BEDLAM (non-commercial) | B/C |

### 1.5 Can synthetic data give perfectly labelled twist counts across camera angles? What is the gap?

- **Labels: yes, trivially.** In a procedural or kinematic generator, flip count, half-twist count, direction and cork tilt are parameters. Their labels are exact, and camera azimuth and elevation, fps and crop can be swept freely. [C, true by construction]
- **Observability: synthetic data is the only cheap way to measure it.** A twist-count classifier trained and tested on *clean* synthetic 2D projections, per azimuth, gives an upper bound on how recoverable 1.5+ twists are from one camera. That answers claim 3 of the question doc ("twist ≥ 1.5 is invisible from one camera") directly, at zero data cost. No published observability study for twist from 2D was found. [C]
- **Sim-to-real gap for twist counting: not found (no measured result).** The analog numbers (S17–S19) put synthetic-only at roughly 15–25 points below real-trained, with gains of 4–17 points when mixed with real data. [B]
- **Three gap sources specific to PkVision.** [C, reasoning]
  1. **Detector error distribution.** RTMPose on real inverted, blurred bodies produces left/right swaps and collapsed limbs (S18). Each left/right swap mimics a spurious half twist. Clean projections do not contain this. Fixes: either render meshes and re-run RTMPose (≤ $20 of GPU), or inject left/right-swap and dropout noise calibrated on the real RTMPose shards.
  2. **Motion realism.** Twist timing (early versus late), arm wrap, off-axis "cork" technique and takeoff style from walls or rails. Analytic or optimal-control profiles (S12–S13) cover twist timing; context is not synthesized.
  3. **Label semantics.** Real twist counts are judged by landing orientation. Parkour naming conventions (tricking degree names like "cork 900") need a convention decision before synthetic labels match real ones.
- **No SMPL is needed for 2D-keypoint training.** A COCO-17 stick skeleton with limb-length priors avoids the SMPL, AMASS and BEDLAM non-commercial licences entirely, which matters for a FIG judge-assist product. [C]

---

## 2. VLM and LLM weak labelling, and noise-robust training

### 2.1 What VLMs can and cannot label (2025–2026)

| # | Finding | Grade |
|---|---|---|
| V1 | In-project July probe: Claude on frame grids scored flip 11/11, twist 10/11 and direction 11/11 against corrected references. The Wilson 95% CI for twist 10/11 is **0.62–0.98** (calc). Relabelling 65 clips (4%) raised held-out flip F1 0.407 → 0.492, positive in all 3 seeds. Bulk labelling stalled at 74/1,519 when the subscription quota ran out. | A (from brief) |
| V2 | **PushupBench** (arXiv 2604.23407, Apr 2026): "The best frontier model achieves 42.1% exact accuracy" on repetition counting, and 4B open models score ~6%. "Weaker models exploit the modal count rather than reason temporally." Twist counting is a harder counting problem. | B |
| V3 | **VideoNet** (CVPR 2026 Highlight, arXiv 2605.02834):<br>• Across 37 domains (Parkour is one: 40 actions, 200 benchmark clips), Gemini 3.1 Pro gets 69.9% and Qwen3-VL-8B 45.0% on multiple choice.<br>• Parkour, Molmo2-4B base: multiple choice 46.88 (4,000-question benchmark), binary 56.88. | A (parkour domain) |
| V4 | **VLMs as weak annotators in active learning** (arXiv 2605.00480, May 2026): VLM reliability "varies significantly with label granularity … poor on fine-grained labels but can provide accurate coarse-grained labels". They "model the systematic noise in VLM-generated labels using a small set of trusted full labels". This maps directly onto PkVision: flip (coarse) is fine, twist (fine) is where noise concentrates. | B |
| V5 | **MoHallBench** (arXiv 2607.01117, Jul 2026): VideoLLMs hallucinate motions from co-occurrence priors, and "sequential inference hallucination is the most severe". Design consequence: VLM prompts must be *blind to the trick name*, otherwise VLM labels copy the parser's errors and the two sources are no longer independent. | B (finding), C (consequence) |
| V6 | **Open VLMs on Olympic diving** (arXiv 2609.19354, Sep 2026, code MIT): zero-shot quality scores reach Spearman ≤ 0.32. Four VLM families ensembled through a regressor reach 0.67. This supports multi-model ensembling (it is quality scoring, not counting). | B |

### 2.2 Multi-vote consensus: does agreement predict correctness?

| # | Finding | Grade |
|---|---|---|
| V7 | **"When Consistency Does Not Mean Reliability"** (arXiv 2609.13824, Sep 2026): local LLM judges had exact self-consistency of 97.3% (LLaMA-3-8B) and 92.3% (Qwen2.5-7B), with Pearson 0.275 and 0.340 against humans. **Self-agreement alone is not a quality signal.** | B |
| V8 | **LLM-human collaboration** (Tavakoli & Zamani, arXiv 2507.00543, Jul 2025): LLM labels are "inconsistent, poorly calibrated, and highly sensitive to prompt variations". Routing by confidence threshold plus *inter-model disagreement* to human review "improves annotation reliability while reducing human effort by up to 45%". | B |
| V9 | **VLM-alone vs human verification** (arXiv 2609.27327, Sep 2026): across 15 human-centred video annotation tasks, VLM-alone reaches HNS 97.0 (human = 100). **Human verification of VLM outputs reaches HNS 121.5** while cutting human time by 48.9%. | B |
| V10 | **LLM ensembles** (arXiv 2501.08413, Jan 2025): heterogeneous open LLMs differ (some high-precision/low-sensitivity, some the reverse), and the ensemble gives the best precision-sensitivity trade-off. **CROWDLAB** (arXiv 2210.06812, in cleanlab, Apache-2.0) combines any number of annotators *plus a trained classifier* into consensus labels and per-label confidence, and beats Dawid-Skene/GLAD on real multi-annotator data. | B |
| V11 | **Decomposing LLM-judge uncertainty** (arXiv 2609.06444, Sep 2026, code MIT): "stated confidence is no guide to its actual error". A small Bayesian model fitted on already-verified labels finds where the judge is ignorant, and escalating by that "removes 83% more error than total uncertainty". | B |

**Synthesis: agreement rates that predict usable labels.** No paper gives a universal threshold (not found). The measured pattern is:
1. Raw agreement or self-consistency does not predict accuracy (V7, V11).
2. The accuracy *of the agreed subset*, measured on a small verified set, is what matters.
3. Structured noise must be corrected with that verified set (section 2.3).

For PkVision's go rule, I propose the following [C]. Unanimous 3-vote twist labels are usable for training only if their verified precision is at least 10 points above the all-votes precision **and** each true twist class keeps more mass on the correct label than on any single wrong label. The second condition is the diagonal-dominance intuition behind loss correction (Patrini et al. CVPR 2017; GLC below).

### 2.3 Noise-robust training: what the numbers say

| # | Finding | Grade |
|---|---|---|
| N1 | **Gold Loss Correction** (Hendrycks et al., NeurIPS 2018, arXiv 1802.05300, code Apache-2.0). Area under the error curve across corruption strengths, CIFAR-10, "flip" noise:<br>• 5% trusted: No-correction 53.3, Confusion-matrix 8.1, **GLC 6.6**.<br>• 10% trusted: 53.2 → 6.2.<br>Weak-classifier labels (60% wrong) with **1% trusted**: GLC 26.94% test error vs 28.32% with no correction. The authors note those weak labels "have low bias", so uncorrected training was already OK.<br>Lesson: **systematic noise is catastrophic without correction; random noise is not.** | B |
| N2 | **Rolnick et al. 2017** (arXiv 1705.10694): DNNs tolerate massive *random* noise (MNIST >90% with 100 random labels per clean label), but this "requires a significant but manageable increase in dataset size" proportional to the dilution. PkVision has ~1.6k clips, so it cannot buy robustness with scale. | B |
| N3 | **Chen et al., ICML 2019** (arXiv 1905.05040): under symmetric noise, test accuracy is a *quadratic* function of the noise ratio. They then use cross-validation to find clean samples, followed by Co-teaching. | B |
| N4 | **Skeleton action recognition with noisy labels** (NoiseEraSAR, arXiv 2403.09975, Mar 2024, venue not checked; NTU-60 with 56,880 clips):<br>• Plain CTR-GCN goes 86.8 → 81.7 (X-Sub) from 20% to 50% symmetric noise, and 64.8 at 80%.<br>• Co-teaching per modality plus a cross-modal mixture of experts reaches 74.9 / 79.5 at 80%.<br>• Classic label-denoising (SOP, NPC) gives "only marginal performance" on sparse skeleton data.<br>• Symmetric noise only, at 35× PkVision's data size. | B |
| N5 | **Co-teaching** (Han et al., NeurIPS 2018, arXiv 1804.06872). **Label smoothing** is competitive with loss correction under noise (Lukasik et al., ICML 2020, arXiv 2003.02819). **Confident learning / cleanlab** (Northcutt et al., JAIR 2021, arXiv 1911.00068; cleanlab repo **Apache-2.0**, verified via the GitHub API). All three are free. | B |
| N6 | **Partial labels.** When some attributes are unknown per sample, ignoring the missing ones (a masked loss) and scaling by the known fraction trains well on COCO, NUS-WIDE and Open Images (Durand et al., CVPR 2019, arXiv 1902.09720). The parser's abstentions map to this. | B |

### 2.4 Cost of VLM weak labels per tier (Claude API)

Prices (A-doc, claude-api skill table cached 2026-06-24):
- Sonnet 5: $2 / $10 per M tokens (input / output)
- Opus 5.5: $4 / $20
- Haiku 4.5: $1 / $5
- The Batch API costs 50%.

Image tokens = ⌈w/28⌉·⌈h/28⌉, so a 1092×1092 grid costs 1,521 tokens (A-doc, Claude vision docs).

Assumptions (calc): 2 grids per clip plus an 800-token prompt gives ~3.85k input tokens. Output is 0.5k tokens (low effort) to 1.5k tokens (adaptive thinking).

| Model (Batch) | $/clip/vote | 1,519 clips, 1 vote | 1,519 clips, 3 votes |
|---|---|---|---|
| Haiku 4.5 | $0.003–0.006 | $5–9 | $15–26 |
| Sonnet 5 | $0.006–0.011 | $10–17 | $29–52 |
| Opus 5.5 | $0.013–0.023 | $19–35 | $58–104 |

Free tier: the Claude Max subscription quota (July got 74 clips before stalling). Local open VLMs are much weaker: Qwen3-VL-8B scores 45.0% vs Gemini 3.1 Pro 69.9% on VideoNet (V3). Gemini API free-tier limits and prices for Sep 2026 were **not verified**. The API minimum credit purchase was not verified.

---

## 3. Name supervision: noisy video-level names and parser labels

| # | Finding | Grade |
|---|---|---|
| NS1 | **Diving48** (Li, Li, Vasconcelos, ECCV 2018; `svcl.ucsd.edu/projects/resound/dataset.html`) has ~18k clips and 48 classes, each "defined by a combination of takeoff (dive groups), movements in flight (somersaults and/or twists), and entry (dive positions)", for example `['Reverse', '15som', '25Twis', 'FREE']`.<br>• "The ground-truth labels are transcribed from the information board before the start of each dive." That is pure name supervision with a compositional parse.<br>• A V2 release (30 Oct 2020) manually cleaned annotations and removed poorly segmented videos, so the name labels needed a cleaning pass.<br>• Licence: not stated on the page. | A-doc |
| NS2 | **FineGym** (Shao et al., CVPR 2020, arXiv 2004.06704) categories are taken from official documents. Gym99 includes FX "salto backward stretched with 1.5 / 2 / 2.5 / 3 twist", "double salto backward tucked with 1 / 2 twist", and VT "stretched salto backward with 1 / 1.5 / 2 / 2.5 turn off" (from `gym99_categories.txt`).<br>• In 2020, ST-GCN on Gym99 reached only 25.2%, and the paper shows "detections and pose estimations of the gymnast are missed" in flight.<br>• PoseConv3D (Duan et al., CVPR 2022, arXiv 2104.13586, pyskl Apache-2.0), using top-down HRNet with GT boxes, reaches **93.2 (J) / 94.3 (J+L) mean top-1**.<br>• So with good pose extraction and clean official labels, twist-count-distinguished classes are solvable from 2D skeletons in broadcast gymnastics.<br>• Per-class twist accuracy was not reported (not found).<br>• FineGym annotations: **CC BY-NC 4.0**. | A-doc (labels), B (accuracies) |
| NS3 | **MMFS** (figure skating, arXiv 2307.02730, 11,671 clips; code MIT) takes its labels from the broadcast information board plus experts. HRNet 2D skeleton models, 4,113 train clips:<br>• MMFS-63: CTR-GCN 78.8, PoseC3D 75.0.<br>• **TL22 (temporal / rotation labels): 76.7 / 78.4** vs **SL24 (spatial): 92.2 / 95.4**.<br>• **Pretraining PoseC3D on FineGym99 did not help** (75.8 vs 77.4 from scratch).<br>Two conclusions: rotation count is the hard part even with strong labels, and naive cross-sport transfer is not a free lunch. | B |
| NS4 | **FineDiving** (CVPR 2022, arXiv 2204.03646; code MIT, data behind a signed release agreement): flight is annotated with sub-action types for **twist count and somersault count**, derived from dive numbers. That is a name-to-attribute parse analogous to PkVision's parser. | A-doc |
| NS5 | **VideoNet** web-mined names (arXiv 2605.02834). Training data was mined from YouTube titles and transcripts with three filters. On parkour (Molmo2-4B, base → SingleAction filter):<br>• Multiple choice **46.88 → 56.88**<br>• Binary 0-shot **56.88 → 74.37**<br>The data is gated on Hugging Face (`raivn/VideoNet`, auto-approval), with no licence stated and research-only usage terms. The parkour action list could not be read (gated), so whether it includes twist variants is not found. | A |
| NS6 | **Weak supervision label models** combine several noisy "labelling functions" (parser, VLM votes, a kinematic heuristic) and estimate each source's accuracy without ground truth. Snorkel: Ratner et al., VLDB 2018, arXiv 1711.10160, Apache-2.0. Prompted LLMs as labelling functions: Smith et al., arXiv 2205.02318. They assume the sources' errors are conditionally independent, which is why V5 (blind VLM prompts) matters. | B/C |
| NS7 | **Tumbling nomenclature as a constraint.** In trampolining, "front twisting somersaults will always have an odd half twist while back twisting somersaults will always have a round number of twists". Examples: barani = front + ½, rudy = front + 1½, randy = front + 2½, full = back + 1, double full = back + 2, full-in = double back with a full in the first somersault (Wikipedia, *Trampolining terms*). This rule could constrain the (direction, twist) output space and the parser lexicon. Whether it holds for parkour landings (for example "back half" or wall tricks) is unverified. | A-doc (trampoline), C (parkour) |

**Twist-specific quality of the parser route.** The parser is at 0.98 precision on flip and direction and abstains on twist. How many of the 1,618 names carry an explicit twist token (full, double full, half, arabian, rudy, randy, b-twist, cork…), and the precision of a twist lexicon, are both **not measured**. This is the first thing the pilot measures. Tricking degree names ("cork 900", "gainer switch 720") need an explicit convention before they can be mapped to half-twists [C].

---

## 4. Active and verify-only human-in-the-loop: how few verified examples?

| # | Finding | Grade |
|---|---|---|
| H1 | In-project: 65 relabels (4% of the data) gave +0.085 flip F1 across 3 seeds. | A |
| H2 | **GLC** (N1): a 1–5% trusted fraction corrects severe structured noise. At PkVision scale that is **15–76 clips**. To estimate a K×K transition matrix, each true twist class needs its own trusted examples. With ~5–7 twist classes (0, ½ … 3+) and ≥ 8–10 per class, that is **~50–70 verified clips per attribute** (calc / C). | B + calc |
| H3 | **TypiClust** (Hacohen et al., ICML 2022, arXiv 2202.02794, code MIT): "typical examples are best queried when the budget is low, while unrepresentative examples are best queried when the budget is large". 10 TypiClust-selected CIFAR-10 labels with semi-supervised learning reach 93.2% (+39.4 over random). For tens of clips per round, pick *representative* clips first, not the most uncertain ones. | B |
| H4 | **ActiveLab** (Goh & Mueller, arXiv 2301.11856, in cleanlab) decides whether to re-label an existing example or label a new one. It fits the "re-verify disputed clips vs verify new clips" choice. | B |
| H5 | **Valid accuracy estimates from few verified clips.**<br>• Prediction-powered inference (Angelopoulos et al., *Science* 2023, arXiv 2301.09633; `ppi_py` MIT).<br>• Active statistical inference (Zrnic & Candès, arXiv 2403.03208): verify where the model is uncertain, with valid CIs.<br>• Confidence-driven inference (Gligorić et al., arXiv 2408.15204): ">25% fewer human annotations" with guaranteed validity.<br>These turn weak labels on all 1,519 clips plus tens of verified clips into an honest accuracy estimate with a confidence interval. | B |
| H6 | **How many verified clips to *certify* a twist threshold?** (calc, Wilson 95% lower bound ≥ 0.85)<br>• 30 clips: all 30 must be correct<br>• 50 clips: 48 correct (0.96)<br>• 60 clips: 57 correct (0.95)<br>• 100 clips: 92 correct (0.92)<br>• 200 clips: 180 correct (0.90)<br>Tens of clips can *reject* a source or *steer* training. Certifying the ≥ 85% twist target needs ~100 verified twist clips in total, for example over several 20–30-clip queues. | calc |
| H7 | **Verification is faster and more accurate than labelling** (V9: −48.9% time, HNS 121.5 vs 100). Caveat for PkVision: the "gold" labels were ~35% wrong on twist, so *human* twist verification is itself error-prone. Offer slow-motion or frame stepping, a "can't tell" option, and prefer own-footage clips where the performer knows the trick. | B + C |

---

## 5. Public datasets usable for pretraining, transfer or evaluation

| Dataset | Domain | Twist or rotation labels | Size | Licence | Use for PkVision |
|---|---|---|---|---|---|
| FineGym (Gym99/288/530) | Gymnastics broadcast | Yes (salto twists ½–3, "turn off" on vault) | 530 element classes; Gym99 counts not re-verified | Annotations **CC BY-NC 4.0**; videos via YouTube IDs | Twist-count pretraining or eval analog. Transfer to skating did not help (NS3) |
| MMFS | Figure skating broadcast | Rotation level (TL22) | 11,671 clips, 256 categories; HRNet skeletons | Code MIT; data licence not stated | Rotation-count analog, 2D skeleton |
| FineDiving | Diving | Twist and somersault counts per flight | 3,000 clips, 52 action types | Release agreement (research); code MIT | Name-to-attribute analog |
| Diving48 (V2) | Diving | Twist half-counts in class codes | ~18k clips, 48 classes | Not stated on page | Compositional analog |
| FS-Jump3D | Figure skating, 3D mocap + 12 views | Jump type and rotations | 253 jumps | **CC BY-NC-SA 4.0** | Projection pretraining (VIFSS recipe) |
| AthletePose3D | Athletics / skating 3D | No | ~1.3 M frames | Non-commercial, agreement | Pose robustness only |
| TramPoseFit / SynTramPose | Trampoline, SMPL | Real twisting somersaults | 10 sequences / 2.5–3.6k images | **Release not found** | Best twisting-somersault 3D seed if released |
| CMU mocap / AMASS | General mocap | A few backflips and cartwheels, no twisting saltos | — | CMU free; AMASS non-commercial | Seed for takeoff and landing poses |
| Kinetics-700 | Web video | No (classes include `parkour`, `somersaulting`, `backflip (human)`, `gymnastics tumbling`, `cartwheeling`, `capoeira`) | ~700 classes | Annotations CC BY 4.0 (standard; not re-verified here); YouTube availability decays | Unlabelled parkour-ish video for SSL |
| VideoNet Parkour | Web video + QA | Unknown (list gated) | 40 actions, 200 benchmark clips, ~4–7k mined training clips | Gated, research terms, no licence | In-domain parkour naming signal (NS5) |
| LAAS Parkour | Mocap + force plates | No (kong vault, safety vault, pull-up, muscle-up) | 5 subjects | Not checked | Context (vault) kinematics only |
| Mixamo | Animator-made FBX | Backflip, front flip, etc.; no labelled twist counts | — | ML-training terms not verified | Synthetic seed (as in S6) |

2025–2026 parkour, tricking or freerunning dataset with twist labels: **not found** (arXiv title/abstract search for "parkour" + dataset/video/recognition returned nothing for 2024–2026). The only in-domain public resource found is VideoNet's parkour domain.

Licence risk [C]: FineGym, FS-Jump3D, AthletePose3D and AMASS are non-commercial. That is fine for a research prototype. It needs a decision before any commercial FIG judge-assist, because weights trained on NC data are a grey zone.

---

## 6. Ranked label and data strategies for twist

Expected twist label quality is on the training labels, not the model.

| Rank | Strategy | Expected twist label quality | Free | ≤ $20 | ≤ $50 | ≤ $150 | Effort | Main risk |
|---|---|---|---|---|---|---|---|---|
| 1 | **Parser twist lexicon + verify-only audit** (explicit tokens only; abstain otherwise) | Unknown. Likely high on explicit tokens (analogs NS1–NS4 are name-derived), zero coverage elsewhere. Must be measured (pilot). | All of it | — | — | — | Low (days) | Parkour naming is less standardized than FINA/FIG codes; coverage may be small; tricking degree names are ambiguous |
| 2 | **Own-footage named tricks as trusted and eval set** | ~100% by construction (minus under-rotation and naming slips) | Filming time only | — | — | — | Medium (sessions) | Small n; one athlete's repertoire (few ≥ 2 twists); biased toward one body and style |
| 3 | **Blinded multi-vote VLM labels** on parser-abstain and hard clips, consensus via CROWDLAB/Dawid-Skene, confidence-driven verification | Twist 10/11 on clean clips (A, wide CI). Expect a drop on ≥ 1.5 twists and corks (B: V2, V4). | Max-subscription quota (slow, ~74 clips per quota window in July) | 1 Sonnet-5 batch vote on all clips ($10–17) | 3 Sonnet votes ($29–52) or 1 Opus 5.5 vote ($19–35) | 3 votes + Opus tie-breaks + re-runs | Low (pipeline exists) | Systematic, correlated errors; name leakage into the prompt (V5); agreement ≠ accuracy (V7) |
| 4 | **Noise-aware training** (GLC transition matrix from verified clips, co-teaching, label smoothing, masked partial labels). An enabler for 1–3. | Converts 60–85%-accurate labels into near-clean training signal *if* the trusted set is ≥ ~50 clips (N1, H2) | All of it | — | — | — | Low-medium | Transition-matrix estimates on ~10 clips per class are noisy |
| 5 | **Procedural synthetic 2D twist data** (kinematic COCO-17 skeleton, analytic flip/twist/cork profiles, random cameras, RTMPose-calibrated left/right-swap noise) for pretraining + an observability map | 100% labels. Transfer unknown (not found); analogs −15 to −25 points synthetic-only, +4 to +17 mixed (S17–S19) | Generator + projection on Mac/2060 | Render meshes + RTMPose re-detect on a rented 4090 (~20–40 GPU-h ≈ $6–28, estimate) | Optimal-control motion (bioptim) for realism | — | Medium-high (1–2 weeks) | Learns synthetic shortcuts (constant twist rate, clean left/right); context (wall/bar) missing |
| 6 | **Analog public datasets** (FineGym, MMFS, FS-Jump3D projections) for pretraining | Clean labels, other domain | Download | Compute ≤ $20 | — | — | Medium | Transfer neutral or negative (NS3: −1.6 points); NC licences |
| 7 | Text-to-motion (Kimodo with root keyframes) | Counts not controllable by text (S3). Keyframe route untested. | Code free; GPU needs not found | — | — | — | High | No acrobatic training data confirmed; plausibility of forced rotations |
| 8 | Physics RL (DeepMimic/MaskedMimic) or video generation | Needs references (S9–S11); video count fidelity not claimed (S7) | — | — | — | Yes | Very high | Effort; for 2D keypoint training it adds little over kinematic synthesis |

Not recommended: HY-Motion (S2, EU-excluded and training on outputs forbidden).

---

## 7. Can synthetic data realistically break the label wall within budget?

**Not on its own.** [B for the analog evidence, C for the conclusion]

- It *can* be built within the free to ≤ $20 tiers, and it gives unlimited perfectly labelled twist counts at any camera angle.
- It cannot provide the thing the wall is actually missing: **trusted real twist labels to validate against**, and the **real RTMPose error distribution** on inverted bodies.
- Every measured analog shows synthetic-only models well below real-trained ones (S17–S19). The gains come from *mixing* synthetic with real labels, or from using synthetic views as *pretraining* before a real fine-tune.
- No measured twist sim-to-real result exists (not found).

The realistic role is:
- **Diagnostic.** Train and test on clean vs noise-injected synthetic projections per azimuth. This tells you, for $0, whether ≥ 1.5 twists are recoverable from single-view 2D keypoints at all, and how much left/right swapping destroys them. It decides the single-phone vs two-phone question on physics, not opinion.
- **Pretraining and augmentation.** VIFSS-style view-invariant pretraining, then fine-tune on the verified real set.
- **Budget.** The first two roles fit in ≤ $20. They reduce, but do not remove, the need for ~50–100 verified real twist clips.

---

## 8. ≤ $5 pilot: twist label-source audit

**Question.** On parkourtheory clips, what is the precision and coverage of each cheap twist label source, and does multi-vote agreement predict correctness?

**Sample (60 clips, stratified, fixed before any source is run).**
- Include the 12 July reference clips (corrected labels) and the 26-clip dispute queue.
- Top up with parkourtheory clips so that there are ≥ 15 clips with twist ≥ 1.5, ≥ 10 corks, ≥ 10 zero-twist flips, and ≥ 10 clips whose names carry explicit twist tokens (full, double full, half/arabian, rudy/randy).

**Sources, run independently. The VLM never sees the trick name.**
1. Parser twist lexicon: map explicit tokens to half-twists and abstain otherwise. $0.
2. Claude Sonnet 5 via Batch API: 3 votes with different frame samplings (for example offsets of 0, ⅓ and ⅔ frame stride), same prompt, structured output {flips, half_twists, direction, cork, confidence}. 60 × 3 × $0.006–0.011 = **$1.15–2.05**.
3. Claude Opus 5.5 via Batch API: 1 vote. 60 × $0.013–0.023 = **$0.76–1.36**.
4. Before the batch, run `count_tokens` on 3 clips (free) to confirm the ~3.85k input-token assumption, and abort if the projected total is above $5.
5. **Total: $1.9–3.4.**

**Human step (verify-only, ~30–45 min).**
- For each clip the owner sees the clip (slow-motion or frame-step available) and **one** proposed twist value: the majority vote, or the parser value where it exists. The answer is yes / no / can't tell.
- If the answer is "no" and the sources disagree, show the competing value as a binary choice. That is still verification, not labelling.
- Log time per clip.

**Metrics.**
- Per source: twist precision at its own coverage, with Wilson 95% CIs, reported separately for twist ≥ 1.5 and for corks.
- Precision of unanimous 3-vote vs split votes.
- Per-true-class confusion rows. This is the first GLC transition-matrix estimate.
- Human "can't tell" rate.
- Minutes per verified clip.

**Preregistered decisions** (write them into the analysis plan before running):
1. **Parser twist lexicon.** If precision ≥ 0.95 on ≥ 20 covered clips (lower bound ≥ 0.76 at 20/20), adopt it as the primary twist label where it fires.
2. **Bulk VLM labels (move to the ≤ $50 tier: 3 Sonnet votes on 1,519 clips, $29–52).** Go only if **all** of these hold:
   - unanimous-vote precision ≥ 0.85 overall and ≥ 0.75 on twist ≥ 1.5;
   - unanimous precision ≥ all-vote precision + 10 points;
   - each true-class row is diagonally dominant.
3. If agreement does *not* predict accuracy, or if ≥ 1.5-twist precision is < 0.6, **no bulk VLM spend**. Twist labels then come from the parser lexicon plus own footage, and the synthetic observability probe below becomes the priority.
4. **Human verifier reliability.** If the human "can't tell" rate is > 25% on twist ≥ 1.5, verification of hard twists needs own-footage or 240 fps clips. Record that as a real-conditions finding.

**$0 side-probe (optional, CPU only).**
- Build the procedural COCO-17 twist generator and project it at 8 azimuths × 3 elevations. Train a small temporal classifier on clean vs left/right-swap-injected keypoints and report twist accuracy per azimuth and per twist count.
- If clean single-view accuracy for 1.5–3 twists is < 80% at common filming azimuths, single-iPhone twist ≥ 1.5 is geometrically marginal, and the two-phone capture decision follows.

---

## 9. Gaps and not found

- Measured sim-to-real transfer for twist or rotation **counting from 2D keypoints**: not found.
- Any 2024–2026 parkour, tricking or freerunning dataset with twist labels: not found. VideoNet parkour's action list is gated, so unknown.
- TramPoseFit / SynTramPose code or data release: not found. Release of the 2,000-video text-to-skeleton synthetic acrobatics dataset: not found.
- Per-class twist accuracy on FineGym99 (PoseConv3D) and per-attribute twist accuracy on Diving48: not found in the sources read (track 06 covers TQN and NS-AQA).
- Controllable-count text-to-motion for flips or twists: not found. Kimodo's acrobatic training coverage and VRAM needs: not found.
- Gemini API free-tier limits and prices for Sep 2026: not verified. Anthropic API minimum credit purchase: not verified. Mixamo ML-training terms: not verified. Diving48 licence: not stated.
- A universal "agreement rate → usable label" threshold: not found. Section 2.2 gives the measured pattern and a proposed rule.
- Coverage and precision of a parkour twist lexicon over the 1,618 names: not measured (pilot item 1).

---

## Sources

Papers (arXiv id, venue/date, code and licence where checked):

1. Drolet-Roy et al., *Human Pose Estimation in Trampoline Gymnastics: Improving Performance Using a New Synthetic Dataset*, arXiv 2604.01322 (Apr 2026). Code: stated open-source, repo not found.
2. Tanaka, Suzuki, Fujii, *VIFSS: View-Invariant and Figure Skating-Specific Pose Representation Learning for TAS*, arXiv 2508.10281 (Aug 2025). Code `github.com/ryota-skating/VIFSS`, Apache-2.0.
3. Tanaka et al., *3D Pose-Based TAS for Figure Skating* (FS-Jump3D), arXiv 2408.16638, ACM MMSports 2024. Data `github.com/ryota-skating/FS-Jump3D`, CC BY-NC-SA 4.0.
4. Varol, Laptev, Schmid, Zisserman, *Synthetic Humans for Action Recognition from Unseen Viewpoints* (SURREACT), IJCV 2021, arXiv 1912.04070. Code `gulvarol/surreact`, no licence asserted.
5. Taghipour et al., *Controllable Complex Human Motion Video Generation via Text-to-Skeleton Cascades*, arXiv 2603.08028 (Mar 2026).
6. Tencent Hunyuan, *HY-Motion 1.0*, arXiv 2512.23464 (Dec 2025). Code/weights `github.com/Tencent-Hunyuan/HY-Motion-1.0`, Tencent HY-Motion 1.0 Community License (EU/UK/KR excluded).
7. Rempe et al., *Kimodo: Scaling Controllable Human Motion Generation*, arXiv 2603.15546 (Mar 2026). Code `nv-tlabs/kimodo`, Apache-2.0; weights under the NVIDIA Open Model / R&D licences.
8. *Go to Zero: Towards Zero-shot Motion Generation with Million-scale Data*, arXiv 2507.07095 (ICCV 2025).
9. Jung & Lee, *Per-Stroke Temporal Control for Text-to-Motion via Action Units*, arXiv 2607.15717 (Jul 2026).
10. Yan, He, Tai, *Beyond MoCap: Scaling Motion Tokenizers with Synthetic Human Motion*, arXiv 2606.27547 (Jun 2026).
11. Dong et al., *Generative Data Augmentation for Skeleton Action Recognition*, arXiv 2604.14933 (Apr 2026).
12. Kuang, Wang, Gui, *GenPrior*, arXiv 2608.02236 (Aug 2026). Code `jidongkuang/GenPrior`, no licence asserted.
13. Peng et al., *DeepMimic*, SIGGRAPH 2018, arXiv 1804.02717. Code MIT.
14. Xu, Shi, Yin, Peng, *PARC*, SIGGRAPH 2025, arXiv 2505.04002. Code `mshoe/PARC`, no licence asserted.
15. Tessler et al., *MaskedMimic*, arXiv 2409.14393. Code `NVlabs/ProtoMotions`, Apache-2.0.
16. Choi, Leem, Won, *InstantMimic*, arXiv 2609.09821 (Sep 2026).
17. Charbonneau et al., *Optimal Control as a Tool for Innovation in Aerial Twisting on a Trampoline*, Applied Sciences 2020, doi 10.3390/app10238363. *Including visual criteria into predictive simulation of acrobatics…*, Sports Biomechanics 25(5), 2025, doi 10.1080/14763141.2025.2577924. bioptim: `pyomeca/bioptim`, MIT.
18. Dullin & Tong, *Twisting Somersault*, arXiv 1510.08046; Tong & Dullin, *A New Twisting Somersault – 513XD*, arXiv 1612.06455.
19. Kawaguchi et al., *UnrealPose*, arXiv 2601.00991 (Jan 2026). Bright et al., *Avatar4D*, arXiv 2512.16199 (Dec 2025).
20. Li et al., *PushupBench: Your VLM is not good at counting pushups*, arXiv 2604.23407 (Apr 2026).
21. Yadav et al., *VideoNet: A Large-Scale Dataset for Domain-Specific Action Recognition*, arXiv 2605.02834, CVPR 2026 Highlight. Data `huggingface.co/datasets/raivn/VideoNet` (gated, no licence stated); repo `RAIVNLab/VideoNet`.
22. Nguyen et al., *Leveraging Vision-Language Models as Weak Annotators in Active Learning*, arXiv 2605.00480 (May 2026).
23. Chen et al., *MoHallBench*, arXiv 2607.01117 (Jul 2026).
24. Velesaca et al., *Can Vision-Language Models Judge Olympic Diving?*, arXiv 2609.19354 (Sep 2026). Code `hvelesaca/olympic_diving_judge_vlm`, MIT.
25. Tiwari, *When Consistency Does Not Mean Reliability*, arXiv 2609.13824 (Sep 2026).
26. Tavakoli & Zamani, *Reliable Annotations with Less Effort*, arXiv 2507.00543 (Jul 2025).
27. Shen et al., *Can Vision-Language Models Analyze Human-Centered Video?*, arXiv 2609.27327 (Sep 2026).
28. Qiu et al., *Labeling Free-text Data using Language Model Ensembles*, arXiv 2501.08413 (Jan 2025).
29. Goh, Tkachenko, Mueller, *CROWDLAB*, arXiv 2210.06812; Goh & Mueller, *ActiveLab*, arXiv 2301.11856. Both in `cleanlab/cleanlab`, Apache-2.0.
30. Lail, *Decomposing LLM-Judge Uncertainty to Target Expert Labels*, arXiv 2609.06444 (Sep 2026). Code `composo-ai/judge-uncertainty-decomposition`, MIT.
31. Hendrycks et al., *Using Trusted Data to Train Deep Networks on Labels Corrupted by Severe Noise* (GLC), NeurIPS 2018, arXiv 1802.05300. Code `mmazeika/glc`, Apache-2.0.
32. Rolnick et al., *Deep Learning is Robust to Massive Label Noise*, arXiv 1705.10694.
33. Chen et al., *Understanding and Utilizing Deep Neural Networks Trained with Noisy Labels*, ICML 2019, arXiv 1905.05040.
34. Xu et al., *Skeleton-Based Human Action Recognition with Noisy Labels* (NoiseEraSAR), arXiv 2403.09975. Code `xuyizdby/NoiseEraSAR`, no licence asserted.
35. Han et al., *Co-teaching*, NeurIPS 2018, arXiv 1804.06872. Lukasik et al., *Does label smoothing mitigate label noise?*, ICML 2020, arXiv 2003.02819. Northcutt et al., *Confident Learning*, JAIR 2021, arXiv 1911.00068.
36. Durand, Mehrasa, Mori, *Learning a Deep ConvNet for Multi-label Classification with Partial Labels*, CVPR 2019, arXiv 1902.09720.
37. Ratner et al., *Snorkel*, VLDB 2018, arXiv 1711.10160 (`snorkel-team/snorkel`, Apache-2.0). Smith et al., *Language Models in the Loop*, arXiv 2205.02318.
38. Hacohen, Dekel, Weinshall, *Active Learning on a Budget* (TypiClust), ICML 2022, arXiv 2202.02794. Code MIT.
39. Angelopoulos et al., *Prediction-Powered Inference*, Science 2023, arXiv 2301.09633 (`aangelopoulos/ppi_py`, MIT). Zrnic & Candès, *Active Statistical Inference*, arXiv 2403.03208. Gligorić et al., *Can Unconfident LLM Annotations Be Used for Confident Conclusions?*, arXiv 2408.15204.
40. Shao et al., *FineGym*, CVPR 2020, arXiv 2004.06704. Annotations CC BY-NC 4.0; class list `sdolivia.github.io/FineGym/resources/dataset/gym99_categories.txt`.
41. Duan et al., *Revisiting Skeleton-based Action Recognition* (PoseConv3D), CVPR 2022, arXiv 2104.13586. Code `kennymckormick/pyskl`, Apache-2.0.
42. Liu et al., *Fine-grained Action Analysis: A Multi-modality and Multi-task Dataset of Figure Skating* (MMFS), arXiv 2307.02730. Code `dingyn-Reno/MMFS`, MIT.
43. Xu et al., *FineDiving*, CVPR 2022, arXiv 2204.03646. Code `xujinglin/FineDiving`, MIT; data by release agreement.
44. Li, Li, Vasconcelos, *RESOUND* (Diving48), ECCV 2018. Page `svcl.ucsd.edu/projects/resound/dataset.html` (no licence stated; V2 labels 30 Oct 2020).
45. Yeung et al., *AthletePose3D*, CVPRW 2025, arXiv 2503.07499. Non-commercial research.
46. Gao et al., *FSBench*, CVPR 2025, arXiv 2504.19514 (figure-skating MLLM benchmark; no rotation-count result extracted).

Documents and web:

- HY-Motion 1.0 `License.txt`, read via the GitHub API 2026-09-28.
- Claude pricing and vision token formula: claude-api skill model table (cached 2026-06-24) and `platform.claude.com/docs/en/build-with-claude/vision`.
- CMU mocap index: `github.com/una-dinosauria/cmu-mocap` (`cmu-mocap-index-text.txt`).
- Kinetics-700 class list: `open-mmlab/mmaction2` `label_map_k700.txt`.
- OpenAI, *Sora 2 is here*, `openai.com/index/sora-2/` (30 Sep 2025).
- Wikipedia, *Trampolining terms*.
