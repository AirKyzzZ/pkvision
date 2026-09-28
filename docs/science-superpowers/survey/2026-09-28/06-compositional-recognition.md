# 06: Compositional, zero-shot and open-vocabulary fine-grained action recognition

Survey track 06 for the Sep-2026 feasibility question (`docs/science-superpowers/questions/2026-09-28-pkvision-feasibility.md`), sub-question 3 ("Composition").
Desk research only. No code was run on project data and no money was spent.
Date: 2026-09-28.

**Grading.** A = in-domain (parkour) measured, or an in-domain primary source. B = analog domain measured (diving, gymnastics, figure skating, generic compositional video/image benchmarks). C = claim, method description, or unmeasured statement.

**Tools used.** alphaXiv discovery + full text, arXiv search/abstracts, WebSearch/WebFetch, `pdftotext` on two downloaded PDFs (the FIG PK Table of Tricks 2025 and a diving paper). Where a number comes from a table I read in full text, it is quoted as printed. Where I could not find something I say "not found".

---

## TL;DR

1. **Nobody gets compositional generalization "for free".** In the most comparable video benchmark (zero-shot compositional action recognition, Sth-com), the best 2026 model reaches 44.0% top-1 on unseen verb-object compositions in the honest open-world setting, and its composition accuracy is within about 1 point of the product of its verb and object accuracies (ΔCG = +0.95). Earlier SOTA sits slightly *below* the product (ΔCG = −0.42 to −0.63). So the "0.5³ ≈ 13.6%" compounding PkVision observed is the normal behaviour, not a bug. The lever is per-attribute accuracy plus abstention, not the composition step. [B]
2. **Rules on pose can be compositional by construction and very accurate in a constrained analog.** NS-AQA (CVPRW 2024) counts somersaults and twists with hand-written rules on 2D pose and reaches 93–99% per attribute on broadcast Olympic diving without training on labelled dives. That matches a supervised C3D multi-task model (93–97%). Full dive-code exact-match was not reported. [B]
3. **Naive product-of-marginals decoding loses badly on seen classes.** On Diving48-v2, a multi-label attribute model decoded by multiplying attribute probabilities gets 50.3% per video. A plain 48-way classifier gets 80.4%, and TQN (per-attribute heads plus a global class head) gets 81.8%. Use a joint/global head for known tricks and keep factorized decoding for unseen ones. [B]
4. **Open-world evaluation is where numbers collapse.** CSP on C-GQA goes from 26.8% unseen (closed-world) to 5.2% (open-world). On the small-vocabulary UT-Zappos, the 2025 best open-world unseen accuracy is 63.7% (LOGICZSL, CVPR 2025). PkVision's attribute space is small like UT-Zappos, so that is the most honest external anchor for a "≥60% on unseen compositions" target. [B]
5. **Feasibility masks from external knowledge are the standard fix for the open-world output space.** KG-SP predicts primitives independently and removes infeasible compositions with a knowledge prior. It also defines a partial-supervision setting (only some primitives labelled), which matches PkVision's "some cues abstain" weak labels. [B/C]
6. **Selective-prediction reference points.** With a ~75%-accurate classifier (ImageNet ResNet-50 top-1), SGR guarantees 10% error at 65% coverage and 15% at 77%. With a perfect ranker, precision at coverage c is at most acc/c, so "≥90% precision at ≥70% coverage" needs at least 63% full-coverage accuracy, and realistically about 78–80%. [B]
7. **Confidence-based abstention breaks under domain shift.** A skeleton model trained on NTU drops from 63.2% to 1.6% on real gym 2D-pose footage and stays confidently wrong: about 98–99% risk at 50% coverage, even though OOD AUROC is 0.94. Gemini's confidence on video QA does not drop when evidence is cut 3×. Abstention thresholds must be calibrated on in-domain iPhone footage. [B]
8. **Procedure-aware sports methods mostly assume the element is known.** FineDiving's TSA picks exemplars using the ground-truth dive number, and AQA pipelines take the degree of difficulty from metadata. No sports dataset reports generalization to *held-out elements* (unseen dive codes, unseen gymnastics elements, unseen jump/rotation combos): not found. [B]
9. **No machine-readable parkour or tricking grammar exists in the literature (not found).** The FIG PK Table of Tricks 2025 is a flat list of guiding values in 4 categories, with semi-systematic names ("Swing Gainer 720 (2 twist)", "Wall Double Cork"). Judges then upscale by placement, form, entry and exit, and slanted-axis moves take −0.5. Trampoline (FIG shorthand like `42/`, `821/`) and diving (dive numbers like `5253B`) *do* have machine-readable codes and are the templates to copy. [A for the FIG document, B/C for the analogs]
10. **The thresholds.** The per-attribute target (90% at 70% coverage, 85% for twist) is in line with what constrained analogs achieve, but it is not measurable on the current gold labels (~35% twist label error) or on 13 own clips. The compositional target (≥60% top-1, ≥80% top-3 exact on held-out *clips*) is (a) not an open-vocabulary test unless compositions are held out, (b) inconsistent with the per-attribute target unless it also allows abstention, and (c) the top-3 part is close to the in-project oracle ceiling (85.1% top-3 with perfect attributes on FIG-149). Section 6 has concrete changes.

---

## 1. Attribute-based and compositional zero-shot action recognition (2024–2026)

### 1.1 Zero-shot compositional action recognition (ZS-CAR) in video

**Task.** Recognise unseen verb-object pairs from seen verbs and objects. The benchmark is Sth-com, built from Something-Something V2. Label coverage (the fraction of the verb×object space seen in training) is 12.8% for Sth-com and 7.5% for the new EK100-com.

| Method (backbone) | Setting | Verb@unseen | Obj@unseen | Unseen comp. top-1 | ΔCG unseen | HM | Grade |
|---|---|---|---|---|---|---|---|
| C2C enhanced (CLIP + CoOp prompts) | closed-world, test-bias sweep | 56.3 (verb, all) | 58.9 (obj, all) | 56.6 | n/a | 44.5 (AUC 26.0) | B |
| C2C enhanced (VideoSwin-T + fastText) | closed-world | 48.8 (verb, all) | 52.8 (obj, all) | 44.1 | n/a | 36.7 | B |
| LogicCAR (first-order-logic constraints) | closed-world | n/a | n/a | n/a | n/a | 45.2 (AUC 27.0) | B |
| RCORE enhanced | closed-world | n/a | n/a | n/a | n/a | 46.1 (AUC 27.5) | B |
| C2C (CLIP) | **open-world, unbiased** | 54.36 | 56.10 | 30.08 | −0.42 | 36.47 | B |
| RCORE (CLIP) | open-world, unbiased | 59.00 | 56.34 | 33.90 | +0.66 | 38.67 | B |
| C2C (InternVideo2) | open-world, unbiased | 63.29 | 63.44 | 39.53 | −0.63 | 44.73 | B |
| RCORE (InternVideo2) | open-world, unbiased | 66.65 | 64.56 | **43.98** | +0.95 | 46.88 | B |
| RCORE (InternVideo2), EK100-com | open-world, unbiased | 59.78 | 56.00 | 37.31 | +3.83 | 39.08 | B |

Sources: C2C Tables 2–3 [S1]; RCORE Tables 1–4 [S2]; LogicCAR as reported in RCORE Table 3 [S2, S3].

Key facts:
- **ΔCG is RCORE's "compositional gap"**: composition accuracy minus (verb accuracy × object accuracy). Every baseline has ΔCG < 0 on unseen compositions, so joint modelling does *worse* than independent multiplication. RCORE's two regularisers make it slightly positive, at +0.66 to +3.83 points [S2]. **Implication:** on unseen combinations, assume exact-match ≈ product of per-attribute accuracies (B).
- **False Seen Prediction (FSP)**: the share of misclassified unseen samples that get predicted as a *seen* composition. For C2C with CLIP it grows from 53% to 63% during training, and most of these are "verb-collapse" (right object, wrong verb). RCORE holds it at 47→44% [S2]. The PkVision analog is an unseen trick being snapped to the nearest known ontology trick. (B)
- The **closed-world, test-tuned protocol inflates numbers.** With CLIP, C2C scores 56.6% unseen under the closed-world bias sweep but 30.1% under the open-world unbiased protocol [S1, S2]. (B)
- Asymmetric learning: objects are learned faster than verbs, so the model uses the object to guess the verb [S2]. The PkVision analog is context (wall/bar, visible in a single frame) being used as a shortcut for flip/twist (which need temporal evidence). (B, by analogy)

### 1.2 Image CZSL as a calibration of expectations

| Benchmark (setting) | Best unseen top-1 | HM | Method, venue | Grade |
|---|---|---|---|---|
| UT-Zappos (closed) | 75.8 | 55.3 | PLO, ACM MM 2025 | B |
| UT-Zappos (open-world) | 63.7 | 50.8 | LOGICZSL, CVPR 2025 | B |
| MIT-States (open-world) | 21.8 | 22.1 | CDS-CZSL, CVPR 2024 | B |
| C-GQA (closed) | 40.7 | 34.0 | Troika+FlowComposer, 2026 | B |
| C-GQA (open-world) | 10.4 | 13.9 | PLO, 2025 | B |
| C-GQA, CSP (ICLR 2023): closed → open | 26.8 → 5.2 | 20.5 → 6.9 | CSP | B |
| C-GQA, CLIP zero-shot (open) | 4.6 | 4.0 | CLIP | B |

Sources: FlowComposer Table 1 (which also reprints LOGICZSL, PLO, CDS-CZSL and IMAX) [S4]; CSP Table 2 [S5]; survey [S6].

Takeaways:
- Open-world numbers are far lower than closed-world ones, and the drop is largest when the primitive vocabulary is large (C-GQA). UT-Zappos has 16 attributes × 12 objects, which is about the size of PkVision's attribute space: flip 4 × twist ~7 × direction ~4 × axis 2 × context ~4–10. Open-world unseen accuracy there reaches ~64% (B).
- In the open world, visual disentanglement methods are more reliable than cross-modal ones [S6] (C, survey synthesis).
- CLIP zero-shot open-world is ~4–5% unseen even on images. PkVision's "CLIP zero-shot 0% exact" is consistent with this (B supports A).

### 1.3 Held-out-composition splits reveal the gap

Something-Else (CVPR 2020) splits Something-Something so that train and test verb-noun combinations do not overlap. I3D drops from 61.7% top-1 on a random ("shuffled") split to 46.8% on the compositional split, a loss of about 15 points. The box-geometry model STIN (ground-truth boxes) drops less, from 54.0% to 47.1% [S7]. **The lesson for PkVision:** structure-based inputs (skeleton, geometry) degrade less on novel compositions than appearance-based inputs, and a random clip split overstates open-vocabulary ability by roughly 7–15 points in that benchmark (B).

### 1.4 How the field avoids error compounding

| Mechanism | What it does | Evidence | Grade |
|---|---|---|---|
| **Independent primitive heads + feasibility mask** (KG-SP, CompCos) | Predict each primitive separately, then zero out compositions that external knowledge says are infeasible. Also trains under **partial supervision** (only some primitives labelled). | KG-SP was SOTA on open-world CZSL and on pCZSL (numbers not extracted) [S8]. Open-world CZSL methods apply feasibility calibration by default [S5, S9]. | B/C |
| **Conditional / joint composition scoring** (C2C composition-inference module, CDS-CZSL) | Scores the pair jointly, so the attribute is conditioned on the object. | On unseen compositions it gains ≤ +4 points over independence and is often negative (ΔCG) [S2]. | B |
| **Symbolic constraints** (LogicCAR) | First-order-logic compositional and hierarchical constraints embedded as losses. | Closed-world HM 45.2 vs 44.5 for C2C enhanced, i.e. +0.7 [S2, S3]. | B |
| **Synthesised compositions / label-space expansion** (RCORE CPR, C2C CutMix) | Mixes clips to create supervised novel pairs and penalises frequent seen "hard negatives". | +3.8 to +7.0 points unseen accuracy [S2]. | B |
| **Seen/unseen calibration bias** ("calibrated stacking", Chao et al. 2016) | Subtracts a constant from the scores of seen classes, tuned on a validation set that contains *its own* unseen compositions. | Standard in the CZSL protocol (bias sweep → AUC, best HM) [S5, S10, S12]. | B/C |
| **Global class head + attribute heads** (TQN) | Attribute queries share data across classes, and a "global" query predicts the known class. | Diving48-v2: 81.8% vs 80.4% for plain multi-class vs 50.3% for the attribute-product decode [S11]. | B |
| **Direct attribute prediction (DAP)** decoding | Scores each class by the product of its attribute-signature probabilities, which is how an ontology decoder works. | Classic (Lampert et al., TPAMI 2014). The TQN multi-label row above is essentially DAP and underperforms on seen classes [S11]. | B/C |
| **Top-k / set-valued output + abstention** | Returns a candidate set or a coarser node instead of a single name. | Section 3 [S16–S19]. | B/C |

### 1.5 Implications for PkVision (sub-question 3)

- With per-cue accuracy of ~0.5, no composition trick can reach 60% exact. RCORE shows at most +4 points above independence. First raise per-attribute accuracy and add abstention (B, strong).
- Use **both** decoders. (i) Known tricks: a closed-set head over ontology entries, or a joint log-likelihood over their attribute signatures, masked by feasibility. (ii) Unseen tricks: fall back to the attribute tuple. Tune the seen-vs-novel switch on held-out *compositions* (calibrated stacking), never on test (B/C).
- Track **FSP**: how often a truly novel trick is snapped to a known ontology name. This is the specific failure an ontology decoder creates (B).

---

## 2. Procedure-aware and element-decomposition methods in fine-grained sports

### 2.1 Diving (closest analog: dive codes are compositional and machine-readable)

**Dive code structure.** A dive code encodes group (forward, back, reverse, inward, twisting, armstand), number of half somersaults, number of half twists, and position (A straight, B pike, C tuck, D free). Examples: `201B`, `5253B` [S13, S14, S15]. Diving48 factorises its 48 classes into takeoff (4) × somersault (8) × twist (8) × flight position (4) [S11, S26].

| System | Input / supervision | Takeoff or armstand | Rotation type / direction | Position | # somersaults | # twists | Exact-match full code | Grade |
|---|---|---|---|---|---|---|---|---|
| Nibali et al. 2017, multi-head 3D CNN, fixed camera, 4,716 training dives, **isolated** (GT tracking) | supervised | 100.0 | 89.81 | 90.78 | 86.89 | 95.15* | not reported | B |
| Same, **combined pipeline** (auto-localised) | supervised | 99.67 | 77.54 | 82.36 | 66.72 | 93.51* | not reported | B |
| C3D-MTL (Parmar & Morris CVPR 2019), MTL-AQA | supervised | 99.72 | 97.45 | 96.32 | 96.88 | 93.20 | not reported | B |
| **NS-AQA** (Okamoto & Parmar CVPRW 2024), MTL-AQA | **rules on 2D pose, no dive-label training** | 99.79 | 99.37 | 97.28 | 97.31 | 93.27 | not reported | B |

\*Twists are rare in Nibali's data. The paper plots a "mode" baseline, and the twist accuracy should be read against it (B).
Sources: [S14] Table 4 and Fig. 9; [S15] Table 3 (which also reprints the C3D-MTL and Nibali numbers on MTL-AQA).

- **Independence arithmetic** (my calculation, not reported by the papers): product of per-part accuracies = 0.875 (NS-AQA), 0.845 (C3D-MTL), 0.674 (Nibali isolated), 0.397 (Nibali combined). The last one shows how upstream localisation errors compound: a 2017 fixed-camera system with 87–95% isolated parts drops to ~40% implied full-code accuracy once end-to-end (B, derived).
- **NS-AQA's counters** (C, method): somersaults are counted as half-rotations of the pelvis→thorax vector crossing ±75° of vertical. Twists are counted as "petals" traced by the right-hip→left-hip vector, 0.5 twist per petal. Position comes from hip and knee angles, and group from facing plus initial rotation sense. The same system gives temporal segmentation AIoU@0.5 = 93.92 and @0.75 = 77.17, against TSA's 82.51 and 34.31. Experts agreed with its outputs ≥90% of the time and preferred its report over the neural score 96.1% of the time [S15]. The failure mode was bad pose under motion blur or half-submerged bodies [S15]. **This is the closest published precedent for PkVision's `rotation_tracker`-style cue extraction, and it is compositional by construction** (B).
- **FineDiving** (CVPR 2022 oral): 3,000 videos, 52 action types, 29 sub-action types, 23 difficulty degrees. It has a two-level semantic structure (sub-action types combine into an action type) and step-level temporal boundaries [S13]. Its procedure-aware TSA uses **the ground-truth dive number to select exemplars** ("w/ DN"). Spearman ρ is 0.9203 with DN vs 0.8925 without, and step segmentation AIoU@0.5/0.75 is 82.51/34.31 [S13]. FineDiving does not recognise the dive number (B). **Implication:** the diving AQA literature treats element identity as given (from the start list). Element recognition and quality are separate problems, and the recognition side is less studied (B).
- **Zero-shot VLM diving judging (Sep 2026).** Six open VLMs (Qwen2.5-VL 3–32B, InternVL3-8B, LLaVA-NeXT-Video, MiniCPM-V) prompted with phase-level rubrics give Spearman ≤ 0.32 against official scores. A supervised regressor on their text reaches 0.67 [S20]. This is about quality, not element identity, but it confirms raw VLM outputs are weak judges (B).
- **Diving48 attribute supervision**: attribute pre-training (takeoff, somersault, twist, position) is used as a mid-level target [S26]. Per-attribute Diving48 accuracies: not found.

### 2.2 Gymnastics and trampoline

- **FineGym** (CVPR 2020): 3-level hierarchy (10 events → 15 sets → 530 elements). Element labels were annotated through **expert decision trees of attribute questions**, so attribute ground truth exists [S21]. Event-level recognition reaches 98.47% top-1 and set-level 91.97%, but Gym288 element-level mean-class top-1 is only 46.5% (TSM two-stream). The listed failure modes include "degrees of rotation, counting", direction, and intense motion. Skeleton ST-GCN "performs poorly" because 2020-era pose failed on gymnastics [S21] (B).
- **TQN** (CVPR 2021): Gym99 90.6% per class / 93.8% per video, Gym288 61.9% / 89.6%, Diving48-v2 74.5% / 81.8% [S11] (B).
- **PoseC3D** (skeleton heatmap volumes, HRNet 2D pose): Gym99 mean-class top-1 93.8% (joint) and 94.1% (two-stream) [S22] (B). Modern 2D pose + temporal CNN solves *closed-set* FineGym-99.
- **Trampoline**: nearest-neighbour matching of 2D-pose joint-angle trajectories gets 80.7% over 20 skills (714 examples) from a single camera [S23] (B, older and small).
- **Held-out element generalization** on FineGym (train on some elements, test on unseen ones described by attributes): **not found**.

### 2.3 Figure skating (jump type × rotation count)

- **VIFSS** (arXiv Aug 2025). Element-level temporal action segmentation has 23 labels combining jump type and rotation count (for example "3 Axel", "4 Salchow"). With view-invariant contrastive pose embeddings it reaches frame accuracy 85.82% and F1@50 92.56%. The raw 2D pose baseline gets 71.34% and 78.78%, and lifted 3D pose gets 70.17% and 76.57% [S24] (B).
- **Unseen rotation count.** The 3D pose lifter (MotionAGFormer, trained on FS-Jump3D, which is mostly doubles and triples) failed on a **quadruple** toe loop absent from training. The authors attribute it to the rotation count and fps being outside the training distribution [S24] (B, qualitative). This is the only direct evidence I found on extrapolating to an unseen rotation count, and it is negative for learned temporal models.

### 2.4 Unseen-element generalization in sports: summary

Not found: any sports benchmark (diving, gymnastics, trampoline, figure skating) that holds out *elements* or *attribute combinations* and reports accuracy on them. Every sports number above is closed-set. The only approaches that generalize to unseen combinations **by construction** are rule/counter pipelines on pose (NS-AQA) and factorized heads decoded over a code grammar (Nibali). Neither paper evaluates held-out codes (B/C).

### 2.5 Implications

- For **countable kinematic attributes** (flip count, twist count, direction), a geometric counter on pose, NS-AQA style, is (a) compositional by construction, (b) label-free, and (c) the best measured analog (93–99% on diving). It depends entirely on pose quality in the air. The pose side is the twist-crux track (B).
- For **visual-context attributes** (wall, bar, vault, palm, kong entry), which are single-frame and appearance-heavy, a learned head is appropriate. Watch for the object-shortcut failure from §1.1 (B by analogy).

---

## 3. Selective prediction and abstention for judge-assist

### 3.1 Reference risk-coverage numbers (in-distribution)

| Setting | Full-coverage error | Operating points | Source | Grade |
|---|---|---|---|---|
| ImageNet ResNet-50 top-1, SGR, δ = 0.001 | ~25% | 5% error @ 48.8% cov; **10% @ 65.0%**; 15% @ 76.8%; 20% @ 86.8% | [S16] Table 5 | B |
| ImageNet VGG-16 top-5 | ~7% | 2% @ 53.5% cov | [S16] Table 4 | B |
| CIFAR-100 VGG-16 | ~25–30% | 10% @ 59.5%; 15% @ 67.5% | [S16] Table 2 | B |
| CIFAR-10 VGG-16 | 6.5% | 1% @ 78.6% | [S16] Table 1 | B |
| NExT-QA video QA, Gemini 2.0 Flash, verbalised confidence | 23.6% @ 98.7% cov | 9.4% @ 63.7% cov (ECE 0.018) | [S17] | B |

**Arithmetic bound** (exact): for any ranker, selective precision at coverage c ≤ min(1, acc/c). For ≥90% precision at 70% coverage you need full-coverage accuracy ≥ 63% even with a perfect confidence ranking. For twist at ≥85% precision and 70% coverage you need ≥ 59.5%. The empirical curves above (75% accuracy → 90% precision at ~65% coverage) suggest **~78–80% full-coverage accuracy** in practice for the 90%/70% operating point, and ~72–75% for the twist target (B, derived).

### 3.2 Under domain shift (the handheld-iPhone case)

- **Skeleton recognizer in a real gym (2026).** Trained on NTU-120 at 63.2%, it drops to 1.6% zero-shot on Gym2D, which is RTMPose 2D pose from real gym footage. Risk at 50% coverage is ~98–99% for MSP, MC dropout, deep ensembles and temperature scaling, although ID-vs-OOD AUROC is 0.94 for MSP and 0.98 for the ensemble. Fine-tuned gating on in-domain data raises accuracy to 27.0 ± 0.6% and lowers risk@50% to 68.2%. Energy and Mahalanobis scores are the best gates [S18] (B, very close to PkVision's input modality).
- **VLM confidence is not epistemic.** Median self-reported confidence stays at 0.9 when video evidence drops from 18 to 6 frames [S17] (B).
- **Implication:** abstention thresholds (and conformal calibration, which assumes exchangeability) must be fitted on data from the *deployment* distribution (iPhone footage), not on parkourtheory clips. "High OOD AUROC" is not evidence of safe selection (B).

### 3.3 Conformal and hierarchical set-valued outputs

- **Conformal Structured Prediction** (Zhang, Li, Bastani; arXiv 2410.06296): prediction sets for structured outputs, represented as a few DAG nodes (coarse labels that implicitly contain their fine descendants), with a marginal coverage guarantee [S19a] (C, method).
- **Hierarchical Conformal Classification** (arXiv 2508.13288, 2025): sets made of nodes at mixed levels of a class hierarchy with preserved coverage. A user study found annotators **significantly prefer hierarchical over flat sets** [S19b] (B for the user-study finding, C otherwise).
- **SCoRE** (arXiv 2603.24704, 2026): selective conformal risk control with e-values, giving finite-sample control of the risk among trusted cases for any bounded risk [S19c] (C).
- **Conformal prediction for HAR with CLIP-style VLMs** (arXiv 2502.06631): CP shrinks candidate sets substantially at guaranteed coverage, but set sizes are long-tailed. Softmax temperature tuning trades average size against the tail [S19d] (B, numbers not extracted).
- **Multi-attribute joint coverage:** a published method for conformal sets over *several discrete attributes jointly* in action or sports recognition was **not found**. The practical choices are: (i) one conformal score on the joint composition (for example 1 − decoded probability of the composition) over the feasibility-masked product space, or (ii) Bonferroni-split per-attribute sets (α/k each), which is conservative (C).

### 3.4 What this means for a judge-assist

- The workflow "system proposes, judge confirms" suits **set-valued output**: a top-3 or conformal set of trick names, and when the set is too large, a coarsened name ("backflip, twist ∈ {1, 1.5}, off wall"). HCC's user study supports that humans prefer this (B/C).
- Coverage is the budget of judge attention. 70% auto-confirmable and 30% flagged is a reasonable product target, but the error guarantee only holds for exchangeable calibration data (C).

---

## 4. Measuring open-vocabulary success

### 4.1 What the field does

| Protocol element | Where it comes from | Grade |
|---|---|---|
| Split the **label space**, not just clips: train compositions and unseen test compositions are disjoint, and all primitives are seen in training | CZSL generalized splits (TMN, ICCV 2019) [S10]; Something-Else compositional split [S7]; ZS-CAR Sth-com [S1] | B |
| **Validation set with its own unseen compositions**, used to tune the seen/unseen bias and thresholds, never the test unseen set | TMN [S10]; CSP protocol [S5]; RCORE's "unbiased" protocol forbids test-tuned bias [S2] | B |
| **Generalized** evaluation (seen and unseen together) with Seen, Unseen, **harmonic mean**, and **AUC** over a bias sweep | Chao et al. 2016; Xian et al. TPAMI 2018 (per-class averaged accuracy + HM) [S12]; CZSL standard [S5, S6] | B |
| **Open-world**: predict over the full primitive Cartesian product, not only the feasible listed compositions | Mancini et al. CVPR 2021 [S9]; RCORE uses it by default for video [S2] | B |
| **Shuffled vs compositional split gap**: same videos, random vs composition-disjoint split | Something-Else (−15 points for I3D) [S7] | B |
| **Diagnostics**: FSP (novel predicted as seen), FCP (collapse onto frequent seen), **ΔCG** = exact − Π(primitive accuracies) | RCORE [S2] | B |
| **Partial credit**: per-primitive accuracy; hierarchical precision/recall for less-specific but correct answers | ZS-CAR reports verb@ and object@unseen [S2]; taxonomy-aware evaluation (arXiv 2504.05457) [S25] | B/C |
| Exact-match ("subset accuracy") vs Hamming accuracy | Standard multi-label metrics. Composition top-1 is exact match | C |

### 4.2 Recommended evaluation protocol for PkVision's open-vocabulary claim

1. **Canonical code first.** Map every ontology entry and every label to a canonical attribute tuple (see §5.5). Evaluate on tuples, then render names.
2. **Three test splits, frozen before any model is run:**
   - **T-seen**: held-out *clips* of tricks whose tuple appears in training.
   - **T-unseen**: held-out *tuples*. Every clip of those tricks is removed from train and val, and each primitive value appears ≥ N times in training. Stratify by distance to the nearest training tuple (1 attribute changed vs ≥2). **Only this split supports the "open-vocabulary" claim.**
   - **T-own**: own iPhone footage (domain shift). It is reported separately and never pooled.
   - Plus **V-unseen**, a separate set of held-out tuples in validation, used to tune the seen/novel bias, abstention thresholds and conformal calibration.
3. **Open-world decoding.** The decoder searches the full feasibility-masked product space, not just the 2,677 listed tricks. Report the **FSP rate** on T-unseen.
4. **Metrics, each with Wilson 95% CIs:**
   - Per attribute: balanced accuracy (twist = 0 dominates), selective precision at the stated coverage, the risk-coverage curve and AURC. Report twist separately for ≥1.5 twists and for corks, as the framing already requires.
   - Composition: exact-match top-1 and top-3 on T-seen and T-unseen, their **harmonic mean**, and ΔCG. Report exact-match both at full coverage and at the selective operating point.
   - Partial credit: the distribution of the number of correct parts, plus hierarchical precision/recall.
   - Product metric: D-score absolute error and the % of tricks within the tolerance a judge would accept. The end product is judge-assist, and exact naming can be stricter than D needs.
5. **Equivalence classes.** Count a prediction as correct if it lands in the ground truth's equivalence class (15 known aliases; the 13 cue-degenerate pk_basics collapse to one class unless context disambiguates). Otherwise the metric penalises what attributes cannot express. The in-project oracle is only 73.1% top-1 on FIG-149 (A, internal).
6. **Label quality gate.** Eval labels on twist must be *verified* (the owner's verify-only queue, slowed or two-view playback). With ~35% twist error in the current "gold 99", a true 90%-precise model would measure around 60%, so the threshold cannot be tested on those labels (A internal, derived).
7. **Sample sizes** (Wilson, derived):
   - To show a lower bound ≥ 85% at an observed 90% precision you need ~184 *accepted* items per attribute.
   - To show exact-match ≥ 50% at an observed 65% you need ~41 clips per split.
   - The ~13 own-footage clips give a 95% CI of [29%, 77%] for 7/13 and [67%, 99%] for 12/13. Treat them as a smoke test, not evidence.

---

## 5. Parkour and tricking notation grammars

### 5.1 FIG Parkour Table of Tricks 2025 (primary in-domain source) [S27]

I extracted the text of the official PDF ("PK Code of Points 2025-2028 – Table of Tricks 2025", April 2025, Parkour Technical Committee). Grade A as a primary document; none of it is a measurement.
- **D-score reference list:** guiding values for tricks "in their most basic form" in 4 columns: **Swing Moves, Wall Moves, Acrobatics Moves, PK Basics**. Values run from 0.1 (Stride, Drop, Roll, Precision) to 7.7 (Swing Double Gainer 1080 "Miller").
- **Judges upscale** the base value by **Placement** (travel distance or height, narrow take-off, narrow or elevated landing), **Form** (layout, pike, pistol, spider or stall shapes; full-up and full-down twist timing; reversed twist direction such as "unfull"; added touchdowns or kicks), **Entry** (a difficult move flowing directly in; round-off, scoot, cartwheel and kip do not count) and **Exit** (a difficult move directly after).
- **Slanted axis:** "moves performed out of the longitudinal plane in slanted axis are decreased by 0.5 points". Repeated tricks do not count "even if they differ in form, entry, placement, or exit". Failed tricks (lying landing or major crash) do not count in D.
- **Naming is semi-compositional with no formal grammar.** Names follow the pattern [context/entry prefix] [multiplicity] [base] [twist]:
  - Context/entry prefixes: Swing, Wall, Palm, Castaway, Gaet Pimp, Hang, Caster, Cast, Kong, Sitting dash, Handstand, Pop, Pimp.
  - Multiplicity: Double, Triple, Quad.
  - Base: Backflip, Frontflip, Sideflip, Gainer, Cork, Kroc, Arabian, Webster, Raiz, Tsukahara, Frisbee, B, A.
  - Twist in degrees, with the count in brackets from 720 up: 180, 360, 540, 720 (2 twist), … 1440 (4 twist).
  - Many entries are idiomatic and do not follow the pattern: Raiden, Ginger, Geinger, Devil drop, Roll Bomb, Tunnel Flip, Kash/Dong vault, Italian Job, "Miller".
- **Implications:**
  - The **context/entry** attribute carries much of D. Examples: Backflip 1.5, Palm Backflip 2.0, Castaway Backflip 2.6, Swing Castaway Backflip 2.7, Wall Backflip 1.3. It needs more than 4 values (ground/wall/bar/vault).
  - **Axis** directly moves D (−0.5 slanted), and "cork" names change the base.
  - **Twist resolution is 180°**, matching the half-twist attribute.
  - The final D-score is **not** a function of the name alone, because judges apply discretionary upscaling. A tuple-to-name decoder can only propose the *base* value (A document, C for the implication).

### 5.2 Analog machine-readable grammars (templates to copy)

- **FIG trampoline shorthand:** a leading digit (or digits) gives quarter somersaults, then one digit per somersault gives half twists in it, then a position symbol (`o` tuck, `<` pike, `/` straight). Examples: `42/` is a back somersault with full twist, straight; `800o` is a double back tuck; `821/` is a double back with a full twist in the first somersault and a half in the second [S28] (C, rule definition from USA Gymnastics and Wikipedia; stable, well known).
- **Diving dive numbers:** group digit, a flying/half-somersault field, a half-twist digit for twisting dives, and a position letter A–D. The degree of difficulty is looked up from the code (FINA/World Aquatics tables) [S13, S14, S15] (B/C).
- Both grammars encode **flip count, twist distribution and position**, and diving adds **direction/group**. Neither encodes context/apparatus entry. Trampoline's per-somersault twist placement is richer than PkVision's single twist count and resolves "full-in" vs "full-out", which FIG PK scales under Form (C).

### 5.3 Tricking

- Community convention names vertical kicks by **takeoff** (pop, cheat, swing, …), **rotational value** (540, 720, …) and **kick/landing**, with modifier prefixes such as swipe, gyro, missleg, switch, hyper [S29] (C).
- "There are no official or standardized names in tricking", and no governing body exists [S29] (C).
- Machine-readable tricking grammar: **not found**.

### 5.4 Parkour Theory (source of the 2,677-trick ontology)

- A community reference that links each move to related moves, in the spirit of The Tricking Bible. It began as a MySQL database and v3 added 200+ moves [S30] (C; the primary Medium post returned 403, so these facts come from search snippets).
- Public API, dump or formal notation: **not found**. The `parkourtheory` GitHub account has no public repositories. A GitHub search for "parkour trick" found only a small tracker repo and PkVision itself.

### 5.5 Verdict

- A machine-readable parkour or tricking trick grammar in the academic or technical literature: **not found**. PkVision's compositional trick-name parser (flip/direction precision 0.98) appears to be novel (A internal / C).
- **Recommendation:** define a canonical code modelled on trampoline and diving, for example `{context}.{entry}.{dir}.{axis}.{flips}.{twist_halfs[/distribution]}.{shape}`. Make it the single join key for (a) the 2,677 ontology, (b) FIG-149 with base D, and (c) labels. Store feasibility rules (for example cork ⇒ off-axis and twist ≥ 1; "Double" ⇒ flips = 2) as the KG-SP-style mask (C).

---

## 6. Assessment of the proposed thresholds

| Threshold (from the framing doc) | Field anchor | Verdict | Suggested change |
|---|---|---|---|
| Per attribute: precision ≥ 90% at ≥ 70% coverage (flip, direction, context) | Needs ~78–80% full-coverage accuracy (SGR curves; bound ≥ 63%). Constrained analogs reach 93–99% at full coverage (diving: NS-AQA, C3D-MTL). Real-footage 2D-pose models can be confidently wrong under shift [S18]. | **Realistic in principle, demanding for handheld parkour.** About the right bar for a judge-assist. | Keep. State the coverage as the fraction of *clips*. Add a lower-CI requirement (for example 95% lower bound ≥ 85%, which needs ~184 accepted items). Use balanced metrics. Calibrate thresholds on iPhone-distribution data. |
| Twist: ≥ 85% at ≥ 70%, measured separately on ≥1.5 twists and corks | Diving twist count 93% (fixed broadcast view, 10 m platform). No analog for off-axis corks: not found. Figure-skating rotation count degrades on unseen counts [S24]. | Reasonable as the gate, but **not measurable today**: gold twist labels are ~35% wrong. | First build a verified twist eval set (verify-only). Allow **set-valued twist** output (for example {1.5, 2}) for the ≥1.5 stratum and score set coverage and size, not just point precision. |
| Compositional: exact ≥ 60% top-1 on held-out parkourtheory **clips** | Best open-world unseen-composition video result: 44% (Sth-com). Small-vocabulary image result: 64% (UT-Zappos, 2025). Random vs composition-disjoint split gap: 7–15 points [S7]. Independence: 5 attributes at 0.90 give 0.59. | **Ambiguous and partly inconsistent.** On held-out *clips* of mostly seen tricks it is an in-distribution test, not open-vocabulary. At full coverage it implies every attribute ≥ ~0.90 at full coverage, which is stricter than the per-attribute rule. On truly unseen tuples no video system has reached 60% open-world. | Split into **T-seen** and **T-unseen** (§4.2). Suggested bars: T-seen exact ≥ 60% at full coverage *or* ≥ 85% exact precision at ≥ 50% clip coverage; T-unseen exact ≥ 40–50% at full coverage (above the best published video ZS-CAR); report HM and FSP. Score modulo equivalence classes. |
| Compositional: exact ≥ 80% top-3 | In-project oracle ceiling on FIG-149: 85.1% top-3 with *perfect* attributes (A internal). Something-Else compositional top-5 ~72–83% [S7]. | **Too strict** if scored on raw FIG names, since it demands ~94% of the oracle ceiling. More reasonable over equivalence classes or canonical tuples. | Define the label space (tuples vs 2,677 names vs FIG-149). Normalise against the oracle ceiling: "top-3 ≥ 0.9 × oracle top-3". |
| Own footage: pass on the majority of ~13 clips | 7/13 has a 95% CI of [29%, 77%] | **Uninformative** as a pass/fail criterion. | Keep as a smoke test. Require ≥ 40 own clips (or a two-phone filming session of ~50 labelled-by-performance tricks) before any GO. |

Overall:
- The per-attribute bar is well chosen. The problem is measurability: labels and sample size.
- The compositional bar should (1) separate seen from unseen compositions, (2) state its coverage, (3) be scored modulo aliases, and (4) be normalised to the oracle ceiling.
- I would also add a **product bar**: D-score base value within ±0.1 of the judge's base value on ≥ 80% of *accepted* tricks. That is what a judge-assist is used for.

---

## 7. Recommended architecture: from attributes to name

1. **Cue extraction with explicit abstention per attribute:**
   - *Geometric counters on pose* for flip count, twist half-count and direction, following NS-AQA's vector-rotation counting: compositional by construction and label-free.
   - *Learned heads* for context/entry and axis, trained in the **partial-supervision** regime (KG-SP pCZSL), since weak labels only cover some cues.
   - Each attribute outputs a calibrated distribution (temperature-scaled on verified iPhone data) or a set.
2. **Joint decoding over a feasibility-masked product space.** Score = Σ log p(attr) + log prior(tuple) + seen/novel bias γ.
   - The prior comes from the ontology (known tricks), with a floor for feasible-but-unlisted tuples.
   - Hard rules come from the canonical-code grammar (§5.5).
   - Tune γ on **V-unseen** (calibrated stacking) so unseen tricks are not collapsed onto known names; monitor FSP.
3. **Known-trick head as a second opinion.** A closed-set head over frequent known tricks (TQN's global query) raises seen-class accuracy well above factorized decoding (81.8 vs 50.3 on Diving48). Fuse it with the attribute decoder only on the "known" branch.
4. **Output a set, not a point.** Return a conformal or top-k set of canonical tuples and render names. When the set is large, **coarsen along the hierarchy** (HCC / conformal structured prediction). Map each tuple to its FIG base D. A human confirms.
5. **Why this pattern:**
   - (a) Evidence shows the composition step adds at most a few points (ΔCG ≈ 0 to +4), so effort belongs in per-attribute accuracy and calibrated abstention.
   - (b) Feasibility masking is the standard, cheap open-world fix.
   - (c) Rule-based counters are the only measured approach that generalizes to unseen combinations by construction, and they hit 93–99% on the closest analog.
   - (d) Sets and coarsening turn abstention into useful partial output, which judges prefer.

---

## 8. Gaps (not found)

- A benchmark or result for **held-out element / attribute-combination generalization in acrobatic sports** (diving, gymnastics, trampoline, figure skating, parkour).
- **Full-code exact-match** accuracy for dive-code recognition. Only per-part accuracies are reported.
- A **machine-readable parkour or tricking grammar**, or a public Parkour Theory dump or API.
- **Conformal or selective prediction for multi-attribute sports-element recognition**, and joint multi-attribute conformal sets in action recognition generally.
- Any per-attribute accuracy for **off-axis (cork) twist counting** in any sport.
- Per-attribute accuracies on **Diving48**.
- Compositional zero-shot **skeleton-based** action recognition with held-out attribute combinations. Skeleton zero-shot work (for example GenPrior, arXiv 2608.02236) holds out whole class names on NTU and PKU-MMD, not attribute tuples.
- Numbers for KG-SP and for the CP-for-HAR set sizes were not extracted (only the methods are cited).

---

## 9. Sources

| ID | Source | Venue / date | Link | Used for | Grade |
|---|---|---|---|---|---|
| S1 | Li et al., "C2C: Component-to-Composition Learning for Zero-Shot Compositional Action Recognition" | ECCV 2024 (arXiv Jul 2024) | arXiv:2407.06113 | Sth-com benchmark, closed-world numbers | B |
| S2 | Ahn et al., "Why Can't I Open My Drawer? Mitigating Object-Driven Shortcuts in Zero-Shot Compositional Action Recognition" (RCORE) | arXiv Jan 2026 | arXiv:2601.16211 | Open-world ZS-CAR numbers, ΔCG, FSP/FCP, EK100-com | B |
| S3 | Ye et al., "Zero-shot Compositional Action Recognition with Neural Logic Constraints" (LogicCAR) | arXiv Aug 2025 | arXiv:2508.02320 | Logic constraints; HM 45.2 (via S2) | B |
| S4 | "FlowComposer: Composable Flows for Compositional Zero-Shot Learning" (HKUST) | arXiv Mar 2026 | arXiv:2603.16641 | 2024–2026 CZSL closed/open-world table (incl. LOGICZSL CVPR25, PLO ACMMM25, CDS-CZSL CVPR24) | B |
| S5 | Nayak, Yu, Bach, "Learning to Compose Soft Prompts for Compositional Zero-Shot Learning" (CSP) | ICLR 2023 | arXiv:2204.03574 | Closed vs open-world drop; bias-sweep protocol; feasibility calibration | B |
| S6 | Munir et al., "Compositional Zero-Shot Learning: A Survey" | arXiv Oct 2025 | arXiv:2510.11106 | Taxonomy; open-world trends | C |
| S7 | Materzynska et al., "Something-Else: Compositional Action Recognition with Spatial-Temporal Interaction Networks" | CVPR 2020 | arXiv:1912.09930 | Shuffled vs compositional split gap | B |
| S8 | Karthik, Mancini, Akata, "KG-SP: Knowledge Guided Simple Primitives for Open World CZSL" | CVPR 2022 | arXiv:2205.06784 | Independent primitives + feasibility prior; partial supervision | B/C |
| S9 | Mancini et al., "Open World Compositional Zero-Shot Learning" | CVPR 2021 | arXiv:2101.12609 | Open-world protocol, feasibility | B/C |
| S10 | Purushwalkam et al., "Task-Driven Modular Networks for Zero-Shot Compositional Learning" | ICCV 2019 | arXiv:1905.05908 | Generalized CZSL splits with validation unseen compositions | C |
| S11 | Zhang, Gupta, Zisserman, "Temporal Query Networks for Fine-grained Video Understanding" | CVPR 2021 | arXiv:2104.09496 | Attribute-query factorisation; Diving48/FineGym numbers; product decode 50.3% | B |
| S12 | Xian et al., "Zero-Shot Learning – A Comprehensive Evaluation of the Good, the Bad and the Ugly"; Chao et al., "An Empirical Study and Analysis of Generalized Zero-Shot Learning for Object Recognition in the Wild" (ECCV 2016) | TPAMI 2018; ECCV 2016 | arXiv:1707.00600; arXiv:1605.04253 | GZSL harmonic mean, per-class accuracy, calibrated stacking | C |
| S13 | Xu et al., "FineDiving: A Fine-grained Dataset for Procedure-aware Action Quality Assessment" | CVPR 2022 | arXiv:2204.03646 | Dataset stats; TSA uses GT dive number; AIoU | B |
| S14 | Nibali, He, Morgan, Greenwood, "Extraction and Classification of Diving Clips from Continuous Video Footage" | CVPR Workshops 2017 | arXiv:1705.09003 | Multi-head dive-code classification, per-part accuracy | B |
| S15 | Okamoto, Parmar, "Hierarchical NeuroSymbolic Approach for Comprehensive and Explainable Action Quality Assessment" (NS-AQA) | CVPR Workshops 2024 | arXiv:2403.13798; code github.com/laurenok24/NSAQA | Rule-based somersault/twist counters, per-attribute accuracy on MTL-AQA (incl. C3D-MTL, Parmar & Morris CVPR 2019, arXiv:1904.04346) | B |
| S16 | Geifman, El-Yaniv, "Selective Classification for Deep Neural Networks" (SGR) | NeurIPS 2017 | arXiv:1705.08500 | Risk-coverage reference numbers | B |
| S17 | Ortiz, "Explicit Abstention Knobs for Predictable Reliability in Video Question Answering" | arXiv Dec 2025 / Jan 2026 | arXiv:2601.00138 | VLM abstention in- and out-of-distribution | B |
| S18 | Khanal, Zhou, "Severe Domain Shift in Skeleton-Based Action Recognition: A Study of Uncertainty Failure in Real-World Gym Environments" | arXiv Mar 2026 | arXiv:2603.15574 | Confidently-wrong skeleton models under 2D-pose shift | B |
| S19a | Zhang, Li, Bastani, "Conformal Structured Prediction" | arXiv Oct 2024 | arXiv:2410.06296 | DAG/hierarchical conformal sets | C |
| S19b | den Hengst et al., "Hierarchical Conformal Classification" | arXiv Aug 2025 | arXiv:2508.13288 | Mixed-level sets; user preference | B/C |
| S19c | Bai, Jin, "Conformal Selective Prediction with General Risk Control" (SCoRE) | arXiv Mar 2026 | arXiv:2603.24704 | Selective risk control | C |
| S19d | Bary, Fuchs, Macq, "Conformal Predictions for Human Action Recognition with Vision-Language Models" | arXiv Feb 2025 | arXiv:2502.06631 | CP set sizes for HAR, temperature | B (numbers not extracted) |
| S20 | Velesaca et al., "Can Vision-Language Models Judge Olympic Diving? From Reasoning to Scores in Zero-Shot AQA" | arXiv Sep 2026 | arXiv:2609.19354 | Raw VLM Spearman ≤ 0.32 | B |
| S21 | Shao, Zhao, Dai, Lin, "FineGym: A Hierarchical Video Dataset for Fine-grained Action Understanding" | CVPR 2020 | arXiv:2004.06704 | Decision-tree attribute labels; element-level difficulty | B |
| S22 | Duan et al., "Revisiting Skeleton-based Action Recognition" (PoseC3D), pyskl model zoo | CVPR 2022 | arXiv:2104.13586; github.com/kennymckormick/pyskl/blob/main/configs/posec3d/README.md | FineGym-99 skeleton 93.8–94.1% | B |
| S23 | Connolly, Silvestre, Bleakley, "Automated Identification of Trampoline Skills Using Computer Vision Extracted Pose Estimation" | IMVIP 2017 | arXiv:1709.03399 | 80.7% on 20 skills | B |
| S24 | Tanaka, Suzuki, Fujii, "VIFSS: View-Invariant and Figure Skating-Specific Pose Representation Learning for Temporal Action Segmentation" | arXiv Aug 2025 | arXiv:2508.10281 | Jump type × rotation element TAS; unseen-quad failure | B |
| S25 | Snæbjarnarson et al., "Taxonomy-Aware Evaluation of Vision-Language Models" | arXiv Apr 2025 | arXiv:2504.05457 | Hierarchical precision/recall for partial credit | C |
| S26 | Kanojia et al., "Attentive Spatio-Temporal Representation Learning for Diving Classification"; Diving48 dataset page (Li et al., RESOUND, ECCV 2018) | CVPR Workshops 2019; ECCV 2018 | gagankanojia.github.io/files/CVPRW2019.pdf; svcl.ucsd.edu/projects/resound/dataset.html | Diving48 attribute structure (4/8/8/4); attribute pre-training | C |
| S27 | FIG Parkour Technical Committee, "PK Code of Points 2025-2028 – Table of Tricks 2025" | FIG, April 2025 | gymnastics.sport/publicdir/rules/files/en_1.1.1%20-%20PK%20Code%20of%20Points%202025-2028%20-%20Table%20of%20tricks%202025.pdf | D-score reference list, scaling rules, naming | A (document) |
| S28 | USA Gymnastics "Trampoline Difficulty" form; Wikipedia "Trampolining terms" | n.d. | static.usagym.org/PDFs/Forms/T&T/DD_TR.pdf; en.wikipedia.org/wiki/Trampolining_terms | FIG trampoline shorthand | C |
| S29 | Loopkicks "What is Tricking?"; Tricking Wiki (Fandom) "Kicks"; Wikipedia "Tricking (martial arts)" | 2021; n.d. | loopkickstricking.com/learn-tricking/how-to-start-tricking-a-comprehensive-beginners-guide; tricking.fandom.com/wiki/Kicks; en.wikipedia.org/wiki/Tricking_(martial_arts) | Tricking naming conventions, no standard | C |
| S30 | Chen, "17 Years of Parkour and the 9 Year Anniversary of Parkour Theory" (Medium; content seen via search snippet only, page returned 403) | n.d. | medium.com/@ch3njust1n/17-years-of-parkour-and-the-9-year-anniversary-of-parkour-theory-e5efd2de68ce | Parkour Theory origins | C |
| S31 | Lampert, Nickisch, Harmeling, "Attribute-Based Classification for Zero-Shot Visual Object Categorization" (DAP) | TPAMI 2014 | doi:10.1109/TPAMI.2013.140 | Attribute-signature decoding | C |
| Internal | PkVision project memory and framing doc: oracle FIG decoder 73.1% top-1 / 85.1% top-3; per-cue ~0.5 → 13.6% end-to-end; gold labels ~17% (flip) / ~35% (twist) wrong; parser flip/dir precision 0.98 | 2026 | docs/science-superpowers/questions/2026-09-28-pkvision-feasibility.md | Ceilings and label-noise arguments | A (internal) |
