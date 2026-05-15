# PkVision — Parkour Trick Recognition Design (Approach A′)

**Date:** 2026-05-15
**Status:** Validated design (brainstorming complete). Ready for implementation planning.
**Supersedes strategy in:** `NEXT_SESSION_PROMPT.md`, `CLAUDE_CODE_PROMPT_2026-04-12.md` (kept as history).

## 1. Problem & Goal

Automated FIG Parkour competition judging: input a run video (single iPhone near-term; synced multi-camera deferred — no 2nd device available now) → identify each trick from the FIG Code of Points 2025 (149 tricks across 4 categories: acrobatics / wall / swing / pk_basics) → output the official D-score. Must generalize across a wide trick variety on real competition footage.

**Build scope:** full-run scoring end-to-end, with the recognition core kept independently measurable inside the pipeline.

## 2. Context: why prior approaches failed (do not revisit)

All scored <20% top-1 on a 13-clip benchmark: GVHMR/SMPL 3D physics (AMASS has zero acrobatics), metric learning (R@1=9%, collapse), VideoMAE attribute classifier (~55% mean, twist collapses), CLIP zero-shot (0% exact), frontier VLM judges (0/13, 1/5, 0/7 — fail twist counting + takeoff disambiguation). May-2026 external research verdict: VLM-as-judge and monocular 3D HMR for acrobatics are dead ends; the evidence-backed direction is 2D-skeleton structured-cue prediction + a structured FIG decoder (cf. VIFSS 92% F1 figure-skating analog, FineGym gym288).

## 3. Critical finding from independent cross-model challenge (Codex + Gemini 3.1 Pro)

The "decoder works on oracle cues (8–9/9) → only cue extraction remains" premise is **circular and currently unproven**: `core/recognition/fig_decoder.py` has `CUE_WEIGHTS` + a hardcoded `CANONICAL_NAMES`/`CANONICAL_BONUS=0.5` hand-tuned toward POOL-B ground truth (in-distribution tuning, N=13). Independently corroborated by the internal benchmark-exploration pass. **Consequence:** the decoder must be validated blind before any model is built. Both models returned RECONSIDER → reshaped into Approach A′ below.

## 4. Architecture (A′)

Pipeline shape (≈70% reuses existing, working code):

```
RAW RUN VIDEO
 [1] YOLO/RTMPose athlete track            core/video.py            REUSE
 [2] Inversion trick segmenter             core/segmentation.py     REUSE
 [3] Per-segment skeleton seq → normalize (hip-center, torso-scale), conf as channel, resample T≈64   NEW (small)
 [4] CUE-EXTRACTION CORE → {context, direction, flip, twist, takeoff} + calibrated confidences        NEW (only new ML)
 [5] Frozen, P0-validated FIG decoder (constrained reranker, abstention-aware)   core/recognition/fig_decoder.py   REUSE
 [6] Run assembler → real CompetitionScorer (top-3 unique adjusted)              core/scoring/competition.py        REUSE
```

Recognition core = [3]→[5] on one segment, benchmarked standalone via a new `SkeletonRecognizer` (one class + one `build_recognizer()` branch in `paper/experiments/recognizers.py`) through `run_structured.py`. Full-run = [1]→[6]; segmentation precision/recall reported separately from recognition.

## 5. Data plan (honest)

- **No skeleton corpus exists yet.** Prior CPU extraction died at clip ~200.
- **Trustworthy:** the 1,618 single-trick parkourtheory clips; `data/fig_tricks_2025.json` (149-trick ontology: flip/twist/direction/axis/score/aliases/disambiguation groups) — the only sound label/decoder authority.
- **Not trustworthy:** per-clip flip/twist/direction labels (only 99/1618 FIG-grounded; rest filename-parsed; twist=1.5 has 2 samples); `fig_to_parkourtheory_map_v2.json` "100% mapped" is fake many-to-one.
- **Gold supervision** = the 99 FIG-grounded clips + the ~75 P0 blind-annotated clips, only.
- **De-contamination guard (hard requirement):** code-enforced filter excluding `final_clips/`, `vlm_clips/`, `run_testing/`, any `test_*` slug, same-type near-dupes (`*_in_back_out`…). Scripts assert it or refuse to run.
- **Serialization:** keypoint corpus stored as numeric `.npz` shards + a JSON manifest. **No pickle anywhere** (arrays + JSON only) — safe, resumable, inspectable.
- **Gate set** = de-contaminated held-out parkourtheory split (stratified by FIG family) + 13 POOL-B. **Demo set** = 4 POOL-A runs. Immutable; never tuned on.

## 6. Cue-extraction model & training

- **Pose substrate (audited, not assumed):** one-time offline Colab extraction with the *heaviest* estimator (RTMPose-x vs YOLO11x-pose, chosen by a 20–30 hard-clip head-to-head). Standardized COCO-17 `(T,17,3)`, written as resumable/checkpointed `.npz` shards + JSON manifest. Hard gate: if even the heavy model's keypoint confidence collapses through the flip/twist apex, that is a surfaced decisive negative finding (no workarounds).
- **Train/inference pose parity (no skew):** the skeleton sequence fed to the cue model (step [3]) is *always* produced by this same heavy estimator, at both training and inference. The fast tracker in `core/video.py` (step [1]) is used only for athlete tracking + segmentation signals, never as the cue-model input.
- **Backbone:** existing `SkeletonTransformer` scaffold (~0.5–2M params), input normalized `(T≈64,17×3)` with confidence as an input channel.
- **Representation = self-supervised** Masked Skeleton Modeling on all ~1618 *unlabeled* sequences (+ FineGym gym288 as *unlabeled* auxiliary only; promote to supervised flip/twist source only if the pose-audit proves domains comparable). The filename fuzzy-matcher (`overlap>=0.6`) is removed entirely.
- **Cue heads:** freeze backbone; small heads (context-4, direction-3, flip-bin, twist-bin, takeoff) trained **strictly on gold**. Class-weighted losses. Calibrated confidences (temperature scaling); below threshold → abstain (`None`). Twist head expected to abstain often on single camera (honest, accepted).
- **Decoder:** frozen at P0-validated state; constrained top-k reranker; narrows on known cues when others abstain; D-score via real `CompetitionScorer`.

## 7. Evaluation & gates

- **P0 blind decoder gate (decisive, cheap, first):** freeze decoder+CANONICAL_NAMES+ontology; ~50–75 fresh stratified cue annotations; frozen decoder + ablations (no canonical bonus / no disambiguation / ontology-only). Metrics: top-1, top-3, D-score MAE, confusion-group accuracy. **Gate: blind top-1 ≥ ~80% → proceed; else rework decoder before any ML.**
- **Recognition-core gate (post-model):** cue→decoder top-1/top-3 + per-cue accuracy on gate set, vs reproduced ~15% baseline and oracle-cue ceiling; report N + CIs; exact-clip vs same-type separated.
- **Full-run demo:** 4 POOL-A runs, order-aware sequence accuracy + D-score error; segmentation P/R separate. Reported, not gated (N=4).

## 8. Failure modes (fail-fast)

Abstention on low-confidence cues (graceful decoder degradation); pose collapse surfaced not patched; de-contamination asserted in code; Colab jobs checkpointed; seeds fixed; decoder output-hash pinned post-P0 to catch accidental retuning. Every gate has an explicit fail-branch that changes the plan.

## 9. Build sequence

| Phase | Exit / decision gate |
|---|---|
| **P0** Blind decoder validation (no model) | GATE: blind top-1 ≥ ~80% or decoder-rework branch |
| **P1** Heavy offline pose + audit, extract 1618 → `.npz` shards + manifest | Audited apex-confidence report; decisive-negative branch if collapse |
| **P2** Self-supervised pretrain (MSM) on ~1618 unlabeled (+FineGym aux) | Backbone checkpoint + reconstruction sanity |
| **P3** Calibrated cue heads on gold only, frozen backbone | Per-cue accuracy table on gate set |
| **P4** Wire SkeletonRecognizer→decoder→CompetitionScorer; core gate | GATE: clear beat over ~15% baseline + per-cue diagnostics → overall go/no-go |
| **P5** Full-run demo (4 runs); conditional retrieval/multimodal fusion only for cues P4 proves weak | Scorecard + decision on C-evolution |

Compute: Google Colab Pro via the `colab` MCP (CLI/notebook fallback); all phases notebook-runnable and checkpoint-friendly.

## 10. Testing (critical-path only)

Unit: de-contamination filter (assert known benchmark slugs excluded — eval-integrity-critical), cue normalization, FIG-alias mapping, decoder cue contract, abstention thresholding. Integration: `SkeletonRecognizer.recognize()` contract + end-to-end smoke (1 clip + 1 short run). Regression: frozen gate sets, per-cue accuracy across phases, decoder hash pinned. Seeds fixed, eval sets immutable.

## 11. Accepted limitations / open questions

- Single-camera twist is a geometric ceiling for multi-twist tricks (Backflip 360/540/720…); twist head abstains rather than guesses. Quantified at P0/P4; the known payoff of the deferred 2-camera path is what it would unlock.
- FineGym→parkour domain gap: mitigated by using FineGym as unlabeled SSL only unless the pose audit proves otherwise.
- Tiny eval (13+4): gates are directional; durable proof needs the deferred human-in-the-loop labeling investment (revisit after P4).
- P0 outcome may force a decoder rework — that is an accepted, valuable branch, not a failure.
