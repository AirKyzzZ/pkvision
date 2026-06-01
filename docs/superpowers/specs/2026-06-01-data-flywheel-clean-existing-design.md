# PkVision Data Flywheel — "Clean-Existing" Verification Loop (v1)

**Date:** 2026-06-01
**Status:** Design (awaiting review → implementation plan)

## 1. Context & motivation

Two days of diagnostics established that the pkvision recognition stack is **viable but blocked by label quality, not by the model**. On the 1,618 parkourtheory clips:

- A skeleton→cue model learns every cue weakly (per-cue val macro-F1 ~0.4) but compounds to ~13% end-to-end.
- A direct skeleton→D-score regressor barely beats a mean baseline (MAE 0.937 vs 1.032).
- The auto-derived labels themselves are ~1.36 MAE off the gold D-scores — they barely track truth.

Both a cue-pipeline and a direct regressor hit the **same ceiling**: noisy automatic labels (slug→trick match errors) + only 99 clean gold labels + a small corpus. This is the quantified cost of the hard **no-manual-labels-at-scale** constraint.

The resolution is a **human-in-the-loop data flywheel**: the human *verifies* machine proposals (never labels from scratch), active-learning focuses that attention, and the model improves round over round. Most of the machinery already exists from a prior session (`active_learn.py`, `label_server.py`, `core/vlm/`, scraped FIG competition footage). This spec covers **v1: cleaning the data we already have.**

## 2. Goal & non-goals

**Goal:** Turn the existing clips from noisy auto-labels into a clean, human-verified `{trick, cues}` labeled set via a propose→verify→retrain loop, minimizing human time, and demonstrate that per-cue accuracy on a clean held-out set climbs round over round.

**Non-goals (v1):**
- No new video scraping / data scaling (that is the *next* sub-project).
- No multi-camera / twist-from-multiview.
- No sophisticated active-learning math (start with a simple confidence/disagreement sort).
- No exact-D-score production scorer yet — v1 proves the loop lifts label quality.

## 3. Key decisions (brainstorm outcomes)

1. **Label unit = hybrid (trick + cues).** The VLM proposes a trick name *and* its cues; the human one-click-confirms the trick (cues auto-fill via the FIG ontology) or overrides individual cues for novel/misread tricks. Bounded cue vocabulary handles the open/evolving trick space; trick-confirm keeps it fast.
2. **First loop = clean existing data**, not scrape-new.
3. **Proposer = cheap VLM, gated by a ~$1 probe**, with the free local v2 model as fallback. The VLM is an accelerant, never a dependency.
4. **Human time is the scarce resource** — every design choice optimizes for fewest clips touched and fastest confirm.

## 4. Architecture — the loop

```
            ┌──────────── retrain v2 on verified labels ───────────┐
            ▼                                                       │
clips ─► PROPOSER ─► proposals{trick,cues,conf} ─► SELECTOR ─► VERIFY UI ─► VERIFIED STORE
(+frames/skeletons)    (cached)               (active-learning)   (human)     (clean labels)
```

Each round: propose → rank by informativeness → human verifies the top-K → append clean labels → retrain → repeat with sharper proposals and ranking.

## 5. Components (units, interfaces, reuse vs build)

Each unit has one purpose and a narrow interface.

**`Proposer` (build — `core/labeling/proposer.py`)**
`propose(clip_ref) -> Proposal{trick: str|None, cues: dict, confidence: float, source: str}`
Two implementations behind one interface:
- `LocalModelProposer` — wraps the v2 cue model (free, $0). Used to validate the loop and as VLM fallback.
- `VLMProposer` — wraps existing `core/vlm/` (prompt → provider → `consensus` → `fig_matcher`). Caches results to disk so a clip is never re-billed.

**`probe` (build — `scripts/probe_vlm_proposer.py`)**
Runs `VLMProposer` on the 99 gold clips, compares to gold `{trick, cues}`, reports trick-match-rate, per-cue accuracy, and total \$ spent. Output gates whether the VLM touches the bulk.

**`Selector` (build — `core/labeling/selector.py`)**
`rank(unverified, proposals) -> ordered slugs`. v1 heuristic: ascending proposer confidence + disagreement between proposers / the clip's existing weak label. Pluggable so it can get smarter later.

**Verify UI (adapt — `scripts/label_server.py`)**
Existing web UI gains the **hybrid** flow: shows clip preview + proposed trick + cues; keys for confirm / correct-trick (re-derives cues via `fig_matcher`) / override-cue. Writes to the verified store. Keyboard-first for throughput.

**Cue model module (refactor — `core/labeling/cue_model.py`)**
Extract the proven v2 pieces (featurize + augment + encoder + heads + train/eval) out of the research script `scripts/diagnose_skeleton_cues_v2.py` into a clean importable module shared by `LocalModelProposer` and the retrain hook. (Targeted cleanup the work needs — the model logic currently lives only inside a one-off script.)

**Retrain hook (build — `scripts/train_cues_verified.py`)**
Trains `cue_model` on the verified clean labels; evaluates on the held-out clean eval set; writes a checkpoint the proposer/selector pick up.

**Reused as-is:** `core/recognition/fig_decoder.py`, `oracle_cues.py`, `dscore_equiv.py`; the clips/frames/skeletons; `core/vlm/*`.

## 6. Data flow & stores

- **Proposals cache** — `data/labeling/proposals/<proposer>/<slug>.json` (so VLM calls are never repeated).
- **Verified store** — `data/labeling/verified.jsonl`, append-only, one record per verification:
  `{slug, trick, cues:{context,direction,flip,twist,axis,...}, d_score, proposer_source, action, verified_at}`
  where `action ∈ {confirm, correct_trick, override_cue}` (full provenance, auditable, undoable).
- **Held-out clean eval** — a frozen split seeded from the 99 gold (+ a reserved fraction of early verifications), never trained on, used to measure round-over-round lift. Decontamination is asserted in code.
- **Checkpoints** — `data/models/cue_model_rN.pt` per round.

## 7. The probe-gate (cost safety)

Before any bulk VLM spend (build phase P-D, after the loop is validated free):
1. Run `VLMProposer` on the 99 gold clips (~\$0.50–1).
2. Report trick-match-rate + per-cue accuracy vs gold, and \$ spent.
3. **Gate:** accept the VLM as bulk proposer only if it clearly beats `LocalModelProposer` on the gold set *and* a majority of its proposals would be confirm-only (target: ≥60% of clips need no correction). Otherwise stay on the free local proposer. Either way the loop runs; downside before we *know* is ~\$1.

## 8. Success metrics & testing

- **Primary:** per-cue accuracy on the held-out clean eval set climbs round over round; end-to-end (cues→decoder) and direct-difficulty MAE improve as the clean set grows.
- **Secondary:** verification throughput (clips/hour) — the human-cost metric.
- The **probe is the proposer's test.** Loop units (proposer interface, selector, verified store schema, decontamination assert) get focused unit tests; the verify UI is validated by a real labeling session.

## 9. Build sequence (phases — detailed in the plan)

- **P-A** Refactor v2 → `core/labeling/cue_model.py`; build `Proposer` interface + `LocalModelProposer`.
- **P-B** Adapt `label_server.py` to the hybrid UX + verified store; run a small **free** verification batch (~30–50 clips) to validate loop mechanics at \$0.
- **P-C** `Selector` v1 (confidence/disagreement sort).
- **P-D** `probe` the VLM on the 99 gold (~\$1) → gate → wire `VLMProposer` if it passes.
- **P-E** First real verification round → clean seed.
- **P-F** Retrain hook → measure round-over-round lift on held-out eval.

## 10. Risks & open questions

- **VLM proposer quality is unknown** → fully mitigated by the probe-gate (P-D before any bulk spend).
- **Mixed clip formats:** parkourtheory = `.mp4` + skeletons; gold/v5 = `.npy` RGB frames; FIG-comp = segmented `.npy`. The proposer + UI must handle both a video and a frame-array clip ref. Define a single `clip_ref` abstraction in P-A.
- **Verification throughput is unproven** until P-B; if it is too slow, the selector and confirm-UX are the levers.
- **Decontamination:** the held-out clean eval must never leak into training — asserted in code (reuse `core/data/decontam`).
