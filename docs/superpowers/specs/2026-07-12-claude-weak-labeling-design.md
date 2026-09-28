# Claude Weak-Labeling — Design (2026-07-12)

## Objective

Use Claude (Fable 5, via Max subscription — $0 marginal) as a consensus weak-labeler to produce the highest-quality cue labels achievable for the parkour clip corpus, so the cue model can be retrained past the label-noise wall. Calibration (12 gold clips, 2 runs) passed: flip 11/11 exact, twist 10/11, direction 11/11 vs corrected FIG references.

The design goal is **relabel-proofing**: capture every field we could plausibly need in one pass, versioned, with provenance, so the corpus never needs a second bulk pass for schema reasons.

## Label schema `claude_weak_label_v1`

Per-clip agent output (structured, forced schema):

| Field | Type / vocab | Notes |
|---|---|---|
| `flip` | number, 0–4 in 0.5 steps | rotations about horizontal axis |
| `twist` | number, 0–4 in 0.5 steps | rotations about longitudinal axis |
| `direction` | `forward\|backward\|side\|none` | manifest + FIG vocab |
| `axis` | `lateral\|sagittal\|longitudinal\|off_axis\|none` | FIG CoP vocab; new — decoder uses it |
| `context` | `acrobatics\|wall\|swing\|pk_basics` | manifest vocab |
| `body_shape` | `tuck\|pike\|layout\|open\|unclear` | manifest vocab |
| `entry` | `running\|standing\|edge\|wall\|one_leg\|castaway\|caster\|other\|unclear` | manifest vocab |
| `hand_contact` | boolean | hands touch surface during the trick |
| `landing` | `feet_controlled\|feet_deep_impact\|roll\|stumble\|fall\|unclear` | new — future E-score/quality layer |
| `multiple_tricks` | boolean + `reps` int | calibration found chained-rep clips |
| `trick_guess` | free text | high value: calibration got ~10/12 names right |
| `visibility` | free text | occlusion/blur/framing issues |
| `confidence` | per-cue 0–1 | drives verification queue ordering |

Record metadata added by the harness: `slug`, `schema_version`, `model`, `frames` (24, even-span `round(i*(nb-1)/23)`), `votes[]` (full per-vote outputs), `consensus`, `consensus_method`, `label_provenance` per cue.

## Consensus protocol

- Vote 1 + Vote 2: independent blind agents (different reviewer preamble).
- Agreement on a cue → accept, confidence = mean.
- Disagreement on `flip`/`twist`/`direction`/`context` → Vote 3 tiebreak; 2-of-3 majority wins; no majority → cue marked `disputed`.
- Disputed cues and any clip with consensus confidence < 0.6 go to the human `verify_server` queue (human = final authority, per project rule).

## Phases & gates

1. **Gold audit (99 clips, 2 votes + tiebreak all):** compare consensus vs manifest vs FIG CoP (exact-name match only — aliases are proven unreliable). Output: `data/labeling/claude_probe/gold99_votes.jsonl`, accuracy report, corrected-gold proposal + dispute queue. **Gate: ≥85% flip and direction agreement with FIG-derivable references.**
2. **Bulk (~1,519 remaining clips, value-ordered: acrobatics → wall/swing → pk_basics):** Vote 1 for all; Vote 2 when vote-1 confidence < 0.75 on any core cue, or flip ≥ 1.5, or twist ≥ 1, or `multiple_tricks`; tiebreak on disagreement. Checkpointed per ~50-clip batch to `data/labeling/claude_labels/labels_v1.jsonl` (append-only, resumable).
3. **Retrain + measure:** `train_cue_model.py` on a claude-label manifest; compare core-mF1 vs 0.390 baseline; end-to-end FIGDecoder top-1/3/5 on the audited gold.

## Known risks

- Calibration clips are curated reference footage; competition footage is harder — bulk accuracy will be measured, not assumed (the audited gold set is the instrument).
- Rate limits may interrupt bulk — append-only JSONL + per-batch checkpoints make every clip durable.
- Taxonomy ambiguities (e.g. wall-assisted acro context, wall_spin flip-vs-twist) are routed to the human queue, not guessed.
- Labels produced by a model must not be used to *evaluate* that same model as a judge — human-verified subsets remain the evaluation yardstick.
