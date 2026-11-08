# Do invented skeletons hurt the CueModel? Results (2026-11-08)

Pre-registration and code: PR #4 (`2026-11-08-clean-pose-cue-prereg.md`, `scripts/probe_clean_pose_cue.py`), committed before any CueModel saw the new skeletons. Numbers: `2026-11-08-clean-pose-cue-results.json` (with all confusion matrices) and `2026-11-08-clean-pose-cue-audit.json`. Run on the RTX 2060, $0. No clip left the PC and no VLM ran.

## Answer

**No detectable effect.** Replacing invented skeletons with detector-first ones changed the mean gold macro-F1 by −0.003 (per seed: −0.029, +0.006, −0.042, +0.023, +0.026). The pre-registered rule needs a mean of at least ±0.03 with 4 of 5 seeds agreeing. The paired clip bootstrap of accuracy gives −0.013, 95% CI [−0.040, +0.015].

Under the pre-registered consequence, the bug stays fixed, but the macro-F1 plateau should no longer be attributed to invented skeletons. The label work stays the priority. The detector-first shards do not become the default.

"No detectable effect" is not "no effect". The per-seed spread is about as large as the 0.03 bar, so effects below it cannot be resolved here.

## Corpus audit

1,582 clips (the 36 blind-queue clips excluded), 117,957 frames, 3 h 54 min of extraction.

| Measure | Value |
|---|---|
| Frames with nobody detected, holding an invented skeleton in the old shards | 11,727 (9.9%) |
| Clips with at least one invented frame | 1,329 of 1,582 (84%) |
| Clips with over 10% frames undetected | 560 of 1,582 (35%) |
| Clips with ≥ 2 identity jumps: old → detector-first | 13.1% → 11.2% |
| Clips with 0 jumps: old → detector-first | 78.1% → 80.2% |

On detected frames the new skeletons matched the old ones in a three-clip smoke check, so the two differ only where the old extractor invented a person. A skeleton that stays on one bystander is not counted as a jump.

## CueModel on the gold set

Five seeds, each arm trained and scored on its own skeletons. Scored on the concordant gold clips, per cue where present: flip 82, twist 64, direction 85, context 78.

| Arm | Gold macro-F1 (mean of 4 cues) | Flip acc | Twist acc | Direction acc | Context acc | Internal val core macro-F1 (noisy labels) |
|---|---|---|---|---|---|---|
| **O** old shards | **0.373** | 0.456 | 0.584 | 0.482 | 0.523 | 0.405 |
| **D** detector-first, gaps interpolated | **0.370** | 0.451 | 0.591 | 0.466 | 0.521 | 0.385 |
| D0 detector-first, gaps as zeros | 0.354 | 0.439 | 0.569 | 0.365 | 0.533 | 0.387 |
| S motion selector (exploratory) | 0.336 | 0.417 | 0.509 | 0.447 | 0.444 | 0.365 |
| T drop ≥ 2-jump clips from training (exploratory) | 0.364 | 0.439 | 0.553 | 0.438 | 0.544 | 0.396 |
| Train-majority constant | 0.195 | 0.683 | 0.797 | 0.659 | 0.218 | n/a |

**Read the baseline row carefully.** The class-weighted model beats the constant on macro-F1 in every arm, and on accuracy only for context. Always answering the majority class scores higher accuracy on flip, twist and direction. On twist, the model is exact on 0.58 where always answering 0 twists gets 0.80. This matches the rule-counter finding: a constant is a hard baseline on this set.

Twist confusion, seeds pooled, rows are truth in bins (0, 0.5, 1, ≥ 1.5), columns are predictions:

| Truth bin | O | D |
|---|---|---|
| 0 (51 clips × 5 seeds) | 169 / 37 / 39 / 10 | 171 / 32 / 39 / 13 |
| 0.5 (3 clips) | 3 / 2 / 9 / 1 | 7 / 5 / 2 / 1 |
| 1 (5 clips) | 14 / 3 / 8 / 0 | 16 / 3 / 5 / 1 |
| ≥ 1.5 (5 clips) | 1 / 5 / 11 / 8 | 1 / 4 / 12 / 8 |

The gold bins above 0 are tiny (3–5 clips each), so no per-bin claim is possible.

## Exploratory notes (no decision weight)

- **Selector (S)** has the lowest accuracy on every cue except direction. That fits the audit's verdict that the motion selector is not production-ready.
- **Dropping frames as zeros (D0)** lost 10 points of direction accuracy against interpolated gaps (0.365 vs 0.466). Zeros create large spurious velocities after hip-centring, a plausible cause, not verified.
- **Dropping jump-heavy clips (T)** did not help: 1,318 training clips, 0.364.
- The internal noisy-label split favours the old shards: O scored 0.405 and D 0.385, a gap that gold does not show.

## Disclosures

- The pre-registered training set was 1,487 clips by arithmetic. It is 1,483 in practice because 4 manifest rows (`test_*`) are own-footage entries with no corpus clip.
- A numpy-integer crash in the audit report was fixed (`scripts/probe_clean_pose_cue.py`) after extraction and before any training ran. It touched reporting only.
- The 36 blind-queue clips stayed out of extraction, training and evaluation. Their poses were not computed by this probe.
- Truth is not owner-verified: gold is manifest–Claude agreement. The gold set has been used since July, so this compares extractors only.

## What this does not settle

The wrong-person problem. No arm has clean athlete selection, and the audit's 12–37% wrong-person range is still unmeasured against model quality.
