# Does a Gemini second opinion on flip and direction improve the training labels? Pre-registration (2026-11-26)

Written and committed **before** any training clip was sent to a model. Thresholds are not to be changed after seeing results; anything added later is reported as exploratory.

## Why this test

Two results of the same day:

- Correcting the training labels with Parkour Theory's categories raised the CueModel's gold macro-F1 from 0.374 to 0.456 (`2026-11-26-pt-label-fix-results.md`). Labels matter.
- On the gold clips Gemini 3.8 Flash reads flip at 0.805 and direction at 0.776, against 0.612 and 0.534 for that CueModel (`2026-11-26-vlm-gold-exploratory.md`, exploratory). It reads twist no better than a constant.

**Question:** if the training labels for flip and direction are kept only where Gemini does not contradict them, does the CueModel's gold macro-F1 rise?

The owner approved sending clips through his OpenRouter account. About $15, capped at $20, then about 6 minutes of the RTX 2060 behind the GPU guard.

## Data

- **Blind queue protected.** The 36 queue clips are never sent and never read, as before.
- **Train:** the corrected labels of `scripts/probe_pt_label_fix.py` (arm P), 1,487 rows.
- **Second opinion:** one vote per training clip from arm G-flash of the VLM blind probe, with its frozen prompt and settings (`google/gemini-3.8-flash`, video retimed 10 times slower). Gold clips are not part of this collection.
- **Evaluate:** `eval_gold_concordant.json`, the same 99 gold clips. Truth is not owner-verified.

## The rule, frozen

Applied by `consensus()` in `scripts/probe_vlm_second_opinion.py`, at the CueModel's bins:

1. Where Gemini gives a flip class different from the label, the flip label is ignored in the loss. Same for direction.
2. An unsure, malformed or missing answer is no opinion: the label stays.
3. Twist, context and axis labels are never touched. Gemini's answer is never used as a label.

Training starts only if votes exist for at least 95% of the training clips.

## Arms

| Arm | Labels | Seeds | Role |
|---|---|---|---|
| **P** | corrected by Parkour Theory categories | 0–4 | baseline: the five runs of 2026-11-26 in `pt_label_fix_runs.jsonl` (mean 0.456), not retrained |
| **C** | P, minus flip and direction labels that Gemini contradicts | 0–4 | primary |

Recipe unchanged from `scripts/train_cue_model.py`. Seed `s` of C uses the same clips and internal split as seed `s` of P.

## Metric and decision rule

**Primary metric:** mean over flip, twist, direction and context of macro-F1 on the gold set, per seed.

**Decision rule, C minus P, per seed:**
- **helps:** mean delta ≥ +0.03 and at least 4 of 5 seeds positive;
- **hurts:** mean delta ≤ −0.03 and at least 4 of 5 seeds negative;
- **no detectable effect:** otherwise.

**Prediction:** no detectable effect, or a small gain under the bar. Gemini is wrong on about one gold clip in five, so the rule also removes correct labels, and the earlier learning curve suggests the model is limited by its skeletons more than by label count.

Reported alongside: per-cue accuracy and macro-F1, how many labels were kept, masked and left without an opinion, and the cost.

## What follows

- **helps:** Gemini becomes the second opinion for the label repair, and the same rule is tried on twist once a model reads twist.
- **no detectable effect or hurts:** the CueModel is not limited by flip and direction labels any more. Gemini is then used as the reader itself, on cropped and slowed clips, and the skeleton model stops being the main path for these two attributes.

## Known limits, stated in advance

- About 80 gold clips, used for comparison since July, and the same ones on which Gemini was scored this morning.
- One vote per clip. Repeated calls can disagree.
- Masking changes the class weights of the two heads, which moved twist in the previous test.
- Agreement between Gemini and a label is not proof that the label is right.
