# Does giving the training labels a "no flip" class lift the CueModel? Pre-registration (2026-11-26)

Written and committed **before** any CueModel was trained on the corrected labels. Thresholds are not to be changed after seeing results; anything added later is reported as exploratory.

## Why this test

The CueModel has sat at a gold macro-F1 of about 0.37 since July and does not beat a constant on accuracy for flip, twist or direction. A label audit on 2026-11-26 (`2026-11-26-direction-review.md`, exploratory) found a systematic defect in the training labels, not random noise:

- The training manifest gives a flip to 1,519 of 1,523 clips and a direction to all but 4. It has no "no flip" class.
- Parkour Theory's own categories say 295 of the 1,503 matched training clips (19.6%) are not flips: vaults, lachés, precisions, rolls. Examples: `split_vault`, `wall_kong`, `double_kong_precision` (labelled 2 flips).
- The gold set has 16 of 82 clips at flip 0 and 16 of 85 at direction "none". Over the five 2026-11-25 runs the model answered flip 0 on 5 of those 80 clip-runs.
- On the gold set, Parkour Theory's Flip category agrees with gold "has a flip" on 77 of 82 clips, and its Twist category with gold "has a twist" on 61 of 64.

**Question:** does correcting the training labels with Parkour Theory's Flip and Twist categories raise the gold macro-F1?

No clip goes to any outside service, no VLM runs, $0, about 15 minutes of the RTX 2060 through the GPU guard.

## Data

- **Blind queue protected.** The 36 clips of `data/labeling/retest_2026-11/queue.json` stay out of training and evaluation, read programmatically and never printed.
- **Train:** `data/v5_attribute_training/manifest_baseline_noeval.json` minus the queue, 1,487 label rows, joined to the detector-first skeletons with gaps interpolated (arm D of the 2026-11-08 probe).
- **Evaluate:** `data/v5_attribute_training/eval_gold_concordant.json`, the same 99 gold clips, scored per cue where present: flip 82, twist 64, direction 85, context 78. Truth is not owner-verified.
- **Categories:** the `type` field of `data/parkourtheory_detailed.json` (March 2026 scrape, already tracked), matched to clips by slugified move name.

## The label correction, frozen

Applied by `corrected()` in `scripts/probe_pt_label_fix.py`, to the training labels only:

1. If the move's Parkour Theory type has no `flip`: flip becomes 0, direction becomes `none`, axis is ignored in the loss.
2. If the type has `twist` and the manifest twist is 0: the twist label is ignored in the loss (the count is unknown).
3. Everything else is unchanged, including clips with no Parkour Theory match.

Counts, measured before any training: 262 flips set to 0, 282 directions set to `none`, 132 twist labels masked, 19 clips unmatched. Gold labels and binning are not touched.

## Arms

| Arm | Labels | Training clips | Seeds | Role |
|---|---|---|---|---|
| **B** | manifest as is | all | 0–4 | primary baseline: the five arm-D runs of 2026-11-25 in `cue_runs.jsonl` (mean 0.374), not retrained |
| **P** | corrected | all | 0–4 | primary |
| P50 | corrected | random 50% | 0–2 | exploratory learning curve |
| P25 | corrected | random 25% | 0–2 | exploratory learning curve |

Recipe unchanged from `scripts/train_cue_model.py`. Seed `s` of P uses the same internal split and training clips as seed `s` of B.

## Metric and decision rule

**Primary metric:** mean over flip, twist, direction and context of macro-F1 on the gold set, per seed, as on 2026-11-08.

**Decision rule, P minus B, per seed:**
- **helps:** mean delta ≥ +0.03 and at least 4 of 5 seeds positive;
- **hurts:** mean delta ≤ −0.03 and at least 4 of 5 seeds negative;
- **no detectable effect:** otherwise.

**Prediction:** helps, through flip and direction. Twist and context are expected to stay within ±0.03.

Reported alongside: per-cue accuracy and macro-F1, the train-majority constant, confusion matrices pooled over seeds, and the learning curve.

## What follows

- **helps:** the label-repair direction continues. Next come twist counts (name parser plus the Parkour Theory progression graph, checked by the owner on a small queue) and a larger evaluation set.
- **no detectable effect or hurts:** the direction's first step is killed. The missing "no flip" class is real and stays fixed, but the plateau is no longer attributed to it, and the plan moves to athlete selection, where the rule counter lost the most (see the review).

## Known limits, stated in advance

- The hypothesis came from looking at the gold label distribution and at the published confusion matrices. Part of any gain is mechanical: the model gains a class that the gold set contains. The size of the gain, and whether accuracy passes the constant, is the informative part.
- About 80 gold clips, used for model comparison since July. Twist has 13 gold clips above 0, so this test says almost nothing about twist counting.
- Parkour Theory's categories are not truth: they disagree with gold on 5 of 82 flip clips.
- The learning curve changes the training set and the internal split together, so it only shows a trend.
