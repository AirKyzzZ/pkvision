# Do invented skeletons hurt the CueModel? Pre-registration (2026-11-08)

Written and committed **before** any CueModel was trained on detector-first skeletons. The corpus extraction starts without waiting for this commit because it produces no cue output. Thresholds are not to be changed after seeing results; anything added later is reported as exploratory.

**Question:** the 2026-11-04 audit found that the corpus shards contain invented skeletons (full-frame boxes on frames with nobody detected, 11.4% of frames on the 47 reference clips). Does removing them change CueModel performance? The audit marked the effect "not measured".

No clip goes to any outside service, no VLM runs. $0, on the RTX 2060. This does not revive the killed counter and does not replace the VLM blind probe, which is still waiting for the owner.

## Data

- **Blind queue protected.** The 36 clips of `data/labeling/retest_2026-11/queue.json` are excluded from extraction, training and evaluation, read programmatically and never printed. Their pose stays unextracted by this probe.
- **Train:** `data/v5_attribute_training/manifest_baseline_noeval.json` (the July baseline manifest, gold-99 removed), minus the queue. Labels are the noisy manifest.
- **Evaluate:** `data/v5_attribute_training/eval_gold_concordant.json`, 99 gold clips held out of training, where the manifest and Claude agree. Scored per cue where present: flip 82, twist 64, direction 85, context 78 clips. Truth is not owner-verified.
- **Pose:** RTMPose-x body7 + YOLOX-x (`rtmlib performance`), detector first (`core/pose/detect_first.py`), the same widest-spread selection as the old shards. Output is gitignored under `data/keypoints_detfirst/`.

## Arms

Each arm trains and evaluates on its own skeletons, because an extractor is deployed as a whole.

| Arm | Skeletons | Role |
|---|---|---|
| **O** | old corpus shards (invented frames included) | primary, baseline |
| **D** | detector-first, frames with no detection linearly interpolated with confidence 0 | primary |
| D0 | detector-first, undetected frames left as zeros | disclosure: the raw alternative to interpolation |
| S | `core/pose/athlete_select.py` (frozen 2026-11-04) over all detected people, then as D | exploratory: the selector is marked not production-ready |
| T | D skeletons; training drops clips with ≥ 2 identity jumps | exploratory |

## Training and metric

Recipe unchanged from `scripts/train_cue_model.py` (70 epochs, batch 64, AdamW, class weights, early stop on a random internal 20% split of the noisy labels). Five seeds, 0–4; each seed uses the same split and same training set across O, D, D0 and S.

**Primary metric:** mean over flip, twist, direction and context of macro-F1 on the gold set, at the model's bin granularity, per seed. Macro-F1 is taken over classes present in the truth.

**Decision rule, D minus O, per seed:**
- **helps:** mean delta ≥ +0.03 and at least 4 of 5 seeds positive;
- **hurts:** mean delta ≤ −0.03 and at least 4 of 5 seeds negative;
- **no detectable effect:** otherwise.

Always reported alongside: per-cue accuracy and macro-F1, the train-majority constant baseline, the macro-F1 on the noisy internal split (comparable to the 0.39 plateau), confusion matrices pooled over seeds, and a paired clip bootstrap 95% CI of the accuracy difference.

## What follows

- **helps:** the detector-first shards become the default for any later cue training or weak-label work.
- **hurts / no detectable effect:** the bug stays fixed regardless, but we stop attributing the macro-F1 plateau to invented skeletons, and the label work stays the priority.

## Known limits, stated in advance

- About 80 gold clips and noisy training labels: small differences are noise. The bootstrap CI is on accuracy, not on the primary macro-F1.
- Interpolation (D) replaces invented skeletons with smoothed motion: D is a claim about the extractor plus this gap-filling, and D0 shows the raw alternative.
- The gold set has been used for model comparison since July, so this compares extractors only; it adds no new information about absolute performance.
- A skeleton that stays on one bystander produces no jump and is not removed by any arm.
- The audit's "wrong person" problem is only partly addressed: the selector is exploratory and known to fail on some clips.
