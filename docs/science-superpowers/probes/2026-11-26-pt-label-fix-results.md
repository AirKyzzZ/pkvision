# Does giving the training labels a "no flip" class lift the CueModel? Results (2026-11-26)

Pre-registration: `2026-11-26-pt-label-fix-prereg.md`, commit `8e0c34d`, before any model saw the corrected labels. Numbers: `2026-11-26-pt-label-fix-results.json`, with all confusion matrices. Eleven runs, about 6 minutes of the RTX 2060 behind the GPU guard, $0. No clip left the PC and no VLM ran.

## Answer

**Helps.** Correcting the training labels with Parkour Theory's Flip and Twist categories raised the mean gold macro-F1 from 0.374 to 0.456, a gain of +0.082. All five seeds are positive: +0.133, +0.109, +0.070, +0.032, +0.067. The rule needed a mean of at least +0.03 with 4 of 5 seeds positive.

It is the first change since July that moves this number. The model still does not beat a constant on accuracy for flip, twist or direction.

## Per cue, mean of five seeds

| Cue | Macro-F1: B → P | Accuracy: B → P | Constant accuracy |
|---|---|---|---|
| Flip (82 gold clips) | 0.276 → **0.516** | 0.427 → 0.612 | 0.683 |
| Direction (85) | 0.297 → **0.476** | 0.449 → 0.534 | 0.659 |
| Twist (64) | 0.366 → **0.279** | 0.575 → 0.566 | 0.797 |
| Context (78) | 0.557 → 0.552 | 0.546 → 0.538 | 0.218 |

B is the manifest as it was, P the corrected labels. 262 flips were set to 0, 282 directions to `none`, 132 twist labels masked.

## Where the gain comes from

Pooled over five seeds, exact answers per true class:

| True class | B | P |
|---|---|---|
| Flip 0 (16 clips × 5) | 5 / 80 | 46 / 80 |
| Flip 1 (56 × 5) | 160 / 280 | 191 / 280 |
| Flip 2 (10 × 5) | 10 / 50 | 14 / 50 |
| Direction `none` (16 × 5) | 0 / 80 | 45 / 80 |
| Direction `backward` (56 × 5) | 150 / 280 | 142 / 280 |

Most of the gain is the class the model could not answer before, which the pre-registration announced as partly mechanical. Single flips also improve, from 57% to 68%, because fewer of them are confused with class 0.

## The prediction that failed

The pre-registration expected twist to stay within ±0.03. Its macro-F1 fell by 0.087. The top bin (1.5 twists and more, 5 clips) went from 9 to 4 exact answers out of 25, the half-twist bin (3 clips) from 7 to 2 out of 15. With 13 gold clips above 0 this cannot be read as a real loss or as noise. Masking 132 "0 twist" labels changed the class weights of the twist head, which is a plausible cause, not verified. Twist needs its own labels and a larger evaluation set before any further change.

## Learning curve (exploratory, no decision weight)

| Training clips | Gold macro-F1 per seed | Mean |
|---|---|---|
| 368 (25%) | 0.347, 0.346, 0.383 | 0.359 |
| 737 (50%) | 0.448, 0.434, 0.443 | 0.442 |
| 1,474 (100%) | 0.487, 0.501, 0.430, 0.429, 0.432 | 0.456 |

Doubling from 368 to 737 clips adds 0.083. Doubling again adds 0.014 (0.031 on seeds 0 to 2). With these skeletons and one demo clip per move, more clips of the same kind would add little. The training set and the internal split change together, so this is a trend only.

## Disclosures

- The hypothesis came from the gold label distribution and the published confusion matrices, as stated in advance.
- The baseline was not retrained: it is the five arm-D runs of 2026-11-25, same code, same machine, same clips.
- Truth is not owner-verified. The gold set has been used for comparison since July.
- The 36 blind-queue clips stayed out of training and evaluation.

## What follows

Under the pre-registered consequence the label-repair direction continues. Next: twist counts from the name parser and the Parkour Theory progression graph, a larger evaluation set, then athlete selection.
