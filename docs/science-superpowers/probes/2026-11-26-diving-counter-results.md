# Does our rule counter work on diving with our own skeletons? Results (2026-11-28)

Pre-registration: `2026-11-26-diving-counter-prereg.md`, commits `a7861b5` and `433c3b6`, before the counter ran on any dive. Numbers: `2026-11-26-diving-counter-results.json`. 200 MTL-AQA dives with official counts, $0, about 50 minutes of the RTX 2060. The downloaded segments were deleted after scoring; the skeletons are kept.

## Answer

**Skeleton or counter is the problem**, as predicted. On arm C (best of four rotations) the counter is exact on 24% of somersault counts and 35.5% of twist counts. Both are below a constant answer (51% and 39.5%). NS-AQA reports 97.3% and 93.3% on the same dataset with a pose model trained on divers.

| Arm | Somersault exact | Twist exact | Within half a turn | Coverage |
|---|---|---|---|---|
| **C**, four rotations | 0.240 | 0.355 | 0.300 / 0.620 | 1.00 |
| A, plain pose | 0.210 | 0.320 | 0.265 / 0.600 | 1.00 |
| Constant | 0.510 | 0.395 | | |

Twist by true count, arm C: 53/79 untwisted dives, 3/3 at half a twist, 3/14 at one, 10/35 at one and a half, and 2/69 at two or more.

Two of the 200 drawn dives could not be fetched and were replaced by the next of their stratum, as the rule says.

## Disclosure

The first 36 dives were posed on the CPU: the script did not load the CUDA libraries and onnxruntime fell back silently. The PC then restarted. The remaining 164 ran on the GPU after a fix that also stops the job if the GPU is not used. Same models and code on both, so the keypoints are equivalent.

## Exploratory: where it fails (no decision weight)

- **It undercounts by whole turns.** Predicted minus true somersaults: −1.0 on 68 dives, −2.0 on 35, between −2.5 and −4.0 on 25, exact on 48, over on 2.
- **The diver is not small.** Median torso length is 101 pixels (87 to 115 for the middle 80%), so resolution is not the cause, contrary to the pre-registration's guess.
- **The skeleton drops out during the fast tucked rotation.** On skeleton overlays of six random dives, the pose follows the diver at take-off and entry, and in the tucked or inverted frames it is missing, or sits on a spectator. Nobody is detected on 29% of the annotated frames (this also counts the underwater frames at the end). The counter then interpolates across the gap and loses the turns made inside it.

Reading: the first thing that fails is detection and pose on tucked, inverted, blurred bodies, not the counting rule. This is the November parkour finding again ("keypoints lost while inverted"), now on 200 clips with trusted labels.

## What follows

Under the pre-registered consequence, filming more parkour would not fix this. Two candidate fixes, each to be pre-registered on dives not used here (about 780 remain in the 720p videos):

1. a counter that carries the measured rotation rate across frames with no skeleton, instead of interpolating positions;
2. a pose model that holds on tucked and inverted bodies, starting with the diver pose data that NS-AQA publishes.

These 200 dives are now a development set: any change tuned on them must be confirmed on fresh dives.
