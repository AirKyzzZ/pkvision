# Rule-counter probe: results (2026-11-04)

Pre-registration: `2026-11-04-rule-counter-prereg.md` (commit `6982cc0`, before any counter output). Raw numbers: `2026-11-04-rule-counter-results.json`. Run on the RTX 2060, $0.

## Pre-registered result

**Decision-12 exact twist: A baseline 3/12, B upright 3/12, C 4-way 3/12. All are ≤ 6/12, so the rule reads: architecture #1 is killed as specified.**

| Arm | Decision-12 exact | within ±0.5 | All-47 twist exact | All-47 flip exact | Coverage |
|---|---|---|---|---|---|
| A baseline | 3/12 | 7/12 | 22/47 | 30/47 | 42/47 |
| B upright | 3/12 | 8/12 | 22/47 | 31/47 | 45/47 |
| C 4-way | 3/12 | 6/12 | 22/47 | 31/47 | 45/47 |

Twist 2+ was 0/4 on every arm. The upright crop and 4-way rotation raised coverage from 42 to 45 clips and did not raise twist accuracy.

Own footage (smoke, no decision weight, arms A/B/C): frontflip and gainer correct on flip, twist and direction. Backflip wrong on all arms: flip 0, twist 1/1.5/0.5. Back double full: flip 1 correct, twist 2.5 on all arms (truth 2), direction wrong on arm A only.

## Exploratory: why it failed (not pre-registered, no decision weight)

Skeleton overlays on 7 evenly spaced frames of each Decision-12 clip, arm B, read by the agent and not checked by the owner:

| Observed cause | Clips |
|---|---|
| **Wrong person:** skeleton on spectators for much of the clip | `swing_castaway_double_back`, `swing_castaway_double_full` (also `wall_double_back`, outside the 12) |
| **Keypoints lost while inverted** in the sampled frames | `palm_spin`, `kong_gainer_full`, `wall_double_full`, `stomach_webster` |
| Tracking looks usable | `raiz_dismount_push_half_twist`, `flyaway_half_down`, `lache_dark_arabian`, `castaway_back`, `360_precision`, `side_double_full` |

Of the 6 usable clips, 2 are exact under arm B (`flyaway_half_down`, `360_precision`). Of the other 4, 3 are off by 0.5 and `castaway_back` is off by 1.0. `stomach_webster` is the third exact clip, despite lost keypoints.

Reading, with n this small:
1. The kill line measures this pipeline, not twist geometry. About half of Decision-12 had no usable athlete signal through the rotation.
2. A better athlete selection and better inverted-pose inference are prerequisites. The counter is not cleared either: it missed on 4 of the 6 clips with usable tracking.
3. Rotation inference (B, C) did not recover enough inverted frames. Handheld camera roll is visible in many clips, as is the athlete leaving frame.

## Finding beyond this probe

The bystander problem is not specific to this probe's box rule. `scripts/extract_pose.py` picks the person with the widest keypoint spread, so a close spectator can win there too. That is the extractor behind the 33 corpus shards, so CueModel training may include skeletons of the wrong person. **The rate is not measured.** This may contribute to the macro-F1 ~0.4 plateau, alongside label noise.

## Limits

- Truth is provisional: Claude consensus, manifest and parser agree, but no owner verified it. It excludes the hardest clips, and it is VLM-anchored.
- Odd ≥1.5 twists and corks could not be evaluated.
- n = 12 can only rank or kill.
- The thresholds were not tuned, so a tuned counter could do better. Any retest on these clips is exploratory, because they have now been seen.

## Owner decision needed

The rule says kill, but the failure is mostly upstream of the counter. Options:

1. **Amend and retest (recommended):** pre-register an athlete-selection fix (the track with the most motion through the clip, or a SAMURAI mask) and test on **fresh clips**, ideally owner-verified or from the filming test.
2. **Accept the kill:** move to architecture #2 (VLM reader) or #3 (two phones).
