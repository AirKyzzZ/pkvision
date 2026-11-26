# PkVision direction review (2026-11-26)

Why the system is stuck near macro-F1 0.37, and which single bet to make next. Everything measured today is **exploratory** unless it points to a pre-registration. The 36 blind-queue clips were excluded from every computation and never printed. $0, no clip left the PC.

## 1. Where the project stands

**Proven**
- CueModel gold macro-F1 is 0.37 and reproduces on the rebuilt Windows setup: 0.374 on 2026-11-25 against 0.370 on 2026-11-08 (five seeds each).
- Removing invented skeletons does not move it (PR #6, pre-registered, delta −0.003).
- The rule-based twist counter is no better than a constant: 3/12 on the decision clips (PR #1, pre-registered kill).
- The old extractor invented a skeleton on 9.9% of frames (PR #4).

**Disproven**
- "Invented skeletons cause the plateau."
- "Twist of 1.5 or more cannot be seen from one camera" as a geometric claim (September survey, measured on diving and skating only).

**Believed, not measured on parkour**
- A frontier VLM reads flips and twists from frame grids: 11/11 and 10/11 in July, not blind, n = 11.
- Label noise is the wall. Until today this rested on the September audit of the old gold set, not on the training labels.

## 2. Diagnosis

Data: the frozen rule counter (`core/pose/rule_counters.py`) run on the "dev" half of the training clips (737 clips, split by slug hash, the other half is untouched), and a label audit against Parkour Theory's own Flip and Twist categories. Scripts: `scripts/probe_flip_yardstick.py`, and the label counts in the pre-registration.

| # | Cause | Evidence | Confidence |
|---|---|---|---|
| 1 | **Training labels are wrong in a systematic way.** The manifest has no "no flip" class: 1,519 of 1,523 clips carry a flip and all but 4 a direction. | Parkour Theory says 295 of 1,503 matched clips (19.6%) are not flips (`split_vault`, `wall_kong`, `double_kong_precision` labelled 2 flips). 132 clips are labelled 0 twist while Parkour Theory files them under Twist (`inward_webster_rudi`). The gold set has 16/82 clips at flip 0; the model answered 0 on 5 of 80 such clip-runs. Parkour Theory's categories agree with gold on 77/82 (flip) and 61/64 (twist). | High that the defect exists. Its share of the plateau is what the pre-registered test measures. |
| 2 | **The yardstick is too small.** | Gold has 13 clips with twist above 0 (3, 5 and 5 per bin) and is not owner-verified. No twist result can be defended on it. | High |
| 3 | **Wrong person in the frame.** | Rule flip counter on clips whose flip label the name parser confirms (157 dev clips): 64% exact with nobody else detected (n = 22), 53% with others in up to 30% of frames (n = 58), 26% beyond that (n = 58). Confounded with shot width. The 2026-11-04 audit put wrong-person clips at 12–37%. | Medium |
| 4 | **Pose and counter fail on the rotation itself.** | A single flip is read as 0 on 40 of 116 dev clips and exact on 55. Torso keypoint validity does not predict success (42%, 48%, 36%, 40% across four validity bands). Where the flip is read right (n = 60) the twist counter is exact on 25% against 65% for a constant, and answers 0 on only 10 of 39 untwisted clips. | Medium |
| 5 | Too few clips per trick (about one). | Not measured before today. The exploratory learning curve in the pre-registered run gives a first trend. | Unknown |
| 6 | 2D loses the rotation (viewpoint). | The shoulder-to-torso ratio, a rough view proxy, shows no trend with flip accuracy (31%, 56%, 35%, 50%). | Low either way |
| 7 | Metric choice. | Below-constant accuracy is partly expected from class-weighted training. On macro-F1 the model is above the constant (0.37 against 0.195). | Low |

The overall flip counter result on the dev half is 30% exact against 73% for a constant over all labels, and 43.5% against 84% on parser-confirmed labels. That confirms the November kill at n = 138 instead of n = 12.

**Reading.** Every learned model since the v5 manifest was trained to believe that every move is a flip with a direction. Cause 1 is the only one that is measured on the whole corpus with an independent source, and it can be fixed today for free. Causes 3 and 4 are real and come next: they limit rule counters whatever the labels are.

## 3. Research scan

The September survey (`docs/science-superpowers/survey/`) stays the reference. This scan only adds what bears on today's diagnosis. Figures marked "not verified" were not read from the primary source.

| Source | What it measured | Use for us |
|---|---|---|
| NS-AQA, [arXiv 2403.13798](https://arxiv.org/abs/2403.13798) | Rule-based twist 93.3% and somersault 97.3% on diving, fixed side camera | Our counter copies it and fails: the gap is perception and viewpoint, not the idea |
| Trampoline synthetic pose, [arXiv 2604.01322](https://arxiv.org/abs/2604.01322) | Fine-tuned ViTPose on extreme poses, 3D MPJPE down 42.7% | Inverted-body fix. Released weights: not verified |
| AthletePose3D, [arXiv 2503.07499](https://arxiv.org/abs/2503.07499) | Fine-tuning cut MPJPE from 214 to 65 mm on athletic motion | Weights listed in the repo, non-commercial. Which models: not verified |
| SAM 3D Body, [repo](https://github.com/facebookresearch/sam-3d-body) | Checkpoints of 631M and 840M parameters | Fit in 6 GB is doubtful (inference, not verified) |
| Sapiens2, [arXiv 2604.21681](https://arxiv.org/abs/2604.21681) | Pose checkpoints from 0.4B parameters | Candidate small pose swap. VRAM: not verified |
| V-JEPA 2, [arXiv 2506.09985](https://arxiv.org/abs/2506.09985) | Frozen-feature probe on Diving48: 89.0% for ViT-L, 90.2% for the largest (read from a mirror, not verified) | A skeleton-free arm that sidesteps causes 3 and 4 |
| VLM4D, [arXiv 2508.02095](https://arxiv.org/abs/2508.02095), and MotionBench, [arXiv 2501.02955](https://arxiv.org/abs/2501.02955) | VLMs on rotation and fine motion in generic video. Accuracies not read | No benchmark counts flips or twists. Our blind probe would be the first number |
| Can VLMs Judge Action Quality?, [arXiv 2604.08294](https://arxiv.org/abs/2604.08294) | Score prediction on diving and skating, not counting | Does not settle the VLM question |
| Gold Loss Correction, [arXiv 1802.05300](https://arxiv.org/abs/1802.05300) | Noise correction from a small trusted set (from memory, not verified) | Only useful once a trusted set exists |
| GVHMR, [arXiv 2409.06662](https://arxiv.org/abs/2409.06662) | Gravity-aligned motion recovery. No flip or twist evaluation found | 3D stays unproven for rotation counting |

## 4. Directions

| | A. Repair the labels | B. Perception first | C. 3D for rotation counting | D. Narrow the scope | E. Coaching metrics | F. Skeleton-free features |
|---|---|---|---|---|---|---|
| Hypothesis | The plateau comes from labels that are wrong in a structured way | Skeletons of the wrong person or of an upright ghost hide the rotation | A 3D body orientation counts turns where 2D cannot | One judgeable sub-problem done well beats five done badly | Airtime, height and rotation speed are useful without a trick name | Frozen video features carry the rotation and skip the skeleton |
| First experiment | Retrain with Parkour Theory categories as labels, 15 GPU-minutes | Athlete selection scored by the flip counter on the dev half, then one pose swap on 100 clips | GVHMR on 20 clips, unwrap the root rotation | "Is it a flip, how many, which way" on single-trick clips | Airtime and peak rotation speed on the owner's own side-view clips | V-JEPA 2 ViT-L features and a linear probe, a few GPU-hours |
| Kill criterion | Gain under +0.03 gold macro-F1 or fewer than 4 of 5 seeds up | Flip counter under 60% exact on multi-person clips after selection | Under 8 of 12 against hand counts | Flip-presence accuracy under 90% on held-out clips | Airtime error over 2 frames on 10 hand-timed clips | Flip, twist and direction not above the constant by 5 points |
| Payoff | Unblocks every learned model, costs nothing | Needed by any skeleton method | Twist on off-axis tricks, if it works | A first component that meets its bar | Something to show athletes | Tells skeleton failure from label failure |
| Owner time | 0 now, about 20 min a week of checks | 0 | 0 | 0 | 30 min of filming notes | 0 |
| Risk | Parkour Theory categories are binary: they do not give counts | May need a pose fine-tune beyond 8 GPU-hours | GVHMR already returned half the flips here; VRAM | Smaller claim | Hard to score, drifts from judging | Inherits the labels, so it only makes sense after A |

A also has two owner-side options that stay open: finishing the 36 blind labels (24 left), and the VLM blind probe that those labels unlock (at most $5, needs approval and a decision on sending clips to an API).

## 5. Decision

**Direction A, label repair, starting with the free fix.**

- It is the only cause measured on the whole corpus (1,503 clips) against an independent source, and the defect is categorical: a class is missing.
- It costs 15 GPU-minutes and none of the owner's time, and it has a numeric kill line that matches the two earlier probes.
- B, C and F all train or evaluate on these labels, so they would inherit the defect. A goes first whatever comes after.
- D is how A is scoped: flip presence, count and direction first, twist counts once a trusted twist set exists.
- If A is killed, B is next, with its yardstick already built (`probe_flip_yardstick.py`, dev and test halves).

## 6. Plan

| Week | Work | Owner |
|---|---|---|
| 1 | Pre-registered label fix (`2026-11-26-pt-label-fix-prereg.md`), report, merge the board | 24 blind labels, 25 min |
| 2 | If it helps: twist counts from the name parser and the Parkour Theory progression graph, conflicts sent to a small check queue. Second evaluation set: the 36 owner labels plus a held-out half scored on Parkour Theory categories. If it does not: athlete selection on the flip yardstick | 20 min of checks from move names |
| 3 | Perception: athlete selection, then one small pose swap on 100 clips | none |
| 4 | Skeleton-free arm on the repaired labels, or the VLM blind probe if labels and budget exist. Then a one-page architecture decision | yes or no on $5 |

## 7. Open pull requests

- **#4 merge.** The extractor fix is real and this work builds on it.
- **#5 merge.** Scorer code and tests only, nothing runs without labels and keys.
- **#6 merge.** A null result is a result, and it is reproduced.
- **#7 update, then merge.** Regenerate the sheet with the 296 new moves and pre-fill "has flip" and "has twist" from Parkour Theory's categories.

## 8. Stale docs

`NEXT_SESSION_PROMPT.md` and `TODO.md` describe the abandoned GVHMR pipeline. Both are rewritten in this branch to point at this review. `README.md` needs the same treatment in a separate change.
