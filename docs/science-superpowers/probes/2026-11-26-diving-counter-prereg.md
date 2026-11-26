# Does our rule counter work on diving with our own skeletons? Pre-registration (2026-11-26)

Written and committed **before** the counter ran on any dive. Thresholds are not to be changed after seeing results; anything added later is reported as exploratory.

## Why this test

The rule counter (`core/pose/rule_counters.py`, frozen since 2026-11-04) copies NS-AQA, which reports 93.3% exact twist count and 97.3% exact somersault count on MTL-AQA dives. On parkour it scored 3/12 in November. On 2026-11-26 it read a 120 fps clip of the owner correctly (1 flip, 2 twists) and failed on a 27 fps clip of a similar trick (exploratory, three clips).

We have almost no parkour footage with trusted labels. MTL-AQA has 1,412 dives with official somersault and twist counts, and its source videos are still online.

**Question:** with our own skeletons (RTMPose, not trained on divers), does the frozen counter come close to the published diving accuracy?

$0. Dive segments are fetched from the public source videos, used on the PC only, and deleted after pose extraction. About one hour of the RTX 2060 behind the GPU guard.

## Data

- **Labels:** `final_annotations_dict.pkl` from the MTL-AQA repository: `ss_no` and `tw_no` in half turns, `start_frame` and `end_frame` per dive. Read with a loader that refuses any import.
- **Videos:** the nine source videos available at 720p (1, 2, 3, 4, 5, 6, 14, 17, 18), all 25 fps. The six others are 360p or 480p and are left out.
- **Pilot:** 200 dives, drawn by twist count in sha1 order: up to 20, 25, 35, 25, 30, 11 and 5 dives for 0.5 to 3.5 twists, then untwisted dives up to 200. With what the nine videos hold this gives 3, 14, 35, 25, 30, 11 and 3 twisting dives and 79 untwisted ones. A dive that cannot be fetched is replaced by the next one of its stratum. Frozen in `pilot()` of `scripts/probe_diving_counter.py`.
- **Frames:** the annotated range of each dive only. Alignment was checked by eye on two dives before this freeze: the dive sits inside the fetched segment.

## Method, frozen

- **Skeletons:** YOLOX-x and RTMPose-x through rtmlib, detector first. The athlete is the largest person box in each frame. A frame with nobody detected stays empty.
- **Arms:** **C**, best of four rotations of the crop by mean keypoint confidence (primary, since divers spend half the flight upside down), and **A**, the plain pose.
- **Counter:** `count(keypoints, 25)` unchanged. Somersaults are its flip count, twists its hip-lobe count, both in steps of 0.5.

## Metric and decision rule

Per arm, on answered dives: exact somersault count and exact twist count. Coverage is the share of dives the counter answers.

Verdict on arm C:
- **counter validated:** both at least 0.80, with coverage at least 0.70;
- **skeleton or counter is the problem:** either under 0.60, or coverage under 0.50;
- **inconclusive:** otherwise.

Reported alongside: accuracy within half a turn, twist accuracy per true twist count, the "always 0 twist" and most-common-somersault constants, mean keypoint validity, and arm A.

**Prediction:** skeleton or counter is the problem. The diver is about 100 pixels tall, tucked and blurred at 25 fps, among spectators, and NS-AQA trained its own pose model on divers. The two alignment clips are what this prediction rests on.

## What follows

- **Validated:** the counting logic is sound and our parkour failures come from capture conditions. The owner's filming session becomes the priority.
- **Problem:** the next step is to find which part fails, on these same dives where the labels are trusted: athlete selection, keypoints on small tucked bodies, or the counting rule. Filming more parkour would not fix it.
- **Inconclusive:** extend to the remaining 720p dives.

## Known limits, stated in advance

- Diving is not parkour: 1.5 s of flight, twists done with an open body before or after tucked somersaults, a broadcast camera that follows the diver.
- The counter has no board, splash or body-openness detector, unlike NS-AQA, so parity is not expected. The bar is 0.80, not 0.93.
- The stratified pilot over-represents twisting dives, so its accuracy is not comparable to a figure on all dives.
- Spectators and judges are in frame. A wrong athlete is counted as a failure of the method.
