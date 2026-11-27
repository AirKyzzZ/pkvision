# Three of the owner's clips, read by a VLM and by the rule counter: exploratory notes (2026-11-26)

**Exploratory, three clips, no decision weight.** The owner gave the true trick after each reading. Nothing here was pre-registered. Clips, crops and skeletons stay on the PC under `G:\pkvision\demo` and are not in the repository.

## The clips

| Clip | Capture | Truth |
|---|---|---|
| 1 | 60 fps, 576×1026, recompressed, four bystanders | standing backflip |
| 2 | 27 fps, 1080×1920, handheld, take-off from a box | double cork |
| 3 | iPhone slow motion exported at 30 fps, 1080×1920, sharp frames | scoot double cork |

Each clip was cropped on the athlete (largest person box, CPU detection). The trick window was set by hand on clips 2 and 3: the automatic airborne detection failed with a moving camera.

## Readings

Claude Opus 5.5 read numbered frame sheets through the owner's subscription, with no context. The rule counter is `core/pose/rule_counters.py`, unchanged since 2026-11-04, on RTMPose-x keypoints.

| Clip | Truth | Opus 5.5, naming alone | Rule counter |
|---|---|---|---|
| 1 | 1 flip, 0 twist, backward | backflip, 1 flip, 0 twist | 1 flip, 0 twist (hips), backward |
| 2 | 1 flip, 2 twists, backward, off-axis | cork, 1 flip, 1 twist (three readings, 24 to 41 frames) | 0 to 0.5 flip, 0 to 0.5 twist |
| 3 | 1 flip, 2 twists, backward, off-axis | front 1.5 twist off one foot, on-axis | 1 flip, 2 twists, backward |

- **Opus** got the flip count on all three, described entry, take-off and landing well, and missed twists and the off-axis reading. On clip 2 larger tiles and a frame-by-frame table did not recover the second twist: it wrote that it could not tell chest from back in the blurred frames.
- **Matching against references.** Clip 3 was compared with eight anonymous Parkour Theory demo clips. Two readings both put the cork first; the double cork came second once and outside the top three once. The reference clips have 3 or 4 frames in the air, and Opus misread their twists too.
- **The counter** was right on the two sharp clips and wrong on the 27 fps clip, where the torso keypoints were reported valid on 47 of 47 frames but did not rotate with the body.

## What it suggests

- Sharp frames matter more than the reader. The same counter that scored 3/12 in November reads a 120 fps clip correctly. The September survey had already put the limit at keypoint reliability under blur, not at frame count.
- The count depends on the window. On clip 3 the range from the set-up to the landing gives 1 flip and 2 twists; the airborne frames alone give 0.5 flip and 1.5 twists, which is also what Opus read. Part of the rotation is taken on the ground, and a trick name counts all of it.
- The two tools fail differently: the VLM on counts and axis, the counter on blurred footage and on finding the window.

## Limits

Three clips, chosen by the owner, windows set by hand after seeing the footage, real frame rate of the slow-motion clip unknown (the result does not change between 60, 120 and 240). The shoulder-lobe count, a secondary signal, showed a spurious half twist on clip 1.

## Also learnt that day

- Through OpenRouter, Gemini samples video at about one frame per second, so clips were retimed 10 times slower.
- The owner's OpenRouter key has a $20 monthly cap, reached during the Gemini second-opinion collection: 549 of 1,474 training clips answered, pre-registration in `2026-11-26-vlm-second-opinion-prereg.md`, collection paused and resumable.
