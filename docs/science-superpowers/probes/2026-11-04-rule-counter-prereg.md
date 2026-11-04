# Rule-counter probe: pre-registration (2026-11-04)

Step 1 of "Validation before coding" (`survey/2026-09-28/00-synthesis.md`), design from `survey/2026-09-28/01-twist-analog-sports.md` §5. Written and committed **before any counter output on real clips exists**. Thresholds below are not to be tuned on these clips; anything changed afterwards is reported as exploratory.

**Question:** does a training-free NS-AQA-style 2D counter recover twist and flip counts on parkour clips, and does upright-crop or 4-way rotation pose inference help?

## Truth set (no owner-verified twist labels exist yet)

Reference R = gold-99 clips (`data/labeling/claude_probe/gold99_votes.jsonl`) where all hold:
- Claude consensus not `multiple_tricks`, with unanimous provenance on flip and twist;
- consensus equals the manifest (`attribute_manifest.json`) on flip and twist;
- the name parser, when it fires on flip or twist, agrees.

This gives 47 clips. `verdict=ok` alone is **not** used: it includes `flyaway_quadruple_full` and `front_full` with consensus twist 0. R is **provisional, not owner-verified**, and it excludes the hardest clips by construction. No odd ≥1.5-twist clip and no cork survives the filter, so the "odd ≥1.5 + cork" clause of the §5 rule **cannot be evaluated** here.

**Decision-12:** within R, order by `sha1(slug)` and take the first 3 of each stratum: twist 0 with flip ≥ 1; twist 0.5; twist 1; twist ≥ 2. All 47 are also reported per stratum.

**Own footage smoke test:** `data/final_clips/` (owner-named): backflip, frontflip, gainer = 0 twists; back_double_full = 2 twists; double_cork raw counts only (twist/flip naming conflict, spec §"Known naming conflicts"). 5 clips; smoke only, no decision weight.

## Pose arms (all RTMPose-x body7 + YOLOX-x via rtmlib `performance`)

- **A baseline:** as `scripts/extract_pose.py`.
- **B upright:** rotate the frame about the person-box centre by minus the previous frame's torso angle, infer, rotate keypoints back. First frame, or torso invalid: use arm C's choice.
- **C 4-way:** infer at 0/90/180/270°, keep the highest mean keypoint score.

Same per-frame person box for all arms: largest YOLOX box by area.

## Counter (`core/pose/rule_counters.py`, pure numpy)

- COCO-17. Torso points valid at conf ≥ 0.3, others linearly interpolated. Scale S = median mid-shoulder to mid-hip distance.
- **Window:** largest run of frames with torso angular speed > 200°/s or |d(|h|/S)/dt| > 1.5/s, merging gaps ≤ 0.2 s, padded 0.1 s. No run: flip = twist = 0. If under 60% of window frames are valid: **abstain**.
- **Flip:** |net unwrapped torso angle over the window| / 360, rounded to nearest 0.5.
- **Twist (primary):** h = r_hip − l_hip. Median-3 filter on |h|/S, reject > 1/1.5 as outlier (held). Rising edges from below inner = 1/3.4 to above outer = 1/3.2, each = 0.5 twist. Length-only, so L/R swaps do not add counts. NS-AQA's origin-crossing rule is deliberately **dropped** for that reason.
- **Twist (secondary, no decision weight):** same on |s|/S with shoulders, thresholds scaled by the median shoulder-to-hip width ratio.
- **Direction:** forward if the head rotates toward the facing side (nose x vs mid-shoulder x at window start). Abstain if |net angle| < 180° or no nose. Only scored where truth is forward/backward.
- **Diagnostic:** per-clip count of frames where h's x-sign flips while |h|/S is above outer (candidate L/R swaps).

## Metrics and decision rule

Primary: exact half-twist accuracy on Decision-12, abstain counted as wrong, per arm. Secondary: off-by-one rate, coverage, flip accuracy, direction accuracy, all-47 per stratum, swap diagnostic.

Per arm, using the stricter kill line from the synthesis:
- **≥ 10/12:** promising. Scale to the 1,618 corpus as weak twist labels, only after owner verification of the reference and a test on odd ≥1.5 and corks.
- **7–9/12:** inspect traces: keypoint failure (then swap repair or better pose) vs view ambiguity (then two-phone).
- **≤ 6/12 on every arm:** architecture #1 ("count, don't learn") is killed as specified.

With n = 12, this can only rank or kill (9/12 has 95% CI 0.47–0.91). It cannot confirm the 85% twist bar.

## Known risks, stated in advance

Landing rolls can inflate flips, handheld roll adds angle, tuck and blur can drop keypoints, truth is VLM-anchored, and R is enriched for easy clips and zero-twist tricks.
