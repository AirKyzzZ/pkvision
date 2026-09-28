# Is iPhone-only, open-vocabulary parkour trick recognition feasible in Sep 2026?

**Research question:** As of September 2026, does any approach that fits a student budget and uses only iPhone cameras recover the compositional attributes of a parkour trick (flip count, twist count, direction, axis, context) from real footage well enough to name known tricks and describe unseen ones, with **twist count** as the gating attribute?

**Background / motivation:** Six approach families failed between March and July 2026 (GVHMR physics, metric learning, VideoMAE attributes, CLIP, zero-shot VLMs, 2D-skeleton cue model). The June verdict was "the wall is labels, not the model"; the July Claude-as-judge probe then showed frame-grid VLM judging reads flip/twist/direction well on clean clips (flip 11/11, twist 10/11, dir 11/11 vs corrected references) and exposed the "gold" labels as ~17% wrong on flip and ~35% wrong on twist. Twist was always the crux: the cue contract says twist ≥ 1.5 is unrecoverable from one camera and must abstain, which July partly contradicted. This investigation decides whether to keep building and on which architecture, before any code is written.

**Objectives (fixed):**
- V1 (proof of concept): open-vocabulary recognition. Output = compositional name ("cork · 2 flips · 1.5 twists · off wall · backward"); correct only if every part is correct. Known tricks map to the 2,677-trick ontology; unseen tricks are described by their parts.
- End product: FIG full-run judge-assist (segment a run, propose trick + D-score, a human judge confirms).

**Hypotheses:**
- H0: No approach within the constraints reaches the feasibility threshold for twist count on single-iPhone footage, and a second synced iPhone does not close the gap.
- H1: At least one approach within the constraints reaches the threshold on single-iPhone footage, or on two audio-synced iPhones.

## Sub-questions (ordered by how much they gate the rest)

1. **Twist (the crux).** Which approach families can count twists 0 to 3+ in half-twist steps, including on corks and off-axis tricks, and at what accuracy?
   - Families to cover: frontier video VLMs (frame grids, tool use, high fps), monocular 3D human mesh recovery released or fine-tuned on acrobatics since June 2026, 2D pose + temporal models, two-view triangulation, slow-motion capture (iPhone 120/240 fps), and synthetic or rendered acrobatic training data.
   - Analog domains to mine: diving (FineDiving and successors), gymnastics (FineGym; Fujitsu Judging Support System used by FIG), trampoline, freestyle ski/snowboard spin counting, figure-skating jump rotation counting, tricking.
   - Re-test the 2026 claims with Sep-2026 evidence:
     1. VLMs can't count flips/twists.
     2. Monocular HMR fails on acrobatics (AMASS has none).
     3. Twist ≥ 1.5 is geometrically invisible from one camera.
     4. 2D pose fails upside-down.
     5. Skeleton SSL is unvalidated on acrobatics.
     6. Twist needs hardware genlock (already superseded by audio sync).
2. **Other attributes.** Flip count, direction, axis (on-axis vs cork/off-axis), and context (ground/wall/bar/vault). Which are already solved and which are not?
3. **Composition.** How to go from attributes to a name without per-attribute errors compounding. July's result was ~0.5 per cue giving ~13.6% end-to-end. Covers abstention, top-k, and a decoder over the 2,677 ontology, with the trick-name parser as the bridge.
4. **Labels without manual labelling.** Covers VLM weak labels (quality, cost, and quota as the July blocker), synthetic data, SSL, and verify-only human loops. What is the cheapest route to a training/eval set that is trustworthy on twist?
5. **Real conditions.** The domain gap from curated clips to handheld iPhone footage: distance, framing, lighting, occlusion, fps. For the end product, full-run temporal segmentation.
6. **Prior art.** Any parkour/freerunning auto-judging system, commercial app, or FIG pilot. What does Fujitsu JSS need (lidar, multi-cam) that an iPhone setup cannot provide?

## Method

- **Desk research first:** papers (arXiv 2025–2026), benchmarks, model releases and cards, open datasets, and deployed systems. Every claim is graded: in-domain measurement > analog-domain measurement > vendor/claim.
- **Micro-probes, allowed and capped:** ≤ 12 clips per candidate model, free tier or ≤ $5 per probe, on existing clips only.
  - The clips are the 12 July reference clips and the ~13 own-footage clips.
  - Probes are exploratory. They inform the design and do not count as confirmatory evidence.
  - Probe clips must include hard cases: twist ≥ 1.5, corks, bad angles.
- **No model training and no bulk labelling** in this phase.

## Constraints

- **Budget:** student. The total is not yet fixed, so report the cost of every candidate in tiers: free / ≤ $20 / ≤ $50 / ≤ $150. Claude Max subscription quota is a real limit. Modal credits are exhausted. Rented 4090s cost ~$0.29–0.69/h. There is a local Mac and an RTX 2060 Windows desktop.
- **Capture:** iPhones only. Two phones synced by audio cross-correlation is an available, solved option. No wearables, lidar, or broadcast rigs.
- **Labels:** no manual labelling. The owner will verify small queues (verify-not-label) but will not annotate datasets. Naming a trick you performed while filming counts as free ground truth.

## Assets on hand (structure and provenance only)

- **Clips:** 1,618 cropped parkourtheory clips (single view, one trick each) and 33 RTMPose-x keypoint shards on Modal volume `pkvision-data`. Local copies were deleted 2026-07-17.
- **Ontologies:** `data/fig_tricks_2025.json` (149 tricks, 15 known alias duplicates) and `data/unified_tricks.json` (2,677 tricks).
- **Labels:**
  - The "gold 99" are FIG-join rows, audited as unreliable on twist.
  - A 26-clip dispute queue.
  - 74 Claude bulk labels in `data/labeling/claude_labels/labels_v1.jsonl`.
  - 12 July reference clips with corrected labels.
  - The trick-name parser: flip/direction precision 0.98, other cues abstain.
- **Own footage:**
  - 5 single-trick clips + 2 full runs, recoverable from git.
  - More (`IMG_5985.mov`, `elis-final-run-japan.mp4`, `double-pov/`) probably on the GPU desktop.

## What counts as an answer (proposed thresholds, to confirm)

- **Per attribute:** at a coverage of ≥ 70% (the system may abstain on the rest), precision ≥ 90% on flip count, direction and context, and ≥ 85% on twist count. Twist must be measured separately on ≥ 1.5 twists and on corks.
- **Compositional:** exact match on all parts ≥ 60% top-1 and ≥ 80% top-3 on held-out parkourtheory clips. On own footage, it must pass on the majority of the ~13 clips.
- **Verdict per sub-question:** GO / CONDITIONAL (names the condition, e.g. "only with two phones" or "only at 240 fps") / NO-GO, each with its evidence grade.

## Deliverables

1. A feasibility report with a verdict per sub-question and the re-test status of each of the six 2026 claims.
2. 2–3 candidate end-to-end architectures ranked. For each: expected accuracy with source, cost per budget tier, capture requirements, the main risk, and what would kill it.
3. A minimal "real conditions" validation plan to run before coding: the probe or filming session that would confirm or kill the top candidate cheaply.
4. Only then, an implementation plan for the chosen V1.

## Scope & exclusions

- **Out of scope for now:** execution scoring (E-score), full-run segmentation beyond noting its feasibility, the paper in `paper/`, and ontology alias cleanup (CDF Task 4) unless the chosen architecture depends on it.
- **Not revisited unless new evidence appears:** GVHMR physics, metric learning, force-from-video.

## Open questions for the owner

- Total budget cap (deferred until costs per tier are known).
- FIG stakeholder: is there a date or a specific demo they expect?
