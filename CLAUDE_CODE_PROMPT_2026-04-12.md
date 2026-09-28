# Claude Code Prompt For Sunday, April 12, 2026

This file supersedes the assumptions in `NEXT_SESSION_PROMPT.md`.

The old prompt is useful as project history, but it is no longer an accurate guide for what to build next. In particular, it overstates the success of the VLM-first route. The prompt below is designed to be pasted into Claude Code as the working brief for the day.

## Paste Into Claude Code

```text
You are continuing work on the `pkvision` repository.

Read `AGENTS.md` and `CLAUDE.md`, then follow this brief. Do not treat `NEXT_SESSION_PROMPT.md` as the source of truth for today. It is outdated and too optimistic about the VLM path.

Your mission for Sunday, April 12, 2026 is not to brainstorm or write paper prose. Your mission is to make the recognition system measurably more real on the actual benchmark and actual run files in this repo.

The core task is:
Build or substantially advance a system that can identify FIG parkour tricks better than the current pipelines on real competition footage.

The important reframing is:
- The problem is not open-world trick naming.
- The label set is fixed by the FIG dataset.
- The main failure is identifying the correct FIG trick from video, especially under twist ambiguity, takeoff ambiguity, and wall/apparatus ambiguity.

You must work from the actual codebase and actual provided clips, not from generic papers or vague model suggestions.

## Ground Truth You Should Assume

These are the strongest findings from the last deep review of this repo:

1. Segmentation is not the main bottleneck anymore.
- `scripts/vlm_judge.py --preview` produced plausible segments on real runs.
- `data/run_testing/IMG_5985.mov` segmented into 3 tricks with good timings.
- `data/run_testing/test_run_2.mp4` segmented into 7 tricks with plausible timings.
- This means recognition is currently weaker than segmentation.

2. The FIG ontology is the real target, not free-form naming.
- Source of truth: `data/fig_tricks_2025.json`
- It defines 149 FIG tricks across 4 categories and includes structured fields like `flip`, `twist`, `direction`, `axis`, `entry`, aliases, and explicit disambiguation groups.
- The disambiguation section is especially important because it tells you which cues actually separate confusing tricks.

3. The current exact-trick training path is not trustworthy enough as the main route.
- `data/v5_full_training/trick_manifest.json` has 96 classes but effectively only one clip per class.
- `scripts/build_v5_full_dataset.py` collapses multiple FIG labels onto approximate ParkourTheory clips.
- That builder also leaks benchmark clips from `data/final_clips/` into training by default.
- The resulting exact dataset is too thin and too contaminated to treat as a real exact-class benchmark.

4. The attribute path is more useful than the exact-class path, but it is still coarse.
- `data/v5_attribute_training/attribute_manifest.json` has 1622 clips.
- Only a small subset is FIG-grounded; most labels come from weak `unified_tricks` heuristics.
- This is still useful for coarse pretraining or structured prediction.
- The current attribute trainer is too coarse on twist: `0` vs `has_twist` is not enough for high-twist FIG disambiguation.

5. The current VLM path is not solved.
- Benchmark source of truth is `paper/experiments/run_benchmark.py` with labels in `paper/experiments/ground_truth.csv`.
- The checked-in benchmark results are poor:
  - `claude-cli-strip`: 0/13 exact on POOL-B
  - `gemini-2.5-flash`: 1/5 exact on checked-in rows
  - `qwen2.5-vl-72b`: 0/7 exact on checked-in rows
- The recurring failures are:
  - twist under-counting
  - takeoff confusion
  - wall-context loss
- Therefore VLM may still be useful as a component or weak labeler, but it is not a complete judge.

6. The current 3D mesh path is also not solved.
- GVHMR/SMPL-based paths can be useful for ideas and heuristics, but not as a trusted measurement front end for high-twist competition footage.
- High-twist and off-axis tricks are still unreliable there.

7. Multi-camera is allowed and should be treated as a serious option.
- The product does not have to be single-camera only.
- If higher accuracy requires two synchronized cameras, that is acceptable.
- Synchronized iPhone cameras are a valid deployment assumption for competitions.

## What Success Looks Like Today

By the end of this session, aim to produce all of the following:

1. A cleaner benchmark path for comparing recognizers on the existing eval set.
2. A structured FIG decoder that ranks valid FIG candidates from predicted cues instead of relying on free-form naming and fuzzy cleanup.
3. At least one stronger baseline recognizer path than the current VLM-only or metric-only route.
4. Measured results on actual repo clips, especially POOL-B and at least one full run from POOL-A.
5. A short write-up in the repo describing:
   - what changed
   - what worked
   - what still fails
   - what the next step should be

Do not stop at architecture talk if you can implement and test.

## Highest-Priority Source Files

Read these first and treat them as the operational map:

- `paper/experiments/ground_truth.csv`
- `paper/experiments/run_benchmark.py`
- `paper/experiments/results.jsonl`
- `data/fig_tricks_2025.json`
- `core/vlm/fig_matcher.py`
- `core/video.py`
- `core/segmentation.py`
- `core/pose/features.py`
- `scripts/vlm_judge.py`
- `scripts/inference_v5.py`
- `scripts/build_attribute_dataset.py`
- `scripts/train_attributes.py`
- `scripts/build_v5_full_dataset.py`
- `data/v5_attribute_training/attribute_manifest.json`
- `data/v5_full_training/trick_manifest.json`
- `data/trick_families.json`

Also review these for failure analysis and repo history:

- `paper/sections/04_journey.tex`
- `paper/sections/05_experiments.tex`
- `paper/sections/06_discussion.tex`

## Files And Assets You Should Reuse

These are likely worth keeping:

- `core/video.py`
  - usable YOLO-based athlete tracking and stable 2D signals
- `core/segmentation.py`
  - segmentation already works better than recognition
- `core/pose/features.py`
  - best reusable 2D feature path for structured cues
- `scripts/build_attribute_dataset.py`
  - usable builder for coarse structured labels
- `paper/experiments/run_benchmark.py`
  - existing benchmark harness, but it needs to become less VLM-specific

These are useful for ideas but should not be trusted as the main measurement path:

- `core/pose/rotation_tracker.py`
- `scripts/analyze_3d.py`
- `scripts/motionbert_3d_judge.py`
- `scripts/stereo_3d_judge.py`

## Things You Must Not Do

Do not waste the day on any of the following:

- Do not assume VLM solved the problem.
- Do not train another end-to-end exact trick classifier on the current exact dataset and call that progress.
- Do not contaminate evaluation by training on `data/final_clips/`.
- Do not treat `POOL-A` full-run scoring and `POOL-B` single-trick recognition as the same task.
- Do not spend most of the session writing paper sections or broad research notes.
- Do not rely on free-form VLM trick names plus fuzzy matching as the main recognition strategy.

## The Build Strategy To Follow

If you have to choose one path, choose this one:

### Step A: Fix evaluation before model work

Use `paper/experiments/ground_truth.csv` as the fixed benchmark.

Split the evaluation conceptually:
- `POOL-B` = single-trick recognition benchmark
- `POOL-A` = full-run segmentation plus ordering benchmark

Improve `paper/experiments/run_benchmark.py` or add an adjacent adapter so non-VLM recognizers can be evaluated without pretending to be VLMs.

You should end up with a generic recognizer interface or equivalent adapter so a structured model can be benchmarked the same way as the current VLM path.

### Step B: Build a structured FIG decoder

Create a decoder that takes predicted cues and ranks FIG tricks.

At minimum, use:
- context
- family
- direction
- flip count or flip bin
- twist count or twist bin
- takeoff / entry
- hand-contact or wall/apparatus cues if available

The decoder should use `data/fig_tricks_2025.json` directly.

The point is:
Do not wait for a model to output the exact trick name. Predict structured evidence and decode against the fixed FIG table.

### Step C: Build a stronger single-camera baseline

Use the existing 2D front end:
- tracking from `core/video.py`
- segmentation from `core/segmentation.py`
- 2D structured features from `core/pose/features.py`

Build specialist predictors rather than one exact labeler:
- `ContextNet` or context module
- `TakeoffNet` or takeoff/entry classifier
- `TwistNet` with more useful bins than `0` vs `has_twist`
- optionally a family classifier

Heuristics are acceptable as part of the first version if they are measurable and improve benchmark behavior.

If you need a fast initial implementation, start with:
- context detector
- gainer-vs-standing takeoff logic
- better twist grouping
- FIG candidate reranking

That alone may beat the current VLM path on the hardest confusions.

### Step D: Evaluate on real footage

Run on these actual files:

Single-trick benchmark:
- `data/final_clips/backflip.mp4`
- `data/final_clips/gainer.MOV`
- `data/final_clips/double_cork.mp4`
- `data/final_clips/frontflip.mp4`
- `data/final_clips/back_double_full.mp4`
- plus the segmented POOL-B clips under `data/vlm_clips/`

Full runs:
- `data/run_testing/IMG_5985.mov`
- `data/run_testing/test_run_2.mp4`
- `data/run_testing/IMG_4243.MOV`
- `data/run_testing/elis-final-run-japan.mp4`

Report actual outputs, not just code changes.

### Step E: If time remains, spike the multi-camera path

Because multi-camera is allowed, do a scoped feasibility check for synchronized smartphone capture.

Do not try to solve full multi-camera production today.

Instead answer:
- what minimal synchronized two-camera setup is needed
- whether the existing stereo path is worth salvaging
- whether a simpler iPhone-based calibration/sync path should be built instead

If a spike is possible, focus on whether two views materially improve twist observability on the known high-twist problem clips.

## Concrete Problems To Solve First

Prioritize these confusions:

1. `Backflip` vs `Gainer` vs `Running Gainer` vs `Kong Gainer`
2. `Backflip 720` vs lower-twist backward flips
3. `Gainer 360` / `Gainer 720` / `Gainer 1080`
4. `Kroc` vs `Double Cork` vs `Triple Cork`
5. `Wall Inward Frontflip` vs ground frontflip variants

These are high-value because they match the actual failure gallery already documented in the repo.

## Benchmark Facts You Should Use

- `ground_truth.csv` currently has 17 labeled rows total
- `POOL-A` contains full runs with comma-separated ordered trick labels
- `POOL-B` contains single-trick clips used for quantitative checks
- the current results file is append-only, so avoid double-counting old runs

When you benchmark:
- use a fresh output file for new runs
- do not rely only on exact top-1
- also inspect top-k candidate quality and confusion behavior

If you improve candidate ranking but not top-1 yet, that is still meaningful progress.

## Data Quality Warnings

Keep these in mind while building:

- `data/v5_full_training/trick_manifest.json` is not a reliable exact-trick training source
- `scripts/build_v5_full_dataset.py` can leak benchmark clips into training
- `data/v5_attribute_training/attribute_manifest.json` is useful but mostly weakly labeled
- `frame_cat` and `context` disagree for many attribute rows, so do not assume directory category is clean ontology
- the current attribute trainer collapses twist too aggressively

## External References Worth Considering

Use these only if they help implementation today:

- MMPose: https://github.com/open-mmlab/mmpose
  - relevant because it supports stronger pose extraction and 3D whole-body options
- MMAction2: https://github.com/open-mmlab/mmaction2
  - relevant because it supports skeleton-based action recognition and localization
- AthletePose3D: https://github.com/calvinyeungck/AthletePose3D
  - relevant because it shows sports-specific pose fine-tuning matters a lot for athletic motion
- OpenCap: https://github.com/opencap-org/opencap-core
  - relevant because it demonstrates multi-camera smartphone capture is practical

Prefer official or primary sources only if you browse.

## Deliverables Required Before You Finish

Do not finish without producing:

1. Code changes or a clearly scoped prototype in the repo
2. A fresh benchmark or evaluation output on actual repo clips
3. A short markdown summary of:
   - what was built
   - exact commands run
   - results
   - limitations
   - recommended next step

## Decision Rule If You Get Stuck

If you get pulled in multiple directions, choose the path that creates the most trustworthy feedback loop:

benchmark harness + structured FIG decoder + real clip evaluation

That is more valuable than another speculative model experiment.

Ask the user a question only if truly blocked. Otherwise make reasonable assumptions, implement, test, and report concrete evidence.
```

