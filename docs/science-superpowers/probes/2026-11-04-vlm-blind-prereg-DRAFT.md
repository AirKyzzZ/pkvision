# VLM blind probe: draft pre-registration (2026-11-04)

**Status: draft, not run.** No clip has been sent anywhere. It needs the owner's OK to spend money and send clips to outside services, plus his blind labels. Freeze this file in a commit before the first API call and drop `DRAFT` from the name. Step 2 of "Validation before coding" (`survey/2026-09-28/00-synthesis.md`).

**Question:** can a frontier VLM read twist count on unseen parkour clips better than a constant answer, without guessing from trick priors?

## Truth

The 36 clips in `data/labeling/retest_2026-11/queue.json` (do not open it before labeling), labeled by the owner on `label.html` **before** any VLM output is seen, so that proposals cannot anchor him. The labels' sha256 goes in the freeze commit, and the scorer requires it (`--labels-sha256`) and stops if the file differs. Labels must not change after that. Clips left unlabeled, such as unplayable ones, must be accepted explicitly with `--accept-unlabeled`.

Scored: finished labels marked "one trick" with numeric flip and twist. Others are reported and excluded. **With under 20 scored clips an arm is not evaluable**, however well it scores. Labels from outside the queue stop the run.

The queue was chosen from manifest and name-parser strata by sha1 order, with no pose or VLM output. Its strata are deliberately hard: corks, likely odd twists, twist 2+.

## Inputs

Built by `scripts/prepare_vlm_blind.py` into a gitignored folder: opaque ids, re-encoded to H.264 with container metadata dropped, audio removed.

**On-screen text:** nothing is stripped, although the synthesis asks for it. An AI agent viewed 8 evenly spaced frames of each forward clip and saw no captions or trick names, only gym logos, clothing and graffiti. A caption shown between those frames could still leak. The label page has a "name on screen" answer, and those clips are excluded.

Only `upload/` may go to a service. Its file names hide clip and direction, and the name mapping sits outside it.

## Arms

| Arm | Model | Input | Settings |
|---|---|---|---|
| G-pro | Gemini 3.1 Pro | native video | 10 fps, medium resolution |
| G-flash | Gemini 3.8 Flash | native video | 10 fps, medium resolution |
| G-grid | Gemini 3.8 Flash | 24-frame grids | settles grid vs native |
| C-grid | Claude Opus 5.5 | 24-frame grids | |

One vote per clip per arm, using this exact prompt:

```text
This shows one person performing an acrobatic movement, in time order. Grids, when used, are numbered frames.
Track the person's body orientation frame by frame. Do not try to name the movement.
Count, for the main airborne movement only:
- flip: full head-over-heels rotations, in steps of 0.5
- twist: full rotations about the head-to-toe axis, in steps of 0.5
- direction: the flip direction at takeoff relative to the chest: "forward", "backward", "side", or null if there is no flip
- cork: true if the rotation axis is off-axis, as in a corkscrew
Set unsure to true if you cannot see enough to count.
Reply with JSON only: {"flip": number, "twist": number, "direction": string or null, "cork": boolean, "unsure": boolean}
```

## Staged labeling and collection (added 2026-11-17, before any API call)

The owner labels in batches. Collection may follow the batches under these rules, which keep the design above intact:

1. **A clip is sent forward only after its label is frozen.** Each batch file gets its sha256 recorded in a commit; `run_vlm_blind.py` refuses a labels file whose hash differs and writes the hash into every vote row. At final scoring, `score_vlm_blind.py --stage PATH=SHA256` checks that every batch appears unchanged in the final labels and that every vote was requested against a frozen file.
2. **Prompt, models, settings and prices are frozen in `run_vlm_blind.py` before the first call** and do not change between batches. If a model is withdrawn between batches, that arm is reported incomplete; it is not replaced.
3. **Reversed clips go only at the end**, once `control-ids` can be computed from the full truth set.
4. **Votes are append-only and never re-requested.** An answered (arm, id, variant) is final; only failed calls are retried. Early votes are reused in the final scoring, never discarded, so staging adds no cost.
5. **One verdict, on the full set.** Nothing computed on a partial set has decision weight, and no VLM output, aggregate or per clip, is shown to the owner until all 36 clips are labeled. The remaining clips must be labeled with the owner as blind as for the first ones.
6. **Settings fixed here:** Claude arm at `output_config.effort = "medium"`, `max_tokens = 1024`, thinking left at the model default, no server-side fallback (a refusal is a missing vote); Gemini arms at 10 fps, medium media resolution, JSON response mode. The first live call per arm is a `--limit 1` smoke test counted in the budget.

Batch 1: 12 clips, sha256 `b3fe8f68cc0efca5dea2aa23e57f824d53173c4d03539b74ba7969106d8751d8`, labeled 2026-11-17 without seeing the queue.

## Reversed-clip control

Each arm also sees 12 clips played backwards: scored clips the owner labeled with at least one flip and whole twists, ordered by a salted sha1 of `slug`. `score_vlm_blind.py control-ids` lists them. Odd half twists are left out because the reversed clip starts facing the other way, which cancels the reversal. Direction means the flip direction at takeoff, relative to the chest.

A clip counts when an unsure-free forward pass answers `forward` or `backward`. **Fail** if a usable reversed pass gives the opposite direction on under 75% of those, or if fewer than 6 clips count.

**What this shows:** failing suggests the arm is not reading motion. Passing clears direction only. It does not show the twist count was read rather than guessed from a recognised trick, and an arm that notices the reversal could pass by inverting its answer.

## Metrics and decision rule

Primary, per arm: exact half-twist accuracy on all scored clips. Unsure and malformed answers count as wrong. A missing vote also scores as wrong in the accuracy figure, but makes the verdict incomplete. A duplicate vote or an unknown arm, id or variant stops the run, so retries cannot pick a result.

Always report these alongside, because the rule-counter probe showed a constant can match a counter:
- always-0-twist and most-common-twist baselines;
- accuracy by the owner's labels, especially twist ≥ 1.5 and corks, because queue strata can be wrong;
- flip accuracy and the same control results.

`scripts/score_vlm_blind.py` applies this rule from the labels and a votes file. It prints no clip names by default. Per arm, with **incomplete** for missing votes or unfinished labels and **not evaluable** for under 20 scored clips or under 6 eligible control clips, in place of a verdict:
- **Killed:** below 75% twist (the synthesis's 9/12), **or** not above the best constant baseline, **or** reversed-clip control fails.
- **Promising:** at least 75%, above baselines, control passes. Next is the label-source audit, not bulk labeling.

With about 30 clips, 75% has a 95% CI near 0.56–0.89, and the bar is really 23/30. This can rank or kill. It cannot confirm the 85% bar. With four arms, a 60% arm has about a 16% chance that at least one clears the accuracy bar, so a single passing arm needs confirming on new clips.

## Cost estimate

About $4–7 at standard prices, or $2–3.50 with 50% batch pricing, using 90 s of forward video, 30 s reversed, 48 clip passes per arm, and September survey prices (`survey/2026-09-28/02-video-vlms.md`). The range depends on output length. Those prices need re-checking before any run. Stop at a hard cap of $5, so use batch pricing or drop an arm.

## Open before running

Scorer work is done: `check-labels` ingests the owner's download, stops on partial labels and prints the hash to freeze; `score` and `control-ids` require `--labels-sha256` and stop on a mismatch, `control-ids` refuses a set that cannot reach a verdict, `parse_reply` gives the future runner a strict parser, and the thresholds and CLI flags have boundary tests.

Also:

- The runner exists (`scripts/run_vlm_blind.py`, rehearsed end to end with `--provider fake`). Its model ids (`gemini-3.1-pro-preview`, `gemini-3.8-flash`, `claude-opus-5-5`) and prices must be confirmed with `list-models` and the price pages once keys exist.
- API keys: none on this PC.
- Model availability and prices.
- Whether the clips' licence allows sending them to third-party APIs.
