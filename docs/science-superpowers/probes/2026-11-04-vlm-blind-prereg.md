# VLM blind probe: pre-registration (drafted 2026-11-04, frozen 2026-11-26)

**Status: frozen on 2026-11-26, before any queue clip was sent.** The owner approved sending clips to outside models through his OpenRouter account that day. Changes made at the freeze are listed in "Frozen on 2026-11-26" below. Step 2 of "Validation before coding" (`survey/2026-09-28/00-synthesis.md`).

**Question:** can a frontier VLM read twist count on unseen parkour clips better than a constant answer, without guessing from trick priors?

## Truth

The 36 clips in `data/labeling/retest_2026-11/queue.json` (do not open it before labeling), labeled by the owner on `label.html` **before** any VLM output is seen, so that proposals cannot anchor him. The labels' sha256 goes in the freeze commit, and the scorer requires it (`--labels-sha256`) and stops if the file differs. Labels must not change after that. Clips left unlabeled, such as unplayable ones, must be accepted explicitly with `--accept-unlabeled`.

Scored: finished labels marked "one trick" with numeric flip and twist. Others are reported and excluded. **With under 20 scored clips an arm is not evaluable**, however well it scores. Labels from outside the queue stop the run.

The queue was chosen from manifest and name-parser strata by sha1 order, with no pose or VLM output. Its strata are deliberately hard: corks, likely odd twists, twist 2+.

## Inputs

Built by `scripts/prepare_vlm_blind.py` into a gitignored folder: opaque ids, re-encoded to H.264 with container metadata dropped, audio removed. Each video also exists retimed 10 times slower, same frames.

**On-screen text:** nothing is stripped, although the synthesis asks for it. An AI agent viewed 8 evenly spaced frames of each forward clip and saw no captions or trick names, only gym logos, clothing and graffiti. A caption shown between those frames could still leak. The label page has a "name on screen" answer, and those clips are excluded.

Only `upload/` may go to a service. Its file names hide clip and direction, and the name mapping sits outside it.

## Arms

| Arm | Model on OpenRouter | Input | Settings |
|---|---|---|---|
| G-pro | `google/gemini-3.1-pro-preview` | video retimed 10 times slower | provider defaults |
| G-flash | `google/gemini-3.8-flash` | video retimed 10 times slower | provider defaults |
| G-grid | `google/gemini-3.8-flash` | 24-frame grids | settles grid vs video |
| C-grid | `anthropic/claude-opus-5.5` | 24-frame grids | reasoning effort medium, `max_tokens` 8192 |

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
6. **Settings fixed here:** superseded by "Frozen on 2026-11-26" below. No server-side fallback: a refusal is a missing vote. The first live call per arm is a `--limit 1` smoke test counted in the budget.

Batch 1: 12 clips, sha256 `b3fe8f68cc0efca5dea2aa23e57f824d53173c4d03539b74ba7969106d8751d8`, labeled 2026-11-17 without seeing the queue.

### Exploratory local arm L-grid (added 2026-11-18, before any API call)

`Qwen/Qwen3-VL-4B-Instruct`, 4-bit NormalFloat (fp16 compute) on the RTX 2060, greedy decoding, 160 new
tokens at most, the same two grids and the same prompt as G-grid and C-grid. It costs nothing and is run
by the owner's machine, so it goes through the same runner, votes file and hash checks as the paid arms.
It is **not a decision arm**: MotionBlind puts models of this size near chance on physical motion, so it
is scored and reported under `exploratory_arms` with `decision_weight: none`, and the kill rule above
reads only the four decision arms. Its purpose is to exercise the pipeline with a real model, and to give a
$0 floor on the full dataset (`scripts/eval_local_vlm_gold.py`, which never touches the 36 queue clips).
Rule 5 applies to it unchanged: its answers on queue clips are sealed until all 36 are labeled.

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

## Frozen on 2026-11-26

Written before any queue clip was sent. `scripts/run_vlm_blind.py` holds the same values.

- **Gateway.** Every paid arm goes through OpenRouter with `data_collection: deny` and `allow_fallbacks: false`. The model that served each vote is recorded in its row.
- **Video arms see the clip 10 times slower.** OpenRouter exposes no frame-rate setting, and Gemini then samples about one frame per second: a 2.8 s clip cost 385 input tokens, the same clip retimed cost 1,921. Retiming gives the 10 frames per second this design planned. The prompt is unchanged and does not mention slow motion.
- **Media resolution and JSON mode are left at the provider default.** Neither can be set through the gateway. The parser already strips code fences.
- **Claude gets `max_tokens` 8192.** With 1,024 the reply was empty: reasoning used the whole budget.
- **Cost** is the amount the gateway reports per call. The hard cap is $15 instead of $5: the owner lifted the budget limit, and the cap stays as a safety stop.
- **Plumbing test, disclosed.** Five calls were made on one clip that is not in the queue, to check that images and video are accepted. They set the two changes above and nothing else. No arm counted that clip's twists correctly, which carries no weight at n = 1.
- **Exploratory gold run.** `scripts/eval_vlm_gold.py` sends the 99 concordant gold clips, none of them in the queue, to the four arms with these exact settings. It runs after this freeze so that nothing here can be tuned on it. It has no decision weight: the gold is not owner-verified and was partly built from Claude's July answers.
- **Rule 5 is unchanged.** No answer on a queue clip, per clip or aggregated, is shown to the owner before all 36 are labeled.

## Cost

About $8 to $12 expected for the 192 calls (36 forward and 12 reversed clips, four arms), from the plumbing test's 1 to 3 cents per call and a larger Claude budget.
