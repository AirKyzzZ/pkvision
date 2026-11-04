# VLM blind probe: draft pre-registration (2026-11-04)

**Status: draft, not run.** No clip has been sent anywhere. It needs the owner's OK to spend money and send clips to outside services, plus his blind labels. Freeze this file in a commit before the first API call and drop `DRAFT` from the name. Step 2 of "Validation before coding" (`survey/2026-09-28/00-synthesis.md`).

**Question:** can a frontier VLM read twist count on unseen parkour clips better than a constant answer, without guessing from trick priors?

## Truth

The 36 clips in `data/labeling/retest_2026-11/queue.json`, labeled by the owner on `label.html` **before** any VLM output is seen, so that proposals cannot anchor him. Scored: clips he marks "one trick" with numeric flip and twist. Others are reported and excluded.

The queue was chosen from manifest and name-parser strata by sha1 order, with no pose or VLM output. Its strata are deliberately hard: corks, likely odd twists, twist 2+.

## Inputs

Built by `scripts/prepare_vlm_blind.py` into a gitignored folder: opaque ids, re-encoded with container metadata dropped, audio removed.

**Known gap:** on-screen text is not stripped, although the synthesis asks for it. Burned-in trick names would leak, so the owner's pass or a scan of the grids must catch these first.

## Arms

| Arm | Model | Input | Settings |
|---|---|---|---|
| G-pro | Gemini 3.1 Pro | native video | 10 fps, medium resolution |
| G-flash | Gemini 3.8 Flash | native video | 10 fps, medium resolution |
| G-grid | Gemini 3.8 Flash | 24-frame grids | settles grid vs native |
| C-grid | Claude Opus 5.5 | 24-frame grids | |

One vote per clip per arm. Prompt: count flips and twists by tracking body orientation frame by frame, no trick naming, JSON output with `flip`, `twist`, `direction`, `cork`, `unsure`. Exact prompt text is frozen with this file.

## Reversed-clip control

Each arm also sees the first 12 scored clips (order: sha1 of `slug`) played backwards. **Fail** if predicted direction flips on under 9 of the clips with a forward or backward forward-pass answer. Models that recognise a trick instead of watching it will repeat the same direction.

## Metrics and decision rule

Primary, per arm: exact half-twist accuracy on all scored clips. Unsure and malformed count as wrong.

Always report these alongside, because the rule-counter probe showed a constant can match a counter:
- always-0-twist and most-common-twist baselines;
- accuracy per stratum, especially twist ≥ 1.5 and corks;
- flip accuracy and the same control results.

Per arm:
- **Killed:** below 75% twist (the synthesis's 9/12), **or** not above the best constant baseline, **or** reversed-clip control fails.
- **Promising:** at least 75%, above baselines, control passes. Next is the label-source audit, not bulk labeling.

With about 30 clips, 75% has a 95% CI near 0.56–0.89. This can rank or kill. It cannot confirm the 85% bar.

## Cost estimate

About $2–4 total, using 90 s of forward video, 48 clip passes per arm, and September survey prices (`survey/2026-09-28/02-video-vlms.md`). Those prices need re-checking before any run. Stop at a hard cap of $5.

## Open before running

- API keys: none on this PC.
- Model availability and prices.
- Whether the clips' licence allows sending them to third-party APIs.
