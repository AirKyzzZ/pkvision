# Frontier VLMs on the gold clips: exploratory results (2026-11-26)

**Exploratory, no decision weight.** The four decision arms of the VLM blind probe, with the settings frozen in `2026-11-04-vlm-blind-prereg.md`, on the 99 concordant gold clips. None is in the blind queue. Numbers: `2026-11-26-vlm-gold-exploratory.json`. Script: `scripts/eval_vlm_gold.py`. One vote per clip per arm, 396 answers, $8.39 through OpenRouter.

The gold is not owner-verified and was built from the agreement of the manifest with Claude's July answers. Unsure, malformed and missing answers count as wrong, as in the blind probe.

## Results

Same clips, bins and metrics as the CueModel probes: flip on 82 clips, twist on 64, direction on 85.

| | Flip acc | Twist acc | Direction acc | Mean macro-F1 | Unsure | Cost |
|---|---|---|---|---|---|---|
| Constant (train majority) | 0.683 | 0.797 | 0.659 | 0.230 | | |
| CueModel, corrected labels (2026-11-26) | 0.612 | 0.566 | 0.534 | 0.424 | | |
| **G-flash**, Gemini 3.8 Flash, slowed video | **0.805** | 0.750 | **0.776** | 0.587 | 8 | $1.00 |
| **G-grid**, Gemini 3.8 Flash, frame grids | 0.732 | 0.781 | 0.682 | **0.589** | 9 | $1.33 |
| G-pro, Gemini 3.1 Pro, slowed video | 0.707 | 0.625 | 0.647 | 0.449 | 6 | $3.82 |
| C-grid, Claude Opus 5.5, frame grids | 0.488 | 0.438 | 0.471 | 0.400 | 47 | $2.25 |

Exact answers on the 13 gold clips with a twist: G-flash 3, G-grid 4, G-pro 4, C-grid 0. On the 10 clips with two flips: 4, 3, 1 and 1.

## Reading

- **Flip and direction: Gemini Flash is the first system here above the constant on accuracy**, by 12 points on both, and about 20 points above the CueModel. It costs one cent a clip.
- **Twist: nothing reads it.** No arm beats "always 0", and the best arm finds 4 of the 13 twisting clips. This matches every earlier result. With 13 clips it cannot rank the arms.
- **Claude answers "unsure" on 47 of 99 clips**, which the rule scores as wrong. Its low score is abstention, not a measured reading error.
- **The larger Gemini model is not better** than Flash here, and costs four times more.
- Video and grids are close for Flash: video is better on flip and direction, grids on twist.

## Limits

- 99 clips, one vote each. Two repeated calls on the same clip gave different twist counts during the plumbing checks.
- The gold shares its origin with Claude's July answers, so it is not a neutral judge between arms.
- Nothing here says whether an arm reads the motion or recognises the trick. Only the blind probe's reversed-clip control addresses that.

## What follows

Nothing is decided by this. It supports two next steps, each to be pre-registered: use Gemini Flash as a second opinion on flip presence, count and direction in the label repair, and test a higher frame rate on the airborne phase for twist. The blind probe still needs the owner's 24 remaining labels: votes on the first 12 clips are collected and sealed (48 calls, $0.95).
