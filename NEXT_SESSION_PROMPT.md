# Next session

Start here: `docs/research/2026-11-26-direction-review.md`. It holds the diagnosis, the chosen direction and the four-week plan. The measured truth of the project lives in `docs/science-superpowers/` (survey, probes). The GVHMR and 3D-mesh pipeline described in older versions of this file is abandoned.

## State on 2026-11-26

- 2D skeletons (rtmlib, detector first) on 1,589 clips, a small transformer (CueModel) at gold macro-F1 0.37.
- Direction: repair the labels before touching the model. First step: `docs/science-superpowers/probes/2026-11-26-pt-label-fix-prereg.md`.
- Living branch: `chore/restart-2026-09`. Never push to `main`.

## Rules that do not change

- Pre-register before any run that could change a decision. A null result is reported as a result.
- The 36 clips of `data/labeling/retest_2026-11/queue.json` stay out of every computation. Never open that file in a way that shows names, never print them.
- No clip leaves the machine and no paid call runs without the owner's approval.
- The GPU is shared: launch through `gpu-guard.ps1`, low priority, one job at a time.

## Commands

    python scripts/probe_pt_label_fix.py plan      # label changes, no training
    python scripts/probe_pt_label_fix.py train     # 11 runs, about 15 GPU-minutes
    python scripts/probe_pt_label_fix.py report    # verdict against the frozen rule
    python scripts/probe_flip_yardstick.py explore # rule counter on the dev half
