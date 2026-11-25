# The clean-pose CueModel result reproduces on the Windows rebuild (2026-11-25)

WSL was wiped on 2026-11-21 and the detector-first shards went with it. They were regenerated natively on Windows (same RTX 2060, `onnxruntime-gpu==1.22.0`, code unchanged on `probe/clean-pose-cue-2026-11`), then arm D was retrained with 5 seeds before anything else used them.

Criterion, written at 01:20 before training: reproduced if the arm D gold macro-F1 (mean of 4 cues) lands within ±0.03 of the 2026-11-08 value, 0.370.

| | 2026-11-08 (WSL) | 2026-11-25 (Windows) |
|---|---|---|
| Gold macro-F1, mean of 4 cues | 0.370 | 0.374 |
| Per seed | 0.367, 0.401, 0.315, 0.395, 0.370 | 0.354, 0.392, 0.360, 0.397, 0.365 |
| Flip / twist / direction / context acc | 0.451 / 0.591 / 0.466 / 0.521 | 0.427 / 0.575 / 0.449 / 0.546 |
| Internal val core macro-F1 (noisy labels) | 0.385 | 0.404 |
| Clips extracted / training clips | 1,582 / 1,483 | 1,589 / 1,474 |

Reproduced, +0.004. The 2026-11-08 conclusion stands and the regenerated shards can be used as before.

- Arm O cannot be rerun, the old shards are lost, so only D was checked.
- The corpus was re-cropped from the originals: 7 more clips extracted, 9 fewer training clips. Not investigated.
- The 36 blind-queue clips stayed out of extraction, training and evaluation.
