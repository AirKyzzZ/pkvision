# Session log, 2026-11-08 (RTX 2060, unattended)

Decisions and reasons, in order. Results are in `docs/science-superpowers/probes/2026-11-08-clean-pose-cue-*`.

## Setup
- `.venv` already held torch 2.7.1+cu126 with CUDA working, so the brief's "torch is not installed yet" was out of date. Kept the env; did not reinstall. CUDA verified with a matrix product on the GPU; `pytest tests -q` passed (396 before changes). Run `. .venv/gpu_env.sh` first or onnxruntime falls back to CPU.
- Git had no identity on this machine. Commits use `Maxime Mansiet <AirKyzzZ@users.noreply.github.com>` per command, as in the existing history; no global config was changed.

## Direction
- `NEXT_SESSION_PROMPT.md` is from the GVHMR era (mesh rendering, 2D cleanup). The September synthesis lists GVHMR physics as not to be revisited without new evidence and the brief says to follow the probes, so the GVHMR tasks were left aside.
- The rule-counter probe is killed and its 47 clips have been seen, so re-scoring it would be exploratory only. The VLM blind probe cannot run: no blind labels, no API keys, no spend approval, and the clip-licence question is open. No VLM was called and no clip left the PC.
- Chosen GPU work: the one question the audit left open that needs no owner input, namely what the invented skeletons in the corpus shards cost the CueModel. Pre-registered in PR #4 before any CueModel saw the new shards.
- The 36 blind-queue clips are excluded from extraction, training and evaluation, read only programmatically, and never printed.
- PR base is `chore/restart-2026-09`, the live branch, not `main` (stale since 2026-06-07).
- New shards go under `data/keypoints_detfirst/`, gitignored and outside `data/keypoints/`, because several scripts glob that folder recursively and would silently mix old and new shards.
- Parallel CPU work while the GPU extracted: the VLM scorer items the draft requires before freezing (PR #5). Light editing and tests only, in line with "no parallel heavy jobs".

## Runs on the GPU
- Extraction: `scripts/extract_pose.py --exclude-json queue.json --people`, 1,582 clips, 117,957 frames, 3 h 54 min at about 8 images/s, about 2.5 GB VRAM observed. Output gitignored in `data/keypoints_detfirst/` (51 MB).
- Training: 25 CueModel runs (5 arms × 5 seeds), one at a time, 20 minutes in total.
- Pre-registered result on the D vs O comparison: no detectable effect. See the results file.

## Calls made along the way
- Frames with nobody detected are interpolated with confidence 0 in the primary arm, and left as zeros in D0, disclosed in advance.
- A numpy-integer crash in the audit report was fixed between extraction and training: reporting only.
- 4 manifest rows named `test_*` are own-footage with no corpus clip: training set 1,483 rather than 1,487.
- Did not adopt the new shards as the default: the pre-registered rule requires "helps" for that.

## Blockers for the owner
- VLM blind probe: 36 blind labels from `label.html` (do not open `queue.json` first), then spend approval, API keys, the clip-licence question, and a runner that does not exist yet.
- PRs #4, #5 and the results PR are open and not merged.
