# Blind retest queue

`queue.json` is frozen and contains names and counts, so do not open it before labeling. Its strata come from manifest and name parser and are known to be wrong for at least one clip, so use owner labels, not strata, as truth.

`label.html` plays H.264 copies in the gitignored `clips_h264/`. On a new machine, make them with `pip install imageio-ffmpeg`, then `python scripts/build_retest_queue.py`, which reads the frozen queue and does not change it. Tested in headless Chrome over `file://`: all 36 clips load, and saving, navigation, the error message and the download trigger work.

## Ingesting the owner's labels

Copy the downloaded `blind_labels.json` here (gitignored). Then, from the repo root:

    python scripts/score_vlm_blind.py check-labels

It stops while any clip is unlabeled and lists the unfinished ones by page number, never by name. Once all 36 are labeled and a verdict is reachable, it prints the file's sha256; record that hash in the freeze commit and pass it to `control-ids` and `score`. No VLM is called by any of this.

## Running the probe (after the freeze)

1. `python scripts/score_vlm_blind.py check-labels LABELS` until it passes; record the hash in a commit.
2. `python scripts/run_vlm_blind.py plan --labels LABELS --labels-sha256 HASH` shows what would be sent.
3. `python scripts/run_vlm_blind.py run ... --provider fake` rehearses with no network; delete the fake `votes.jsonl` afterwards.
4. With keys in the environment: `run ... --limit 1` per arm as a smoke test, then `run ... --cap-usd 5`.
5. Once all 36 are labeled: `control-ids`, then `run --control-ids ids.json` for the reversed clips, then `score --stage BATCH=HASH ...`.

Labels may arrive in batches: see "Staged labeling and collection" in the pre-registration.

## Free local arm (exploratory)

`run_vlm_blind.py run --arms L-grid --provider local ...` answers with Qwen3-VL-4B in 4-bit on the GPU, no key,
$0; `. .venv/gpu_env.sh` first and set `PKVISION_HF_CACHE` if the weights are not in `~/hf-cache`. It is scored
under `exploratory_arms` and has no decision weight. The same model runs over the rest of the dataset with
`scripts/eval_local_vlm_gold.py run` (queue clips excluded) and `report` for the numbers against the noisy gold.
