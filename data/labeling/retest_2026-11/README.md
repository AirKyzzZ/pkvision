# Blind retest queue

`queue.json` is frozen and contains names and counts, so do not open it before labeling. Its strata come from manifest and name parser and are known to be wrong for at least one clip, so use owner labels, not strata, as truth.

`label.html` plays H.264 copies in the gitignored `clips_h264/`. On a new machine, make them with `pip install imageio-ffmpeg`, then `python scripts/build_retest_queue.py`, which reads the frozen queue and does not change it. Tested in headless Chrome over `file://`: all 36 clips load, and saving, navigation, the error message and the download trigger work.

## Ingesting the owner's labels

Copy the downloaded `blind_labels.json` here (gitignored). Then, from the repo root:

    python scripts/score_vlm_blind.py check-labels

It stops while any clip is unlabeled and lists the unfinished ones by page number, never by name. Once all 36 are labeled and a verdict is reachable, it prints the file's sha256; record that hash in the freeze commit and pass it to `control-ids` and `score`. No VLM is called by any of this.
