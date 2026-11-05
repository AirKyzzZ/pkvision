# Blind retest queue

`queue.json` is frozen and contains names and counts, so do not open it before labeling. Its strata come from manifest and name parser and are known to be wrong for at least one clip, so use owner labels, not strata, as truth.

`label.html` plays H.264 copies in the gitignored `clips_h264/`. On a new machine, make them with `pip install imageio-ffmpeg`, then `python scripts/build_retest_queue.py`, which reads the frozen queue and does not change it. Tested in headless Chrome over `file://`: all 36 clips load, and saving, navigation, the error message and the download trigger work.
