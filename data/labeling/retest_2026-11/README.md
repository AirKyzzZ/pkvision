# Blind retest queue

`queue.json` is frozen, and its strata come from manifest and name parser, so they can be wrong: `540_dive_roll` is filed as `twist0_flip` but is probably 1.5 twists. Use owner labels, not strata, as truth.

`label.html` plays H.264 copies in the gitignored `clips_h264/`. On a new machine, make them with `pip install imageio-ffmpeg`, then `python scripts/build_retest_queue.py`. Tested in headless Chrome over `file://`: all 36 clips load, and saving, navigation, the error message and the download trigger work.
