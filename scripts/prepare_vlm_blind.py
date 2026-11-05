#!/usr/bin/env python3
"""Build anonymised inputs for the VLM blind probe. Makes no network calls.

Per clip: forward and reversed H.264 mp4 for native-video models, and 24-frame grids
for image-only models. Names are opaque and re-encoding drops container metadata.
Output stays local and gitignored because the clips are copyrighted. Encoding needs
`pip install imageio-ffmpeg`. Design:
docs/science-superpowers/probes/2026-11-04-vlm-blind-prereg-DRAFT.md
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
QUEUE = ROOT / "data/labeling/retest_2026-11/queue.json"
CLIPS = ROOT / "data/parkourtheory_clips_cropped"
OUT = ROOT / "data/labeling/retest_2026-11/vlm_inputs"
GRID_FRAMES, ROWS, COLS, TILE = 24, 3, 4, 300


def opaque_id(slug: str) -> str:
    return "c" + hashlib.sha1(("vlm-blind:" + slug).encode()).hexdigest()[:10]


def read_frames(path: Path) -> tuple[list[np.ndarray], float]:
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return frames, fps


def write_video(frames: list[np.ndarray], fps: float, path: Path) -> None:
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    for frame in frames:
        writer.write(frame)
    writer.release()


def to_h264(source: Path, dest: Path) -> None:
    import imageio_ffmpeg
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(source),
                    "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                    "-an", "-map_metadata", "-1", str(dest)], check=True)
    source.unlink()


def grids(frames: list[np.ndarray]) -> list[np.ndarray]:
    """Evenly sampled frames as numbered tiles on 3x4 sheets, in time order."""
    index = np.linspace(0, len(frames) - 1, GRID_FRAMES).astype(int)
    tiles = []
    for number, i in enumerate(index, 1):
        height, width = frames[i].shape[:2]
        scale = TILE / max(height, width)
        small = cv2.resize(frames[i], (max(1, round(width * scale)), max(1, round(height * scale))))
        tile = np.zeros((TILE, TILE, 3), np.uint8)
        tile[:small.shape[0], :small.shape[1]] = small
        cv2.putText(tile, str(number), (6, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        tiles.append(tile)
    per_sheet = ROWS * COLS
    sheets = [tiles[i:i + per_sheet] for i in range(0, len(tiles), per_sheet)]
    return [np.vstack([np.hstack(s[r * COLS:(r + 1) * COLS]) for r in range(ROWS)]) for s in sheets]


def main() -> None:
    queue = json.loads(QUEUE.read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    mapping = {}
    for clip in queue:
        slug, cid = clip["slug"], opaque_id(clip["slug"])
        frames, fps = read_frames(CLIPS / f"{slug}.mp4")
        if not frames:
            sys.exit(f"could not decode {slug}")
        for variant, seq in (("fwd", frames), ("rev", frames[::-1])):
            raw = OUT / f"{cid}_{variant}.raw.mp4"
            write_video(seq, fps, raw)
            to_h264(raw, OUT / f"{cid}_{variant}.mp4")
            for n, sheet in enumerate(grids(seq), 1):
                cv2.imwrite(str(OUT / f"{cid}_{variant}_grid{n}.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
        mapping[cid] = slug
    (OUT / "mapping_DO_NOT_SEND.json").write_text(json.dumps(mapping, indent=1))
    print(f"{len(mapping)} clips in {OUT}")


if __name__ == "__main__":
    main()
