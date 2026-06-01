"""One handle over the project's clip formats: .mp4 video (parkourtheory) and
.npy RGB frame arrays (v5/gold/comp). Skeletons (T,17,3) attach optionally."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class ClipRef:
    slug: str
    video_path: Path | None
    frames_path: Path | None
    skeleton: np.ndarray | None = None

    @classmethod
    def from_path(cls, path: str | Path) -> "ClipRef":
        path = Path(path)
        is_video = path.suffix.lower() in {".mp4", ".mov", ".webm"}
        return cls(
            slug=path.stem,
            video_path=path if is_video else None,
            frames_path=path if not is_video else None,
        )

    def get_frames(self, max_frames: int = 8) -> np.ndarray:
        """Uniformly sampled RGB frames (N,H,W,3) uint8, for VLM + UI preview."""
        if self.frames_path is not None:
            arr = np.load(self.frames_path)
        else:
            import cv2

            cap = cv2.VideoCapture(str(self.video_path))
            got = []
            while True:
                ok, f = cap.read()
                if not ok:
                    break
                got.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
            cap.release()
            arr = np.stack(got) if got else np.zeros((0, 1, 1, 3), np.uint8)
        if len(arr) <= max_frames or len(arr) == 0:
            return arr
        idx = np.linspace(0, len(arr) - 1, max_frames).astype(int)
        return arr[idx]

    def get_skeleton(self) -> np.ndarray | None:
        return self.skeleton
