#!/usr/bin/env python3
"""Count per-clip identity jumps in the corpus shards, for the person-selection audit.

A frame is usable with at least 6 joints above confidence 0.3. A jump is a move of the
joints' bounding-box centre above 1 median body diagonal between usable frames at most
2 frames apart. Writes nothing; reads gitignored shards.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SHARDS = ROOT / "data/keypoints/parkourtheory_pose"
CONF_VALID, MIN_JOINTS, MAX_GAP, JUMP_DIAGONALS = 0.3, 6, 2, 1.0


def count_jumps(seq: np.ndarray) -> int:
    valid = seq[..., 2] > CONF_VALID
    frames = np.flatnonzero(valid.sum(1) >= MIN_JOINTS)
    if len(frames) < 5:
        return 0
    low = np.array([seq[t, valid[t], :2].min(0) for t in frames])
    high = np.array([seq[t, valid[t], :2].max(0) for t in frames])
    centres, diagonal = (low + high) / 2, np.median(np.linalg.norm(high - low, axis=1))
    step = np.linalg.norm(np.diff(centres, axis=0), axis=1)
    return int(((step > JUMP_DIAGONALS * diagonal) & (np.diff(frames) <= MAX_GAP)).sum())


def main() -> None:
    jumps = []
    for shard in sorted(SHARDS.glob("shard_*.npz")):
        with np.load(shard) as z:
            jumps += [count_jumps(z[name]) for name in z.files if z[name].shape[0] >= 5]
    if not jumps:
        sys.exit(f"no shards in {SHARDS}")
    values = np.array(jumps)
    print(f"clips {len(values)}  >=2 jumps {(values >= 2).mean():.3f}  1 jump {(values == 1).mean():.3f}  0 jumps {(values == 0).mean():.3f}")


if __name__ == "__main__":
    main()
