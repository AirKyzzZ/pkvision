#!/usr/bin/env python3
"""P1 full pose extraction: RTMPose-x over the parkourtheory clip corpus.

For every clip, extracts COCO-17 keypoints as a (T, 17, 3) array (x, y,
confidence) and writes resumable .npz shards plus a JSON manifest -- numeric
arrays and JSON only, no unsafe serialization. A' spec section 6 / P1.

Resumable: a shard whose .npz exists and whose clips are all in the manifest
is skipped. Interrupt and rerun safely.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DEFAULT_CLIPS_DIR = ROOT / "data" / "parkourtheory_clips_cropped"
DEFAULT_OUT_DIR = ROOT / "data" / "keypoints" / "parkourtheory_pose"
SHARD_SIZE = 50
CONF_VALID = 0.3

# Contaminated source dirs the corpus must never include (A' spec section 5).
FORBIDDEN_DIR_TOKENS = ("final_clips", "vlm_clips", "run_testing")


def resolve_device(requested: str) -> str:
    """Expose torch's bundled CUDA/cuDNN DLLs so onnxruntime-gpu finds them."""
    if requested == "cpu":
        return "cpu"
    if sys.platform == "win32":
        try:
            import torch
            torch_lib = Path(torch.__file__).parent / "lib"
            if torch_lib.is_dir():
                os.add_dll_directory(str(torch_lib))
        except Exception as e:
            print(f"[warn] could not expose torch CUDA DLLs: {e}")
    import onnxruntime as ort
    providers = ort.get_available_providers()
    print(f"onnxruntime providers: {providers}", flush=True)
    if "CUDAExecutionProvider" not in providers:
        print("[warn] CUDAExecutionProvider unavailable -> running on CPU")
        return "cpu"
    return "cuda"


def _largest_person(keypoints: np.ndarray, scores: np.ndarray) -> int:
    """Index of the person with the widest keypoint spread (the athlete)."""
    spans = []
    for p in range(len(keypoints)):
        valid = scores[p] > CONF_VALID
        if valid.sum() < 4:
            spans.append(0.0)
            continue
        pts = keypoints[p][valid]
        spans.append(float(np.ptp(pts[:, 0]) * np.ptp(pts[:, 1])))
    return int(np.argmax(spans)) if spans else 0


def extract_clip(video_path: Path, body) -> tuple[np.ndarray, float]:
    """RTMPose over every frame -> ((T, 17, 3) float32, fps)."""
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    seq = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        joints = np.zeros((17, 3), dtype=np.float32)
        keypoints, scores = body(frame)
        if keypoints is not None and len(keypoints) > 0:
            best = _largest_person(keypoints, scores)
            if keypoints[best].shape[0] >= 17:
                joints[:, :2] = keypoints[best][:17]
                joints[:, 2] = scores[best][:17]
        seq.append(joints)
    cap.release()
    return np.stack(seq) if seq else np.zeros((0, 17, 3), np.float32), fps


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clips-dir", type=Path, default=DEFAULT_CLIPS_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--shard-size", type=int, default=SHARD_SIZE)
    args = parser.parse_args()

    clips_dir = args.clips_dir.resolve()
    token = clips_dir.name.lower()
    if any(t in str(clips_dir).lower() for t in FORBIDDEN_DIR_TOKENS):
        sys.exit(f"refusing to extract from contaminated dir: {clips_dir}")

    clips = sorted(clips_dir.glob("*.mp4"))
    if not clips:
        sys.exit(f"no .mp4 clips in {clips_dir}")
    print(f"{len(clips)} clips in {clips_dir}", flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {
        "estimator": "rtmpose-x (rtmlib performance)",
        "shard_size": args.shard_size,
        "source": token,
        "clips": [],
    }
    done = {c["name"] for c in manifest["clips"]}

    device = resolve_device(args.device)
    from rtmlib import Body
    print(f"Loading RTMPose (device={device})...", flush=True)
    body = Body(mode="performance", backend="onnxruntime", device=device)

    shards = [clips[i:i + args.shard_size] for i in range(0, len(clips), args.shard_size)]
    t_start = time.time()
    for si, chunk in enumerate(shards):
        shard_name = f"shard_{si:03d}.npz"
        shard_path = args.out_dir / shard_name
        if shard_path.exists() and all(c.stem in done for c in chunk):
            print(f"[shard {si + 1}/{len(shards)}] skip (done)", flush=True)
            continue

        t0 = time.time()
        arrays: dict[str, np.ndarray] = {}
        entries = []
        for clip in chunk:
            seq, fps = extract_clip(clip, body)
            arrays[clip.stem] = seq
            mean_conf = float(seq[..., 2].mean()) if seq.size else 0.0
            entries.append({
                "name": clip.stem,
                "shard": shard_name,
                "n_frames": int(seq.shape[0]),
                "fps": round(fps, 3),
                "mean_conf": round(mean_conf, 4),
                "near_dup": clip.stem.endswith("_in_back_out"),
            })

        np.savez_compressed(shard_path, **arrays)
        manifest["clips"] = [c for c in manifest["clips"]
                             if c["name"] not in {e["name"] for e in entries}]
        manifest["clips"].extend(entries)
        manifest["generated"] = datetime.now(timezone.utc).isoformat()
        manifest["device"] = device
        manifest_path.write_text(json.dumps(manifest, indent=2))
        done.update(e["name"] for e in entries)
        print(
            f"[shard {si + 1}/{len(shards)}] {len(chunk)} clips -> {shard_name} "
            f"[{time.time() - t0:.1f}s, total {time.time() - t_start:.0f}s]",
            flush=True,
        )

    print(f"\nDone. {len(manifest['clips'])} clips in {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
