#!/usr/bin/env python3
"""P1 pose extraction on Modal — RTMPose-x over the parkourtheory corpus, parallel.

Serverless port of scripts/extract_pose.py. Clips live in the `pkvision-data`
Modal Volume (push once with `modal volume put`); each shard runs on its own GPU
container and writes a resumable .npz back to the Volume, so the downstream GFP
pretrain reads the skeletons in-cloud without a round-trip. Output schema is
identical to the local extractor: shard_NNN.npz keyed by clip stem -> (T,17,3)
float32 (x, y, conf), plus manifest.json.

Usage:
    uvx modal volume put pkvision-data ./data/parkourtheory_clips_cropped /parkourtheory_clips_cropped
    uvx modal run scripts/extract_pose_modal.py
    uvx modal volume get pkvision-data /keypoints/parkourtheory_pose ./data/keypoints/parkourtheory_pose
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import modal

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04", add_python="3.11"
    )
    .apt_install("libgl1-mesa-glx", "libglib2.0-0", "ffmpeg")
    # rtmlib hard-depends on CPU onnxruntime; install it, purge that, then add the
    # GPU build fresh. Order matters: both packages share the onnxruntime/ dir, so
    # removing CPU *after* GPU is installed corrupts the module — purge before.
    .pip_install("rtmlib", "opencv-python-headless", "numpy")
    .run_commands("pip uninstall -y onnxruntime || true")
    .pip_install("onnxruntime-gpu==1.19.2")
)

app = modal.App("pkvision-pose", image=image)
volume = modal.Volume.from_name("pkvision-data", create_if_missing=True)

VOL = "/vol"
CLIPS_DIR = f"{VOL}/parkourtheory_clips_cropped"
OUT_DIR = f"{VOL}/keypoints/parkourtheory_pose"
SHARD_SIZE = 50
CONF_VALID = 0.3


def _largest_person(keypoints, scores) -> int:
    import numpy as np

    spans = []
    for p in range(len(keypoints)):
        valid = scores[p] > CONF_VALID
        if valid.sum() < 4:
            spans.append(0.0)
            continue
        pts = keypoints[p][valid]
        spans.append(float(np.ptp(pts[:, 0]) * np.ptp(pts[:, 1])))
    return int(np.argmax(spans)) if spans else 0


def _extract_clip(video_path, body):
    import cv2
    import numpy as np

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
    return (np.stack(seq) if seq else np.zeros((0, 17, 3), np.float32)), fps


@app.function(gpu="T4", timeout=1800, volumes={VOL: volume})
def extract_shard(shard_idx: int, clip_names: list[str]) -> list[dict]:
    import time
    from pathlib import Path

    import numpy as np
    import onnxruntime as ort
    from rtmlib import Body

    providers = ort.get_available_providers()
    if "CUDAExecutionProvider" not in providers:
        raise RuntimeError(f"GPU unavailable to onnxruntime, refusing CPU run: {providers}")

    t0 = time.time()
    body = Body(mode="performance", backend="onnxruntime", device="cuda")
    load_s = time.time() - t0
    out = Path(OUT_DIR)
    out.mkdir(parents=True, exist_ok=True)
    shard_name = f"shard_{shard_idx:03d}.npz"

    t1 = time.time()
    arrays, entries = {}, []
    for name in clip_names:
        seq, fps = _extract_clip(Path(CLIPS_DIR) / name, body)
        stem = Path(name).stem
        arrays[stem] = seq
        mean_conf = float(seq[..., 2].mean()) if seq.size else 0.0
        entries.append(
            {
                "name": stem,
                "shard": shard_name,
                "n_frames": int(seq.shape[0]),
                "fps": round(fps, 3),
                "mean_conf": round(mean_conf, 4),
                "near_dup": stem.endswith("_in_back_out"),
            }
        )
    infer_s = time.time() - t1
    total_frames = sum(e["n_frames"] for e in entries)

    np.savez_compressed(out / shard_name, **arrays)
    volume.commit()
    fps_eff = total_frames / infer_s if infer_s > 0 else 0.0
    print(
        f"[shard {shard_idx:03d}] {len(clip_names)} clips, {total_frames} frames | "
        f"load {load_s:.1f}s infer {infer_s:.1f}s ({fps_eff:.1f} fps)",
        flush=True,
    )
    return entries


@app.function(volumes={VOL: volume})
def write_manifest(entries: list[dict], shard_size: int) -> int:
    from pathlib import Path

    mpath = Path(OUT_DIR) / "manifest.json"
    manifest = (
        json.loads(mpath.read_text())
        if mpath.exists()
        else {
            "estimator": "rtmpose-x (rtmlib performance)",
            "shard_size": shard_size,
            "source": "parkourtheory_clips_cropped",
            "clips": [],
        }
    )
    fresh = {e["name"] for e in entries}
    manifest["clips"] = [c for c in manifest["clips"] if c["name"] not in fresh] + entries
    manifest["generated"] = datetime.now(timezone.utc).isoformat()
    manifest["device"] = "cuda"
    mpath.write_text(json.dumps(manifest, indent=2))
    volume.commit()
    return len(manifest["clips"])


@app.local_entrypoint()
def main(shard_size: int = SHARD_SIZE, force: bool = False, limit: int = 0):
    clips = sorted(
        e.path.split("/")[-1]
        for e in volume.listdir("parkourtheory_clips_cropped")
        if e.path.endswith(".mp4")
    )
    if not clips:
        raise SystemExit(
            "no clips in volume — run: "
            "uvx modal volume put pkvision-data ./data/parkourtheory_clips_cropped /parkourtheory_clips_cropped"
        )
    print(f"{len(clips)} clips in volume")

    existing = set()
    if not force:
        try:
            existing = {
                e.path.split("/")[-1]
                for e in volume.listdir("keypoints/parkourtheory_pose")
                if e.path.endswith(".npz")
            }
        except Exception:
            existing = set()

    shards = [clips[i : i + shard_size] for i in range(0, len(clips), shard_size)]
    jobs = [
        (i, names)
        for i, names in enumerate(shards)
        if force or f"shard_{i:03d}.npz" not in existing
    ]
    if limit > 0:
        jobs = jobs[:limit]
    print(f"{len(jobs)}/{len(shards)} shards to extract this run ({len(existing)} already present)")
    if not jobs:
        print("all shards present — pass --force to re-extract")
        return

    all_entries: list[dict] = []
    for entries in extract_shard.starmap(jobs):
        all_entries.extend(entries)

    total = write_manifest.remote(all_entries, shard_size)
    print(f"done — {total} clips in manifest, {len(jobs)} shards written this run")
