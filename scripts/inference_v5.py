#!/usr/bin/env python3
"""PkVision v5 — Video trick recognition and FIG scoring.

Processes a competition video through the full pipeline:
1. YOLO-pose: detect athlete, extract keypoints
2. Segment trick boundaries from angular velocity peaks
3. Crop athlete in each trick segment
4. Metric model: embed each trick, match nearest families (kNN)
5. FIG lookup: output D-scores

Usage:
    python scripts/inference_v5.py --input video.mp4
    python scripts/inference_v5.py --input data/run_testing/IMG_5985.mov --top-k 5
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.pose.angles import frame_result_to_analysis
from core.pose.detector import PoseDetector
from core.recognition.segmentation import RunSegmenter

ROOT = Path(__file__).parent.parent
DEFAULT_METRIC_MODEL = ROOT / "data" / "models" / "pkvision_metric.pt"
DEFAULT_REFS_DB = ROOT / "data" / "models" / "pkvision_metric_refs.json"
FIG_TRICKS_PATH = ROOT / "data" / "fig_tricks_2025.json"
FIG_MAP_PATH = ROOT / "data" / "fig_to_parkourtheory_map_v2.json"


# ── Model ────────────────────────────────────────────────────────────


class MetricModel(nn.Module):
    def __init__(self, backbone, hidden_size, embed_dim):
        super().__init__()
        self.backbone = backbone
        self.projector = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, embed_dim),
        )

    def forward(self, pixel_values):
        outputs = self.backbone.videomae(pixel_values=pixel_values)
        cls_token = outputs.last_hidden_state[:, 0]
        embedding = self.projector(cls_token)
        return F.normalize(embedding, p=2, dim=1)


# ── Device ───────────────────────────────────────────────────────────


def get_device(override=None):
    if override:
        return torch.device(override)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ── FIG Scoring ──────────────────────────────────────────────────────


def load_fig_scores():
    """Load FIG trick name -> D-score mapping."""
    scores = {}
    if not FIG_TRICKS_PATH.exists():
        return scores
    with open(FIG_TRICKS_PATH) as f:
        data = json.load(f)
    for category in data.get("categories", {}).values():
        for trick in category.get("tricks", []):
            name = trick["name"]
            score = trick.get("score", 0.0)
            scores[name.lower()] = score
            for alias in trick.get("aliases", []):
                scores[alias.lower()] = score
    return scores


# ── Video Utilities ──────────────────────────────────────────────────


def read_video_frames(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  ERROR: Cannot open video: {video_path}")
        sys.exit(1)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames, fps


def detect_person_boxes(frames):
    """YOLO person detection, returns (T, 4) bounding boxes."""
    from ultralytics import YOLO

    yolo_path = ROOT / "yolo11n-pose.pt"
    model = YOLO(str(yolo_path))
    T = len(frames)
    boxes = np.full((T, 4), np.nan)
    for i, frame in enumerate(frames):
        results = model(frame, conf=0.3, verbose=False)
        if results and results[0].boxes is not None and len(results[0].boxes) > 0:
            box_data = results[0].boxes.xyxy.cpu().numpy()
            areas = (box_data[:, 2] - box_data[:, 0]) * (box_data[:, 3] - box_data[:, 1])
            boxes[i] = box_data[np.argmax(areas)]
    return boxes


def smooth_boxes(boxes, window=7):
    T = len(boxes)
    valid = ~np.isnan(boxes[:, 0])
    if np.sum(valid) < 2:
        return boxes
    for col in range(4):
        valid_idx = np.where(valid)[0]
        boxes[:, col] = np.interp(np.arange(T), valid_idx, boxes[valid_idx, col])
    kernel = np.ones(window) / window
    for col in range(4):
        boxes[:, col] = np.convolve(boxes[:, col], kernel, mode="same")
    return boxes


def crop_person_frames(frames, boxes, start, end, crop_size=256, padding=1.5):
    cropped = []
    for i in range(start, min(end + 1, len(frames))):
        frame = frames[i]
        h, w = frame.shape[:2]
        if np.isnan(boxes[i, 0]):
            side = min(h, w)
            y1 = (h - side) // 2
            x1 = (w - side) // 2
            crop = frame[y1 : y1 + side, x1 : x1 + side]
        else:
            x1, y1, x2, y2 = boxes[i].astype(int)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            side = int(max(x2 - x1, y2 - y1) * padding)
            half = side // 2
            crop = frame[
                max(0, cy - half) : min(h, cy + half),
                max(0, cx - half) : min(w, cx + half),
            ]
        if crop.size == 0:
            crop = frame
        resized = cv2.resize(crop, (crop_size, crop_size))
        cropped.append(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
    return cropped


def sample_frames(frames, num_frames=16):
    T = len(frames)
    if T == 0:
        return []
    if T >= num_frames:
        indices = np.linspace(0, T - 1, num_frames, dtype=int)
    else:
        indices = list(range(T))
        while len(indices) < num_frames:
            indices.append(indices[-1])
        indices = indices[:num_frames]
    return [frames[i] for i in indices]


def format_time(seconds):
    m = int(seconds) // 60
    s = seconds - m * 60
    return f"{m}:{s:04.1f}"


# ── Matching ─────────────────────────────────────────────────────────


def match_embedding(query_emb, ref_db, fig_scores, top_k=5):
    """Match a query embedding against the reference DB.

    Uses kNN (max similarity across all reference embeddings per family)
    for robust matching, with FIG D-score lookup.
    """
    matches = []

    for fname, fdata in ref_db.items():
        # kNN: best similarity across all reference clips
        if "embeddings" in fdata and fdata["embeddings"]:
            sims = [float(np.dot(query_emb, np.array(e))) for e in fdata["embeddings"]]
            sim = max(sims)
        else:
            sim = float(np.dot(query_emb, np.array(fdata["centroid"])))

        # Find best FIG D-score for this family
        fig_tricks = fdata.get("fig_tricks", [])
        d_score = 0.0
        best_fig = ""
        for ft in fig_tricks:
            s = fig_scores.get(ft.lower(), 0.0)
            if s > d_score:
                d_score = s
                best_fig = ft

        matches.append({
            "family": fname,
            "similarity": sim,
            "fig_trick": best_fig,
            "d_score": d_score,
            "category": fdata.get("category", ""),
            "num_refs": fdata.get("num_refs", 0),
        })

    matches.sort(key=lambda x: -x["similarity"])
    return matches[:top_k]


# ── Main Pipeline ────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="PkVision v5 — AI Parkour Judge")
    parser.add_argument("--input", required=True, help="Input video path")
    parser.add_argument("--model", type=Path, default=DEFAULT_METRIC_MODEL)
    parser.add_argument("--refs", type=Path, default=DEFAULT_REFS_DB)
    parser.add_argument("--device", default=None)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--min-confidence", type=float, default=0.3,
                        help="Minimum cosine similarity to report")
    args = parser.parse_args()

    video_path = Path(args.input)
    device = get_device(args.device)

    if not video_path.exists():
        print(f"\n  ERROR: Video not found: {video_path}\n")
        sys.exit(1)

    t_total = time.time()

    # ── Header ───────────────────────────────────────────────────────

    print()
    print("  PkVision v5 — AI Parkour Judge")
    print("  " + "=" * 48)
    print(f"  Video:   {video_path.name}")
    print(f"  Device:  {device}")

    # ── Step 1: Load metric model + refs ─────────────────────────────

    print(f"\n  [1/5] Loading metric model...", end=" ", flush=True)
    t0 = time.time()

    if not args.model.exists():
        print(f"\n  ERROR: Model not found at {args.model}")
        print(f"  Train with: python scripts/train_metric_local.py\n")
        sys.exit(1)

    ckpt = torch.load(args.model, map_location=device, weights_only=False)
    config = ckpt["config"]
    processor = VideoMAEImageProcessor.from_pretrained(config["model_name"])
    base = VideoMAEForVideoClassification.from_pretrained(config["model_name"])
    model = MetricModel(base, ckpt["hidden_size"], config["embed_dim"])
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    if not args.refs.exists():
        print(f"\n  ERROR: Reference DB not found at {args.refs}")
        print(f"  Build with: python scripts/build_metric_refs.py\n")
        sys.exit(1)

    with open(args.refs) as f:
        ref_db = json.load(f)

    fig_scores = load_fig_scores()
    families = list(ref_db.keys())
    print(f"OK ({len(families)} families, {len(fig_scores)} FIG entries, {time.time() - t0:.1f}s)")

    # ── Step 2: Read video + keypoints ───────────────────────────────

    print(f"\n  [2/5] Reading video...", end=" ", flush=True)
    t0 = time.time()
    all_frames, fps = read_video_frames(str(video_path))
    total_frames = len(all_frames)
    duration = total_frames / fps
    print(f"{total_frames} frames ({duration:.1f}s @ {fps:.0f}fps, {time.time() - t0:.1f}s)")

    print(f"         YOLO-pose keypoints...", end=" ", flush=True)
    t0 = time.time()
    detector = PoseDetector()
    frame_results = list(detector.process_video(str(video_path)))
    print(f"{len(frame_results)} detections ({time.time() - t0:.1f}s)")

    # ── Step 3: Segment tricks ───────────────────────────────────────

    print(f"\n  [3/5] Segmenting tricks...", end=" ", flush=True)
    t0 = time.time()
    frame_analyses = [frame_result_to_analysis(fr) for fr in frame_results]
    segmenter = RunSegmenter()
    segments = segmenter.segment(frame_analyses)

    if len(segments) == 0:
        print(f"single-trick clip ({time.time() - t0:.1f}s)")
        seg_list = [(0, total_frames - 1)]
    else:
        print(f"{len(segments)} tricks ({time.time() - t0:.1f}s)")
        seg_list = [(s.start_frame, s.end_frame) for s in segments]

    # ── Step 4: Person detection + cropping ──────────────────────────

    print(f"\n  [4/5] Person detection...", end=" ", flush=True)
    t0 = time.time()
    boxes = detect_person_boxes(all_frames)
    detected = int(np.sum(~np.isnan(boxes[:, 0])))
    boxes = smooth_boxes(boxes)
    print(f"{detected}/{total_frames} frames ({time.time() - t0:.1f}s)")

    # ── Step 5: Embed + match each trick ─────────────────────────────

    print(f"\n  [5/5] Matching tricks...")
    t0 = time.time()

    results = []

    for trick_idx, (start, end) in enumerate(seg_list):
        start = max(0, start)
        end = min(total_frames - 1, end)

        cropped = crop_person_frames(all_frames, boxes, start, end)
        sampled = sample_frames(cropped, num_frames=16)
        if not sampled:
            continue

        inputs = processor(sampled, return_tensors="pt")
        pv = inputs["pixel_values"].to(device)

        with torch.no_grad():
            query_emb = model(pv).cpu().numpy()[0]

        matches = match_embedding(query_emb, ref_db, fig_scores, top_k=args.top_k)

        results.append({
            "trick_num": trick_idx + 1,
            "start_s": start / fps,
            "end_s": end / fps,
            "matches": matches,
        })

    match_time = time.time() - t0
    total_time = time.time() - t_total

    # ── Output ───────────────────────────────────────────────────────

    n = len(results)
    label = "trick" if n == 1 else "tricks"

    print()
    print("  PkVision v5 — Results")
    print("  " + "=" * 48)
    print(f"  Video: {video_path.name} ({n} {label})")
    print()

    total_d = 0.0

    for r in results:
        t_start = format_time(r["start_s"])
        t_end = format_time(r["end_s"])
        print(f"  Trick {r['trick_num']}  [{t_start} - {t_end}]")

        for rank, m in enumerate(r["matches"]):
            if m["similarity"] < args.min_confidence and rank > 0:
                break
            sim = m["similarity"]
            family = m["family"]
            fig = m["fig_trick"]
            d = m["d_score"]

            d_str = f"D={d:.1f}" if d > 0 else ""
            fig_str = f"  [{fig}]" if fig else ""
            bar = "█" * int(sim * 20) + "░" * (20 - int(sim * 20))
            print(f"    {rank + 1}. {family:<22s} {bar} {sim:.3f}{fig_str}  {d_str}")

        # Top prediction D-score
        if r["matches"] and r["matches"][0]["d_score"] > 0:
            total_d += r["matches"][0]["d_score"]
        print()

    # ── Summary ──────────────────────────────────────────────────────

    print(f"  {'─' * 48}")
    if total_d > 0:
        print(f"  D-score (top predictions): {total_d:.1f}")
    else:
        print(f"  D-score: N/A (model needs training)")
    print(f"  Time: {total_time:.1f}s")
    print()


if __name__ == "__main__":
    main()
