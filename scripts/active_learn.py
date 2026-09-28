#!/usr/bin/env python3
"""Active learning pipeline for PkVision.

Full loop: scrape → crop → segment → suggest → human confirms → save to training data.

Steps:
  1. Search YouTube for parkour competition videos
  2. Download with yt-dlp
  3. YOLO-pose: detect athlete, segment trick boundaries
  4. Crop each trick to 256x256 frames
  5. CLIP suggests trick attributes (or trained model if available)
  6. Present to user for confirmation (Y/N/correct label)
  7. Save confirmed clips as .npy frames + labels → training data

Usage:
    python scripts/active_learn.py search "FISE parkour speed run"
    python scripts/active_learn.py download <youtube_url>
    python scripts/active_learn.py process <video_path>    # segment + crop + suggest
    python scripts/active_learn.py label <session_dir>     # human review UI
    python scripts/active_learn.py stats                   # show collected data stats
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SCRAPE_DIR = ROOT / "data" / "active_learning"
CONFIRMED_DIR = SCRAPE_DIR / "confirmed"
PENDING_DIR = SCRAPE_DIR / "pending"
VIDEOS_DIR = SCRAPE_DIR / "videos"


def ensure_dirs():
    for d in [SCRAPE_DIR, CONFIRMED_DIR, PENDING_DIR, VIDEOS_DIR]:
        d.mkdir(parents=True, exist_ok=True)


# ── Step 1: Search YouTube ───────────────────────────────────────────


def search_youtube(query, max_results=10):
    """Search YouTube and return video URLs."""
    print(f"\n  Searching YouTube: '{query}' (max {max_results})")
    cmd = [
        "yt-dlp", "--flat-playlist", "--no-download",
        "-j", f"ytsearch{max_results}:{query}",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

    videos = []
    for line in result.stdout.strip().split("\n"):
        if not line.strip():
            continue
        try:
            data = json.loads(line)
            videos.append({
                "id": data.get("id", ""),
                "title": data.get("title", "Unknown"),
                "url": data.get("url", f"https://youtube.com/watch?v={data.get('id', '')}"),
                "duration": data.get("duration"),
                "channel": data.get("channel", ""),
            })
        except json.JSONDecodeError:
            continue

    print(f"  Found {len(videos)} results:")
    for i, v in enumerate(videos):
        dur = f" ({v['duration']}s)" if v.get("duration") else ""
        print(f"    {i + 1}. {v['title'][:60]}{dur}")
        print(f"       {v['url']}")
    return videos


# ── Step 2: Download ─────────────────────────────────────────────────


def download_video(url, output_dir=None):
    """Download a YouTube video."""
    output_dir = output_dir or VIDEOS_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  Downloading: {url}")
    cmd = [
        "yt-dlp",
        "-f", "best[height<=720]",  # 720p max to save space
        "-o", str(output_dir / "%(title)s.%(ext)s"),
        "--no-playlist",
        url,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

    if result.returncode != 0:
        print(f"  ERROR: {result.stderr[:200]}")
        return None

    # Find the downloaded file
    for line in result.stdout.split("\n"):
        if "Destination:" in line or "has already been downloaded" in line:
            path = line.split("Destination:")[-1].strip() if "Destination:" in line else ""
            if path:
                print(f"  Saved: {path}")
                return Path(path)

    # Fallback: find most recent file in output_dir
    files = sorted(output_dir.glob("*.*"), key=lambda f: f.stat().st_mtime, reverse=True)
    video_exts = {".mp4", ".webm", ".mkv", ".mov"}
    for f in files:
        if f.suffix.lower() in video_exts:
            print(f"  Saved: {f}")
            return f

    print("  ERROR: Could not find downloaded file")
    return None


# ── Step 3: Process video (segment + crop) ───────────────────────────


def process_video(video_path, min_trick_duration=0.3, max_trick_duration=4.0):
    """Segment ONLY real acrobatic tricks from a video.

    A real trick = the athlete's body goes inverted (head below hips).
    This filters out vaults, cat leaps, walking, cameramen, spectators.

    Detection method:
    1. YOLO-pose on every frame → track the MAIN athlete (largest consistent person)
    2. Compute head-hip inversion signal (head_y > hip_y in image coords = inverted)
    3. Compute body angular velocity (rotation speed)
    4. A trick = segment where inversion happens + high angular velocity
    """
    from ultralytics import YOLO

    video_path = Path(video_path)
    if not video_path.exists():
        print(f"  ERROR: {video_path} not found")
        return None

    session_name = video_path.stem.replace(" ", "_")[:50]
    session_dir = PENDING_DIR / session_name
    session_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  Processing: {video_path.name}")
    print(f"  Session: {session_dir}")

    # Read video
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    total_frames = len(frames)
    duration = total_frames / fps
    print(f"  Video: {total_frames} frames, {duration:.1f}s @ {fps:.0f}fps")

    # YOLO-pose detection with simple tracking
    print(f"  Running YOLO-pose...", end=" ", flush=True)
    yolo_path = ROOT / "yolo11n-pose.pt"
    if not yolo_path.exists():
        yolo = YOLO("yolo11n-pose.pt")
    else:
        yolo = YOLO(str(yolo_path))

    boxes = np.full((total_frames, 4), np.nan)
    head_y = np.full(total_frames, np.nan)      # nose Y position
    hip_y = np.full(total_frames, np.nan)        # hip midpoint Y position
    body_angle = np.full(total_frames, np.nan)   # nose-hip angle
    ankle_y = np.full(total_frames, np.nan)      # ankle Y (for airborne detection)

    prev_box = None  # for simple IoU tracking

    for i, frame in enumerate(frames):
        results = yolo(frame, conf=0.25, verbose=False)
        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            continue

        box_data = results[0].boxes.xyxy.cpu().numpy()
        areas = (box_data[:, 2] - box_data[:, 0]) * (box_data[:, 3] - box_data[:, 1])

        # Track main athlete:
        # 1. IoU with previous frame (continuity)
        # 2. Prefer center of frame (camera follows athlete)
        # 3. Filter out tiny people (background spectators)
        frame_h, frame_w = frame.shape[:2]
        frame_cx, frame_cy = frame_w / 2, frame_h / 2
        min_area = (frame_h * frame_w) * 0.005  # person must be > 0.5% of frame

        # Score each detection
        scores = np.zeros(len(box_data))
        for j, b in enumerate(box_data):
            if areas[j] < min_area:
                scores[j] = -999  # too small, likely spectator
                continue
            # Center proximity score (0-1, 1=center)
            bcx = (b[0] + b[2]) / 2
            bcy = (b[1] + b[3]) / 2
            dist = np.sqrt((bcx - frame_cx)**2 + (bcy - frame_cy)**2)
            max_dist = np.sqrt(frame_cx**2 + frame_cy**2)
            center_score = 1 - (dist / max_dist)
            scores[j] = center_score * 0.4 + (areas[j] / areas.max()) * 0.3

            # IoU with previous (continuity bonus)
            if prev_box is not None:
                ix1 = max(prev_box[0], b[0])
                iy1 = max(prev_box[1], b[1])
                ix2 = min(prev_box[2], b[2])
                iy2 = min(prev_box[3], b[3])
                inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                a1 = (prev_box[2] - prev_box[0]) * (prev_box[3] - prev_box[1])
                union = a1 + areas[j] - inter
                iou = inter / max(union, 1)
                scores[j] += iou * 0.3

        best = int(np.argmax(scores))

        boxes[i] = box_data[best]
        prev_box = box_data[best]

        # Extract keypoints for the tracked person
        if results[0].keypoints is not None and len(results[0].keypoints) > 0:
            kp = results[0].keypoints.xy.cpu().numpy()
            if best < len(kp):
                kp_person = kp[best]
                # COCO keypoints: 0=nose, 5=l_shoulder, 6=r_shoulder, 11=l_hip, 12=r_hip, 15=l_ankle, 16=r_ankle
                if kp_person.shape[0] >= 17:
                    # Head position (nose, or average of eyes/ears if nose missing)
                    if kp_person[0].sum() > 0:
                        head_y[i] = kp_person[0][1]
                    elif kp_person[1].sum() > 0 and kp_person[2].sum() > 0:
                        head_y[i] = (kp_person[1][1] + kp_person[2][1]) / 2

                    # Hip midpoint
                    if kp_person[11].sum() > 0 and kp_person[12].sum() > 0:
                        hip_y[i] = (kp_person[11][1] + kp_person[12][1]) / 2
                    elif kp_person[11].sum() > 0:
                        hip_y[i] = kp_person[11][1]
                    elif kp_person[12].sum() > 0:
                        hip_y[i] = kp_person[12][1]

                    # Body angle (nose relative to hip midpoint)
                    if not np.isnan(head_y[i]) and not np.isnan(hip_y[i]):
                        hx = kp_person[0][0] if kp_person[0].sum() > 0 else (kp_person[1][0] + kp_person[2][0]) / 2
                        hip_x = (kp_person[11][0] + kp_person[12][0]) / 2 if kp_person[11].sum() > 0 else kp_person[12][0]
                        body_angle[i] = np.arctan2(head_y[i] - hip_y[i], hx - hip_x)

                    # Ankle Y (for airborne detection)
                    if kp_person[15].sum() > 0 and kp_person[16].sum() > 0:
                        ankle_y[i] = (kp_person[15][1] + kp_person[16][1]) / 2

    detected = int(np.sum(~np.isnan(boxes[:, 0])))
    print(f"{detected}/{total_frames} detections")

    # Smooth boxes for cropping
    valid = ~np.isnan(boxes[:, 0])
    if np.sum(valid) >= 2:
        for col in range(4):
            valid_idx = np.where(valid)[0]
            boxes[:, col] = np.interp(np.arange(total_frames), valid_idx, boxes[valid_idx, col])
        kernel_box = np.ones(7) / 7
        for col in range(4):
            boxes[:, col] = np.convolve(boxes[:, col], kernel_box, mode="same")

    # ── TRICK DETECTION: inversion + rotation ────────────────────────
    smooth_k = max(int(fps * 0.1), 3)
    kernel = np.ones(smooth_k) / smooth_k

    # Signal 1: INVERSION — head below hips (in image coords: head_y > hip_y)
    # This is THE key signal. A vault/cat leap never goes inverted.
    inversion = np.zeros(total_frames)
    for i in range(total_frames):
        if not np.isnan(head_y[i]) and not np.isnan(hip_y[i]):
            # In image coords, Y increases downward. head_y > hip_y means inverted.
            margin = 0  # head at same height as hips = starting to invert
            inversion[i] = max(0, head_y[i] - hip_y[i] + margin)

    # Normalize inversion signal
    if inversion.max() > 0:
        inversion_norm = inversion / np.percentile(inversion[inversion > 0], 90) if np.any(inversion > 0) else inversion
    else:
        inversion_norm = inversion
    inversion_smooth = np.convolve(np.clip(inversion_norm, 0, 2), kernel, mode="same")

    # Signal 2: Angular velocity (body rotation speed)
    angles_clean = np.copy(body_angle)
    # Interpolate NaNs
    valid_ang = ~np.isnan(angles_clean)
    if np.sum(valid_ang) >= 2:
        valid_idx = np.where(valid_ang)[0]
        angles_clean = np.interp(np.arange(total_frames), valid_idx, angles_clean[valid_idx])
    else:
        angles_clean = np.zeros(total_frames)

    ang_vel = np.abs(np.diff(angles_clean, prepend=angles_clean[0]))
    # Unwrap large jumps (angle wrapping)
    ang_vel = np.where(ang_vel > np.pi, 2 * np.pi - ang_vel, ang_vel)
    ang_vel_smooth = np.convolve(ang_vel, kernel, mode="same")
    if ang_vel_smooth.max() > 0:
        ang_vel_norm = ang_vel_smooth / np.percentile(ang_vel_smooth[ang_vel_smooth > 0], 90) if np.any(ang_vel_smooth > 0) else ang_vel_smooth
    else:
        ang_vel_norm = ang_vel_smooth

    # Combined trick score: MUST have inversion, rotation is a bonus
    # Inversion is weighted heavily — no inversion = no trick
    trick_score = inversion_smooth * 0.7 + ang_vel_norm * 0.3
    trick_score = np.convolve(trick_score, kernel, mode="same")

    # Threshold: only keep frames with significant inversion
    # Use a HIGH threshold to avoid false positives
    inv_threshold = 0.15  # must have meaningful inversion signal
    active = (inversion_smooth > inv_threshold) | (trick_score > 0.5)

    # Find contiguous active regions
    segments = []
    in_segment = False
    start = 0
    for i in range(total_frames):
        if active[i] and not in_segment:
            start = i
            in_segment = True
        elif not active[i] and in_segment:
            segments.append((start, i))
            in_segment = False
    if in_segment:
        segments.append((start, total_frames - 1))

    # Add padding (0.3s before and after)
    pad_frames = int(fps * 0.3)
    segments = [(max(0, s - pad_frames), min(total_frames - 1, e + pad_frames)) for s, e in segments]

    # Merge nearby segments (gap < 0.4s)
    merged = []
    for seg in segments:
        if merged and (seg[0] - merged[-1][1]) / fps < 0.4:
            merged[-1] = (merged[-1][0], seg[1])
        else:
            merged.append(seg)
    segments = merged

    # Filter: must have actual inversion within the segment
    filtered = []
    for start, end in segments:
        dur = (end - start) / fps
        if dur < min_trick_duration or dur > max_trick_duration:
            continue
        # Check if there's real inversion in this segment
        seg_inv = inversion[start:end + 1]
        seg_ang = ang_vel_smooth[start:end + 1]
        has_inversion = np.any(seg_inv > 0)
        has_rotation = np.max(seg_ang) > np.percentile(ang_vel_smooth[ang_vel_smooth > 0], 70) if np.any(ang_vel_smooth > 0) else False
        if has_inversion or has_rotation:
            filtered.append((start, end))
    segments = filtered

    if not segments:
        print(f"  Segments: 0 tricks detected (no inversions found)")
        # Don't create a fake segment — empty is better than noise
        segments = []

    print(f"  Segments: {len(segments)} tricks detected")

    # Crop each segment
    tricks = []
    for trick_idx, (start, end) in enumerate(segments):
        # Crop person frames
        crop_frames = []
        for i in range(start, min(end + 1, total_frames)):
            frame = frames[i]
            h, w = frame.shape[:2]
            if np.isnan(boxes[i, 0]):
                side = min(h, w)
                y1 = (h - side) // 2
                x1 = (w - side) // 2
                crop = frame[y1:y1 + side, x1:x1 + side]
            else:
                x1, y1, x2, y2 = boxes[i].astype(int)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                side = int(max(x2 - x1, y2 - y1) * 1.5)
                half = side // 2
                crop = frame[
                    max(0, cy - half):min(h, cy + half),
                    max(0, cx - half):min(w, cx + half),
                ]
            if crop.size == 0:
                crop = frame
            resized = cv2.resize(crop, (256, 256))
            crop_frames.append(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))

        # Sample 16 frames
        T = len(crop_frames)
        if T >= 16:
            indices = np.linspace(0, T - 1, 16, dtype=int)
        else:
            indices = list(range(T))
            while len(indices) < 16:
                indices.append(indices[-1])
            indices = indices[:16]

        sampled = np.stack([crop_frames[i] for i in indices])

        # Save as .npy
        trick_name = f"trick_{trick_idx + 1:03d}"
        npy_path = session_dir / f"{trick_name}.npy"
        np.save(npy_path, sampled)

        # Save preview image (middle frame)
        preview_path = session_dir / f"{trick_name}_preview.jpg"
        mid_frame = crop_frames[len(crop_frames) // 2]
        cv2.imwrite(str(preview_path), cv2.cvtColor(mid_frame, cv2.COLOR_RGB2BGR))

        start_s = start / fps
        end_s = end / fps
        tricks.append({
            "name": trick_name,
            "npy_path": str(npy_path),
            "preview": str(preview_path),
            "start_s": start_s,
            "end_s": end_s,
            "duration_s": end_s - start_s,
            "num_frames": T,
            "suggested_label": None,
            "confirmed_label": None,
            "status": "pending",
        })
        print(f"    Trick {trick_idx + 1}: [{start_s:.1f}s - {end_s:.1f}s] ({T} frames)")

    # Save session manifest
    manifest = {
        "video": str(video_path),
        "video_name": video_path.name,
        "fps": fps,
        "total_frames": total_frames,
        "duration_s": duration,
        "tricks": tricks,
    }
    manifest_path = session_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n  Session saved: {session_dir}")
    print(f"  {len(tricks)} tricks ready for labeling")
    return session_dir


# ── Step 4: Suggest labels with CLIP ─────────────────────────────────


def suggest_labels(session_dir):
    """Use CLIP to suggest trick attributes for pending clips."""
    import open_clip
    import torch
    from PIL import Image

    session_dir = Path(session_dir)
    manifest_path = session_dir / "manifest.json"
    with open(manifest_path) as f:
        manifest = json.load(f)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"\n  Loading CLIP on {device}...", end=" ", flush=True)
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai", device=device,
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    model.eval()
    print("OK")

    # Attribute prompts
    attr_prompts = {
        "direction": {
            "backward": "a person flipping backward, backflip, gainer",
            "forward": "a person flipping forward, frontflip, webster",
            "side": "a person flipping sideways, cartwheel, sideflip, aerial",
            "none": "a person jumping without flipping, vault, precision jump",
        },
        "flip_count": {
            "0": "a person jumping or vaulting without any somersault",
            "1": "a person doing a single flip, one somersault",
            "2+": "a person doing a double or triple flip, multiple somersaults",
        },
        "context": {
            "acrobatics": "a person doing acrobatics on flat ground, tumbling",
            "wall": "a person doing a trick off a wall, wall flip",
            "swing": "a person on a bar, swinging and releasing",
            "pk_basics": "a person vaulting over an obstacle, parkour vault",
        },
    }

    # Encode prompts
    attr_embeddings = {}
    for attr_name, classes in attr_prompts.items():
        class_names = list(classes.keys())
        texts = list(classes.values())
        tokens = tokenizer(texts).to(device)
        with torch.no_grad():
            feats = model.encode_text(tokens)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        attr_embeddings[attr_name] = (class_names, feats)

    # Suggest for each trick
    for trick in manifest["tricks"]:
        if trick["status"] != "pending":
            continue

        npy_path = trick["npy_path"]
        frames = np.load(npy_path)
        # Sample 8 frames for CLIP
        indices = np.linspace(0, frames.shape[0] - 1, 8, dtype=int)
        images = [preprocess(Image.fromarray(frames[i])) for i in indices]
        batch = torch.stack(images).to(device)

        with torch.no_grad():
            img_feats = model.encode_image(batch)
            img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
        avg = img_feats.mean(dim=0, keepdim=True)
        avg = avg / avg.norm(dim=-1, keepdim=True)

        suggestion = {}
        for attr_name, (class_names, text_feats) in attr_embeddings.items():
            sims = (avg @ text_feats.T).squeeze(0).cpu().numpy()
            best_idx = int(np.argmax(sims))
            suggestion[attr_name] = {
                "value": class_names[best_idx],
                "confidence": float(sims[best_idx]),
            }

        trick["suggested_label"] = suggestion
        print(f"  {trick['name']}: "
              f"dir={suggestion['direction']['value']} ({suggestion['direction']['confidence']:.2f})  "
              f"flip={suggestion['flip_count']['value']} ({suggestion['flip_count']['confidence']:.2f})  "
              f"ctx={suggestion['context']['value']} ({suggestion['context']['confidence']:.2f})")

    # Save updated manifest
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n  Suggestions saved to {manifest_path}")


# ── Step 5: Human labeling UI ────────────────────────────────────────


def label_session(session_dir):
    """Interactive labeling: show suggestion, human confirms or corrects."""
    session_dir = Path(session_dir)
    manifest_path = session_dir / "manifest.json"
    with open(manifest_path) as f:
        manifest = json.load(f)

    pending = [t for t in manifest["tricks"] if t["status"] == "pending"]
    if not pending:
        print("  No pending tricks to label.")
        return

    print(f"\n  Labeling {len(pending)} tricks from: {manifest['video_name']}")
    print(f"  Preview images in: {session_dir}")
    print(f"  Commands: Y=accept, N=skip, Q=quit, or type correct label")
    print(f"  Label format: direction,flip_count,context (e.g. backward,1,acrobatics)")
    print()

    confirmed = 0
    for trick in pending:
        suggestion = trick.get("suggested_label", {})
        s_dir = suggestion.get("direction", {}).get("value", "?")
        s_flip = suggestion.get("flip_count", {}).get("value", "?")
        s_ctx = suggestion.get("context", {}).get("value", "?")

        print(f"  {trick['name']} [{trick['start_s']:.1f}s-{trick['end_s']:.1f}s]")
        print(f"    Preview: {trick['preview']}")
        print(f"    Suggestion: dir={s_dir}, flip={s_flip}, ctx={s_ctx}")

        response = input(f"    Accept? [Y/n/quit/label]: ").strip()

        if response.lower() in ("q", "quit"):
            break
        elif response.lower() in ("y", "yes", ""):
            trick["confirmed_label"] = {
                "direction": s_dir,
                "flip_count": s_flip,
                "context": s_ctx,
            }
            trick["status"] = "confirmed"
            confirmed += 1
            print(f"    -> Confirmed: {s_dir}, {s_flip}, {s_ctx}")
        elif response.lower() in ("n", "no", "skip"):
            trick["status"] = "skipped"
            print(f"    -> Skipped")
        elif "," in response:
            parts = [p.strip() for p in response.split(",")]
            if len(parts) >= 3:
                trick["confirmed_label"] = {
                    "direction": parts[0],
                    "flip_count": parts[1],
                    "context": parts[2],
                }
                trick["status"] = "confirmed"
                confirmed += 1
                print(f"    -> Corrected: {parts[0]}, {parts[1]}, {parts[2]}")
        else:
            print(f"    -> Unknown command, skipping")
        print()

    # Save
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    total_confirmed = sum(1 for t in manifest["tricks"] if t["status"] == "confirmed")
    print(f"\n  Session: {confirmed} new confirmations, {total_confirmed} total confirmed")

    # Export confirmed to training format
    if total_confirmed > 0:
        export_confirmed(session_dir)


def export_confirmed(session_dir):
    """Export confirmed tricks to training data format."""
    session_dir = Path(session_dir)
    manifest_path = session_dir / "manifest.json"
    with open(manifest_path) as f:
        manifest = json.load(f)

    confirmed = [t for t in manifest["tricks"] if t["status"] == "confirmed"]
    if not confirmed:
        return

    # Copy .npy files to confirmed dir with labels
    exported = []
    for trick in confirmed:
        src = Path(trick["npy_path"])
        if not src.exists():
            continue

        label = trick["confirmed_label"]
        slug = f"{session_dir.name}_{trick['name']}"
        dst = CONFIRMED_DIR / f"{slug}.npy"

        if not dst.exists():
            import shutil
            shutil.copy2(src, dst)

        exported.append({
            "slug": slug,
            "npy_path": str(dst),
            "direction": label.get("direction", "none"),
            "flip_count": label.get("flip_count", "0"),
            "context": label.get("context", "acrobatics"),
            "source": "active_learning",
            "video": manifest["video_name"],
        })

    # Save/append to confirmed manifest
    confirmed_manifest = CONFIRMED_DIR / "confirmed_clips.json"
    existing = []
    if confirmed_manifest.exists():
        with open(confirmed_manifest) as f:
            existing = json.load(f)

    # Merge (avoid duplicates)
    existing_slugs = {e["slug"] for e in existing}
    for e in exported:
        if e["slug"] not in existing_slugs:
            existing.append(e)

    with open(confirmed_manifest, "w") as f:
        json.dump(existing, f, indent=2)

    print(f"  Exported {len(exported)} clips to {CONFIRMED_DIR}")


# ── Stats ────────────────────────────────────────────────────────────


def show_stats():
    """Show active learning data collection stats."""
    print(f"\n  Active Learning Stats")
    print(f"  {'=' * 40}")

    # Count sessions
    sessions = list(PENDING_DIR.glob("*/manifest.json"))
    total_tricks = 0
    confirmed = 0
    pending = 0
    skipped = 0

    for mp in sessions:
        with open(mp) as f:
            m = json.load(f)
        for t in m["tricks"]:
            total_tricks += 1
            if t["status"] == "confirmed":
                confirmed += 1
            elif t["status"] == "pending":
                pending += 1
            elif t["status"] == "skipped":
                skipped += 1

    print(f"  Sessions:  {len(sessions)}")
    print(f"  Tricks:    {total_tricks}")
    print(f"  Confirmed: {confirmed}")
    print(f"  Pending:   {pending}")
    print(f"  Skipped:   {skipped}")

    # Confirmed clips
    confirmed_manifest = CONFIRMED_DIR / "confirmed_clips.json"
    if confirmed_manifest.exists():
        with open(confirmed_manifest) as f:
            clips = json.load(f)
        print(f"\n  Confirmed training clips: {len(clips)}")
    print()


# ── Main ─────────────────────────────────────────────────────────────


def main():
    ensure_dirs()

    parser = argparse.ArgumentParser(description="PkVision Active Learning")
    sub = parser.add_subparsers(dest="command")

    s = sub.add_parser("search", help="Search YouTube")
    s.add_argument("query", nargs="+")
    s.add_argument("--max", type=int, default=10)

    d = sub.add_parser("download", help="Download a YouTube video")
    d.add_argument("url")

    p = sub.add_parser("process", help="Process video: segment + crop")
    p.add_argument("video")

    l = sub.add_parser("label", help="Label pending tricks")
    l.add_argument("session", help="Session directory path")

    sub.add_parser("stats", help="Show collection stats")

    # Convenience: full pipeline
    f = sub.add_parser("pipeline", help="Full pipeline: download + process + suggest + label")
    f.add_argument("url")

    args = parser.parse_args()

    if args.command == "search":
        search_youtube(" ".join(args.query), args.max)

    elif args.command == "download":
        download_video(args.url)

    elif args.command == "process":
        session = process_video(args.video)
        if session:
            suggest_labels(session)

    elif args.command == "label":
        label_session(args.session)

    elif args.command == "stats":
        show_stats()

    elif args.command == "pipeline":
        video = download_video(args.url)
        if video:
            session = process_video(video)
            if session:
                suggest_labels(session)
                label_session(session)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
