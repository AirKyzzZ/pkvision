#!/usr/bin/env python3
"""3D Parkour Judge using MotionBERT for 2D→3D pose lifting.

Pipeline:
  Video → YOLO 2D keypoints → COCO→H36M conversion → MotionBERT 3D lift
  → 3D rotation measurement → FIG candidate matching → trick identification

Usage:
    python scripts/motionbert_3d_judge.py data/run_testing/double-pov/1.mp4
    python scripts/motionbert_3d_judge.py data/run_testing/test_run_2.mp4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks

ROOT = Path(__file__).resolve().parent.parent
MOTIONBERT_ROOT = ROOT.parent / "MotionBERT"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(MOTIONBERT_ROOT))


# ═══════════════════════════════════════════════════════════════════
# COCO-17 → H36M-17 KEYPOINT CONVERSION
# ═══════════════════════════════════════════════════════════════════

# COCO-17: 0=nose, 1=Leye, 2=Reye, 3=Lear, 4=Rear, 5=Lshoulder,
#          6=Rshoulder, 7=Lelbow, 8=Relbow, 9=Lwrist, 10=Rwrist,
#          11=Lhip, 12=Rhip, 13=Lknee, 14=Rknee, 15=Lankle, 16=Rankle
#
# H36M-17: 0=Hip, 1=RHip, 2=RKnee, 3=RAnkle, 4=LHip, 5=LKnee,
#          6=LAnkle, 7=Spine, 8=Neck, 9=Nose, 10=Head, 11=LShoulder,
#          12=LElbow, 13=LWrist, 14=RShoulder, 15=RElbow, 16=RWrist

def coco_to_h36m(coco_kp: np.ndarray) -> np.ndarray:
    """Convert (T, 17, 3) COCO keypoints to (T, 17, 3) H36M format.

    Third channel is confidence score.
    """
    T = len(coco_kp)
    h36m = np.zeros((T, 17, 3))

    # Direct mappings
    h36m[:, 1] = coco_kp[:, 12]    # RHip
    h36m[:, 2] = coco_kp[:, 14]    # RKnee
    h36m[:, 3] = coco_kp[:, 16]    # RAnkle
    h36m[:, 4] = coco_kp[:, 11]    # LHip
    h36m[:, 5] = coco_kp[:, 13]    # LKnee
    h36m[:, 6] = coco_kp[:, 15]    # LAnkle
    h36m[:, 9] = coco_kp[:, 0]     # Nose
    h36m[:, 11] = coco_kp[:, 5]    # LShoulder
    h36m[:, 12] = coco_kp[:, 7]    # LElbow
    h36m[:, 13] = coco_kp[:, 9]    # LWrist
    h36m[:, 14] = coco_kp[:, 6]    # RShoulder
    h36m[:, 15] = coco_kp[:, 8]    # RElbow
    h36m[:, 16] = coco_kp[:, 10]   # RWrist

    # Computed joints (averages)
    h36m[:, 0] = (coco_kp[:, 11] + coco_kp[:, 12]) / 2    # Hip center
    h36m[:, 7] = (h36m[:, 0] + (coco_kp[:, 5] + coco_kp[:, 6]) / 2) / 2  # Spine
    h36m[:, 8] = (coco_kp[:, 5] + coco_kp[:, 6]) / 2      # Neck (shoulder center)
    h36m[:, 10] = coco_kp[:, 0] + (coco_kp[:, 0] - h36m[:, 8])  # Head (above nose)

    # Fix confidence for computed joints
    h36m[:, 0, 2] = np.minimum(coco_kp[:, 11, 2], coco_kp[:, 12, 2])
    h36m[:, 7, 2] = h36m[:, 0, 2]
    h36m[:, 8, 2] = np.minimum(coco_kp[:, 5, 2], coco_kp[:, 6, 2])
    h36m[:, 10, 2] = coco_kp[:, 0, 2]

    return h36m


# ═══════════════════════════════════════════════════════════════════
# YOLO KEYPOINT EXTRACTION
# ═══════════════════════════════════════════════════════════════════

def extract_yolo_coco(video_path: str, yolo_model) -> tuple[np.ndarray, float]:
    """Extract YOLO keypoints in COCO format with confidence.

    Returns: (keypoints (T, 17, 3) where 3=(x,y,conf), fps)
    """
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    all_kp = []
    for i in range(total):
        ret, frame = cap.read()
        if not ret:
            break

        kp = np.zeros((17, 3))
        results = yolo_model(frame, conf=0.25, verbose=False)

        if results and results[0].keypoints is not None and len(results[0].keypoints) > 0:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            best = int(np.argmax(areas))

            kp_xy = results[0].keypoints.xy.cpu().numpy()
            kp_conf = results[0].keypoints.conf.cpu().numpy() if results[0].keypoints.conf is not None else None

            if best < len(kp_xy) and kp_xy[best].shape[0] >= 17:
                kp[:, :2] = kp_xy[best][:17]
                if kp_conf is not None and best < len(kp_conf):
                    kp[:, 2] = kp_conf[best][:17]
                else:
                    kp[:, 2] = np.where(kp[:, :2].sum(axis=1) > 0, 1.0, 0.0)

        all_kp.append(kp)

        if (i + 1) % 500 == 0:
            print(f"    frame {i+1}/{total}", flush=True)

    cap.release()
    return np.array(all_kp), fps, w, h


# ═══════════════════════════════════════════════════════════════════
# MOTIONBERT 3D LIFTING
# ═══════════════════════════════════════════════════════════════════

def load_motionbert(config_path: str, checkpoint_path: str, device: str = "cpu"):
    """Load MotionBERT model."""
    from lib.utils.tools import get_config
    from lib.utils.learning import load_backbone

    args = get_config(config_path)
    model = load_backbone(args)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    # Handle DataParallel state dict
    state_dict = checkpoint["model_pos"]
    new_state = {}
    for k, v in state_dict.items():
        new_state[k.replace("module.", "")] = v
    model.load_state_dict(new_state, strict=True)
    model = model.to(device)
    model.eval()
    return model, args


def lift_to_3d(model, h36m_kp: np.ndarray, args, device: str = "cpu",
               clip_len: int = 243) -> np.ndarray:
    """Lift 2D H36M keypoints to 3D using MotionBERT.

    Args:
        h36m_kp: (T, 17, 3) H36M keypoints with confidence.

    Returns:
        3D positions (T, 17, 3) in relative coordinates.
    """
    from lib.utils.utils_data import flip_data, crop_scale

    T = len(h36m_kp)

    # Normalize: center and scale to [-1, 1]
    kp_normalized = h36m_kp.copy()
    # Center on hip
    hip = kp_normalized[:, 0:1, :2].copy()
    kp_normalized[:, :, :2] -= hip
    # Scale
    scale = np.max(np.abs(kp_normalized[:, :, :2])) + 1e-6
    kp_normalized[:, :, :2] /= scale

    # Process in clips of clip_len
    results_3d = []
    for start in range(0, T, clip_len):
        end = min(start + clip_len, T)
        clip = kp_normalized[start:end]

        # Pad if needed
        if len(clip) < clip_len:
            pad = clip_len - len(clip)
            clip = np.concatenate([clip, np.repeat(clip[-1:], pad, axis=0)])

        batch = torch.FloatTensor(clip).unsqueeze(0).to(device)  # (1, clip_len, 17, 3)

        with torch.no_grad():
            if hasattr(args, 'flip') and args.flip:
                pred1 = model(batch)
                batch_flip = flip_data(batch)
                pred2 = flip_data(model(batch_flip))
                pred = (pred1 + pred2) / 2
            else:
                pred = model(batch)

        pred = pred.cpu().numpy()[0]  # (clip_len, 17, 3)
        results_3d.append(pred[:end - start])

    return np.concatenate(results_3d)  # (T, 17, 3)


# ═══════════════════════════════════════════════════════════════════
# 3D ROTATION MEASUREMENT
# ═══════════════════════════════════════════════════════════════════

def measure_rotations_3d(skeleton_3d: np.ndarray, fps: float, start: int, end: int):
    """Measure flip and twist from 3D skeleton.

    H36M joints: 0=Hip, 7=Spine, 8=Neck, 11=LShoulder, 14=RShoulder
    V1 approach (unwrap) — best results on Double Cork.
    """
    seg = skeleton_3d[start:end + 1]
    T = len(seg)
    if T < 5:
        return 0, 0, "unknown"

    smooth_w = max(3, int(fps * 0.05))

    # Spine vector: Hip(0) → Neck(8)
    spine = seg[:, 8] - seg[:, 0]
    spine_len = np.linalg.norm(spine, axis=1, keepdims=True)
    spine_norm = spine / (spine_len + 1e-8)

    # === FLIP: unwrapped spine angle ===
    flip_angle_yz = np.arctan2(spine_norm[:, 2], spine_norm[:, 1])
    flip_yz_smooth = uniform_filter1d(flip_angle_yz, size=smooth_w)
    flip_yz_unwrap = np.unwrap(flip_yz_smooth)
    total_flip_yz = abs(flip_yz_unwrap[-1] - flip_yz_unwrap[0])

    flip_angle_xy = np.arctan2(spine_norm[:, 0], spine_norm[:, 1])
    flip_xy_smooth = uniform_filter1d(flip_angle_xy, size=smooth_w)
    flip_xy_unwrap = np.unwrap(flip_xy_smooth)
    total_flip_xy = abs(flip_xy_unwrap[-1] - flip_xy_unwrap[0])

    total_flip = max(total_flip_yz, total_flip_xy)
    flip_count = round(total_flip / np.pi * 2) / 2  # snap to 0.5

    # === TWIST: shoulder rotation around spine axis ===
    l_shoulder = seg[:, 11]
    r_shoulder = seg[:, 14]
    shoulder_vec = r_shoulder - l_shoulder

    # Remove spine component
    shoulder_perp = shoulder_vec - np.sum(shoulder_vec * spine_norm, axis=1, keepdims=True) * spine_norm
    shoulder_perp_norm = shoulder_perp / (np.linalg.norm(shoulder_perp, axis=1, keepdims=True) + 1e-8)

    # Frame-to-frame twist angle
    cross = np.cross(shoulder_perp_norm[:-1], shoulder_perp_norm[1:])
    dot = np.sum(shoulder_perp_norm[:-1] * shoulder_perp_norm[1:], axis=1)
    frame_angles = np.arctan2(np.linalg.norm(cross, axis=1), dot)
    total_twist = np.sum(np.abs(frame_angles))

    twist_count = round(total_twist / np.pi) / 2  # snap to 0.5

    # === DIRECTION ===
    if flip_count < 0.3:
        direction = "none"
    else:
        if total_flip_yz >= total_flip_xy:
            net = flip_yz_unwrap[-1] - flip_yz_unwrap[0]
        else:
            net = flip_xy_unwrap[-1] - flip_xy_unwrap[0]
        direction = "backward" if net < 0 else "forward"

    return flip_count, twist_count, direction


# ═══════════════════════════════════════════════════════════════════
# SEGMENTATION FROM 3D
# ═══════════════════════════════════════════════════════════════════

def segment_tricks_3d(skeleton_3d: np.ndarray, fps: float):
    """Segment tricks from 3D skeleton using spine inversion."""
    T = len(skeleton_3d)
    spine = skeleton_3d[:, 8] - skeleton_3d[:, 0]  # Hip→Neck

    # Spine Y component: positive = upright, negative = inverted
    spine_y = spine[:, 1]
    spine_y_smooth = uniform_filter1d(spine_y, size=max(3, int(fps * 0.1)))

    # Detect inversions: spine_y drops below threshold
    # Try both signs (3D reconstruction may flip Y)
    for sign in [1, -1]:
        signal = sign * (-spine_y_smooth)  # positive when inverted
        if signal.max() <= 0:
            continue

        signal_norm = signal / np.percentile(signal[signal > 0], 90)
        signal_smooth = uniform_filter1d(np.clip(signal_norm, 0, 2), size=max(3, int(fps * 0.1)))
        active = signal_smooth > 0.15

        # Find segments
        segments = []
        in_seg = False
        start = 0
        for i in range(T):
            if active[i] and not in_seg:
                start = i
                in_seg = True
            elif not active[i] and in_seg:
                segments.append((start, i))
                in_seg = False
        if in_seg:
            segments.append((start, T - 1))

        # Pad + merge
        pad = int(fps * 0.3)
        segments = [(max(0, s - pad), min(T - 1, e + pad)) for s, e in segments]
        merged = []
        for seg in segments:
            if merged and (seg[0] - merged[-1][1]) / fps < 0.4:
                merged[-1] = (merged[-1][0], seg[1])
            else:
                merged.append(seg)

        filtered = [(s, e) for s, e in merged if 0.3 <= (e - s) / fps <= 4.0]

        if len(filtered) >= 2:
            return filtered

    # Fallback: use 2D segmentation
    return []


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="MotionBERT 3D parkour judge")
    parser.add_argument("video", help="Video path")
    parser.add_argument("--config", default=str(MOTIONBERT_ROOT / "configs/pose3d/MB_ft_h36m_global.yaml"))
    parser.add_argument("--checkpoint", default=str(MOTIONBERT_ROOT / "checkpoint/pose3d/FT_MB_lite_MB_ft_h36m_global_lite/best_epoch.bin"))
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    from ultralytics import YOLO
    from core.vlm.fig_matcher import FIGMatcher

    matcher = FIGMatcher()
    yolo = YOLO("yolo11n-pose.pt")

    # Step 1: YOLO 2D keypoints
    print(f"\n[1/4] Extracting YOLO keypoints from {Path(args.video).name}...")
    coco_kp, fps, w, h = extract_yolo_coco(args.video, yolo)
    print(f"  {len(coco_kp)} frames @ {fps:.0f}fps ({w}x{h})")

    # Step 2: Convert to H36M + lift to 3D
    print(f"\n[2/4] Converting COCO→H36M and lifting to 3D with MotionBERT...")
    h36m_kp = coco_to_h36m(coco_kp)

    model, mb_args = load_motionbert(args.config, args.checkpoint, args.device)
    skeleton_3d = lift_to_3d(model, h36m_kp, mb_args, args.device)
    print(f"  3D skeleton: {skeleton_3d.shape}")

    # Step 3: Segment tricks
    print(f"\n[3/4] Segmenting tricks from 3D skeleton...")
    segments = segment_tricks_3d(skeleton_3d, fps)

    if not segments:
        print("  3D segmentation failed, falling back to 2D...")
        from core.segmentation import segment_tricks
        head_y = coco_kp[:, 0, 1]
        hip_y = (coco_kp[:, 11, 1] + coco_kp[:, 12, 1]) / 2
        body_angle = np.arctan2(head_y - hip_y, coco_kp[:, 0, 0] - (coco_kp[:, 11, 0] + coco_kp[:, 12, 0]) / 2)
        segs_2d = segment_tricks(
            np.where(coco_kp[:, 0, 2] > 0.3, head_y, np.nan),
            np.where((coco_kp[:, 11, 2] > 0.3) & (coco_kp[:, 12, 2] > 0.3), hip_y, np.nan),
            np.where(coco_kp[:, 0, 2] > 0.3, body_angle, np.nan),
            fps,
        )
        segments = [(s.start_frame, s.end_frame) for s in segs_2d]

    print(f"  {len(segments)} tricks found")

    # Step 4: Measure 3D rotations
    print(f"\n[4/4] Measuring 3D rotations...")
    print(f"\n{'='*70}")
    print(f"3D MOTIONBERT ANALYSIS")
    print(f"{'='*70}")

    total_d = 0
    for idx, (start, end) in enumerate(segments):
        flip, twist, direction = measure_rotations_3d(skeleton_3d, fps, start, end)
        t_start = start / fps
        t_end = end / fps
        dur = (end - start) / fps

        # FIG match
        fig = matcher.match("", flip_count=flip, twist_count=twist, direction=direction)
        fig_name = fig.fig_name if fig else "Unknown"
        d_score = fig.d_score if fig else 0
        total_d += d_score

        # Get all candidates
        candidates = []
        for entry in matcher.get_all_tricks():
            if entry.flip == flip and abs(entry.twist - twist) < 0.3:
                if direction == "unknown" or entry.direction is None or entry.direction == direction:
                    candidates.append(f"{entry.name} (D={entry.score})")

        cands = ", ".join(candidates[:5]) if candidates else "no match"
        print(f"\n  {idx+1}. {fig_name:<30} {flip}f {twist}t {direction}")
        print(f"     {t_start:.1f}s-{t_end:.1f}s ({dur:.1f}s) | D={d_score}")
        print(f"     Candidates: {cands}")

    print(f"\n{'='*70}")
    print(f"TOTAL D-SCORE: {total_d:.1f}")
    print(f"{'='*70}")

    # Save 3D data for analysis
    out_path = ROOT / "data" / "keypoints" / f"{Path(args.video).stem}_3d.npy"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, skeleton_3d)
    print(f"\n3D skeleton saved to {out_path}")


if __name__ == "__main__":
    main()
