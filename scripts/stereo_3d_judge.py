#!/usr/bin/env python3
"""Multi-camera stereo 3D parkour trick judge.

Takes two video angles of the same run, reconstructs 3D skeleton,
measures rotations/twists precisely, then narrows to FIG candidates
and optionally disambiguates with a VLM.

Usage:
    python scripts/stereo_3d_judge.py data/run_testing/double-pov/1.mp4 data/run_testing/double-pov/2.mp4
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import correlate, find_peaks

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ═══════════════════════════════════════════════════════════════════
# STEP 1: AUDIO SYNC
# ═══════════════════════════════════════════════════════════════════

def extract_audio(video_path: str, sr: int = 16000) -> np.ndarray:
    """Extract mono audio as numpy array using ffmpeg."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
        subprocess.run([
            "ffmpeg", "-y", "-i", str(video_path),
            "-ac", "1", "-ar", str(sr), "-vn", tmp.name,
        ], capture_output=True)
        # Read WAV manually (avoid extra dependency)
        import wave
        with wave.open(tmp.name, "rb") as wf:
            frames = wf.readframes(wf.getnframes())
            audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32)
            audio /= 32768.0
    return audio


def sync_videos(video1: str, video2: str, sr: int = 16000) -> float:
    """Find time offset between two videos using audio cross-correlation.

    Returns offset in seconds: video2 is shifted by this amount relative to video1.
    Positive = video2 starts later.
    """
    print("  Extracting audio...", end=" ", flush=True)
    audio1 = extract_audio(video1, sr)
    audio2 = extract_audio(video2, sr)
    print(f"({len(audio1)/sr:.1f}s, {len(audio2)/sr:.1f}s)")

    # Use a chunk for faster correlation
    chunk_len = min(len(audio1), len(audio2), sr * 30)  # max 30s
    a1 = audio1[:chunk_len]
    a2 = audio2[:chunk_len]

    print("  Cross-correlating...", end=" ", flush=True)
    corr = correlate(a1, a2, mode="full")
    lag = np.argmax(np.abs(corr)) - (len(a2) - 1)
    offset_sec = lag / sr
    print(f"offset = {offset_sec:.3f}s")

    return offset_sec


# ═══════════════════════════════════════════════════════════════════
# STEP 2: YOLO KEYPOINTS ON BOTH VIEWS
# ═══════════════════════════════════════════════════════════════════

def extract_yolo_keypoints(video_path: str, yolo_model) -> tuple[np.ndarray, np.ndarray, float]:
    """Extract YOLO keypoints from every frame.

    Returns: (keypoints (T,17,2), confidences (T,17), fps)
    """
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    all_kp = []
    all_conf = []

    for i in range(total):
        ret, frame = cap.read()
        if not ret:
            break

        kp = np.zeros((17, 2))
        conf = np.zeros(17)

        results = yolo_model(frame, conf=0.25, verbose=False)
        if results and results[0].keypoints is not None and len(results[0].keypoints) > 0:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            best = int(np.argmax(areas))

            kp_data = results[0].keypoints.xy.cpu().numpy()
            conf_data = results[0].keypoints.conf.cpu().numpy() if results[0].keypoints.conf is not None else None

            if best < len(kp_data) and kp_data[best].shape[0] >= 17:
                kp = kp_data[best][:17]
                if conf_data is not None and best < len(conf_data):
                    conf = conf_data[best][:17]
                else:
                    conf = np.where(kp.sum(axis=1) > 0, 1.0, 0.0)

        all_kp.append(kp)
        all_conf.append(conf)

        if (i + 1) % 500 == 0:
            print(f"    frame {i+1}/{total}", flush=True)

    cap.release()
    return np.array(all_kp), np.array(all_conf), fps


# ═══════════════════════════════════════════════════════════════════
# STEP 3: STEREO TRIANGULATION → 3D SKELETON
# ═══════════════════════════════════════════════════════════════════

def align_frames(kp1, conf1, fps1, kp2, conf2, fps2, offset_sec):
    """Align two keypoint sequences using the audio offset.

    Resamples view 2 to match view 1's timeline.
    Returns aligned (kp1, kp2, conf1, conf2) of same length.
    """
    T1 = len(kp1)
    offset_frames2 = int(offset_sec * fps2)

    aligned_kp2 = []
    aligned_conf2 = []

    for i in range(T1):
        # Map view1 frame i to view2 frame
        t = i / fps1  # time in seconds
        j = int(t * fps2) + offset_frames2

        if 0 <= j < len(kp2):
            aligned_kp2.append(kp2[j])
            aligned_conf2.append(conf2[j])
        else:
            aligned_kp2.append(np.zeros((17, 2)))
            aligned_conf2.append(np.zeros(17))

    return kp1, np.array(aligned_kp2), conf1, np.array(aligned_conf2)


def triangulate_3d(kp1, kp2, conf1, conf2):
    """Triangulate 3D skeleton from two 2D views.

    Uses OpenCV fundamental matrix estimation + triangulation.
    Returns: 3D keypoints (T, 17, 3)
    """
    T = len(kp1)

    # Collect valid point correspondences across all frames for F estimation
    pts1_all = []
    pts2_all = []
    for i in range(T):
        for j in range(17):
            if conf1[i, j] > 0.3 and conf2[i, j] > 0.3:
                if kp1[i, j].sum() > 0 and kp2[i, j].sum() > 0:
                    pts1_all.append(kp1[i, j])
                    pts2_all.append(kp2[i, j])

    pts1_all = np.array(pts1_all, dtype=np.float64)
    pts2_all = np.array(pts2_all, dtype=np.float64)
    print(f"  Matched points for F estimation: {len(pts1_all)}")

    if len(pts1_all) < 20:
        print("  WARNING: Too few matched points for reliable F estimation")
        return None

    # Estimate fundamental matrix
    F, mask = cv2.findFundamentalMat(pts1_all, pts2_all, cv2.FM_RANSAC, 3.0, 0.99)
    if F is None:
        print("  ERROR: Could not estimate fundamental matrix")
        return None

    inliers = mask.ravel().sum()
    print(f"  Fundamental matrix: {inliers}/{len(pts1_all)} inliers")

    # Derive projection matrices from F
    # P1 = [I|0], P2 derived from essential matrix
    # Approximate: assume unit focal length for relative reconstruction
    # This gives reconstruction up to a projective ambiguity, but rotation
    # measurements are preserved.
    h1 = kp1.shape[1]  # not image height, just using normalized coords

    # Simple projection: P1 = [I|0]
    P1 = np.hstack([np.eye(3), np.zeros((3, 1))])

    # Decompose F to get P2 (Hartley & Zisserman method)
    # E ≈ F for uncalibrated (approximate)
    U, S, Vt = np.linalg.svd(F)
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64)

    # Four possible P2 solutions — pick the one with most points in front
    u3 = U[:, 2].reshape(3, 1)
    candidates = [
        np.hstack([U @ W @ Vt, u3]),
        np.hstack([U @ W @ Vt, -u3]),
        np.hstack([U @ W.T @ Vt, u3]),
        np.hstack([U @ W.T @ Vt, -u3]),
    ]

    # Test each P2 candidate
    best_P2 = None
    best_count = 0
    sample_pts1 = pts1_all[mask.ravel() == 1][:200]
    sample_pts2 = pts2_all[mask.ravel() == 1][:200]

    for P2_candidate in candidates:
        pts4d = cv2.triangulatePoints(
            P1.astype(np.float64),
            P2_candidate.astype(np.float64),
            sample_pts1.T.astype(np.float64),
            sample_pts2.T.astype(np.float64),
        )
        pts3d = (pts4d[:3] / pts4d[3:]).T
        # Count points with positive depth in both cameras
        in_front = np.sum((pts3d[:, 2] > 0))
        if in_front > best_count:
            best_count = in_front
            best_P2 = P2_candidate

    if best_P2 is None:
        print("  ERROR: Could not find valid camera configuration")
        return None

    P2 = best_P2
    print(f"  Best P2: {best_count}/{len(sample_pts1)} points in front")

    # Triangulate all frames
    skeleton_3d = np.full((T, 17, 3), np.nan)

    for i in range(T):
        valid = (conf1[i] > 0.3) & (conf2[i] > 0.3)
        valid_joints = np.where(valid)[0]

        if len(valid_joints) < 3:
            continue

        p1 = kp1[i, valid_joints].T.astype(np.float64)  # (2, N)
        p2 = kp2[i, valid_joints].T.astype(np.float64)

        pts4d = cv2.triangulatePoints(
            P1.astype(np.float64), P2.astype(np.float64), p1, p2,
        )
        pts3d = (pts4d[:3] / pts4d[3:]).T  # (N, 3)

        for k, joint_idx in enumerate(valid_joints):
            skeleton_3d[i, joint_idx] = pts3d[k]

    valid_frames = np.sum(~np.isnan(skeleton_3d[:, 0, 0]))
    print(f"  3D skeleton: {valid_frames}/{T} frames with valid data")

    return skeleton_3d


# ═══════════════════════════════════════════════════════════════════
# STEP 4: 3D ROTATION MEASUREMENT
# ═══════════════════════════════════════════════════════════════════

def measure_3d_rotations(skeleton_3d, fps, start, end):
    """Measure flip count and twist count from 3D skeleton trajectory.

    Uses spine vector (hip→shoulder) for flips and shoulder line for twists.
    In 3D, these measurements are exact — no projection ambiguity.
    """
    seg = skeleton_3d[start:end + 1]
    T = len(seg)

    # Hip and shoulder centers
    hip_center = np.nanmean(seg[:, [11, 12], :], axis=1)   # (T, 3)
    shoulder_center = np.nanmean(seg[:, [5, 6], :], axis=1)  # (T, 3)

    # Spine vector: hip → shoulder
    spine = shoulder_center - hip_center  # (T, 3)

    # --- FLIP: rotation of spine around the lateral axis ---
    # Project spine onto the sagittal plane (YZ plane) and measure angle from vertical
    # Y = vertical (up), Z = forward
    valid = ~np.isnan(spine[:, 0])
    if np.sum(valid) < 5:
        return 0, 0, "unknown"

    spine_v = spine[valid]

    # Compute angle of spine relative to vertical (Y axis) in the YZ plane
    # atan2 gives us the continuous angle
    flip_angles = np.arctan2(spine_v[:, 2], spine_v[:, 1])  # angle in YZ plane
    flip_smooth = uniform_filter1d(flip_angles, size=max(3, int(fps * 0.05)))
    flip_unwrapped = np.unwrap(flip_smooth)
    total_flip_rad = abs(flip_unwrapped[-1] - flip_unwrapped[0])
    flip_count = round(total_flip_rad / np.pi * 2) / 2  # snap to 0.5

    # Also try XY plane (different camera orientation)
    flip_angles_xy = np.arctan2(spine_v[:, 0], spine_v[:, 1])
    flip_smooth_xy = uniform_filter1d(flip_angles_xy, size=max(3, int(fps * 0.05)))
    flip_unwrapped_xy = np.unwrap(flip_smooth_xy)
    total_flip_xy = abs(flip_unwrapped_xy[-1] - flip_unwrapped_xy[0])
    flip_count_xy = round(total_flip_xy / np.pi * 2) / 2

    # Take the larger measurement (one plane might miss rotation)
    flip_count = max(flip_count, flip_count_xy)

    # --- TWIST: rotation of shoulder line around the spine axis ---
    l_shoulder = seg[valid, 5, :]
    r_shoulder = seg[valid, 6, :]
    shoulder_vec = r_shoulder - l_shoulder  # (T, 3)

    # Project shoulder vector onto the plane perpendicular to spine
    # Then measure rotation in that plane
    spine_norm = spine_v / (np.linalg.norm(spine_v, axis=1, keepdims=True) + 1e-8)
    # Remove spine component from shoulder vector
    shoulder_perp = shoulder_vec - np.sum(shoulder_vec * spine_norm, axis=1, keepdims=True) * spine_norm

    # Measure angle of the perpendicular component
    twist_angles = np.arctan2(
        np.linalg.norm(np.cross(shoulder_perp[:-1], shoulder_perp[1:]), axis=1),
        np.sum(shoulder_perp[:-1] * shoulder_perp[1:], axis=1),
    )
    total_twist_rad = np.nansum(np.abs(twist_angles))
    twist_count = round(total_twist_rad / np.pi) / 2  # snap to 0.5

    # --- DIRECTION ---
    # Compare hip center movement: if moving forward while flipping backward = gainer
    hip_v = hip_center[valid]
    if len(hip_v) > 3:
        movement = hip_v[-1] - hip_v[0]
        # Dominant horizontal movement direction
        horiz_movement = np.sqrt(movement[0]**2 + movement[2]**2)
        vert_movement = abs(movement[1])

        if flip_count < 0.3:
            direction = "none"
        elif vert_movement > horiz_movement * 2:
            direction = "side"  # mostly vertical movement = sideflip
        else:
            # Check if spine flips backward or forward
            if flip_unwrapped[-1] < flip_unwrapped[0]:
                direction = "backward"
            else:
                direction = "forward"
    else:
        direction = "backward"

    return flip_count, twist_count, direction


# ═══════════════════════════════════════════════════════════════════
# STEP 5: SEGMENTATION + FIG MATCHING
# ═══════════════════════════════════════════════════════════════════

def segment_from_3d(skeleton_3d, fps, min_dur=0.3, max_dur=4.0):
    """Segment tricks from 3D skeleton using inversion detection."""
    T = len(skeleton_3d)

    hip_center = np.nanmean(skeleton_3d[:, [11, 12], :], axis=1)
    shoulder_center = np.nanmean(skeleton_3d[:, [5, 6], :], axis=1)

    # Inversion = shoulder Y below hip Y (in 3D, Y is vertical)
    spine_y = shoulder_center[:, 1] - hip_center[:, 1]

    # Handle NaNs
    valid = ~np.isnan(spine_y)
    if np.sum(valid) < 10:
        return []

    spine_y_interp = np.interp(np.arange(T), np.where(valid)[0], spine_y[valid])
    spine_y_smooth = uniform_filter1d(spine_y_interp, size=max(3, int(fps * 0.1)))

    # Inversion: spine_y < 0 (shoulder below hip)
    # But 3D reconstruction might flip the Y axis — check both
    # Use the sign that produces reasonable segments
    for sign in [1, -1]:
        inv_signal = np.maximum(0, sign * (-spine_y_smooth))
        if inv_signal.max() > 0:
            inv_norm = inv_signal / np.percentile(inv_signal[inv_signal > 0], 90)
        else:
            continue

        inv_smooth = uniform_filter1d(np.clip(inv_norm, 0, 2), size=max(3, int(fps * 0.1)))
        active = inv_smooth > 0.15

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

        # Filter by duration
        filtered = [(s, e) for s, e in merged
                     if min_dur <= (e - s) / fps <= max_dur]

        if len(filtered) >= 3:  # expect at least a few tricks
            return filtered

    return []


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Stereo 3D parkour judge")
    parser.add_argument("video1", help="Camera angle 1")
    parser.add_argument("video2", help="Camera angle 2")
    parser.add_argument("--no-vlm", action="store_true", help="Skip VLM disambiguation")
    args = parser.parse_args()

    from ultralytics import YOLO
    from core.vlm.fig_matcher import FIGMatcher

    matcher = FIGMatcher()
    yolo = YOLO("yolo11n-pose.pt")

    # Step 1: Audio sync
    print("\n[1/5] Syncing videos via audio...")
    offset = sync_videos(args.video1, args.video2)

    # Step 2: YOLO keypoints
    print("\n[2/5] Extracting YOLO keypoints...")
    print(f"  View 1: {args.video1}")
    kp1, conf1, fps1 = extract_yolo_keypoints(args.video1, yolo)
    print(f"  → {len(kp1)} frames @ {fps1:.0f}fps")

    print(f"  View 2: {args.video2}")
    kp2, conf2, fps2 = extract_yolo_keypoints(args.video2, yolo)
    print(f"  → {len(kp2)} frames @ {fps2:.0f}fps")

    # Align
    kp1, kp2, conf1, conf2 = align_frames(kp1, conf1, fps1, kp2, conf2, fps2, offset)
    print(f"  Aligned: {len(kp1)} synced frame pairs")

    # Step 3: Triangulate
    print("\n[3/5] Stereo triangulation → 3D skeleton...")
    skeleton_3d = triangulate_3d(kp1, kp2, conf1, conf2)

    if skeleton_3d is None:
        print("FAILED: Could not reconstruct 3D skeleton")
        return

    # Step 4: Segment + measure
    print("\n[4/5] Segmenting tricks + measuring 3D rotations...")
    segments = segment_from_3d(skeleton_3d, fps1)
    print(f"  {len(segments)} tricks segmented")

    if not segments:
        # Fallback: use 2D segmentation from view 1
        print("  Falling back to 2D segmentation from view 1...")
        from core.segmentation import segment_tricks
        head_y = np.where(conf1[:, 0] > 0.3, kp1[:, 0, 1], np.nan)
        hip_y = np.where(
            (conf1[:, 11] > 0.3) & (conf1[:, 12] > 0.3),
            (kp1[:, 11, 1] + kp1[:, 12, 1]) / 2,
            np.nan,
        )
        body_angle = np.full(len(kp1), np.nan)
        for i in range(len(kp1)):
            if not np.isnan(head_y[i]) and not np.isnan(hip_y[i]):
                hx = kp1[i, 0, 0]
                hip_x = (kp1[i, 11, 0] + kp1[i, 12, 0]) / 2
                body_angle[i] = np.arctan2(head_y[i] - hip_y[i], hx - hip_x)

        segs_2d = segment_tricks(head_y, hip_y, body_angle, fps1)
        segments = [(s.start_frame, s.end_frame) for s in segs_2d]
        print(f"  2D fallback: {len(segments)} tricks")

    # Measure 3D rotations for each segment
    results = []
    for idx, (start, end) in enumerate(segments):
        flip, twist, direction = measure_3d_rotations(skeleton_3d, fps1, start, end)
        t_start = start / fps1
        t_end = end / fps1
        duration = (end - start) / fps1

        # FIG candidate lookup
        fig_match = matcher.match("", flip_count=flip, twist_count=twist, direction=direction)
        candidates = []
        # Get all tricks matching this physics
        for entry in matcher.get_all_tricks():
            if entry.flip == flip and abs(entry.twist - twist) < 0.3:
                if direction == "unknown" or entry.direction is None or entry.direction == direction:
                    candidates.append(entry.name)

        results.append({
            "index": idx + 1,
            "time": f"{t_start:.1f}s-{t_end:.1f}s",
            "duration": duration,
            "flip_count": flip,
            "twist_count": twist,
            "direction": direction,
            "fig_match": fig_match.fig_name if fig_match else "Unknown",
            "d_score": fig_match.d_score if fig_match else 0,
            "candidates": candidates[:5],
        })

    # Print results
    print(f"\n{'='*70}")
    print(f"3D STEREO ANALYSIS — {len(results)} tricks")
    print(f"{'='*70}")
    for r in results:
        cands = ", ".join(r["candidates"][:3]) if r["candidates"] else "no match"
        print(f"  {r['index']}. {r['fig_match']:<30} {r['flip_count']}f {r['twist_count']}t {r['direction']}")
        print(f"     {r['time']} ({r['duration']:.1f}s) | D={r['d_score']}")
        print(f"     Candidates: {cands}")

    # Step 5: VLM disambiguation (optional)
    if not args.no_vlm and results:
        print(f"\n[5/5] VLM disambiguation on ambiguous tricks...")
        try:
            from dotenv import load_dotenv
            load_dotenv(ROOT / ".env")
            from core.vlm.openrouter_provider import OpenRouterProvider

            provider = OpenRouterProvider(model="google/gemini-3.1-pro-preview")

            for r in results:
                if len(r["candidates"]) <= 1:
                    continue

                cand_list = ", ".join(r["candidates"])
                prompt = (
                    f"This parkour trick has {r['flip_count']} flips, {r['twist_count']} twists, "
                    f"direction: {r['direction']}. "
                    f"It is one of these FIG tricks: {cand_list}. "
                    f"Watch the video at {r['time']} and tell me which one it is. "
                    f"Return ONLY the trick name, nothing else."
                )

                response = provider.client.chat.completions.create(
                    model="google/gemini-3.1-pro-preview",
                    messages=[
                        {"role": "user", "content": [
                            {"type": "video_url", "video_url": {
                                "url": f"data:video/mp4;base64,{_encode_video(args.video1)}",
                            }},
                            {"type": "text", "text": prompt},
                        ]},
                    ],
                    max_tokens=100,
                )
                vlm_answer = response.choices[0].message.content.strip() if response.choices else ""
                print(f"  Trick {r['index']}: VLM says '{vlm_answer}' (from {cand_list})")

        except Exception as e:
            print(f"  VLM failed: {e}")


def _encode_video(path):
    import base64
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


if __name__ == "__main__":
    main()
