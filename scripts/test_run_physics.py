#!/usr/bin/env python3
"""Analyze a parkour run using physics-based trick detection.

Instead of MLP classification, this extracts actual physics from the movement:
- Total body rotation (flip count)
- Twist detection (shoulder oscillation)
- Body shape (tuck/pike/layout from joint angles)
- Entry type (standing/running detection)

Then matches against the trick database using the zero-shot matcher.

Usage:
    python scripts/test_run_physics.py data/mlp_testing/IMG_4738.MOV
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))


FRAME_SCALE = np.array([640.0, 480.0], dtype=np.float32)


def extract_keypoints(video_path: str) -> tuple[np.ndarray, float]:
    """Extract YOLO keypoints. Returns ((3, T, 17), fps)."""
    from ultralytics import YOLO
    import cv2

    model = YOLO("yolo11n-pose.pt")
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    print(f"  Video: {total} frames @ {fps:.1f} FPS ({total/fps:.1f}s)")

    frames = []
    for result in model(video_path, stream=True, verbose=False):
        if result.keypoints is not None and len(result.keypoints) > 0:
            kp = result.keypoints[0]
            xy = kp.xyn[0].cpu().numpy()
            conf = kp.conf[0].cpu().numpy()
            f = np.zeros((3, 17), dtype=np.float32)
            f[0], f[1], f[2] = xy[:, 0], xy[:, 1], conf
            frames.append(f)
        else:
            frames.append(np.zeros((3, 17), dtype=np.float32))

    arr = np.stack(frames).transpose(1, 0, 2).astype(np.float32)
    return arr, fps


def compute_physics_per_frame(kp: np.ndarray) -> dict:
    """Compute physics features from (3, T, 17) keypoints.

    Returns dict with per-frame arrays:
    - body_angle: body tilt angle per frame
    - angular_vel: rotation speed per frame
    - com_y: center of mass vertical position
    - lr_asymmetry: left/right shoulder asymmetry (side flip indicator)
    - knee_angle: average knee angle (tuck/pike/layout)
    - hip_angle: average hip angle
    """
    C, T, V = kp.shape

    # COM from hips + shoulders
    com_x = np.mean(kp[0, :, :][:, [5, 6, 11, 12]], axis=1)
    com_y = np.mean(kp[1, :, :][:, [5, 6, 11, 12]], axis=1)

    # Body angle: nose relative to COM
    body_angle = np.arctan2(kp[1, :, 0] - com_y, kp[0, :, 0] - com_x)

    # Angular velocity (handle wrapping)
    d_angle = np.diff(body_angle)
    d_angle = np.where(d_angle > np.pi, d_angle - 2 * np.pi, d_angle)
    d_angle = np.where(d_angle < -np.pi, d_angle + 2 * np.pi, d_angle)
    angular_vel = np.zeros(T)
    angular_vel[1:] = d_angle

    # Left/right shoulder asymmetry (sideflip = high asymmetry)
    lr_asym = np.abs(kp[1, :, 5] - kp[1, :, 6])  # |left_shoulder_y - right_shoulder_y|

    # Knee angle proxy: distance between hip and ankle relative to hip-knee + knee-ankle
    # Lower values = more tucked
    def limb_angle(a, b, c):
        """Angle at joint b from segments a-b and b-c."""
        T = a.shape[0]
        angles = np.zeros(T)
        for t in range(T):
            v1 = a[t] - b[t]
            v2 = c[t] - b[t]
            cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
            angles[t] = np.degrees(np.arccos(np.clip(cos_a, -1, 1)))
        return angles

    # COCO: 11=left_hip, 13=left_knee, 15=left_ankle, 12=right_hip, 14=right_knee, 16=right_ankle
    pts = np.stack([kp[0], kp[1]], axis=-1)  # (T, 17, 2)
    l_knee = limb_angle(pts[:, 11], pts[:, 13], pts[:, 15])
    r_knee = limb_angle(pts[:, 12], pts[:, 14], pts[:, 16])
    knee_angle = (l_knee + r_knee) / 2

    # Hip angle: shoulder-hip-knee
    l_hip = limb_angle(pts[:, 5], pts[:, 11], pts[:, 13])
    r_hip = limb_angle(pts[:, 6], pts[:, 12], pts[:, 14])
    hip_angle = (l_hip + r_hip) / 2

    return {
        "body_angle": body_angle,
        "angular_vel": angular_vel,
        "com_x": com_x,
        "com_y": com_y,
        "lr_asymmetry": lr_asym,
        "knee_angle": knee_angle,
        "hip_angle": hip_angle,
    }


def segment_by_airtime(kp: np.ndarray, fps: float, physics: dict) -> list[dict]:
    """Segment tricks by detecting high-rotation aerial phases.

    A trick = a phase where the body rotates significantly.
    """
    T = kp.shape[1]
    ang_vel = np.abs(physics["angular_vel"])
    com_y = physics["com_y"]

    # Smooth angular velocity
    kernel = np.ones(max(3, int(fps * 0.1))) / max(3, int(fps * 0.1))
    smooth_vel = np.convolve(ang_vel, kernel, mode="same")

    # Dynamic threshold: mean + 1.0 * std
    threshold = np.mean(smooth_vel) + 1.0 * np.std(smooth_vel)

    # Find high-rotation regions
    active = smooth_vel > threshold

    # Merge nearby active regions (tricks are continuous)
    min_gap = int(0.3 * fps)
    segments = []
    in_seg = False
    start = 0

    for i in range(T):
        if active[i] and not in_seg:
            start = i
            in_seg = True
        elif not active[i] and in_seg:
            # Check if gap is short
            gap_end = min(i + min_gap, T)
            if np.any(active[i:gap_end]):
                continue
            in_seg = False
            segments.append((start, i))

    if in_seg:
        segments.append((start, T - 1))

    # Filter by duration and extend boundaries
    result = []
    for start, end in segments:
        pad = int(0.2 * fps)
        s = max(0, start - pad)
        e = min(T - 1, end + pad)
        duration = (e - s) / fps

        if 0.25 < duration < 5.0:
            # Compute physics for this segment
            seg_ang_vel = physics["angular_vel"][s:e+1]
            seg_com_y = physics["com_y"][s:e+1]
            seg_knee = physics["knee_angle"][s:e+1]
            seg_hip = physics["hip_angle"][s:e+1]
            seg_lr = physics["lr_asymmetry"][s:e+1]

            # Cumulative rotation (signed)
            cum_rotation = np.sum(seg_ang_vel)
            total_rotation_deg = np.degrees(np.abs(cum_rotation))

            # Flip count
            flip_count = total_rotation_deg / 360.0

            # Direction: positive = forward, negative = backward
            direction = "forward" if cum_rotation > 0 else "backward"

            # Body shape from knee/hip angles during peak rotation
            peak_frames = np.abs(seg_ang_vel) > np.percentile(np.abs(seg_ang_vel), 70)
            if np.any(peak_frames):
                avg_knee = np.mean(seg_knee[peak_frames])
                avg_hip = np.mean(seg_hip[peak_frames])
            else:
                avg_knee = np.mean(seg_knee)
                avg_hip = np.mean(seg_hip)

            if avg_knee < 90 and avg_hip < 90:
                body_shape = "tuck"
            elif avg_knee > 140 and avg_hip < 100:
                body_shape = "pike"
            elif avg_knee > 140 and avg_hip > 140:
                body_shape = "layout"
            else:
                body_shape = "open"

            # L/R asymmetry → sagittal (side) vs lateral (front/back)
            avg_lr = np.mean(seg_lr[peak_frames]) if np.any(peak_frames) else np.mean(seg_lr)
            is_side = avg_lr > 0.05  # High asymmetry = side rotation

            # Twist detection: rapid L/R oscillation of shoulders
            # When twisting, shoulders alternate left/right rapidly
            lr_diff = kp[0, s:e+1, 5] - kp[0, s:e+1, 6]  # left_shoulder_x - right_shoulder_x
            lr_vel = np.abs(np.diff(lr_diff))
            twist_intensity = np.mean(lr_vel) if len(lr_vel) > 0 else 0
            # Rough twist count: oscillation frequency
            # A full twist = ~2 zero-crossings of shoulder difference
            zero_crossings = np.sum(np.abs(np.diff(np.sign(np.diff(lr_diff)))) > 0)
            est_twist_count = zero_crossings / 4.0  # ~4 sign changes per full twist

            result.append({
                "start": s,
                "end": e,
                "start_s": s / fps,
                "end_s": e / fps,
                "duration_s": (e - s) / fps,
                "total_rotation_deg": total_rotation_deg,
                "flip_count": round(flip_count, 1),
                "direction": direction,
                "body_shape": body_shape,
                "avg_knee": avg_knee,
                "avg_hip": avg_hip,
                "twist_intensity": twist_intensity,
                "est_twist_count": round(est_twist_count, 1),
                "is_side": is_side,
                "lr_asymmetry": float(avg_lr),
            })

    return result


def classify_from_physics(seg: dict) -> list[tuple[str, str]]:
    """Match physics to known tricks. Returns list of (trick_name, reason)."""
    rot = seg["total_rotation_deg"]
    flips = seg["flip_count"]
    direction = seg["direction"]
    shape = seg["body_shape"]
    twists = seg["est_twist_count"]
    is_side = seg["is_side"]
    twist_int = seg["twist_intensity"]

    candidates = []

    # Low rotation = ground move or transition
    if rot < 90:
        candidates.append(("Transition/Setup", f"low rotation ({rot:.0f}°)"))
        return candidates

    # Very high twist intensity = twist-based trick
    high_twist = twist_int > 0.015 or twists >= 0.8

    # === Rotation-based matching ===

    # ~360° rotation
    if 250 < rot < 500:
        if high_twist and flips >= 0.7:
            if twists >= 1.5:
                candidates.append(("Double Full", f"{rot:.0f}° + {twists:.1f} twists"))
            else:
                candidates.append(("Back Full", f"{rot:.0f}° + {twists:.1f} twists"))
        elif is_side:
            candidates.append(("Side Flip", f"{rot:.0f}° side rotation"))
        elif direction == "backward":
            if shape == "tuck":
                candidates.append(("Back Flip", f"{rot:.0f}° backward tuck"))
            elif shape == "layout":
                candidates.append(("Back Layout", f"{rot:.0f}° backward layout"))
            else:
                candidates.append(("Back Flip", f"{rot:.0f}° backward"))
        else:
            candidates.append(("Front Flip", f"{rot:.0f}° forward"))

    # ~180° = half rotation tricks
    elif 120 < rot <= 250:
        if high_twist:
            candidates.append(("B-Twist", f"{rot:.0f}° + twist, half rotation"))
            candidates.append(("Raiz", f"{rot:.0f}° off-axis half"))
        elif is_side:
            candidates.append(("Aerial", f"{rot:.0f}° side"))
        else:
            if direction == "backward":
                candidates.append(("Macaco", f"{rot:.0f}° backward half"))
                candidates.append(("Krok", f"{rot:.0f}° backward half"))
            else:
                candidates.append(("Front Half", f"{rot:.0f}° forward half"))
                candidates.append(("Krok", f"{rot:.0f}° forward half"))

    # ~540° = 1.5 rotations
    elif 500 < rot < 680:
        if high_twist:
            candidates.append(("Raiz Pop Full", f"{rot:.0f}° + {twists:.1f} twists"))
            candidates.append(("Cork", f"{rot:.0f}° off-axis + twist"))
        else:
            candidates.append(("Castaway", f"{rot:.0f}° 1.5 rotation"))

    # ~720° = double rotation
    elif 680 < rot < 850:
        if high_twist:
            candidates.append(("Double Full", f"{rot:.0f}° + {twists:.1f} twists"))
            candidates.append(("Double Cork", f"{rot:.0f}° off-axis double"))
        else:
            candidates.append(("Double Back", f"{rot:.0f}° double rotation"))

    # >850° = triple or more
    elif rot >= 850:
        if high_twist:
            candidates.append(("Triple Full", f"{rot:.0f}° + {twists:.1f} twists"))
        else:
            candidates.append(("Triple Back", f"{rot:.0f}° triple rotation"))

    # 90-120° = quarter rotation / small movement
    elif 90 <= rot <= 120:
        candidates.append(("Wall Flip", f"{rot:.0f}° (small rotation, possible wall trick)"))
        candidates.append(("Roundoff", f"{rot:.0f}°"))

    if not candidates:
        candidates.append(("Unknown", f"{rot:.0f}° rotation"))

    return candidates


def main():
    parser = argparse.ArgumentParser(description="Physics-based run analysis")
    parser.add_argument("video", help="Path to video")
    args = parser.parse_args()

    print(f"\nPkVision — Physics-Based Run Analysis")
    print(f"{'=' * 60}")

    # Step 1: Extract keypoints
    print(f"\n[1] Extracting YOLO keypoints...")
    kp, fps = extract_keypoints(args.video)

    # Step 2: Compute physics
    print(f"\n[2] Computing physics features...")
    physics = compute_physics_per_frame(kp)

    # Step 3: Segment tricks
    print(f"\n[3] Segmenting trick zones...")
    segments = segment_by_airtime(kp, fps, physics)
    print(f"  Found {len(segments)} trick segments")

    # Step 4: Classify each segment
    print(f"\n[4] Physics-based trick identification")
    print(f"{'─' * 60}")

    for i, seg in enumerate(segments):
        candidates = classify_from_physics(seg)
        top_name = candidates[0][0] if candidates else "Unknown"
        top_reason = candidates[0][1] if candidates else ""

        print(f"\n  Trick #{i+1} @ {seg['start_s']:.1f}s - {seg['end_s']:.1f}s ({seg['duration_s']:.1f}s)")
        print(f"    Rotation:  {seg['total_rotation_deg']:.0f}° ({seg['flip_count']:.1f} flips)")
        print(f"    Direction: {seg['direction']}")
        print(f"    Shape:     {seg['body_shape']} (knee={seg['avg_knee']:.0f}° hip={seg['avg_hip']:.0f}°)")
        print(f"    Twist:     {seg['est_twist_count']:.1f} (intensity={seg['twist_intensity']:.4f})")
        print(f"    Side:      {'yes' if seg['is_side'] else 'no'} (L/R asym={seg['lr_asymmetry']:.3f})")
        print(f"    → Matches:")
        for name, reason in candidates:
            print(f"      • {name:30s} ({reason})")

    # Summary
    print(f"\n{'=' * 60}")
    print(f"DETECTED TRICKS:")
    print(f"{'=' * 60}")
    for i, seg in enumerate(segments):
        candidates = classify_from_physics(seg)
        top = candidates[0][0] if candidates else "?"
        rot = seg["total_rotation_deg"]
        print(f"  {i+1}. {top:35s} ({rot:.0f}°, {seg['direction']}, {seg['body_shape']}) @ {seg['start_s']:.1f}s")


if __name__ == "__main__":
    main()
