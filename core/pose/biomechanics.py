"""3D biomechanical feature extraction from SMPL body mesh parameters.

Converts SMPL output (from GVHMR) into interpretable biomechanical features:
  - Swing-twist decomposition: separates flip rotation from twist rotation
  - Body shape classification: tuck, pike, layout, open from joint angles
  - Entry type detection: standing, running, one-leg from ground contact
  - COM trajectory: jump height, airborne detection

This is the core of v2 — it replaces the 2D feature extraction in features.py
with unambiguous 3D biomechanical analysis.

Usage:
    from core.pose.biomechanics import extract_biomechanics, extract_trick_signature
    bio_frames = extract_biomechanics(smpl_frames)
    signature = extract_trick_signature(bio_frames, start=10, end=50)
"""

from __future__ import annotations

import math

import numpy as np
from scipy.spatial.transform import Rotation

from core.models import BiomechanicalFrame, SMPLFrame, TrickSignature3D


# ── SMPL Joint Indices ──────────────────────────────────────────────

# SMPL has 24 joints in a kinematic tree
SMPL_JOINT = {
    "pelvis": 0, "left_hip": 1, "right_hip": 2, "spine1": 3,
    "left_knee": 4, "right_knee": 5, "spine2": 6,
    "left_ankle": 7, "right_ankle": 8, "spine3": 9,
    "left_foot": 10, "right_foot": 11, "neck": 12,
    "left_collar": 13, "right_collar": 14, "head": 15,
    "left_shoulder": 16, "right_shoulder": 17,
    "left_elbow": 18, "right_elbow": 19,
    "left_wrist": 20, "right_wrist": 21,
    "left_hand": 22, "right_hand": 23,
}

# Ground contact threshold (meters above ground plane)
GROUND_CONTACT_THRESHOLD = 0.10  # 10cm
AIRBORNE_MIN_HEIGHT = 0.15  # Must be at least 15cm above takeoff height


# ── Swing-Twist Decomposition ──────────────────────────────────────

def swing_twist_decompose(rotation: Rotation, twist_axis: np.ndarray) -> tuple[float, float]:
    """Decompose a rotation into swing (flip) and twist components.

    Given a rotation and a twist axis (body longitudinal axis),
    returns the angle of rotation around the twist axis (twist)
    and the angle of rotation perpendicular to it (swing/flip).

    Args:
        rotation: scipy Rotation object
        twist_axis: (3,) unit vector defining the twist axis

    Returns:
        (swing_angle_deg, twist_angle_deg) — both signed
    """
    twist_axis = twist_axis / np.linalg.norm(twist_axis)

    # Convert to quaternion [x, y, z, w]
    q = rotation.as_quat()  # scipy uses [x, y, z, w]
    qx, qy, qz, qw = q

    # Project quaternion onto twist axis
    # Twist quaternion: projection of the rotation onto the twist axis
    projection = qx * twist_axis[0] + qy * twist_axis[1] + qz * twist_axis[2]

    twist_quat = np.array([
        twist_axis[0] * projection,
        twist_axis[1] * projection,
        twist_axis[2] * projection,
        qw,
    ])

    # Normalize twist quaternion
    twist_norm = np.linalg.norm(twist_quat)
    if twist_norm < 1e-10:
        return 0.0, 0.0
    twist_quat /= twist_norm

    # Swing = rotation * inverse(twist)
    twist_rot = Rotation.from_quat(twist_quat)
    swing_rot = rotation * twist_rot.inv()

    # Extract angles
    twist_angle = twist_rot.magnitude() * np.sign(projection) * 180.0 / math.pi
    swing_angle = swing_rot.magnitude() * 180.0 / math.pi

    # Determine swing sign from the swing axis direction
    swing_axis = swing_rot.as_rotvec()
    swing_norm = np.linalg.norm(swing_axis)
    if swing_norm > 1e-6:
        swing_axis /= swing_norm
        # Sign based on whether swing goes forward or backward
        # In SMPL: x = right, y = up, z = forward
        # Lateral axis (flip) is roughly x-axis
        # A forward flip has swing axis pointing to the right (+x)
        # A backward flip has swing axis pointing to the left (-x)
        swing_angle *= np.sign(swing_axis[0]) if abs(swing_axis[0]) > 0.3 else 1.0

    return swing_angle, twist_angle


def compute_frame_to_frame_rotation(
    orient_prev: np.ndarray,
    orient_curr: np.ndarray,
) -> tuple[float, float]:
    """Compute the incremental flip and twist between two frames.

    Args:
        orient_prev: (3,) axis-angle global orientation of previous frame
        orient_curr: (3,) axis-angle global orientation of current frame

    Returns:
        (delta_flip_deg, delta_twist_deg)
    """
    R_prev = Rotation.from_rotvec(orient_prev)
    R_curr = Rotation.from_rotvec(orient_curr)

    # Relative rotation from prev to curr
    R_delta = R_curr * R_prev.inv()

    # In SMPL, the longitudinal axis (head-to-toe) is the y-axis
    # BUT this changes as the body rotates! We need the BODY's longitudinal axis.
    # The body's y-axis in world frame = R_prev applied to [0, 1, 0]
    body_y_in_world = R_prev.apply([0.0, 1.0, 0.0])

    return swing_twist_decompose(R_delta, body_y_in_world)


# ── Joint Angle Extraction ──────────────────────────────────────────

def _axis_angle_to_degrees(axis_angle: np.ndarray) -> float:
    """Convert a single joint's axis-angle rotation to scalar angle in degrees."""
    angle_rad = np.linalg.norm(axis_angle)
    return float(np.degrees(angle_rad))


def extract_joint_angles(body_pose: np.ndarray) -> dict[str, float]:
    """Extract key joint angles from SMPL body_pose parameters.

    SMPL body_pose is (23, 3) axis-angle rotations for each joint
    relative to its parent in the kinematic tree.

    Returns dict of human-readable joint angles in degrees.
    """
    # body_pose indices (0-indexed, joint 0 = pelvis is in global_orient)
    # So body_pose[0] = left_hip, body_pose[1] = right_hip, etc.
    left_hip = _axis_angle_to_degrees(body_pose[0])
    right_hip = _axis_angle_to_degrees(body_pose[1])
    left_knee = _axis_angle_to_degrees(body_pose[3])
    right_knee = _axis_angle_to_degrees(body_pose[4])
    spine1 = _axis_angle_to_degrees(body_pose[2])
    spine2 = _axis_angle_to_degrees(body_pose[5])
    spine3 = _axis_angle_to_degrees(body_pose[8])
    left_shoulder = _axis_angle_to_degrees(body_pose[15])
    right_shoulder = _axis_angle_to_degrees(body_pose[16])
    left_elbow = _axis_angle_to_degrees(body_pose[17])
    right_elbow = _axis_angle_to_degrees(body_pose[18])

    return {
        "knee": (left_knee + right_knee) / 2.0,
        "hip": (left_hip + right_hip) / 2.0,
        "spine": (spine1 + spine2 + spine3) / 3.0,
        "shoulder": (left_shoulder + right_shoulder) / 2.0,
        "elbow": (left_elbow + right_elbow) / 2.0,
    }


# ── Body Shape Classification ──────────────────────────────────────

def classify_body_shape(knee_angle: float, hip_angle: float, shoulder_angle: float) -> dict[str, float]:
    """Classify body shape from joint angles. Returns soft scores.

    Tuck:   knees bent (<90°), hips bent (<90°) — tight ball
    Pike:   knees straight (>140°), hips bent (<90°) — folded at hips
    Layout: knees straight (>140°), hips straight (>140°) — extended
    Open:   layout + arms spread (shoulder >100°)
    """
    scores = {"tuck": 0.0, "pike": 0.0, "layout": 0.0, "open": 0.0}

    # Tuck: both bent
    if knee_angle < 100 and hip_angle < 100:
        scores["tuck"] = 1.0 - (knee_angle + hip_angle) / 200.0
    elif knee_angle < 120 and hip_angle < 120:
        scores["tuck"] = 0.3

    # Pike: straight legs, bent hips
    if knee_angle > 130 and hip_angle < 100:
        scores["pike"] = min(1.0, (knee_angle - 130) / 50.0 * (100 - hip_angle) / 100.0)

    # Layout: both extended
    if knee_angle > 140 and hip_angle > 140:
        base = min(1.0, (knee_angle - 140) / 40.0 * (hip_angle - 140) / 40.0)
        if shoulder_angle > 100:
            scores["open"] = base * min(1.0, (shoulder_angle - 100) / 80.0)
            scores["layout"] = base * (1.0 - scores["open"])
        else:
            scores["layout"] = base

    # Normalize
    total = sum(scores.values())
    if total > 0:
        scores = {k: v / total for k, v in scores.items()}
    else:
        scores["layout"] = 1.0  # Default if nothing matches

    return scores


def dominant_shape(scores: dict[str, float]) -> str:
    """Return the shape with the highest score."""
    return max(scores, key=scores.get)


# ── Entry Type Detection ───────────────────────────────────────────

def detect_entry_type(
    com_velocities: list[np.ndarray],
    is_airborne: list[bool],
    ankle_heights: list[float],
) -> str:
    """Detect how the athlete entered the trick.

    Args:
        com_velocities: COM velocity vectors for frames before and at takeoff
        is_airborne: airborne flags for same frames
        ankle_heights: left/right average ankle height

    Returns:
        "standing", "running", "one_leg", "wall", or "edge"
    """
    # Find takeoff frame (first airborne frame)
    takeoff_idx = 0
    for i, airborne in enumerate(is_airborne):
        if airborne:
            takeoff_idx = i
            break

    if takeoff_idx == 0 or not com_velocities:
        return "standing"

    # Horizontal velocity at takeoff
    v_takeoff = com_velocities[min(takeoff_idx, len(com_velocities) - 1)]
    horizontal_speed = math.sqrt(v_takeoff[0] ** 2 + v_takeoff[2] ** 2)

    # Running if significant horizontal speed
    if horizontal_speed > 1.5:  # m/s
        return "running"

    # Check for wall push (horizontal velocity reversal)
    if takeoff_idx >= 2:
        v_before = com_velocities[takeoff_idx - 2]
        h_before = math.sqrt(v_before[0] ** 2 + v_before[2] ** 2)
        # Approaching wall then pushing away
        if h_before > 0.5 and horizontal_speed > 0.3:
            dot = np.dot(v_before[:3], v_takeoff[:3])
            if dot < 0:  # Velocity reversed
                return "wall"

    return "standing"


# ── Main Extraction Functions ──────────────────────────────────────

def extract_biomechanics(
    smpl_frames: list[SMPLFrame],
    fps: float = 30.0,
    ground_height: float | None = None,
) -> list[BiomechanicalFrame]:
    """Extract full biomechanical features from a sequence of SMPL frames.

    This is the core function that converts GVHMR output into interpretable
    biomechanical features for trick detection.

    Args:
        smpl_frames: List of SMPLFrame from GVHMR
        fps: Video frame rate (for velocity computation)
        ground_height: Y-coordinate of ground. If None, auto-detected from first frame.

    Returns:
        List of BiomechanicalFrame with rotation, shape, and trajectory data.
    """
    if not smpl_frames:
        return []

    dt = 1.0 / fps
    n = len(smpl_frames)

    # Auto-detect ground height from the lowest ankle in the first few frames
    if ground_height is None:
        first_heights = []
        for f in smpl_frames[:min(10, n)]:
            if f.joints_3d is not None:
                left_ankle_y = f.joints_3d[SMPL_JOINT["left_ankle"]][1]
                right_ankle_y = f.joints_3d[SMPL_JOINT["right_ankle"]][1]
                first_heights.append(min(left_ankle_y, right_ankle_y))
            else:
                first_heights.append(f.transl[1])
        ground_height = min(first_heights) if first_heights else 0.0

    # Accumulate rotations
    cumulative_flip = 0.0
    cumulative_twist = 0.0

    bio_frames: list[BiomechanicalFrame] = []

    for i in range(n):
        sf = smpl_frames[i]

        # Joint angles
        angles = extract_joint_angles(sf.body_pose)

        # Body shape
        shape_scores = classify_body_shape(
            angles["knee"], angles["hip"], angles["shoulder"]
        )

        # COM position and velocity
        com_pos = np.asarray(sf.transl, dtype=np.float64)
        com_height = float(com_pos[1] - ground_height)

        if i > 0:
            prev_pos = np.asarray(smpl_frames[i - 1].transl, dtype=np.float64)
            com_vel = (com_pos - prev_pos) / dt
        else:
            com_vel = np.zeros(3)

        # Airborne detection
        is_airborne = False
        if sf.joints_3d is not None:
            left_ankle_y = sf.joints_3d[SMPL_JOINT["left_ankle"]][1] - ground_height
            right_ankle_y = sf.joints_3d[SMPL_JOINT["right_ankle"]][1] - ground_height
            min_ankle = min(left_ankle_y, right_ankle_y)
            is_airborne = min_ankle > AIRBORNE_MIN_HEIGHT
        else:
            is_airborne = com_height > AIRBORNE_MIN_HEIGHT * 2

        # Rotation: swing-twist decomposition
        orient = np.asarray(sf.global_orient, dtype=np.float64)
        rotation_axis = np.zeros(3)

        if i > 0:
            prev_orient = np.asarray(smpl_frames[i - 1].global_orient, dtype=np.float64)
            delta_flip, delta_twist = compute_frame_to_frame_rotation(prev_orient, orient)
            cumulative_flip += delta_flip
            cumulative_twist += delta_twist

            # Current rotation axis
            R_delta = Rotation.from_rotvec(orient) * Rotation.from_rotvec(prev_orient).inv()
            rotvec = R_delta.as_rotvec()
            norm = np.linalg.norm(rotvec)
            if norm > 1e-6:
                rotation_axis = rotvec / norm

        bio_frames.append(BiomechanicalFrame(
            frame_idx=sf.frame_idx,
            timestamp_ms=sf.timestamp_ms,
            rotation_axis=rotation_axis,
            accumulated_flip_deg=cumulative_flip,
            accumulated_twist_deg=cumulative_twist,
            knee_angle=angles["knee"],
            hip_angle=angles["hip"],
            spine_angle=angles["spine"],
            shoulder_angle=angles["shoulder"],
            elbow_angle=angles["elbow"],
            com_position=com_pos,
            com_velocity=com_vel,
            com_height_m=com_height,
            is_airborne=is_airborne,
            body_shape_scores=shape_scores,
        ))

    return bio_frames


def extract_trick_signature(
    bio_frames: list[BiomechanicalFrame],
    start: int = 0,
    end: int | None = None,
) -> TrickSignature3D:
    """Extract a TrickSignature3D from a segment of biomechanical frames.

    This aggregates per-frame data into the trick-level signature used
    for zero-shot matching against TrickDefinitions.

    Args:
        bio_frames: Full sequence of BiomechanicalFrame
        start: Start frame index
        end: End frame index (None = last frame)

    Returns:
        TrickSignature3D ready for matching
    """
    if end is None:
        end = len(bio_frames) - 1
    if end <= start or not bio_frames:
        return TrickSignature3D(
            primary_rotation_axis=np.array([1.0, 0.0, 0.0]),
            total_flip_deg=0.0, total_twist_deg=0.0,
            rotation_direction="forward", body_shape="layout",
        )

    segment = bio_frames[start:end + 1]

    # Find aerial phase (airborne frames)
    aerial = [f for f in segment if f.is_airborne]
    if not aerial:
        aerial = segment  # Use all if no airborne detected

    # Total rotation
    total_flip = segment[-1].accumulated_flip_deg - segment[0].accumulated_flip_deg
    total_twist = segment[-1].accumulated_twist_deg - segment[0].accumulated_twist_deg

    # Primary rotation axis (averaged over aerial phase)
    axes = [f.rotation_axis for f in aerial if np.linalg.norm(f.rotation_axis) > 0.1]
    if axes:
        avg_axis = np.mean(axes, axis=0)
        norm = np.linalg.norm(avg_axis)
        primary_axis = avg_axis / norm if norm > 1e-6 else np.array([1.0, 0.0, 0.0])
    else:
        primary_axis = np.array([1.0, 0.0, 0.0])

    # Direction: sign of flip determines forward vs backward
    rotation_direction = "backward" if total_flip < 0 else "forward"

    # Body shape: dominant shape during aerial phase
    all_knee = [f.knee_angle for f in aerial]
    all_hip = [f.hip_angle for f in aerial]
    all_shoulder = [f.shoulder_angle for f in aerial]
    avg_knee = np.mean(all_knee) if all_knee else 180.0
    avg_hip = np.mean(all_hip) if all_hip else 180.0
    avg_shoulder = np.mean(all_shoulder) if all_shoulder else 0.0
    shape_scores = classify_body_shape(avg_knee, avg_hip, avg_shoulder)
    body_shape = dominant_shape(shape_scores)

    # Peak height
    heights = [f.com_height_m for f in segment]
    peak_height = max(heights) if heights else 0.0

    # Duration
    duration = (segment[-1].timestamp_ms - segment[0].timestamp_ms) / 1000.0

    # Entry type
    entry_frames = segment[:min(10, len(segment))]
    entry_type = detect_entry_type(
        com_velocities=[f.com_velocity for f in entry_frames if f.com_velocity is not None],
        is_airborne=[f.is_airborne for f in entry_frames],
        ankle_heights=[f.com_height_m for f in entry_frames],
    )

    return TrickSignature3D(
        primary_rotation_axis=primary_axis,
        total_flip_deg=total_flip,
        total_twist_deg=total_twist,
        rotation_direction=rotation_direction,
        body_shape=body_shape,
        avg_knee_angle=float(avg_knee),
        avg_hip_angle=float(avg_hip),
        entry_type=entry_type,
        peak_height_m=peak_height,
        duration_s=duration,
        start_frame=segment[0].frame_idx,
        end_frame=segment[-1].frame_idx,
        start_time_ms=segment[0].timestamp_ms,
        end_time_ms=segment[-1].timestamp_ms,
    )
