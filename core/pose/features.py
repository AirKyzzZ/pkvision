"""Camera-invariant feature extraction from YOLO keypoints.

Extracts features robust to camera position, athlete height, and body proportions:
- Joint angles (from 3-point geometry, invariant in 2D plane)
- Angular velocities (first derivative, captures movement dynamics)
- Relative positions (normalized to center of mass and torso length)
- Limb length ratios (body-shape invariant proportions)
- Body tilt (torso angle relative to vertical — detects inversions)
- Body tilt velocity (rotation speed)
- Vertical COM trajectory (jump height, aerial phase detection)
- Vertical COM velocity (ascending vs descending)
- Cumulative rotation (total degrees rotated — single vs double flip)
- Left/right symmetry (distinguishes front/back flips from side flips)
- Angular accelerations (explosive takeoff vs smooth rotation)

These features feed the TCN classifier (temporal) and DTW classifier (raw values).

Usage:
    from core.pose.features import extract_features
    features = extract_features(keypoints_seq, confidences_seq, timestamps_ms)
    array = features.to_array()                    # (n_frames, 75) for DTW/TCN
    tensor = features.to_temporal_array(target_frames=64)  # (75, 64) for TCN
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from core.pose.angles import (
    MIN_KEYPOINT_CONFIDENCE,
    get_joint_angles,
    get_joint_velocities,
)
from core.pose.constants import KEYPOINT_INDEX

# ── Feature Definitions ─────────────────────────────────────────────

# The 9 joint angles used in the feature vector (matches brief spec)
ANGLE_NAMES: list[str] = [
    "left_knee",
    "right_knee",
    "left_hip",
    "right_hip",
    "left_elbow",
    "right_elbow",
    "left_shoulder",
    "right_shoulder",
    "spine",
]

# Limb segments: (proximal_joint, distal_joint)
LIMB_DEFINITIONS: dict[str, tuple[str, str]] = {
    "left_upper_arm": ("left_shoulder", "left_elbow"),
    "right_upper_arm": ("right_shoulder", "right_elbow"),
    "left_forearm": ("left_elbow", "left_wrist"),
    "right_forearm": ("right_elbow", "right_wrist"),
    "left_thigh": ("left_hip", "left_knee"),
    "right_thigh": ("right_hip", "right_knee"),
    "left_shin": ("left_knee", "left_ankle"),
    "right_shin": ("right_knee", "right_ankle"),
}

LIMB_RATIO_NAMES: list[str] = list(LIMB_DEFINITIONS.keys())

# Torso reference points for scale normalization
TORSO_SHOULDER_JOINTS = ["left_shoulder", "right_shoulder"]
TORSO_HIP_JOINTS = ["left_hip", "right_hip"]

# Normalization constants for to_array(normalize=True)
_ANGLE_MAX = 180.0
_VELOCITY_CLIP = 1000.0  # deg/s — clips extreme values before normalizing
_ACCEL_CLIP = 50000.0  # deg/s² — clips extreme accelerations
_TILT_MAX = 180.0  # body tilt range [-180, 180]
_TILT_VEL_CLIP = 2000.0  # deg/s — body rotation speed clip
_COM_VEL_CLIP = 20.0  # torso-lengths/s — vertical COM velocity clip
_CUMROT_CLIP = 1080.0  # deg — max 3 full rotations

# Feature counts
N_ANGLES = 9
N_VELOCITIES = 9
N_POSITIONS = 34  # 17 × 2
N_RATIOS = 8
N_WORLD_SCALARS = 6  # body_tilt, body_tilt_vel, vert_com, vert_com_vel, cumrot, lr_sym
N_ACCELS = 9
FEATURES_PER_FRAME = N_ANGLES + N_VELOCITIES + N_POSITIONS + N_RATIOS + N_WORLD_SCALARS + N_ACCELS  # 75


# ── Data Structures ──────────────────────────────────────────────────


@dataclass
class FrameFeatures:
    """Camera-invariant features for a single frame.

    Feature vector layout (75 total):
        [0:9]     joint_angles (9) — degrees
        [9:18]    angular_velocities (9) — deg/s
        [18:52]   relative_positions (34) — 17 keypoints × 2, torso-normalized
        [52:60]   limb_ratios (8) — relative to torso length
        [60]      body_tilt — torso angle vs vertical, degrees [-180, 180]
        [61]      body_tilt_velocity — rotation speed, deg/s
        [62]      vertical_com — COM y displacement from start, torso-lengths
        [63]      vertical_com_velocity — vertical speed, torso-lengths/s
        [64]      cumulative_rotation — total degrees rotated since start
        [65]      left_right_symmetry — shoulder height asymmetry / torso
        [66:75]   angular_accelerations (9) — deg/s²
    """

    joint_angles: np.ndarray  # (9,)
    angular_velocities: np.ndarray  # (9,)
    relative_positions: np.ndarray  # (17, 2)
    limb_ratios: np.ndarray  # (8,)
    # World-frame features
    body_tilt: float = 0.0
    body_tilt_velocity: float = 0.0
    vertical_com: float = 0.0
    vertical_com_velocity: float = 0.0
    cumulative_rotation: float = 0.0
    left_right_symmetry: float = 0.0
    angular_accelerations: np.ndarray = field(default_factory=lambda: np.zeros(9, dtype=np.float32))


@dataclass
class FeatureSequence:
    """A temporal sequence of camera-invariant features for classification.

    Provides conversion to numpy arrays for DTW (raw) and MLP (normalized, fixed-length).
    """

    frames: list[FrameFeatures]
    angle_names: list[str] = field(default_factory=lambda: list(ANGLE_NAMES))
    limb_ratio_names: list[str] = field(default_factory=lambda: list(LIMB_RATIO_NAMES))

    @property
    def n_frames(self) -> int:
        return len(self.frames)

    @property
    def n_features_per_frame(self) -> int:
        """Total feature count per frame: 9+9+34+8+6+9 = 75."""
        return FEATURES_PER_FRAME

    def to_array(self, normalize: bool = False) -> np.ndarray:
        """Convert to (n_frames, n_features) array.

        Feature ordering per frame (75 total):
            [joint_angles(9) | angular_velocities(9) | relative_positions(34) |
             limb_ratios(8) | world_scalars(6) | angular_accelerations(9)]

        Args:
            normalize: If True, normalize features to approximately [-1, 1] range.
        """
        if not self.frames:
            return np.empty((0, 0), dtype=np.float32)

        rows = []
        for f in self.frames:
            world_scalars = np.array([
                f.body_tilt,
                f.body_tilt_velocity,
                f.vertical_com,
                f.vertical_com_velocity,
                f.cumulative_rotation,
                f.left_right_symmetry,
            ], dtype=np.float32)

            row = np.concatenate([
                f.joint_angles,
                f.angular_velocities,
                f.relative_positions.flatten(),
                f.limb_ratios,
                world_scalars,
                f.angular_accelerations,
            ])
            rows.append(row)

        arr = np.array(rows, dtype=np.float32)

        if normalize:
            n_ang = N_ANGLES
            # Angles: [0, 180] → [0, 1]
            arr[:, :n_ang] /= _ANGLE_MAX
            # Velocities: clip ±1000 → [-1, 1]
            arr[:, n_ang : 2 * n_ang] = np.clip(
                arr[:, n_ang : 2 * n_ang], -_VELOCITY_CLIP, _VELOCITY_CLIP
            ) / _VELOCITY_CLIP
            # Positions [18:52] and ratios [52:60] already scale-normalized

            # World scalars [60:66]
            ws_start = N_ANGLES + N_VELOCITIES + N_POSITIONS + N_RATIOS
            arr[:, ws_start + 0] /= _TILT_MAX  # body_tilt → [-1, 1]
            arr[:, ws_start + 1] = np.clip(arr[:, ws_start + 1], -_TILT_VEL_CLIP, _TILT_VEL_CLIP) / _TILT_VEL_CLIP
            arr[:, ws_start + 2] /= 5.0  # vertical_com: ±5 torso-lengths reasonable max
            arr[:, ws_start + 3] = np.clip(arr[:, ws_start + 3], -_COM_VEL_CLIP, _COM_VEL_CLIP) / _COM_VEL_CLIP
            arr[:, ws_start + 4] /= _CUMROT_CLIP  # cumulative_rotation
            # left_right_symmetry [ws_start + 5] already torso-normalized

            # Angular accelerations [66:75]
            accel_start = ws_start + N_WORLD_SCALARS
            arr[:, accel_start:accel_start + N_ACCELS] = np.clip(
                arr[:, accel_start:accel_start + N_ACCELS], -_ACCEL_CLIP, _ACCEL_CLIP
            ) / _ACCEL_CLIP

        return arr

    def to_temporal_array(self, target_frames: int = 64, normalize: bool = True) -> np.ndarray:
        """Convert to (n_features, n_frames) array for TCN input.

        Interpolates to target_frames, optionally normalizes, returns channels-first.
        """
        interpolated = self.interpolate(target_frames)
        arr = interpolated.to_array(normalize=normalize)  # (T, F)
        return arr.T  # (F, T) — channels first for Conv1d

    def interpolate(self, target_frames: int) -> FeatureSequence:
        """Interpolate to a fixed frame count via linear interpolation.

        NaN gaps are filled by interpolating between valid neighbors.
        """
        if self.n_frames == 0:
            return FeatureSequence(frames=[])

        if self.n_frames == target_frames:
            return self

        arr = self.to_array()  # (n_frames, n_features)
        n_feat = arr.shape[1]

        src_idx = np.arange(self.n_frames)
        dst_idx = np.linspace(0, self.n_frames - 1, target_frames)

        interpolated = np.zeros((target_frames, n_feat), dtype=np.float32)
        for j in range(n_feat):
            col = arr[:, j]
            valid = ~np.isnan(col)
            if valid.sum() >= 2:
                interpolated[:, j] = np.interp(dst_idx, src_idx[valid], col[valid])
            elif valid.sum() == 1:
                interpolated[:, j] = col[valid][0]
            else:
                interpolated[:, j] = np.nan

        return _array_to_sequence(interpolated, self.angle_names, self.limb_ratio_names)

    def to_flat_array(self, target_frames: int = 64, normalize: bool = True) -> np.ndarray:
        """Flatten to 1D array for MLP input.

        Interpolates to target_frames, optionally normalizes, then flattens.
        """
        interpolated = self.interpolate(target_frames)
        return interpolated.to_array(normalize=normalize).flatten()


# ── Internal Helpers ─────────────────────────────────────────────────


def _array_to_sequence(
    arr: np.ndarray,
    angle_names: list[str],
    limb_ratio_names: list[str],
) -> FeatureSequence:
    """Reconstruct a FeatureSequence from a (n_frames, n_features) array.

    Handles both legacy 60-feature arrays and new 75-feature arrays.
    """
    n_ang = len(angle_names)
    n_rat = len(limb_ratio_names)
    n_feat = arr.shape[1] if arr.ndim == 2 else 0
    has_world_features = n_feat >= FEATURES_PER_FRAME

    frames: list[FrameFeatures] = []
    for i in range(arr.shape[0]):
        row = arr[i]
        offset = 0

        angles = row[offset : offset + n_ang].copy()
        offset += n_ang

        velocities = row[offset : offset + n_ang].copy()
        offset += n_ang

        positions = row[offset : offset + N_POSITIONS].reshape(17, 2).copy()
        offset += N_POSITIONS

        ratios = row[offset : offset + n_rat].copy()
        offset += n_rat

        if has_world_features:
            body_tilt = float(row[offset])
            body_tilt_velocity = float(row[offset + 1])
            vertical_com = float(row[offset + 2])
            vertical_com_velocity = float(row[offset + 3])
            cumulative_rotation = float(row[offset + 4])
            left_right_symmetry = float(row[offset + 5])
            offset += N_WORLD_SCALARS
            angular_accelerations = row[offset : offset + N_ACCELS].copy()
        else:
            body_tilt = 0.0
            body_tilt_velocity = 0.0
            vertical_com = 0.0
            vertical_com_velocity = 0.0
            cumulative_rotation = 0.0
            left_right_symmetry = 0.0
            angular_accelerations = np.zeros(N_ACCELS, dtype=np.float32)

        frames.append(FrameFeatures(
            joint_angles=angles,
            angular_velocities=velocities,
            relative_positions=positions,
            limb_ratios=ratios,
            body_tilt=body_tilt,
            body_tilt_velocity=body_tilt_velocity,
            vertical_com=vertical_com,
            vertical_com_velocity=vertical_com_velocity,
            cumulative_rotation=cumulative_rotation,
            left_right_symmetry=left_right_symmetry,
            angular_accelerations=angular_accelerations,
        ))

    return FeatureSequence(
        frames=frames,
        angle_names=angle_names,
        limb_ratio_names=limb_ratio_names,
    )


def _compute_com(
    keypoints: np.ndarray,
    confidences: np.ndarray,
    min_conf: float,
) -> np.ndarray:
    """Compute center of mass from hip midpoint.

    Hip midpoint is the most stable reference in parkour movements.
    Falls back to mean of all confident keypoints if hips are occluded.
    """
    left_hip = KEYPOINT_INDEX["left_hip"]
    right_hip = KEYPOINT_INDEX["right_hip"]

    if confidences[left_hip] >= min_conf and confidences[right_hip] >= min_conf:
        return (keypoints[left_hip] + keypoints[right_hip]) / 2.0

    valid = confidences >= min_conf
    if valid.sum() > 0:
        return keypoints[valid].mean(axis=0)

    return keypoints.mean(axis=0)


def _compute_torso_length(
    keypoints: np.ndarray,
    confidences: np.ndarray,
    min_conf: float,
) -> float:
    """Compute torso length (shoulder midpoint → hip midpoint) for scale normalization."""
    for name in TORSO_SHOULDER_JOINTS + TORSO_HIP_JOINTS:
        if confidences[KEYPOINT_INDEX[name]] < min_conf:
            return float("nan")

    shoulder_mid = np.mean(
        [keypoints[KEYPOINT_INDEX[n]] for n in TORSO_SHOULDER_JOINTS], axis=0
    )
    hip_mid = np.mean(
        [keypoints[KEYPOINT_INDEX[n]] for n in TORSO_HIP_JOINTS], axis=0
    )

    length = float(np.linalg.norm(shoulder_mid - hip_mid))
    return length if length > 1e-6 else float("nan")


def _compute_relative_positions(
    keypoints: np.ndarray,
    confidences: np.ndarray,
    com: np.ndarray,
    torso_length: float,
    min_conf: float,
) -> np.ndarray:
    """Compute positions relative to COM, normalized by torso length.

    Returns (17, 2). Low-confidence keypoints are NaN.
    """
    relative = np.full((17, 2), float("nan"), dtype=np.float32)

    if math.isnan(torso_length):
        return relative

    for i in range(17):
        if confidences[i] >= min_conf:
            relative[i] = (keypoints[i] - com) / torso_length

    return relative


def _compute_limb_ratios(
    keypoints: np.ndarray,
    confidences: np.ndarray,
    torso_length: float,
    min_conf: float,
) -> np.ndarray:
    """Compute limb length ratios relative to torso length."""
    ratios = np.full(len(LIMB_RATIO_NAMES), float("nan"), dtype=np.float32)

    if math.isnan(torso_length):
        return ratios

    for i, name in enumerate(LIMB_RATIO_NAMES):
        joint_a, joint_b = LIMB_DEFINITIONS[name]
        idx_a = KEYPOINT_INDEX[joint_a]
        idx_b = KEYPOINT_INDEX[joint_b]

        if confidences[idx_a] >= min_conf and confidences[idx_b] >= min_conf:
            limb_len = float(np.linalg.norm(keypoints[idx_a] - keypoints[idx_b]))
            ratios[i] = limb_len / torso_length

    return ratios


# ── World-Frame Feature Helpers ──────────────────────────────────────


def _compute_body_tilt(
    keypoints: np.ndarray,
    confidences: np.ndarray,
    min_conf: float,
) -> float:
    """Compute angle of torso relative to vertical in degrees.

    0° = standing upright (shoulders above hips).
    +90° = leaning right, -90° = leaning left.
    ±180° = fully inverted.

    Uses atan2 for continuous [-180, 180] range.
    """
    for name in TORSO_SHOULDER_JOINTS + TORSO_HIP_JOINTS:
        if confidences[KEYPOINT_INDEX[name]] < min_conf:
            return float("nan")

    shoulder_mid = np.mean(
        [keypoints[KEYPOINT_INDEX[n]] for n in TORSO_SHOULDER_JOINTS], axis=0
    )
    hip_mid = np.mean(
        [keypoints[KEYPOINT_INDEX[n]] for n in TORSO_HIP_JOINTS], axis=0
    )

    # Vector from hip to shoulder
    torso_vec = shoulder_mid - hip_mid
    # In image coords y increases downward, so upright = (0, negative_y)
    # atan2(x, -y) gives 0° for upright, positive for clockwise rotation
    angle_rad = math.atan2(torso_vec[0], -torso_vec[1])
    return math.degrees(angle_rad)


def _compute_left_right_symmetry(
    keypoints: np.ndarray,
    confidences: np.ndarray,
    torso_length: float,
    min_conf: float,
) -> float:
    """Compute left/right shoulder height asymmetry normalized by torso length.

    ~0 for front/back flips (symmetric rotation).
    Positive = left shoulder higher (body tilted right).
    Negative = right shoulder higher (body tilted left).
    """
    l_shoulder = KEYPOINT_INDEX["left_shoulder"]
    r_shoulder = KEYPOINT_INDEX["right_shoulder"]

    if confidences[l_shoulder] < min_conf or confidences[r_shoulder] < min_conf:
        return float("nan")
    if math.isnan(torso_length) or torso_length < 1e-6:
        return float("nan")

    # In image coords, lower y = higher in frame
    y_diff = keypoints[r_shoulder][1] - keypoints[l_shoulder][1]
    return float(y_diff / torso_length)


# ── Public API ───────────────────────────────────────────────────────


def extract_frame_features(
    keypoints: np.ndarray,
    confidences: np.ndarray,
    angles: dict[str, float],
    velocities: dict[str, float] | None = None,
    min_confidence: float = MIN_KEYPOINT_CONFIDENCE,
) -> FrameFeatures:
    """Extract camera-invariant features from a single frame.

    World-frame temporal features (body_tilt_velocity, vertical_com, etc.)
    are set to defaults here and computed during sequence extraction.

    Args:
        keypoints: (17, 2) keypoint coordinates.
        confidences: (17,) confidence scores.
        angles: Pre-computed joint angles from get_joint_angles().
        velocities: Pre-computed angular velocities. None → zeros.
        min_confidence: Minimum keypoint confidence threshold.
    """
    angle_values = np.array(
        [angles.get(name, float("nan")) for name in ANGLE_NAMES],
        dtype=np.float32,
    )

    if velocities is not None:
        vel_values = np.array(
            [velocities.get(name, float("nan")) for name in ANGLE_NAMES],
            dtype=np.float32,
        )
    else:
        vel_values = np.zeros(len(ANGLE_NAMES), dtype=np.float32)

    com = _compute_com(keypoints, confidences, min_confidence)
    torso_length = _compute_torso_length(keypoints, confidences, min_confidence)
    relative_pos = _compute_relative_positions(
        keypoints, confidences, com, torso_length, min_confidence
    )
    limb_ratios = _compute_limb_ratios(
        keypoints, confidences, torso_length, min_confidence
    )
    body_tilt = _compute_body_tilt(keypoints, confidences, min_confidence)
    lr_symmetry = _compute_left_right_symmetry(
        keypoints, confidences, torso_length, min_confidence
    )

    return FrameFeatures(
        joint_angles=angle_values,
        angular_velocities=vel_values,
        relative_positions=relative_pos,
        limb_ratios=limb_ratios,
        body_tilt=body_tilt,
        left_right_symmetry=lr_symmetry,
    )


def extract_features(
    keypoints_sequence: list[np.ndarray],
    confidences_sequence: list[np.ndarray],
    timestamps_ms: list[float],
    min_confidence: float = MIN_KEYPOINT_CONFIDENCE,
) -> FeatureSequence:
    """Extract camera-invariant features from a keypoint sequence.

    This is the main entry point for the feature extraction pipeline.
    Computes per-frame features, then adds temporal features:
    body_tilt_velocity, vertical_com, vertical_com_velocity,
    cumulative_rotation, and angular_accelerations.

    Args:
        keypoints_sequence: List of (17, 2) arrays, one per frame.
        confidences_sequence: List of (17,) arrays, one per frame.
        timestamps_ms: Timestamp for each frame in milliseconds.
        min_confidence: Minimum keypoint confidence threshold.

    Returns:
        FeatureSequence ready for TCN (.to_temporal_array()) or DTW (.to_array()).
    """
    n_frames = len(keypoints_sequence)
    if n_frames == 0:
        return FeatureSequence(frames=[])

    # Compute all joint angles
    angles_list = [
        get_joint_angles(kp, conf, min_confidence)
        for kp, conf in zip(keypoints_sequence, confidences_sequence)
    ]

    # Compute angular velocities across time
    velocities_list = get_joint_velocities(angles_list, timestamps_ms)

    # Extract per-frame features (body_tilt and lr_symmetry computed here)
    frame_features = []
    for i in range(n_frames):
        features = extract_frame_features(
            keypoints=keypoints_sequence[i],
            confidences=confidences_sequence[i],
            angles=angles_list[i],
            velocities=velocities_list[i] if i < len(velocities_list) else None,
            min_confidence=min_confidence,
        )
        frame_features.append(features)

    # ── Compute temporal (sequence-level) features ──

    # Compute COM positions for vertical trajectory
    com_positions = []
    torso_lengths = []
    for i in range(n_frames):
        com = _compute_com(keypoints_sequence[i], confidences_sequence[i], min_confidence)
        tl = _compute_torso_length(keypoints_sequence[i], confidences_sequence[i], min_confidence)
        com_positions.append(com)
        torso_lengths.append(tl)

    # Find initial COM y and a stable torso length for normalization
    valid_torso = [t for t in torso_lengths if not math.isnan(t)]
    ref_torso = np.median(valid_torso) if valid_torso else 1.0
    initial_com_y = com_positions[0][1] if n_frames > 0 else 0.0

    cumulative_rot = 0.0

    for i in range(n_frames):
        f = frame_features[i]

        # Vertical COM: displacement from initial position, normalized by torso
        # In image coords, y increases downward — so going UP means decreasing y
        # We invert so positive = went up (jumped)
        if not math.isnan(ref_torso) and ref_torso > 1e-6:
            f.vertical_com = -(com_positions[i][1] - initial_com_y) / ref_torso
        else:
            f.vertical_com = 0.0

        if i == 0:
            f.body_tilt_velocity = 0.0
            f.vertical_com_velocity = 0.0
            f.cumulative_rotation = 0.0
            f.angular_accelerations = np.zeros(N_ACCELS, dtype=np.float32)
            continue

        dt_s = (timestamps_ms[i] - timestamps_ms[i - 1]) / 1000.0
        if dt_s <= 0:
            dt_s = 1.0 / 30.0  # fallback to 30fps

        # Body tilt velocity
        prev_tilt = frame_features[i - 1].body_tilt
        curr_tilt = f.body_tilt
        if not math.isnan(prev_tilt) and not math.isnan(curr_tilt):
            # Handle wraparound at ±180°
            delta_tilt = curr_tilt - prev_tilt
            delta_tilt = (delta_tilt + 180) % 360 - 180
            f.body_tilt_velocity = delta_tilt / dt_s
            cumulative_rot += delta_tilt
        else:
            f.body_tilt_velocity = 0.0

        f.cumulative_rotation = cumulative_rot

        # Vertical COM velocity
        prev_vcom = frame_features[i - 1].vertical_com
        f.vertical_com_velocity = (f.vertical_com - prev_vcom) / dt_s

        # Angular accelerations (2nd derivative of joint angles)
        if i >= 2:
            prev_vel = frame_features[i - 1].angular_velocities
            curr_vel = f.angular_velocities
            accel = np.where(
                np.isnan(curr_vel) | np.isnan(prev_vel),
                0.0,
                (curr_vel - prev_vel) / dt_s,
            )
            f.angular_accelerations = accel.astype(np.float32)
        else:
            f.angular_accelerations = np.zeros(N_ACCELS, dtype=np.float32)

    return FeatureSequence(frames=frame_features)


def extract_features_from_frames(
    frames: list,
) -> FeatureSequence:
    """Extract features from existing FrameResult or FrameAnalysis objects.

    Convenience wrapper for integrating with the existing detection pipeline.
    """
    keypoints = []
    confidences = []
    timestamps = []

    for frame in frames:
        keypoints.append(np.asarray(frame.keypoints, dtype=np.float32))
        confidences.append(np.asarray(frame.keypoint_confidences, dtype=np.float32))
        timestamps.append(frame.timestamp_ms)

    return extract_features(keypoints, confidences, timestamps)
