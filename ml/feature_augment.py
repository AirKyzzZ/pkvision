"""Feature-level augmentation for the DTW + MLP ensemble pipeline.

Generates synthetic FeatureSequence variants from a few real reference clips
by applying physically-plausible perturbations in feature space:

- Time stretching (±30% speed variation)
- Joint angle perturbation (tighter/looser tuck, arm positions)
- Body proportion scaling (simulate different athlete morphologies)
- Noise injection (simulate imperfect YOLO detection)
- Horizontal mirroring (left/right swap)
- Keypoint dropout (simulate occlusion)

From 3-4 real clips per trick → 500-1000 synthetic FeatureSequence variants.

Usage:
    from ml.feature_augment import augment_sequence, AugmentConfig

    config = AugmentConfig(angle_noise_std=8.0, time_stretch_range=(0.6, 1.4))
    variants = augment_sequence(reference_features, n_variants=500, config=config)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.pose.features import (
    ANGLE_NAMES,
    FEATURES_PER_FRAME,
    LIMB_RATIO_NAMES,
    N_ACCELS,
    N_ANGLES,
    N_POSITIONS,
    N_RATIOS,
    N_VELOCITIES,
    N_WORLD_SCALARS,
    FeatureSequence,
    _array_to_sequence,
)

# ── Column Layout ────────────────────────────────────────────────────
# The (n_frames, 75) array from FeatureSequence.to_array() has:
#   [0:9]     joint_angles         (9 values)
#   [9:18]    angular_velocities   (9 values)
#   [18:52]   relative_positions   (17 keypoints × 2 = 34 values)
#   [52:60]   limb_ratios          (8 values)
#   [60:66]   world_scalars        (6 values: body_tilt, body_tilt_vel,
#                                   vert_com, vert_com_vel, cumrot, lr_sym)
#   [66:75]   angular_accelerations (9 values)

_N_ANGLES = N_ANGLES  # 9
_ANGLES_SLICE = slice(0, _N_ANGLES)
_VELOCITIES_SLICE = slice(N_ANGLES, N_ANGLES + N_VELOCITIES)
_POSITIONS_START = N_ANGLES + N_VELOCITIES  # 18
_POSITIONS_END = _POSITIONS_START + N_POSITIONS  # 52
_POSITIONS_SLICE = slice(_POSITIONS_START, _POSITIONS_END)
_N_RATIOS = N_RATIOS  # 8
_RATIOS_SLICE = slice(_POSITIONS_END, _POSITIONS_END + _N_RATIOS)

_WORLD_START = _POSITIONS_END + _N_RATIOS  # 60
_WORLD_SLICE = slice(_WORLD_START, _WORLD_START + N_WORLD_SCALARS)
# Individual world scalar column indices
_BODY_TILT_COL = _WORLD_START + 0       # 60
_BODY_TILT_VEL_COL = _WORLD_START + 1   # 61
_VERT_COM_COL = _WORLD_START + 2        # 62
_VERT_COM_VEL_COL = _WORLD_START + 3    # 63
_CUMROT_COL = _WORLD_START + 4          # 64
_LR_SYM_COL = _WORLD_START + 5         # 65

_ACCELS_START = _WORLD_START + N_WORLD_SCALARS  # 66
_ACCELS_SLICE = slice(_ACCELS_START, _ACCELS_START + N_ACCELS)

# Left/right swap pairs for mirroring (indices within ANGLE_NAMES / LIMB_RATIO_NAMES)
_ANGLE_MIRROR_PAIRS = [(0, 1), (2, 3), (4, 5), (6, 7)]  # left↔right knee, hip, elbow, shoulder
_RATIO_MIRROR_PAIRS = [(0, 1), (2, 3), (4, 5), (6, 7)]  # left↔right upper_arm, forearm, thigh, shin

# COCO left/right keypoint pairs for position mirroring
_COCO_MIRROR_PAIRS = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]


# ── Configuration ────────────────────────────────────────────────────


@dataclass
class AugmentConfig:
    """Configuration for feature-level augmentation strengths.

    All probabilities control how often each augmentation is applied.
    Set a prob to 0.0 to disable that augmentation entirely.
    """

    # Time stretch: resample temporal dimension
    time_stretch_range: tuple[float, float] = (0.7, 1.3)
    time_stretch_prob: float = 0.8

    # Angle perturbation: smooth correlated noise on joint angles
    angle_noise_std: float = 5.0  # degrees
    angle_noise_smoothing: int = 5  # moving-average kernel width (frames)
    angle_noise_prob: float = 0.8

    # Proportion scaling: scale limb ratios (body morphology)
    proportion_scale_range: tuple[float, float] = (0.85, 1.15)
    proportion_scale_prob: float = 0.6

    # Position noise: Gaussian noise on relative positions
    position_noise_std: float = 0.02  # relative to torso length
    position_noise_prob: float = 0.8

    # Mirror: swap left/right features
    mirror_prob: float = 0.5

    # Dropout: set random features to NaN (simulate occlusion)
    dropout_prob: float = 0.3
    dropout_rate: float = 0.05  # fraction of position features per frame

    # Frame rate assumed for velocity recomputation after angle perturbation
    assumed_fps: float = 30.0


# ── Individual Augmentations ─────────────────────────────────────────


def _time_stretch(
    arr: np.ndarray,
    factor: float,
    fps: float,
) -> np.ndarray:
    """Resample temporal dimension and recompute velocities.

    factor > 1.0 = faster (fewer frames), factor < 1.0 = slower (more frames).
    """
    n_frames, n_feat = arr.shape
    new_n = max(4, int(n_frames / factor))

    src_idx = np.arange(n_frames)
    dst_idx = np.linspace(0, n_frames - 1, new_n)

    stretched = np.zeros((new_n, n_feat), dtype=np.float32)
    for j in range(n_feat):
        col = arr[:, j]
        valid = ~np.isnan(col)
        if valid.sum() >= 2:
            stretched[:, j] = np.interp(dst_idx, src_idx[valid], col[valid])
        elif valid.sum() == 1:
            stretched[:, j] = col[valid][0]
        else:
            stretched[:, j] = np.nan

    # Recompute velocities from stretched angles
    _recompute_velocities(stretched, fps)

    return stretched


def _perturb_angles(
    arr: np.ndarray,
    std: float,
    smoothing: int,
    rng: np.random.Generator,
    fps: float,
) -> np.ndarray:
    """Add temporally-smooth noise to joint angles and body_tilt, then recompute velocities.

    Generates iid noise per joint per frame, then applies a moving-average filter
    for temporal coherence (no sudden jumps between frames).

    Also adds small noise to body_tilt (same std) and a small vertical_com offset
    to simulate different jump heights.
    """
    result = arr.copy()
    n_frames = arr.shape[0]
    n_feat = arr.shape[1]

    # Generate raw noise for angles
    raw_noise = rng.normal(0, std, size=(n_frames, _N_ANGLES)).astype(np.float32)

    # Smooth temporally with a moving-average kernel
    if smoothing > 1 and n_frames > smoothing:
        kernel = np.ones(smoothing, dtype=np.float32) / smoothing
        for j in range(_N_ANGLES):
            raw_noise[:, j] = np.convolve(raw_noise[:, j], kernel, mode="same")

    # Apply to angles, clamp to [0, 180]
    angles = result[:, _ANGLES_SLICE]
    valid = ~np.isnan(angles)
    angles[valid] = np.clip(angles[valid] + raw_noise[valid], 0.0, 180.0)
    result[:, _ANGLES_SLICE] = angles

    if n_feat >= FEATURES_PER_FRAME:
        # Perturb body_tilt with smoothed noise (same std as joint angles)
        tilt_noise = rng.normal(0, std, size=n_frames).astype(np.float32)
        if smoothing > 1 and n_frames > smoothing:
            tilt_noise = np.convolve(tilt_noise, kernel, mode="same")
        tilt = result[:, _BODY_TILT_COL]
        tilt_valid = ~np.isnan(tilt)
        tilt[tilt_valid] += tilt_noise[tilt_valid]
        # Wrap to [-180, 180]
        tilt[tilt_valid] = (tilt[tilt_valid] + 180.0) % 360.0 - 180.0
        result[:, _BODY_TILT_COL] = tilt

        # Perturb vertical_com with a small constant offset (simulate jump height variation)
        vcom_offset = np.float32(rng.normal(0, 0.1))  # ±0.1 torso-lengths
        vcom = result[:, _VERT_COM_COL]
        vcom_valid = ~np.isnan(vcom)
        vcom[vcom_valid] += vcom_offset
        result[:, _VERT_COM_COL] = vcom

    # Recompute velocities from perturbed angles / body_tilt / vertical_com
    _recompute_velocities(result, fps)

    return result


def _scale_proportions(
    arr: np.ndarray,
    scale_range: tuple[float, float],
    rng: np.random.Generator,
) -> np.ndarray:
    """Scale limb ratios by random per-limb factors (consistent across all frames).

    Bilateral limbs (left/right) get correlated scaling to maintain realism.
    """
    result = arr.copy()

    # Generate one scale factor per limb pair (4 pairs)
    pair_scales = rng.uniform(scale_range[0], scale_range[1], size=4).astype(np.float32)
    # Add small per-side variation (left vs right not perfectly identical)
    side_jitter = rng.uniform(-0.02, 0.02, size=8).astype(np.float32)

    scales = np.ones(_N_RATIOS, dtype=np.float32)
    for i, (left_idx, right_idx) in enumerate(_RATIO_MIRROR_PAIRS):
        scales[left_idx] = pair_scales[i] + side_jitter[left_idx]
        scales[right_idx] = pair_scales[i] + side_jitter[right_idx]

    # Apply same scales to every frame
    ratios = result[:, _RATIOS_SLICE]
    valid = ~np.isnan(ratios)
    ratios[valid] = (ratios * scales)[valid]
    result[:, _RATIOS_SLICE] = ratios

    return result


def _inject_noise(
    arr: np.ndarray,
    std: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Add Gaussian noise to relative positions (simulates detection imprecision)."""
    result = arr.copy()
    positions = result[:, _POSITIONS_SLICE]
    noise = rng.normal(0, std, size=positions.shape).astype(np.float32)
    valid = ~np.isnan(positions)
    positions[valid] += noise[valid]
    result[:, _POSITIONS_SLICE] = positions
    return result


def _mirror(arr: np.ndarray) -> np.ndarray:
    """Swap left/right features: angles, velocities, positions, limb ratios,
    body_tilt, left_right_symmetry, and angular accelerations."""
    result = arr.copy()
    n_feat = arr.shape[1]

    # Swap angle pairs
    for left, right in _ANGLE_MIRROR_PAIRS:
        result[:, left], result[:, right] = arr[:, right].copy(), arr[:, left].copy()

    # Swap velocity pairs (same indices, offset by _N_ANGLES)
    for left, right in _ANGLE_MIRROR_PAIRS:
        vl, vr = left + _N_ANGLES, right + _N_ANGLES
        result[:, vl], result[:, vr] = arr[:, vr].copy(), arr[:, vl].copy()

    # Swap position pairs and flip x-coordinate
    positions = result[:, _POSITIONS_SLICE].reshape(-1, 17, 2).copy()
    # Flip x for all keypoints (positions are relative to COM, so negate x)
    positions[:, :, 0] *= -1.0
    # Swap left/right keypoint pairs
    for left_kp, right_kp in _COCO_MIRROR_PAIRS:
        positions[:, left_kp], positions[:, right_kp] = (
            positions[:, right_kp].copy(),
            positions[:, left_kp].copy(),
        )
    result[:, _POSITIONS_SLICE] = positions.reshape(-1, 34)

    # Swap limb ratio pairs
    for left, right in _RATIO_MIRROR_PAIRS:
        rl, rr = left + _POSITIONS_END, right + _POSITIONS_END
        result[:, rl], result[:, rr] = arr[:, rr].copy(), arr[:, rl].copy()

    if n_feat >= FEATURES_PER_FRAME:
        # Negate body_tilt (left-right flip)
        result[:, _BODY_TILT_COL] = -arr[:, _BODY_TILT_COL]
        # body_tilt_velocity also negates (derivative of negated tilt)
        result[:, _BODY_TILT_VEL_COL] = -arr[:, _BODY_TILT_VEL_COL]

        # Negate left_right_symmetry
        result[:, _LR_SYM_COL] = -arr[:, _LR_SYM_COL]

        # Swap angular acceleration pairs (same mirror pairs as angles)
        for left, right in _ANGLE_MIRROR_PAIRS:
            al = _ACCELS_START + left
            ar = _ACCELS_START + right
            result[:, al], result[:, ar] = arr[:, ar].copy(), arr[:, al].copy()

    return result


def _dropout(
    arr: np.ndarray,
    rate: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Set random position features to NaN (simulates keypoint occlusion).

    Only affects relative positions, not angles or ratios.
    """
    result = arr.copy()
    positions = result[:, _POSITIONS_SLICE]
    # Dropout entire keypoints (both x and y together)
    n_frames = positions.shape[0]
    drop_mask = rng.random(size=(n_frames, 17)) < rate
    # Expand to (n_frames, 17, 2) then flatten
    drop_mask_2d = np.repeat(drop_mask, 2, axis=1)
    positions[drop_mask_2d] = np.nan
    result[:, _POSITIONS_SLICE] = positions
    return result


def _recompute_velocities(arr: np.ndarray, fps: float) -> None:
    """Recompute derived velocities/accelerations from primary features in-place.

    Recomputes:
    - angular velocities (cols 9:18) from joint angles (cols 0:9)
    - body_tilt_velocity (col 61) from body_tilt (col 60)
    - vertical_com_velocity (col 63) from vertical_com (col 62)
    - angular_accelerations (cols 66:75) from angular velocities (cols 9:18)

    Assumes uniform frame spacing at the given FPS.
    """
    dt = 1.0 / fps  # seconds per frame
    angles = arr[:, _ANGLES_SLICE]
    n_frames = arr.shape[0]
    n_feat = arr.shape[1]

    # First frame: zero all derivatives
    arr[0, _VELOCITIES_SLICE] = 0.0
    if n_feat >= FEATURES_PER_FRAME:
        arr[0, _BODY_TILT_VEL_COL] = 0.0
        arr[0, _VERT_COM_VEL_COL] = 0.0
        arr[0, _ACCELS_SLICE] = 0.0

    for i in range(1, n_frames):
        # Angular velocities from joint angles
        curr = angles[i]
        prev = angles[i - 1]
        vel = np.where(
            np.isnan(curr) | np.isnan(prev),
            np.nan,
            (curr - prev) / dt,
        )
        arr[i, _VELOCITIES_SLICE] = vel

        if n_feat < FEATURES_PER_FRAME:
            continue

        # Body tilt velocity from body_tilt (handle ±180 wraparound)
        curr_tilt = arr[i, _BODY_TILT_COL]
        prev_tilt = arr[i - 1, _BODY_TILT_COL]
        if np.isnan(curr_tilt) or np.isnan(prev_tilt):
            arr[i, _BODY_TILT_VEL_COL] = 0.0
        else:
            delta_tilt = curr_tilt - prev_tilt
            delta_tilt = (delta_tilt + 180.0) % 360.0 - 180.0
            arr[i, _BODY_TILT_VEL_COL] = delta_tilt / dt

        # Vertical COM velocity from vertical_com
        curr_vcom = arr[i, _VERT_COM_COL]
        prev_vcom = arr[i - 1, _VERT_COM_COL]
        if np.isnan(curr_vcom) or np.isnan(prev_vcom):
            arr[i, _VERT_COM_VEL_COL] = 0.0
        else:
            arr[i, _VERT_COM_VEL_COL] = (curr_vcom - prev_vcom) / dt

        # Angular accelerations from angular velocities
        if i >= 2:
            curr_vel = arr[i, _VELOCITIES_SLICE]
            prev_vel = arr[i - 1, _VELOCITIES_SLICE]
            accel = np.where(
                np.isnan(curr_vel) | np.isnan(prev_vel),
                0.0,
                (curr_vel - prev_vel) / dt,
            )
            arr[i, _ACCELS_SLICE] = accel
        else:
            arr[i, _ACCELS_SLICE] = 0.0


# ── Augmenter Class ──────────────────────────────────────────────────


class FeatureAugmenter:
    """Generates augmented FeatureSequence variants from reference sequences.

    Applies a random combination of feature-space augmentations per variant,
    controlled by AugmentConfig.
    """

    def __init__(
        self,
        config: AugmentConfig | None = None,
        seed: int | None = None,
    ):
        self.config = config or AugmentConfig()
        self.rng = np.random.default_rng(seed)

    def augment_one(self, features: FeatureSequence) -> FeatureSequence:
        """Apply a random combination of augmentations to produce one variant."""
        if features.n_frames == 0:
            return FeatureSequence(frames=[])

        arr = features.to_array()  # (n_frames, 75)
        cfg = self.config

        # 1. Time stretch
        if self.rng.random() < cfg.time_stretch_prob:
            factor = self.rng.uniform(*cfg.time_stretch_range)
            arr = _time_stretch(arr, factor, cfg.assumed_fps)

        # 2. Angle perturbation (with velocity recomputation)
        if self.rng.random() < cfg.angle_noise_prob:
            arr = _perturb_angles(
                arr, cfg.angle_noise_std, cfg.angle_noise_smoothing,
                self.rng, cfg.assumed_fps,
            )

        # 3. Proportion scaling
        if self.rng.random() < cfg.proportion_scale_prob:
            arr = _scale_proportions(arr, cfg.proportion_scale_range, self.rng)

        # 4. Position noise
        if self.rng.random() < cfg.position_noise_prob:
            arr = _inject_noise(arr, cfg.position_noise_std, self.rng)

        # 5. Mirror
        if self.rng.random() < cfg.mirror_prob:
            arr = _mirror(arr)

        # 6. Dropout
        if self.rng.random() < cfg.dropout_prob:
            arr = _dropout(arr, cfg.dropout_rate, self.rng)

        return _array_to_sequence(arr, features.angle_names, features.limb_ratio_names)

    def augment_many(
        self,
        features: FeatureSequence,
        n_variants: int,
    ) -> list[FeatureSequence]:
        """Generate n augmented variants from a single reference sequence."""
        return [self.augment_one(features) for _ in range(n_variants)]

    def augment_references(
        self,
        references: list[FeatureSequence],
        n_total: int,
    ) -> list[FeatureSequence]:
        """Generate n_total variants, sampling evenly from multiple references.

        With 3 references and n_total=600, each reference generates ~200 variants.
        """
        if not references:
            return []

        per_ref = n_total // len(references)
        remainder = n_total % len(references)

        variants: list[FeatureSequence] = []
        for i, ref in enumerate(references):
            count = per_ref + (1 if i < remainder else 0)
            variants.extend(self.augment_many(ref, count))

        return variants


# ── Public API ───────────────────────────────────────────────────────


def augment_sequence(
    features: FeatureSequence,
    n_variants: int,
    config: AugmentConfig | None = None,
    seed: int | None = None,
) -> list[FeatureSequence]:
    """Generate augmented FeatureSequence variants from a reference.

    Args:
        features: Reference feature sequence to augment.
        n_variants: Number of synthetic variants to generate.
        config: Augmentation configuration. Uses defaults if None.
        seed: Random seed for reproducibility.

    Returns:
        List of n_variants augmented FeatureSequence objects.
    """
    augmenter = FeatureAugmenter(config=config, seed=seed)
    return augmenter.augment_many(features, n_variants)


def augment_from_keypoints(
    keypoints_sequence: list[np.ndarray],
    confidences_sequence: list[np.ndarray],
    timestamps_ms: list[float],
    n_variants: int = 500,
    config: AugmentConfig | None = None,
    seed: int | None = None,
) -> list[FeatureSequence]:
    """Augment from raw YOLO keypoints: extract features first, then augment.

    Convenience bridge between the raw keypoint pipeline and feature augmentation.

    Args:
        keypoints_sequence: List of (17, 2) arrays, one per frame.
        confidences_sequence: List of (17,) arrays, one per frame.
        timestamps_ms: Timestamp per frame in milliseconds.
        n_variants: Number of synthetic variants to generate.
        config: Augmentation configuration. Uses defaults if None.
        seed: Random seed for reproducibility.

    Returns:
        List of n_variants augmented FeatureSequence objects.
    """
    from core.pose.features import extract_features

    reference = extract_features(keypoints_sequence, confidences_sequence, timestamps_ms)
    return augment_sequence(reference, n_variants, config=config, seed=seed)
