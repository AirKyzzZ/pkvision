"""Pure physics-based synthetic feature sequence generator.

Generates (T, 75) feature arrays directly from TrickDefinitions, replacing the
need for manually labeled video data. Each trick is defined by its kinematics
(rotation axis, body shape, phase timing), and realistic feature trajectories
are simulated from first principles.

Feature layout per frame (75 total):
    [0:9]     joint_angles (9) — degrees
    [9:18]    angular_velocities (9) — deg/s
    [18:52]   relative_positions (34) — 17×2 body-centric
    [52:60]   limb_ratios (8) — relative to torso
    [60]      body_tilt — degrees [-180, 180], 0=upright
    [61]      body_tilt_velocity — deg/s
    [62]      vertical_com — torso-lengths above start
    [63]      vertical_com_velocity — torso-lengths/s
    [64]      cumulative_rotation — total degrees
    [65]      left_right_symmetry — L/R shoulder diff / torso
    [66:75]   angular_accelerations (9) — deg/s²
"""

from __future__ import annotations

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
)
from ml.trick_physics import (
    TRICK_DEFINITIONS,
    BodyShape,
    Direction,
    EntryType,
    RotationAxis,
    TrickDefinition,
)

# ── Body shape target angles ────────────────────────────────────────
# Order: left_knee, right_knee, left_hip, right_hip,
#        left_elbow, right_elbow, left_shoulder, right_shoulder, spine

_STANDING_ANGLES = np.array(
    [170.0, 170.0, 170.0, 170.0, 160.0, 160.0, 30.0, 30.0, 175.0],
    dtype=np.float64,
)

_BODY_SHAPE_ANGLES: dict[BodyShape, np.ndarray] = {
    BodyShape.TUCK: np.array(
        [45.0, 45.0, 45.0, 45.0, 45.0, 45.0, 60.0, 60.0, 80.0],
        dtype=np.float64,
    ),
    BodyShape.PIKE: np.array(
        [170.0, 170.0, 45.0, 45.0, 160.0, 160.0, 120.0, 120.0, 90.0],
        dtype=np.float64,
    ),
    BodyShape.LAYOUT: np.array(
        [170.0, 170.0, 170.0, 170.0, 150.0, 150.0, 170.0, 170.0, 175.0],
        dtype=np.float64,
    ),
    BodyShape.OPEN: np.array(
        [150.0, 150.0, 150.0, 150.0, 140.0, 140.0, 130.0, 130.0, 160.0],
        dtype=np.float64,
    ),
}

# Takeoff angles: knees deeply bent, hips slightly bent, arms loaded
_TAKEOFF_START_ANGLES = np.array(
    [90.0, 90.0, 140.0, 140.0, 100.0, 100.0, 60.0, 60.0, 165.0],
    dtype=np.float64,
)
_TAKEOFF_END_ANGLES = np.array(
    [170.0, 170.0, 170.0, 170.0, 130.0, 130.0, 140.0, 140.0, 175.0],
    dtype=np.float64,
)

# Landing angles: impact absorption
_LANDING_IMPACT_ANGLES = np.array(
    [120.0, 120.0, 140.0, 140.0, 130.0, 130.0, 90.0, 90.0, 165.0],
    dtype=np.float64,
)

# ── Human limb proportions (relative to torso) ─────────────────────
# Order matches LIMB_RATIO_NAMES:
#   left_upper_arm, right_upper_arm, left_forearm, right_forearm,
#   left_thigh, right_thigh, left_shin, right_shin
_LIMB_RATIO_MEANS = np.array(
    [0.55, 0.55, 0.45, 0.45, 0.75, 0.75, 0.72, 0.72],
    dtype=np.float64,
)
_LIMB_RATIO_STDS = np.array(
    [0.05, 0.05, 0.04, 0.04, 0.06, 0.06, 0.05, 0.05],
    dtype=np.float64,
)

# ── Simple kinematic chain for keypoint generation ──────────────────
# COCO 17 keypoints, indexed 0-16.
# We build body-centric positions from joint angles + body tilt.
# Hip midpoint is the origin (0, 0).

# Segment lengths relative to torso (= 1.0)
_SEG_TORSO = 1.0
_SEG_HEAD = 0.35        # shoulder mid → nose
_SEG_UPPER_ARM = 0.55
_SEG_FOREARM = 0.45
_SEG_THIGH = 0.75
_SEG_SHIN = 0.72


class PhysicsFeatureGenerator:
    """Generate synthetic (T, 75) feature arrays directly from TrickDefinitions."""

    def __init__(self, target_frames: int = 64, fps: float = 30.0, seed: int = 42):
        self.target_frames = target_frames
        self.fps = fps
        self.rng = np.random.default_rng(seed)

    # ── Public API ──────────────────────────────────────────────────

    def generate(
        self, trick_def: TrickDefinition, n: int = 1000
    ) -> list[np.ndarray]:
        """Generate n synthetic feature arrays for a trick.

        Returns list of (target_frames, 75) arrays.
        """
        results: list[np.ndarray] = []
        for _ in range(n):
            arr = self._generate_single(trick_def)
            results.append(arr)
        return results

    def generate_no_trick(self, n: int = 1000) -> list[np.ndarray]:
        """Generate negative class samples (standing, walking, running, crouching)."""
        results: list[np.ndarray] = []
        motion_types = ["standing", "walking", "running", "crouching"]
        for i in range(n):
            motion = motion_types[i % len(motion_types)]
            arr = self._generate_no_trick_single(motion)
            results.append(arr)
        return results

    def generate_all(
        self, samples_per_trick: int = 1000
    ) -> dict[str, list[np.ndarray]]:
        """Generate for all tricks in TRICK_DEFINITIONS + no_trick class."""
        result: dict[str, list[np.ndarray]] = {}
        for trick_id, trick_def in TRICK_DEFINITIONS.items():
            result[trick_id] = self.generate(trick_def, samples_per_trick)
        result["no_trick"] = self.generate_no_trick(samples_per_trick)
        return result

    # ── Single trick generation ─────────────────────────────────────

    def _generate_single(self, td: TrickDefinition) -> np.ndarray:
        """Generate one (target_frames, 75) array for a single trick sample."""
        T = self.target_frames
        dt = 1.0 / self.fps

        # Randomize duration and height
        duration_var = self.rng.uniform(0.8, 1.2)
        height_var = self.rng.uniform(0.75, 1.25)
        tuck_var = self.rng.uniform(-15.0, 15.0)
        rotation_var = self.rng.uniform(-5.0, 5.0)
        noise_std_angles = self.rng.uniform(2.0, 5.0)
        noise_std_pos = self.rng.uniform(0.01, 0.03)
        landing_softness = self.rng.uniform(0.7, 1.3)

        total_duration = td.typical_duration_s * duration_var
        typical_height = td.typical_height_m * height_var

        # Phase boundaries (in normalized time 0..1)
        approach_end = td.takeoff_duration
        takeoff_end = approach_end + 0.1 * td.air_duration
        air_end = approach_end + td.air_duration
        # landing_end = 1.0

        # Target rotation
        total_rotation_deg = td.rotation_count * 360.0 + rotation_var
        if td.direction == Direction.BACKWARD:
            total_rotation_deg = -total_rotation_deg
        elif td.direction == Direction.LEFT:
            total_rotation_deg = -total_rotation_deg
        # FORWARD and RIGHT keep positive sign

        # Time array normalized to [0, 1]
        t_norm = np.linspace(0.0, 1.0, T)

        # Phase masks
        is_approach = t_norm < approach_end
        is_takeoff = (t_norm >= approach_end) & (t_norm < takeoff_end)
        is_air = (t_norm >= takeoff_end) & (t_norm < air_end)
        is_landing = t_norm >= air_end

        # ── Joint angles (T, 9) ────────────────────────────────────
        air_shape_angles = np.clip(
            _BODY_SHAPE_ANGLES[td.body_shape] + tuck_var, 10.0, 180.0
        )

        angles = np.zeros((T, N_ANGLES), dtype=np.float64)
        for i in range(T):
            tn = t_norm[i]
            if is_approach[i]:
                # Transition from standing to takeoff-ready
                frac = tn / max(approach_end, 1e-6)
                angles[i] = _lerp(_STANDING_ANGLES, _TAKEOFF_START_ANGLES, frac)
            elif is_takeoff[i]:
                frac = (tn - approach_end) / max(takeoff_end - approach_end, 1e-6)
                angles[i] = _lerp(_TAKEOFF_START_ANGLES, _TAKEOFF_END_ANGLES, frac)
            elif is_air[i]:
                air_frac = (tn - takeoff_end) / max(air_end - takeoff_end, 1e-6)
                # Shape up in first 20%, hold shape, open in last 20%
                if air_frac < 0.2:
                    local_frac = air_frac / 0.2
                    angles[i] = _lerp(_TAKEOFF_END_ANGLES, air_shape_angles, local_frac)
                elif air_frac > 0.8:
                    local_frac = (air_frac - 0.8) / 0.2
                    angles[i] = _lerp(air_shape_angles, _LANDING_IMPACT_ANGLES, local_frac)
                else:
                    angles[i] = air_shape_angles
            else:  # landing
                frac = (tn - air_end) / max(1.0 - air_end, 1e-6)
                impact_angles = _LANDING_IMPACT_ANGLES.copy()
                impact_angles[:2] *= landing_softness  # knee bend
                impact_angles = np.clip(impact_angles, 30.0, 180.0)
                angles[i] = _lerp(impact_angles, _STANDING_ANGLES, frac)

        # Add noise
        angles += self.rng.normal(0, noise_std_angles, angles.shape)
        angles = np.clip(angles, 0.0, 180.0)

        # ── Body tilt (T,) ──────────────────────────────────────────
        body_tilt = np.zeros(T, dtype=np.float64)

        if td.rotation_axis in (RotationAxis.LATERAL, RotationAxis.OFF_AXIS):
            for i in range(T):
                tn = t_norm[i]
                if is_approach[i]:
                    # Slight lean depending on direction
                    lean = 5.0 if td.direction == Direction.FORWARD else -5.0
                    frac = tn / max(approach_end, 1e-6)
                    body_tilt[i] = lean * frac
                elif is_takeoff[i]:
                    frac = (tn - approach_end) / max(takeoff_end - approach_end, 1e-6)
                    start_lean = 5.0 if td.direction == Direction.FORWARD else -5.0
                    body_tilt[i] = start_lean + frac * (total_rotation_deg * 0.05)
                elif is_air[i]:
                    air_frac = (tn - takeoff_end) / max(air_end - takeoff_end, 1e-6)
                    body_tilt[i] = air_frac * total_rotation_deg
                else:
                    # Landing: wrap to nearest upright
                    target_tilt = total_rotation_deg
                    # Snap to nearest multiple of 360
                    nearest_360 = round(target_tilt / 360.0) * 360.0
                    frac = (tn - air_end) / max(1.0 - air_end, 1e-6)
                    body_tilt[i] = _lerp_scalar(target_tilt, nearest_360, frac)
        elif td.rotation_axis == RotationAxis.SAGITTAL:
            # Side flip: tilt goes through ±180 laterally
            # We encode as body_tilt oscillation through ±180
            for i in range(T):
                tn = t_norm[i]
                if is_air[i]:
                    air_frac = (tn - takeoff_end) / max(air_end - takeoff_end, 1e-6)
                    body_tilt[i] = air_frac * total_rotation_deg
                elif is_landing[i]:
                    target_tilt = total_rotation_deg
                    nearest_360 = round(target_tilt / 360.0) * 360.0
                    frac = (tn - air_end) / max(1.0 - air_end, 1e-6)
                    body_tilt[i] = _lerp_scalar(target_tilt, nearest_360, frac)
        elif td.rotation_axis == RotationAxis.LONGITUDINAL:
            # Pure twist — minimal tilt, rotation is mostly in symmetry
            body_tilt[:] = self.rng.normal(0, 3.0, T)

        # Wrap to [-180, 180] for storage
        body_tilt_wrapped = ((body_tilt + 180.0) % 360.0) - 180.0
        # Add noise
        body_tilt_wrapped += self.rng.normal(0, noise_std_angles * 0.5, T)

        # ── Vertical COM trajectory (T,) ───────────────────────────
        # Parabolic arc during air phase, 0 on ground
        vertical_com = np.zeros(T, dtype=np.float64)
        # Convert typical_height_m to torso-lengths (~0.5m torso)
        peak_height_tl = typical_height / 0.5

        for i in range(T):
            tn = t_norm[i]
            if is_takeoff[i]:
                frac = (tn - approach_end) / max(takeoff_end - approach_end, 1e-6)
                vertical_com[i] = peak_height_tl * 0.15 * frac
            elif is_air[i]:
                air_frac = (tn - takeoff_end) / max(air_end - takeoff_end, 1e-6)
                # Parabola: h(t) = 4*H*t*(1-t) peaks at t=0.5
                vertical_com[i] = peak_height_tl * 4.0 * air_frac * (1.0 - air_frac)
            elif is_landing[i]:
                frac = (tn - air_end) / max(1.0 - air_end, 1e-6)
                vertical_com[i] = peak_height_tl * 0.1 * (1.0 - frac)

        vertical_com += self.rng.normal(0, 0.02, T)

        # ── Left/right symmetry (T,) ───────────────────────────────
        lr_symmetry = np.zeros(T, dtype=np.float64)

        if td.rotation_axis == RotationAxis.SAGITTAL:
            # Side flip: strong asymmetry during air
            for i in range(T):
                if is_air[i]:
                    air_frac = (tn - takeoff_end) / max(air_end - takeoff_end, 1e-6)
                    lr_symmetry[i] = 0.5 * np.sin(air_frac * np.pi * td.rotation_count)
        elif td.rotation_axis == RotationAxis.OFF_AXIS:
            # Cork: intermediate asymmetry
            for i in range(T):
                if is_air[i]:
                    air_frac = (t_norm[i] - takeoff_end) / max(air_end - takeoff_end, 1e-6)
                    lr_symmetry[i] = 0.3 * np.sin(air_frac * np.pi * td.rotation_count)
        # LATERAL: stays ~0 (symmetric)

        # Twist contribution
        if td.twist_count > 0:
            for i in range(T):
                if is_air[i]:
                    air_frac = (t_norm[i] - takeoff_end) / max(air_end - takeoff_end, 1e-6)
                    twist_osc = 0.3 * np.sin(
                        air_frac * 2.0 * np.pi * td.twist_count
                    )
                    lr_symmetry[i] += twist_osc

        lr_symmetry += self.rng.normal(0, 0.03, T)

        # ── Cumulative rotation (T,) ───────────────────────────────
        # This is the unwrapped total rotation from body_tilt
        cumulative_rotation = body_tilt.copy()  # Unwrapped version

        # ── Compute derivatives ─────────────────────────────────────
        # Angular velocities from joint angles
        ang_vel = np.zeros_like(angles)
        ang_vel[1:] = np.diff(angles, axis=0) * self.fps
        ang_vel[0] = ang_vel[1] if T > 1 else 0.0

        # Body tilt velocity from unwrapped body_tilt
        tilt_vel = np.zeros(T, dtype=np.float64)
        tilt_vel[1:] = np.diff(body_tilt) * self.fps
        tilt_vel[0] = tilt_vel[1] if T > 1 else 0.0

        # Vertical COM velocity
        com_vel = np.zeros(T, dtype=np.float64)
        com_vel[1:] = np.diff(vertical_com) * self.fps
        com_vel[0] = com_vel[1] if T > 1 else 0.0

        # Angular accelerations from angular velocities
        ang_accel = np.zeros_like(angles)
        ang_accel[1:] = np.diff(ang_vel, axis=0) * self.fps
        ang_accel[0] = ang_accel[1] if T > 1 else 0.0

        # ── Limb ratios (T, 8) ─────────────────────────────────────
        # Fixed per sample with small variation
        person_ratios = _LIMB_RATIO_MEANS + self.rng.normal(0, 1, N_RATIOS) * _LIMB_RATIO_STDS
        person_ratios = np.clip(person_ratios, 0.2, 1.5)
        limb_ratios = np.tile(person_ratios, (T, 1))
        # Tiny per-frame noise
        limb_ratios += self.rng.normal(0, 0.005, limb_ratios.shape)

        # ── Relative positions (T, 34) ─────────────────────────────
        rel_pos = self._compute_keypoint_positions(
            angles, body_tilt_wrapped, person_ratios, noise_std_pos
        )

        # ── Assemble feature array (T, 75) ─────────────────────────
        features = np.zeros((T, FEATURES_PER_FRAME), dtype=np.float32)
        o = 0
        features[:, o : o + N_ANGLES] = angles
        o += N_ANGLES
        features[:, o : o + N_VELOCITIES] = ang_vel
        o += N_VELOCITIES
        features[:, o : o + N_POSITIONS] = rel_pos
        o += N_POSITIONS
        features[:, o : o + N_RATIOS] = limb_ratios
        o += N_RATIOS
        # World scalars
        features[:, o + 0] = body_tilt_wrapped
        features[:, o + 1] = tilt_vel
        features[:, o + 2] = vertical_com
        features[:, o + 3] = com_vel
        features[:, o + 4] = cumulative_rotation
        features[:, o + 5] = lr_symmetry
        o += N_WORLD_SCALARS
        features[:, o : o + N_ACCELS] = ang_accel

        return features

    # ── No-trick generation ─────────────────────────────────────────

    def _generate_no_trick_single(self, motion: str) -> np.ndarray:
        """Generate a single no-trick feature array for a given motion type."""
        T = self.target_frames
        t_norm = np.linspace(0.0, 1.0, T)
        noise_angles = self.rng.uniform(1.0, 3.0)
        noise_pos = self.rng.uniform(0.005, 0.015)

        angles = np.zeros((T, N_ANGLES), dtype=np.float64)
        body_tilt = np.zeros(T, dtype=np.float64)
        vertical_com = np.zeros(T, dtype=np.float64)
        lr_symmetry = np.zeros(T, dtype=np.float64)

        if motion == "standing":
            for i in range(T):
                angles[i] = _STANDING_ANGLES + self.rng.normal(0, 2.0, N_ANGLES)
            body_tilt[:] = self.rng.normal(0, 2.0, T)
            vertical_com[:] = self.rng.normal(0, 0.02, T)

        elif motion == "walking":
            freq = self.rng.uniform(1.5, 3.0)  # step frequency in cycles
            for i in range(T):
                phase = np.sin(2.0 * np.pi * freq * t_norm[i])
                base = _STANDING_ANGLES.copy()
                # Hip and knee oscillation
                base[0] += 15.0 * phase   # left_knee
                base[1] -= 15.0 * phase   # right_knee (opposite)
                base[2] += 10.0 * phase   # left_hip
                base[3] -= 10.0 * phase   # right_hip
                # Arm swing
                base[6] += 8.0 * phase    # left_shoulder
                base[7] -= 8.0 * phase    # right_shoulder
                angles[i] = base
            vertical_com[:] = 0.03 * np.sin(2.0 * np.pi * freq * 2 * t_norm)
            body_tilt[:] = 3.0 * np.sin(2.0 * np.pi * freq * t_norm) + self.rng.normal(0, 1.0, T)

        elif motion == "running":
            freq = self.rng.uniform(2.5, 5.0)
            for i in range(T):
                phase = np.sin(2.0 * np.pi * freq * t_norm[i])
                base = _STANDING_ANGLES.copy()
                base[0] += 30.0 * phase
                base[1] -= 30.0 * phase
                base[2] += 20.0 * phase
                base[3] -= 20.0 * phase
                base[4] += 15.0 * phase   # left_elbow
                base[5] -= 15.0 * phase   # right_elbow
                base[6] += 15.0 * phase
                base[7] -= 15.0 * phase
                angles[i] = base
            vertical_com[:] = 0.08 * np.abs(np.sin(2.0 * np.pi * freq * 2 * t_norm))
            body_tilt[:] = -8.0 + self.rng.normal(0, 2.0, T)  # slight forward lean

        elif motion == "crouching":
            crouch_depth = self.rng.uniform(0.3, 0.7)
            for i in range(T):
                base = _STANDING_ANGLES.copy()
                # Deep knee/hip bend
                base[0] = _lerp_scalar(170.0, 70.0, crouch_depth)
                base[1] = _lerp_scalar(170.0, 70.0, crouch_depth)
                base[2] = _lerp_scalar(170.0, 90.0, crouch_depth)
                base[3] = _lerp_scalar(170.0, 90.0, crouch_depth)
                base[8] = _lerp_scalar(175.0, 140.0, crouch_depth)  # spine
                angles[i] = base
            vertical_com[:] = -0.2 * crouch_depth + self.rng.normal(0, 0.02, T)
            body_tilt[:] = self.rng.normal(0, 3.0, T)

        # Add noise
        angles += self.rng.normal(0, noise_angles, angles.shape)
        angles = np.clip(angles, 0.0, 180.0)

        lr_symmetry[:] = self.rng.normal(0, 0.03, T)

        # Derivatives
        ang_vel = np.zeros_like(angles)
        ang_vel[1:] = np.diff(angles, axis=0) * self.fps
        ang_vel[0] = ang_vel[1] if T > 1 else 0.0

        tilt_vel = np.zeros(T, dtype=np.float64)
        tilt_vel[1:] = np.diff(body_tilt) * self.fps
        tilt_vel[0] = tilt_vel[1] if T > 1 else 0.0

        com_vel = np.zeros(T, dtype=np.float64)
        com_vel[1:] = np.diff(vertical_com) * self.fps
        com_vel[0] = com_vel[1] if T > 1 else 0.0

        ang_accel = np.zeros_like(angles)
        ang_accel[1:] = np.diff(ang_vel, axis=0) * self.fps
        ang_accel[0] = ang_accel[1] if T > 1 else 0.0

        cumulative_rotation = np.cumsum(
            np.concatenate([[0.0], np.diff(body_tilt)])
        )

        person_ratios = _LIMB_RATIO_MEANS + self.rng.normal(0, 1, N_RATIOS) * _LIMB_RATIO_STDS
        person_ratios = np.clip(person_ratios, 0.2, 1.5)
        limb_ratios = np.tile(person_ratios, (T, 1))
        limb_ratios += self.rng.normal(0, 0.005, limb_ratios.shape)

        body_tilt_wrapped = ((body_tilt + 180.0) % 360.0) - 180.0

        rel_pos = self._compute_keypoint_positions(
            angles, body_tilt_wrapped, person_ratios, noise_pos
        )

        # Assemble
        features = np.zeros((T, FEATURES_PER_FRAME), dtype=np.float32)
        o = 0
        features[:, o : o + N_ANGLES] = angles
        o += N_ANGLES
        features[:, o : o + N_VELOCITIES] = ang_vel
        o += N_VELOCITIES
        features[:, o : o + N_POSITIONS] = rel_pos
        o += N_POSITIONS
        features[:, o : o + N_RATIOS] = limb_ratios
        o += N_RATIOS
        features[:, o + 0] = body_tilt_wrapped
        features[:, o + 1] = tilt_vel
        features[:, o + 2] = vertical_com
        features[:, o + 3] = com_vel
        features[:, o + 4] = cumulative_rotation
        features[:, o + 5] = lr_symmetry
        o += N_WORLD_SCALARS
        features[:, o : o + N_ACCELS] = ang_accel

        return features

    # ── Kinematic chain → relative positions ────────────────────────

    def _compute_keypoint_positions(
        self,
        angles: np.ndarray,
        body_tilt: np.ndarray,
        limb_ratios: np.ndarray,
        noise_std: float,
    ) -> np.ndarray:
        """Compute 17×2 body-centric keypoint positions from joint angles.

        Uses a simplified kinematic chain with hip midpoint at origin.

        Args:
            angles: (T, 9) joint angles in degrees.
            body_tilt: (T,) body tilt in degrees.
            limb_ratios: (8,) limb length ratios for this person.
            noise_std: Position noise standard deviation.

        Returns:
            (T, 34) flattened relative positions.
        """
        T = angles.shape[0]
        positions = np.zeros((T, 17, 2), dtype=np.float64)

        # Extract per-person limb lengths
        l_upper_arm = limb_ratios[0]
        r_upper_arm = limb_ratios[1]
        l_forearm = limb_ratios[2]
        r_forearm = limb_ratios[3]
        l_thigh = limb_ratios[4]
        r_thigh = limb_ratios[5]
        l_shin = limb_ratios[6]
        r_shin = limb_ratios[7]

        shoulder_spread = 0.3  # half-width of shoulders in torso-lengths
        hip_spread = 0.25     # half-width of hips

        for i in range(T):
            tilt_rad = np.radians(body_tilt[i])
            cos_t = np.cos(tilt_rad)
            sin_t = np.sin(tilt_rad)

            # Hip midpoint = origin (0, 0) — keypoints 11 (left_hip), 12 (right_hip)
            # Convention: x = horizontal, y = vertical (up positive)
            hip_mid = np.array([0.0, 0.0])

            # Torso direction: from hip to shoulder (rotated by body tilt)
            # Upright = (0, 1), tilt rotates this
            torso_dir = np.array([sin_t, cos_t])
            torso_perp = np.array([cos_t, -sin_t])  # perpendicular (rightward)

            # Shoulder midpoint
            shoulder_mid = hip_mid + torso_dir * _SEG_TORSO

            # Individual hips
            l_hip_pos = hip_mid - torso_perp * hip_spread
            r_hip_pos = hip_mid + torso_perp * hip_spread

            # Individual shoulders
            l_shoulder_pos = shoulder_mid - torso_perp * shoulder_spread
            r_shoulder_pos = shoulder_mid + torso_perp * shoulder_spread

            # Head / nose: above shoulder midpoint
            nose_pos = shoulder_mid + torso_dir * _SEG_HEAD

            # Eyes and ears (small offsets from nose)
            l_eye_pos = nose_pos - torso_perp * 0.06 + torso_dir * 0.03
            r_eye_pos = nose_pos + torso_perp * 0.06 + torso_dir * 0.03
            l_ear_pos = nose_pos - torso_perp * 0.1
            r_ear_pos = nose_pos + torso_perp * 0.1

            # ── Arms ───────────────────────────────────────────────
            # Shoulder angle: angle between upper arm and torso
            # Larger angle = arms more raised
            l_shoulder_angle_rad = np.radians(angles[i, 6])  # left_shoulder
            r_shoulder_angle_rad = np.radians(angles[i, 7])  # right_shoulder

            # Left arm: extends downward-outward from left shoulder
            l_arm_dir = _rotate_vec(-torso_dir, l_shoulder_angle_rad)
            l_elbow_pos = l_shoulder_pos + l_arm_dir * l_upper_arm

            l_elbow_angle_rad = np.radians(angles[i, 4])  # left_elbow
            l_forearm_dir = _rotate_vec(l_arm_dir, np.pi - l_elbow_angle_rad)
            l_wrist_pos = l_elbow_pos + l_forearm_dir * l_forearm

            # Right arm
            r_arm_dir = _rotate_vec(-torso_dir, -r_shoulder_angle_rad)
            r_elbow_pos = r_shoulder_pos + r_arm_dir * r_upper_arm

            r_elbow_angle_rad = np.radians(angles[i, 5])  # right_elbow
            r_forearm_dir = _rotate_vec(r_arm_dir, -(np.pi - r_elbow_angle_rad))
            r_wrist_pos = r_elbow_pos + r_forearm_dir * r_forearm

            # ── Legs ───────────────────────────────────────────────
            # Hip angle: angle between thigh and torso (downward)
            l_hip_angle_rad = np.radians(angles[i, 2])  # left_hip
            r_hip_angle_rad = np.radians(angles[i, 3])  # right_hip

            l_leg_dir = _rotate_vec(-torso_dir, -(np.pi - l_hip_angle_rad))
            l_knee_pos = l_hip_pos + l_leg_dir * l_thigh

            l_knee_angle_rad = np.radians(angles[i, 0])  # left_knee
            l_shin_dir = _rotate_vec(l_leg_dir, np.pi - l_knee_angle_rad)
            l_ankle_pos = l_knee_pos + l_shin_dir * l_shin

            r_leg_dir = _rotate_vec(-torso_dir, np.pi - r_hip_angle_rad)
            r_knee_pos = r_hip_pos + r_leg_dir * r_thigh

            r_knee_angle_rad = np.radians(angles[i, 1])  # right_knee
            r_shin_dir = _rotate_vec(r_leg_dir, -(np.pi - r_knee_angle_rad))
            r_ankle_pos = r_knee_pos + r_shin_dir * r_shin

            # Assign to COCO order:
            # 0:nose, 1:l_eye, 2:r_eye, 3:l_ear, 4:r_ear,
            # 5:l_shoulder, 6:r_shoulder, 7:l_elbow, 8:r_elbow,
            # 9:l_wrist, 10:r_wrist, 11:l_hip, 12:r_hip,
            # 13:l_knee, 14:r_knee, 15:l_ankle, 16:r_ankle
            positions[i, 0] = nose_pos
            positions[i, 1] = l_eye_pos
            positions[i, 2] = r_eye_pos
            positions[i, 3] = l_ear_pos
            positions[i, 4] = r_ear_pos
            positions[i, 5] = l_shoulder_pos
            positions[i, 6] = r_shoulder_pos
            positions[i, 7] = l_elbow_pos
            positions[i, 8] = r_elbow_pos
            positions[i, 9] = l_wrist_pos
            positions[i, 10] = r_wrist_pos
            positions[i, 11] = l_hip_pos
            positions[i, 12] = r_hip_pos
            positions[i, 13] = l_knee_pos
            positions[i, 14] = r_knee_pos
            positions[i, 15] = l_ankle_pos
            positions[i, 16] = r_ankle_pos

        # Add noise
        positions += self.rng.normal(0, noise_std, positions.shape)

        # Flatten to (T, 34)
        return positions.reshape(T, N_POSITIONS)


# ── Utility functions ───────────────────────────────────────────────


def _lerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    """Linear interpolation between arrays."""
    return a + (b - a) * t


def _lerp_scalar(a: float, b: float, t: float) -> float:
    """Linear interpolation between scalars."""
    return a + (b - a) * t


def _rotate_vec(v: np.ndarray, angle_rad: float) -> np.ndarray:
    """Rotate a 2D vector by angle_rad (counterclockwise)."""
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])
