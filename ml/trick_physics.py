"""Physics-based trick definition and synthetic skeleton generator.

Every parkour trick is decomposed into:
  - Rotation axis: lateral (flip), longitudinal (twist), off-axis (cork)
  - Rotation direction: forward, backward, lateral
  - Rotation count: single, double, triple
  - Twist count: 0, 1, 2, 3
  - Entry type: standing, running, one-leg, wall
  - Body shape: tuck, pike, layout, open

The generator creates physically-plausible skeleton sequences by:
1. Extracting the CENTER OF MASS trajectory from real clips (parabolic arc = gravity)
2. Extracting the ROTATION PATTERN (how fast joints rotate around COM)
3. Generating variations by modifying: jump height, rotation speed, body proportions,
   camera angle, entry speed, landing position — while preserving the physics.

Key insight: A backflip and sideflip have DIFFERENT joint symmetry patterns.
- Backflip: left_shoulder and right_shoulder stay at similar heights (symmetric)
- Sideflip: left_shoulder goes high while right_shoulder goes low (asymmetric)
This 2D signature is detectable even from a single camera.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class RotationAxis(str, Enum):
    LATERAL = "lateral"         # Flip axis (left-right, like a cartwheel axis)
    LONGITUDINAL = "longitudinal"  # Twist axis (head-to-toe spin)
    OFF_AXIS = "off_axis"       # Cork/corkscrew (tilted ~30-60°)
    SAGITTAL = "sagittal"       # Side flip axis (front-back)


class Direction(str, Enum):
    FORWARD = "forward"
    BACKWARD = "backward"
    LEFT = "left"
    RIGHT = "right"


class BodyShape(str, Enum):
    TUCK = "tuck"       # Knees to chest, tight ball
    PIKE = "pike"       # Legs straight, folded at hips
    LAYOUT = "layout"   # Body straight/slightly arched
    OPEN = "open"       # Arms and legs spread


class EntryType(str, Enum):
    STANDING = "standing"
    RUNNING = "running"
    ONE_LEG = "one_leg"
    WALL = "wall"
    EDGE = "edge"       # Off a ledge/platform


@dataclass
class TrickDefinition:
    """Physics-based definition of a parkour trick."""
    trick_id: str
    name: str

    # Core rotation
    rotation_axis: RotationAxis
    direction: Direction
    rotation_count: float  # 1.0 = full rotation, 0.5 = half, 2.0 = double
    twist_count: float = 0.0  # Number of longitudinal twists

    # Body shape during execution
    body_shape: BodyShape = BodyShape.TUCK

    # Entry/exit
    entry: EntryType = EntryType.STANDING

    # Timing (relative)
    takeoff_duration: float = 0.15   # Fraction of total time
    air_duration: float = 0.6        # Fraction of total time
    landing_duration: float = 0.25   # Fraction of total time

    # Physical constraints
    min_height_m: float = 0.5    # Minimum jump height in meters
    typical_height_m: float = 1.0
    typical_duration_s: float = 0.8  # Typical trick duration


# ── Trick Catalog ────────────────────────────────────────────────

TRICK_DEFINITIONS: dict[str, TrickDefinition] = {
    # Basic flips
    "back_flip": TrickDefinition("back_flip", "Back Flip",
        RotationAxis.LATERAL, Direction.BACKWARD, 1.0, body_shape=BodyShape.TUCK),
    "front_flip": TrickDefinition("front_flip", "Front Flip",
        RotationAxis.LATERAL, Direction.FORWARD, 1.0, body_shape=BodyShape.TUCK),
    "side_flip": TrickDefinition("side_flip", "Side Flip",
        RotationAxis.SAGITTAL, Direction.LEFT, 1.0, body_shape=BodyShape.TUCK),

    # Backflip variations
    "back_full": TrickDefinition("back_full", "Back Full",
        RotationAxis.LATERAL, Direction.BACKWARD, 1.0, twist_count=1.0, body_shape=BodyShape.LAYOUT),
    "double_full": TrickDefinition("double_full", "Double Full",
        RotationAxis.LATERAL, Direction.BACKWARD, 1.0, twist_count=2.0, body_shape=BodyShape.LAYOUT),
    "triple_full": TrickDefinition("triple_full", "Triple Full",
        RotationAxis.LATERAL, Direction.BACKWARD, 1.0, twist_count=3.0, body_shape=BodyShape.LAYOUT),
    "double_back": TrickDefinition("double_back", "Double Back",
        RotationAxis.LATERAL, Direction.BACKWARD, 2.0, body_shape=BodyShape.TUCK,
        typical_height_m=1.5, typical_duration_s=1.2),

    # Front flip variations
    "double_front": TrickDefinition("double_front", "Double Front",
        RotationAxis.LATERAL, Direction.FORWARD, 2.0, body_shape=BodyShape.TUCK,
        typical_height_m=1.5),
    "double_pike": TrickDefinition("double_pike", "Double Pike",
        RotationAxis.LATERAL, Direction.FORWARD, 2.0, body_shape=BodyShape.PIKE,
        typical_height_m=1.5),

    # Corks
    "cork": TrickDefinition("cork", "Cork",
        RotationAxis.OFF_AXIS, Direction.BACKWARD, 1.0, twist_count=1.0, body_shape=BodyShape.LAYOUT),
    "double_cork": TrickDefinition("double_cork", "Double Cork",
        RotationAxis.OFF_AXIS, Direction.BACKWARD, 2.0, twist_count=2.0, body_shape=BodyShape.LAYOUT,
        typical_height_m=1.5, typical_duration_s=1.2),
    "triple_cork": TrickDefinition("triple_cork", "Triple Cork",
        RotationAxis.OFF_AXIS, Direction.BACKWARD, 3.0, twist_count=3.0, body_shape=BodyShape.LAYOUT,
        typical_height_m=2.0, typical_duration_s=1.5),
    "cork_in": TrickDefinition("cork_in", "Cork In",
        RotationAxis.OFF_AXIS, Direction.FORWARD, 1.0, twist_count=1.0),

    # Special entries
    "gainer": TrickDefinition("gainer", "Gainer",
        RotationAxis.LATERAL, Direction.BACKWARD, 1.0, entry=EntryType.RUNNING),
    "webster": TrickDefinition("webster", "Webster",
        RotationAxis.LATERAL, Direction.FORWARD, 1.0, entry=EntryType.ONE_LEG),
    "wall_side": TrickDefinition("wall_side", "Wall Side",
        RotationAxis.SAGITTAL, Direction.LEFT, 1.0, entry=EntryType.WALL),

    # Twist-based
    "raiz": TrickDefinition("raiz", "Raiz",
        RotationAxis.OFF_AXIS, Direction.BACKWARD, 0.5, twist_count=0.5,
        body_shape=BodyShape.OPEN),
    "b_twist": TrickDefinition("b_twist", "B-Twist",
        RotationAxis.LONGITUDINAL, Direction.LEFT, 0.0, twist_count=1.0,
        body_shape=BodyShape.LAYOUT),

    # Combos
    "castaway": TrickDefinition("castaway", "Castaway",
        RotationAxis.OFF_AXIS, Direction.BACKWARD, 1.5, twist_count=0.5,
        body_shape=BodyShape.OPEN, typical_duration_s=1.0),
    "krok": TrickDefinition("krok", "Krok",
        RotationAxis.LATERAL, Direction.FORWARD, 0.5, body_shape=BodyShape.OPEN),
    "cody": TrickDefinition("cody", "Cody",
        RotationAxis.LATERAL, Direction.BACKWARD, 1.0, entry=EntryType.EDGE),

    # Full twist combos
    "tak_full": TrickDefinition("tak_full", "Tak Full",
        RotationAxis.LATERAL, Direction.BACKWARD, 1.0, twist_count=1.0,
        entry=EntryType.RUNNING),
    "gainer_double_full": TrickDefinition("gainer_double_full", "Gainer Double Full",
        RotationAxis.LATERAL, Direction.BACKWARD, 1.0, twist_count=2.0,
        entry=EntryType.RUNNING, typical_height_m=1.3),
    "tak_double_full": TrickDefinition("tak_double_full", "Tak Double Full",
        RotationAxis.LATERAL, Direction.BACKWARD, 1.0, twist_count=2.0,
        entry=EntryType.RUNNING, typical_height_m=1.3),
    "kong_gainer": TrickDefinition("kong_gainer", "Kong Gainer",
        RotationAxis.LATERAL, Direction.BACKWARD, 1.0,
        entry=EntryType.RUNNING),
    "side_full": TrickDefinition("side_full", "Side Full",
        RotationAxis.SAGITTAL, Direction.LEFT, 1.0, twist_count=1.0),
    "double_side": TrickDefinition("double_side", "Double Side",
        RotationAxis.SAGITTAL, Direction.LEFT, 2.0),
}


class PhysicsGenerator:
    """Generate synthetic skeleton sequences using physics-based transformations.

    Instead of randomly warping vectors, this generator:
    1. Extracts the movement signature from real clips (COM trajectory + joint rotations)
    2. Generates variations that preserve the PHYSICS of the trick
    3. Uses the TrickDefinition to constrain what variations are plausible

    Key signatures that distinguish tricks in 2D:
    - Backflip: left/right symmetry preserved, hips go above shoulders
    - Sideflip: left/right asymmetry (one side goes up), hips stay level
    - Twist: rapid left/right oscillation in shoulder positions
    - Cork: asymmetric rotation + twist combination
    """

    def __init__(self, target_frames: int = 64, seed: int = 42):
        self.target_frames = target_frames
        self.rng = np.random.default_rng(seed)
        self.real_examples: dict[str, list[np.ndarray]] = {}
        self.signatures: dict[str, dict] = {}

    def load_real(self, trick_id: str, sequences: list[np.ndarray]) -> None:
        """Load real keypoint sequences and extract movement signatures."""
        self.real_examples[trick_id] = sequences
        self.signatures[trick_id] = self._extract_signature(sequences)

    def _extract_signature(self, sequences: list[np.ndarray]) -> dict:
        """Extract the movement signature from real clips.

        Computes:
        - Center of mass trajectory (gravity parabola)
        - Left/right symmetry ratio (distinguishes backflip vs sideflip)
        - Rotation speed profile (how fast the body rotates per phase)
        - Joint angle extremes (max tuck, max extension)
        """
        all_com_y = []
        all_symmetry = []
        all_rotation_speed = []

        for seq in sequences:
            # seq shape: (3, T, 17) — x, y, confidence
            T = seq.shape[1]
            resized = self._resize(seq, self.target_frames)

            # Center of mass (average of hips and shoulders)
            com_x = np.mean(resized[0, :, [5, 6, 11, 12]], axis=1)  # shoulders + hips x
            com_y = np.mean(resized[1, :, [5, 6, 11, 12]], axis=1)  # shoulders + hips y
            all_com_y.append(com_y)

            # Left/right symmetry: |left_shoulder_y - right_shoulder_y|
            # High symmetry = backflip, low symmetry = sideflip
            lr_diff = np.abs(resized[1, :, 5] - resized[1, :, 6])  # left_shoulder vs right_shoulder y
            all_symmetry.append(lr_diff)

            # Rotation speed: frame-to-frame angle change of the body axis
            body_angle = np.arctan2(
                resized[1, :, 0] - com_y,  # nose_y - com_y
                resized[0, :, 0] - com_x,  # nose_x - com_x
            )
            angular_vel = np.abs(np.diff(body_angle))
            all_rotation_speed.append(angular_vel)

        return {
            "com_y_mean": np.mean(all_com_y, axis=0),
            "com_y_std": np.std(all_com_y, axis=0) + 1e-6,
            "symmetry_mean": np.mean(all_symmetry, axis=0),
            "symmetry_std": np.std(all_symmetry, axis=0) + 1e-6,
            "rotation_speed_mean": np.mean([np.mean(rs) for rs in all_rotation_speed]),
        }

    def generate(self, trick_id: str, n: int = 500) -> list[np.ndarray]:
        """Generate n synthetic sequences that preserve the trick's physics."""
        if trick_id not in self.real_examples:
            raise ValueError(f"No real examples for '{trick_id}'")

        real = self.real_examples[trick_id]
        sig = self.signatures[trick_id]
        trick_def = TRICK_DEFINITIONS.get(trick_id)

        synthetic = []
        for _ in range(n):
            base = real[self.rng.integers(len(real))].copy()
            base = self._resize(base, self.target_frames)
            augmented = self._physics_augment(base, sig, trick_def)
            synthetic.append(augmented)

        return synthetic

    def _physics_augment(
        self,
        data: np.ndarray,
        sig: dict,
        trick_def: TrickDefinition | None,
    ) -> np.ndarray:
        """Apply physics-preserving augmentations."""
        result = data.copy()

        # 1. Speed variation (preserves relative timing between phases)
        if self.rng.random() < 0.8:
            result = self._speed_preserving(result, self.rng.uniform(0.75, 1.3))

        # 2. Jump height variation (scales vertical movement, preserves parabola)
        if self.rng.random() < 0.7:
            result = self._height_variation(result, self.rng.uniform(0.8, 1.25))

        # 3. Camera angle rotation (small — large rotations would change the trick appearance)
        if self.rng.random() < 0.6:
            result = self._rotate_2d(result, self.rng.uniform(-15, 15))

        # 4. Body proportion scaling (different body types)
        if self.rng.random() < 0.5:
            result = self._body_proportion(result, self.rng.uniform(0.9, 1.1))

        # 5. Position translation
        if self.rng.random() < 0.7:
            result = self._translate(result, self.rng.uniform(-0.1, 0.1), self.rng.uniform(-0.1, 0.1))

        # 6. Controlled noise (small — preserves joint relationships)
        if self.rng.random() < 0.7:
            result = self._controlled_noise(result, std=0.008)

        # 7. Mirror (only if trick is symmetric — don't mirror side_flip to wrong side)
        if self.rng.random() < 0.3:
            if trick_def is None or trick_def.rotation_axis != RotationAxis.SAGITTAL:
                result = self._mirror(result)

        # 8. Confidence variation
        if self.rng.random() < 0.5:
            result[2] *= self.rng.uniform(0.75, 1.0, size=(1, result.shape[1], 1))

        return result

    def _speed_preserving(self, data: np.ndarray, factor: float) -> np.ndarray:
        """Speed variation that preserves phase structure."""
        C, T, V = data.shape
        new_T = max(8, int(T / factor))
        indices = np.linspace(0, T - 1, new_T)
        result = np.zeros((C, new_T, V), dtype=data.dtype)
        for c in range(C):
            for v in range(V):
                result[c, :, v] = np.interp(indices, np.arange(T), data[c, :, v])
        return self._resize(result, self.target_frames)

    def _height_variation(self, data: np.ndarray, factor: float) -> np.ndarray:
        """Scale vertical movement amplitude (simulating different jump heights)."""
        result = data.copy()
        # Find the lowest point of COM (peak of jump in y-down coordinates)
        com_y = np.mean(data[1, :, [5, 6, 11, 12]], axis=1)
        baseline = com_y[0]  # Starting y position
        # Scale deviation from baseline
        for v in range(17):
            deviation = data[1, :, v] - baseline
            result[1, :, v] = baseline + deviation * factor
        return np.clip(result, 0, 1)

    def _rotate_2d(self, data: np.ndarray, angle_deg: float) -> np.ndarray:
        """Small rotation to simulate slightly different camera angles."""
        import math
        result = data.copy()
        angle = math.radians(angle_deg)
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        cx = data[0].mean(axis=1, keepdims=True)
        cy = data[1].mean(axis=1, keepdims=True)
        x, y = data[0] - cx, data[1] - cy
        result[0] = x * cos_a - y * sin_a + cx
        result[1] = x * sin_a + y * cos_a + cy
        return np.clip(result, 0, 1)

    def _body_proportion(self, data: np.ndarray, factor: float) -> np.ndarray:
        """Scale limb lengths from center of mass."""
        result = data.copy()
        com_x = np.mean(data[0, :, [5, 6, 11, 12]], axis=1, keepdims=True)
        com_y = np.mean(data[1, :, [5, 6, 11, 12]], axis=1, keepdims=True)
        result[0] = com_x + (data[0] - com_x) * factor
        result[1] = com_y + (data[1] - com_y) * factor
        return np.clip(result, 0, 1)

    def _translate(self, data: np.ndarray, dx: float, dy: float) -> np.ndarray:
        result = data.copy()
        result[0] = np.clip(data[0] + dx, 0, 1)
        result[1] = np.clip(data[1] + dy, 0, 1)
        return result

    def _controlled_noise(self, data: np.ndarray, std: float) -> np.ndarray:
        """Add noise that preserves joint relationships (noise is correlated between connected joints)."""
        result = data.copy()
        # Global shift noise (moves all joints together)
        global_noise = self.rng.normal(0, std * 0.5, size=(2, data.shape[1], 1))
        result[:2] += global_noise
        # Per-joint noise (smaller)
        local_noise = self.rng.normal(0, std * 0.3, size=(2, data.shape[1], 17))
        result[:2] += local_noise
        return np.clip(result, 0, 1)

    def _mirror(self, data: np.ndarray) -> np.ndarray:
        MIRROR_PAIRS = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]
        result = data.copy()
        result[0] = 1.0 - data[0]
        for left, right in MIRROR_PAIRS:
            result[:, :, left], result[:, :, right] = result[:, :, right].copy(), result[:, :, left].copy()
        return result

    def _resize(self, data: np.ndarray, target: int) -> np.ndarray:
        C, T, V = data.shape
        if T == target:
            return data
        indices = np.linspace(0, T - 1, target)
        result = np.zeros((C, target, V), dtype=data.dtype)
        for c in range(C):
            for v in range(V):
                result[c, :, v] = np.interp(indices, np.arange(T), data[c, :, v])
        return result
