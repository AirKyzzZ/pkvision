"""Zero-shot trick matching from 3D biomechanical signatures.

Compares a TrickSignature3D against TrickDefinitions using weighted
multi-axis distance. No training needed — purely physics-based matching.

Usage:
    from core.recognition.matcher import Matcher3D
    matcher = Matcher3D(trick_definitions)
    matches = matcher.match(signature)
    print(matches[0].trick_id, matches[0].confidence)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.models import TrickSignature3D
from ml.trick_physics import (
    TRICK_DEFINITIONS,
    TrickDefinition,
    TrickContext,
    RotationAxis,
    Direction,
    BodyShape,
    EntryType,
)


# ── Axis Vectors ────────────────────────────────────────────────────

# Expected 3D rotation axis unit vectors for each RotationAxis category
# In SMPL world coordinates: x=right, y=up, z=forward
AXIS_VECTORS = {
    RotationAxis.LATERAL: np.array([1.0, 0.0, 0.0]),       # Left-right axis (flip)
    RotationAxis.SAGITTAL: np.array([0.0, 0.0, 1.0]),      # Front-back axis (side flip)
    RotationAxis.LONGITUDINAL: np.array([0.0, 1.0, 0.0]),  # Head-toe axis (twist)
    RotationAxis.OFF_AXIS: np.array([0.7, 0.3, 0.7]),      # Tilted axis (cork)
}
# Normalize off-axis
AXIS_VECTORS[RotationAxis.OFF_AXIS] /= np.linalg.norm(AXIS_VECTORS[RotationAxis.OFF_AXIS])

# Direction mapping
DIRECTION_MAP = {
    Direction.FORWARD: "forward",
    Direction.BACKWARD: "backward",
    Direction.LEFT: "forward",   # Side tricks: left/right maps to direction
    Direction.RIGHT: "backward",
}

# Body shape mapping
SHAPE_MAP = {
    BodyShape.TUCK: "tuck",
    BodyShape.PIKE: "pike",
    BodyShape.LAYOUT: "layout",
    BodyShape.OPEN: "open",
}

# Entry mapping
ENTRY_MAP = {
    EntryType.STANDING: "standing",
    EntryType.RUNNING: "running",
    EntryType.ONE_LEG: "one_leg",
    EntryType.WALL: "wall",
    EntryType.EDGE: "edge",
}


# ── Match Result ────────────────────────────────────────────────────

@dataclass
class TrickMatch:
    """Result of matching a 3D signature against a TrickDefinition."""
    trick_id: str
    trick_name: str
    confidence: float
    distance: float

    # Per-property distances (for explainability)
    axis_distance: float = 0.0
    rotation_distance: float = 0.0
    twist_distance: float = 0.0
    shape_distance: float = 0.0
    entry_distance: float = 0.0
    direction_distance: float = 0.0

    # What the matcher saw vs what the trick expects
    measured: dict | None = None
    expected: dict | None = None


# ── Matcher ─────────────────────────────────────────────────────────

class Matcher3D:
    """Zero-shot trick matcher using 3D biomechanical signatures.

    Compares a TrickSignature3D against all TrickDefinitions and returns
    ranked matches by weighted multi-axis distance.
    """

    def __init__(
        self,
        trick_definitions: dict[str, TrickDefinition] | None = None,
        # Matching weights (sum to 1.0)
        w_rotation: float = 0.35,   # Rotation count — strongest discriminator
        w_twist: float = 0.30,      # Twist count — second strongest
        w_direction: float = 0.15,  # Forward/backward — critical for trick identity
        w_axis: float = 0.10,       # Rotation axis
        w_shape: float = 0.05,      # Body shape (unreliable from GVHMR)
        w_entry: float = 0.05,      # Entry type (partially detectable)
        # Tolerances: how far off before distance = 1.0
        # 180° = 0.5 rotations → being 0.5 flips/twists off = complete mismatch
        rotation_tolerance_deg: float = 180.0,
        twist_tolerance_deg: float = 180.0,
        axis_tolerance_deg: float = 45.0,
    ):
        self.definitions = trick_definitions or TRICK_DEFINITIONS
        self.w_rotation = w_rotation
        self.w_axis = w_axis
        self.w_twist = w_twist
        self.w_shape = w_shape
        self.w_entry = w_entry
        self.w_direction = w_direction
        self.rotation_tolerance = rotation_tolerance_deg
        self.twist_tolerance = twist_tolerance_deg
        self.axis_tolerance = axis_tolerance_deg

    def match(
        self,
        signature: TrickSignature3D,
        top_k: int = 5,
        min_confidence: float = 0.1,
        context: TrickContext | None = None,
    ) -> list[TrickMatch]:
        """Match a 3D signature against TrickDefinitions.

        Args:
            signature: The measured trick physics.
            top_k: Return this many matches.
            min_confidence: Minimum confidence threshold.
            context: If set, only match tricks from this context (ground/wall/bar).
                     If None, matches all contexts.

        Returns top_k matches sorted by confidence (highest first).
        """
        matches = []

        for trick_id, trick_def in self.definitions.items():
            # Filter by context (Layer 1)
            if context is not None and hasattr(trick_def, "context"):
                if trick_def.context != context:
                    continue

            match = self._compare(signature, trick_id, trick_def)
            if match.confidence >= min_confidence:
                matches.append(match)

        # Sort by confidence (primary). For ties, prefer the most basic trick
        # (lowest FIG score) since without Layer 3 disambiguation we can't tell
        # apart variants in the same physics family. A "Backflip" (D=1.5) is more
        # likely than "Handstand Gainer" (D=4.5) when both match at 99%.
        def sort_key(m: TrickMatch):
            td = self.definitions.get(m.trick_id)
            fig_score = td.fig_score if td and hasattr(td, "fig_score") else 0
            return (-m.confidence, fig_score, m.trick_name)

        matches.sort(key=sort_key)
        return matches[:top_k]

    def _compare(
        self,
        sig: TrickSignature3D,
        trick_id: str,
        trick_def: TrickDefinition,
    ) -> TrickMatch:
        """Compare a signature against a single TrickDefinition."""

        # 1. Rotation axis distance
        expected_axis = AXIS_VECTORS.get(trick_def.rotation_axis, np.array([1, 0, 0]))
        measured_axis = np.asarray(sig.primary_rotation_axis, dtype=np.float64)
        measured_norm = np.linalg.norm(measured_axis)
        if measured_norm > 1e-6:
            measured_axis /= measured_norm

        # Angular distance between axes (0 to 90 degrees, since axis direction is ambiguous)
        cos_angle = np.clip(np.abs(np.dot(measured_axis, expected_axis)), 0, 1)
        axis_angle_diff = np.degrees(np.arccos(cos_angle))
        axis_dist = min(1.0, axis_angle_diff / self.axis_tolerance)

        # 2. Rotation count distance
        measured_flips = abs(sig.total_flip_deg) / 360.0
        expected_flips = trick_def.rotation_count
        rotation_diff = abs(measured_flips - expected_flips) * 360.0
        rotation_dist = min(1.0, rotation_diff / self.rotation_tolerance)

        # 3. Twist count distance
        measured_twists = abs(sig.total_twist_deg) / 360.0
        expected_twists = trick_def.twist_count
        twist_diff = abs(measured_twists - expected_twists) * 360.0
        twist_dist = min(1.0, twist_diff / self.twist_tolerance)

        # 4. Body shape distance
        expected_shape = SHAPE_MAP.get(trick_def.body_shape, "tuck")
        shape_dist = 0.0 if sig.body_shape == expected_shape else 0.5
        # Tuck and pike are more similar to each other than to layout
        if {sig.body_shape, expected_shape} == {"tuck", "pike"}:
            shape_dist = 0.3

        # 5. Entry type distance
        expected_entry = ENTRY_MAP.get(trick_def.entry, "standing")
        entry_dist = 0.0 if sig.entry_type == expected_entry else 0.5
        # Standing and running are somewhat compatible
        if {sig.entry_type, expected_entry} == {"standing", "running"}:
            entry_dist = 0.3

        # 6. Direction distance
        expected_dir = DIRECTION_MAP.get(trick_def.direction, "backward")
        direction_dist = 0.0 if sig.rotation_direction == expected_dir else 1.0

        # Weighted total distance
        total_dist = (
            self.w_axis * axis_dist
            + self.w_rotation * rotation_dist
            + self.w_twist * twist_dist
            + self.w_shape * shape_dist
            + self.w_entry * entry_dist
            + self.w_direction * direction_dist
        )

        confidence = max(0.0, 1.0 - total_dist)

        return TrickMatch(
            trick_id=trick_id,
            trick_name=trick_def.name,
            confidence=confidence,
            distance=total_dist,
            axis_distance=axis_dist,
            rotation_distance=rotation_dist,
            twist_distance=twist_dist,
            shape_distance=shape_dist,
            entry_distance=entry_dist,
            direction_distance=direction_dist,
            measured={
                "flip_count": round(measured_flips, 2),
                "twist_count": round(measured_twists, 2),
                "axis": sig.axis_classification,
                "direction": sig.rotation_direction,
                "body_shape": sig.body_shape,
                "entry": sig.entry_type,
            },
            expected={
                "flip_count": trick_def.rotation_count,
                "twist_count": trick_def.twist_count,
                "axis": trick_def.rotation_axis.value,
                "direction": trick_def.direction.value,
                "body_shape": trick_def.body_shape.value,
                "entry": trick_def.entry.value,
            },
        )
