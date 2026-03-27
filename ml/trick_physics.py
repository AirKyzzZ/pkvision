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


class TrickContext(str, Enum):
    """What apparatus/environment the trick is performed on."""
    GROUND = "ground"           # Acrobatics moves
    WALL = "wall"               # Wall moves
    BAR_OR_RAIL = "bar_or_rail" # Swing moves
    OBSTACLE = "obstacle"       # PK Basics (vaults, climbs)


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

    # FIG Code of Points
    fig_score: float = 0.0       # Base difficulty value from FIG Table of Tricks
    fig_category: str = ""       # "swing", "wall", "acrobatics", "pk_basics"
    context: TrickContext = TrickContext.GROUND

    # Aliases (same trick, different name)
    aliases: list[str] = field(default_factory=list)

    # Layer 3 disambiguation features
    hand_contact: bool | None = None   # Does the trick involve hand contact with ground/wall?
    takeoff_type: str | None = None    # "two_foot", "one_leg", "running_forward", "gainer", etc.
    entry_pattern: str | None = None   # "castaway", "caster", "pop", "kong", "roundoff", etc.
    kick: bool = False                 # Kick during rotation (frisbee, butterfly)

    # Timing (relative)
    takeoff_duration: float = 0.15   # Fraction of total time
    air_duration: float = 0.6        # Fraction of total time
    landing_duration: float = 0.25   # Fraction of total time

    # Physical constraints
    min_height_m: float = 0.5    # Minimum jump height in meters
    typical_height_m: float = 1.0
    typical_duration_s: float = 0.8  # Typical trick duration


# ── FIG Trick Loader ─────────────────────────────────────────────

def load_fig_tricks(json_path: str | None = None) -> dict[str, TrickDefinition]:
    """Load trick definitions from the FIG Code of Points JSON.

    Returns a dict of TrickDefinition objects keyed by trick_id (snake_case name).
    Falls back to legacy hardcoded definitions if JSON not found.
    """
    import json
    from pathlib import Path

    if json_path is None:
        json_path = str(Path(__file__).parent.parent / "data" / "fig_tricks_2025.json")

    path = Path(json_path)
    if not path.exists():
        # Fall back to legacy definitions
        return _legacy_trick_definitions()

    with open(path) as f:
        data = json.load(f)

    axis_map = {
        "lateral": RotationAxis.LATERAL,
        "sagittal": RotationAxis.SAGITTAL,
        "longitudinal": RotationAxis.LONGITUDINAL,
        "off_axis": RotationAxis.OFF_AXIS,
    }
    dir_map = {
        "forward": Direction.FORWARD,
        "backward": Direction.BACKWARD,
        "side": Direction.LEFT,
    }
    shape_map_rev = {
        "tuck": BodyShape.TUCK,
        "pike": BodyShape.PIKE,
        "layout": BodyShape.LAYOUT,
        "open": BodyShape.OPEN,
    }
    context_map = {
        "ground": TrickContext.GROUND,
        "wall": TrickContext.WALL,
        "bar_or_rail": TrickContext.BAR_OR_RAIL,
        "obstacle": TrickContext.OBSTACLE,
    }

    definitions: dict[str, TrickDefinition] = {}

    for cat_key, cat_data in data.get("categories", {}).items():
        context_str = cat_data.get("context", "ground")
        context = context_map.get(context_str, TrickContext.GROUND)

        for trick in cat_data.get("tricks", []):
            name = trick["name"]
            trick_id = name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")

            # Skip tricks with no rotation data (PK basics locomotion)
            axis_str = trick.get("axis")
            dir_str = trick.get("direction")

            axis = axis_map.get(axis_str, RotationAxis.LATERAL) if axis_str else RotationAxis.LATERAL
            direction = dir_map.get(dir_str, Direction.BACKWARD) if dir_str else Direction.BACKWARD

            flip = trick.get("flip", 0)
            twist = trick.get("twist", 0)
            score = trick.get("score", 0)

            # Determine entry type
            entry = EntryType.STANDING
            if context == TrickContext.WALL:
                entry = EntryType.WALL
            takeoff = trick.get("takeoff")
            if takeoff == "one_leg":
                entry = EntryType.ONE_LEG
            elif takeoff in ("running_forward", "gainer"):
                entry = EntryType.RUNNING

            td = TrickDefinition(
                trick_id=trick_id,
                name=name,
                rotation_axis=axis,
                direction=direction,
                rotation_count=flip,
                twist_count=twist,
                body_shape=BodyShape.TUCK,
                entry=entry,
                fig_score=score,
                fig_category=cat_key,
                context=context,
                aliases=trick.get("aliases", []),
                hand_contact=trick.get("hand_contact"),
                takeoff_type=trick.get("takeoff"),
                entry_pattern=trick.get("entry"),
                kick=trick.get("kick", False),
            )
            definitions[trick_id] = td

    return definitions


# Load FIG tricks by default; fall back to legacy if JSON missing
TRICK_DEFINITIONS: dict[str, TrickDefinition] = load_fig_tricks()


# ── Legacy Trick Catalog (fallback) ──────────────────────────────

def _legacy_trick_definitions() -> dict[str, TrickDefinition]:
    return {
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

    # ── Expanded definitions (auto-generated from unified_tricks.json) ──

    # --- Flip tricks ---
    "gumbi": TrickDefinition("gumbi", "Gumbi",
        RotationAxis.LATERAL, Direction.FORWARD, 1.0,
        typical_height_m=0.7),
    "double_aerial_twist": TrickDefinition("double_aerial_twist", "Double Aerial Twist",
        RotationAxis.LONGITUDINAL, Direction.BACKWARD, 2.0,
        typical_height_m=1.1, typical_duration_s=1.2),
    "double_gainer": TrickDefinition("double_gainer", "Double Gainer",
        RotationAxis.LATERAL, Direction.BACKWARD, 2.0,
        entry=EntryType.RUNNING, typical_height_m=1.1, typical_duration_s=1.2),
    "triple_corkscrew": TrickDefinition("triple_corkscrew", "Triple Corkscrew",
        RotationAxis.OFF_AXIS, Direction.BACKWARD, 3.0,
        twist_count=1.0, typical_height_m=1.4, typical_duration_s=1.6),
    "double_corkscrew": TrickDefinition("double_corkscrew", "Double Corkscrew",
        RotationAxis.OFF_AXIS, Direction.BACKWARD, 2.0,
        twist_count=1.0, typical_height_m=1.1, typical_duration_s=1.3),
    "double_arabian": TrickDefinition("double_arabian", "Double Arabian",
        RotationAxis.LATERAL, Direction.FORWARD, 2.0,
        twist_count=0.5, typical_height_m=1.1, typical_duration_s=1.2),
    "helicoptero": TrickDefinition("helicoptero", "Helicoptero",
        RotationAxis.LONGITUDINAL, Direction.BACKWARD, 0.5,
        typical_height_m=0.5),
    "corkscrew_in_back_out": TrickDefinition("corkscrew_in_back_out", "Corkscrew-In Back-Out",
        RotationAxis.OFF_AXIS, Direction.BACKWARD, 1.0,
        twist_count=1.0, typical_height_m=0.7),
    "double_butterfly_twist": TrickDefinition("double_butterfly_twist", "Double Butterfly Twist",
        RotationAxis.LONGITUDINAL, Direction.BACKWARD, 2.0,
        twist_count=1.0, body_shape=BodyShape.LAYOUT, typical_height_m=1.3, typical_duration_s=1.3),
    "triple_aerial_twist": TrickDefinition("triple_aerial_twist", "Triple Aerial Twist",
        RotationAxis.LONGITUDINAL, Direction.BACKWARD, 3.0,
        typical_height_m=1.4, typical_duration_s=1.5),
    "cat_leap": TrickDefinition("cat_leap", "Cat Leap",
        RotationAxis.LATERAL, Direction.BACKWARD, 1.0,
        entry=EntryType.WALL, typical_height_m=0.7),
    "cheat_1440": TrickDefinition("cheat_1440", "Cheat 1440",
        RotationAxis.LONGITUDINAL, Direction.LEFT, 4.0,
        typical_height_m=1.8, typical_duration_s=1.8),
    "cheat_900": TrickDefinition("cheat_900", "Cheat 900",
        RotationAxis.LONGITUDINAL, Direction.LEFT, 2.5,
        typical_height_m=1.3, typical_duration_s=1.4),
    "quadruple_corkscrew": TrickDefinition("quadruple_corkscrew", "Quadruple Corkscrew",
        RotationAxis.OFF_AXIS, Direction.BACKWARD, 1.0,
        twist_count=1.0, typical_height_m=0.7),
    "kip_up": TrickDefinition("kip_up", "Kip-Up",
        RotationAxis.LATERAL, Direction.FORWARD, 1.0,
        typical_height_m=0.7),
    "540_kick": TrickDefinition("540_kick", "540",
        RotationAxis.LONGITUDINAL, Direction.BACKWARD, 1.5,
        typical_height_m=0.9),
    "webster_half": TrickDefinition("webster_half", "Webster Half",
        RotationAxis.LATERAL, Direction.FORWARD, 0.5,
        entry=EntryType.ONE_LEG, typical_height_m=0.5),
    "butterfly_kick": TrickDefinition("butterfly_kick", "Butterfly Kick",
        RotationAxis.LONGITUDINAL, Direction.BACKWARD, 0.5,
        typical_height_m=0.5),
    "butterfly_twist": TrickDefinition("butterfly_twist", "Butterfly Twist",
        RotationAxis.LONGITUDINAL, Direction.BACKWARD, 1.0,
        twist_count=1.0, typical_height_m=0.7),
    "aerial": TrickDefinition("aerial", "Aerial",
        RotationAxis.SAGITTAL, Direction.BACKWARD, 1.0,
        typical_height_m=0.7),
    "aerial_twist": TrickDefinition("aerial_twist", "Aerial Twist",
        RotationAxis.LONGITUDINAL, Direction.FORWARD, 1.0,
        typical_height_m=0.7),
    "corkscrew": TrickDefinition("corkscrew", "Corkscrew",
        RotationAxis.OFF_AXIS, Direction.BACKWARD, 1.0,
        twist_count=1.0, typical_height_m=0.7),
    "arabian": TrickDefinition("arabian", "Arabian",
        RotationAxis.LATERAL, Direction.FORWARD, 1.0,
        twist_count=0.5, typical_height_m=0.7),
    "gainer_full": TrickDefinition("gainer_full", "Gainer Full",
        RotationAxis.LATERAL, Direction.BACKWARD, 1.0,
        twist_count=1.0, entry=EntryType.RUNNING, typical_height_m=0.7),
    "gainer_arabian": TrickDefinition("gainer_arabian", "Gainer Arabian",
        RotationAxis.LATERAL, Direction.FORWARD, 1.0,
        twist_count=0.5, entry=EntryType.RUNNING, typical_height_m=0.7),
    # double_back_flip → use double_back (above)
    # double_front_flip → use double_front (above)
    "back_layout": TrickDefinition("back_layout", "Back Layout",
        RotationAxis.LATERAL, Direction.BACKWARD, 1.0,
        body_shape=BodyShape.LAYOUT, typical_height_m=0.9),
    "double_layout": TrickDefinition("double_layout", "Double Layout",
        RotationAxis.LATERAL, Direction.BACKWARD, 2.0,
        body_shape=BodyShape.LAYOUT, typical_height_m=1.3, typical_duration_s=1.2),
    "front_double_full": TrickDefinition("front_double_full", "Front Double Full",
        RotationAxis.LATERAL, Direction.FORWARD, 1.0,
        twist_count=2.0, body_shape=BodyShape.LAYOUT, typical_height_m=0.9, typical_duration_s=1.1),
    "double_side_flip": TrickDefinition("double_side_flip", "Double Side Flip",
        RotationAxis.LATERAL, Direction.LEFT, 2.0,
        typical_height_m=1.1, typical_duration_s=1.2),
    "inward_side": TrickDefinition("inward_side", "Inward Side",
        RotationAxis.SAGITTAL, Direction.LEFT, 1.0,
        typical_height_m=0.7),
    "rudi": TrickDefinition("rudi", "Rudi",
        RotationAxis.LATERAL, Direction.FORWARD, 1.0,
        twist_count=0.5, typical_height_m=0.7),
    "front_half": TrickDefinition("front_half", "Front Half",
        RotationAxis.LATERAL, Direction.FORWARD, 0.5,
        twist_count=0.5, body_shape=BodyShape.LAYOUT, typical_height_m=0.7),
    "flash_kick": TrickDefinition("flash_kick", "Flash Kick",
        RotationAxis.LATERAL, Direction.BACKWARD, 1.0,
        typical_height_m=0.7),
    "touchdown_raiz": TrickDefinition("touchdown_raiz", "Touchdown Raiz",
        RotationAxis.OFF_AXIS, Direction.BACKWARD, 0.5,
        twist_count=0.5, typical_height_m=0.5),
    "boxcutter": TrickDefinition("boxcutter", "Boxcutter",
        RotationAxis.OFF_AXIS, Direction.BACKWARD, 1.0,
        typical_height_m=0.7),
    "sideswipe": TrickDefinition("sideswipe", "Sideswipe",
        RotationAxis.LATERAL, Direction.BACKWARD, 1.5,
        typical_height_m=0.9),
    "masterswipe": TrickDefinition("masterswipe", "Masterswipe",
        RotationAxis.LATERAL, Direction.BACKWARD, 1.0,
        typical_height_m=0.7),
    "pimp_flip": TrickDefinition("pimp_flip", "Pimp Flip",
        RotationAxis.LATERAL, Direction.BACKWARD, 1.0,
        typical_height_m=0.7),
    "gainer_switch": TrickDefinition("gainer_switch", "Gainer Switch",
        RotationAxis.LATERAL, Direction.BACKWARD, 1.0,
        entry=EntryType.RUNNING, typical_height_m=0.7),
    "slant_gainer": TrickDefinition("slant_gainer", "Slant Gainer",
        RotationAxis.LATERAL, Direction.BACKWARD, 1.0,
        entry=EntryType.RUNNING, typical_height_m=0.7),
    "palm_flip": TrickDefinition("palm_flip", "Palm Flip",
        RotationAxis.LATERAL, Direction.BACKWARD, 1.0,
        entry=EntryType.WALL, typical_height_m=0.7),
    "tic_tac": TrickDefinition("tic_tac", "Tic Tac",
        RotationAxis.LATERAL, Direction.FORWARD, 1.0,
        entry=EntryType.WALL, typical_height_m=0.7),

    # --- Bar tricks ---
    "lache": TrickDefinition("lache", "Lache",
        RotationAxis.LATERAL, Direction.BACKWARD, 1.0,
        entry=EntryType.EDGE, typical_height_m=0.7),
    "flyaway_full": TrickDefinition("flyaway_full", "Flyaway Full",
        RotationAxis.LONGITUDINAL, Direction.BACKWARD, 1.0,
        twist_count=1.0, entry=EntryType.EDGE, typical_height_m=0.7),

    # --- Vault tricks ---
    "speed_vault": TrickDefinition("speed_vault", "Speed Vault",
        RotationAxis.LATERAL, Direction.FORWARD, 1.0,
        entry=EntryType.RUNNING, typical_height_m=0.7),
    "dash_vault": TrickDefinition("dash_vault", "Dash Vault",
        RotationAxis.LATERAL, Direction.FORWARD, 1.0,
        entry=EntryType.RUNNING, typical_height_m=0.7),
    "kong_vault": TrickDefinition("kong_vault", "Kong Vault",
        RotationAxis.LATERAL, Direction.BACKWARD, 1.0,
        entry=EntryType.RUNNING, typical_height_m=0.7),
    "lazy_vault": TrickDefinition("lazy_vault", "Lazy Vault",
        RotationAxis.LATERAL, Direction.BACKWARD, 1.0,
        entry=EntryType.RUNNING, typical_height_m=0.7),
    "reverse_vault": TrickDefinition("reverse_vault", "Reverse Vault",
        RotationAxis.LATERAL, Direction.BACKWARD, 1.0,
        entry=EntryType.RUNNING, typical_height_m=0.7),

    # --- Ground tricks ---
    "cartwheel": TrickDefinition("cartwheel", "Cartwheel",
        RotationAxis.LATERAL, Direction.LEFT, 1.0,
        typical_height_m=0.7),
    "roundoff": TrickDefinition("roundoff", "Roundoff",
        RotationAxis.LATERAL, Direction.BACKWARD, 1.0,
        typical_height_m=0.7),
    "macaco": TrickDefinition("macaco", "Macaco",
        RotationAxis.LATERAL, Direction.BACKWARD, 1.0,
        typical_height_m=0.7),
    "back_handspring": TrickDefinition("back_handspring", "Back Handspring",
        RotationAxis.LATERAL, Direction.BACKWARD, 1.0,
        typical_height_m=0.7),
    "front_handspring": TrickDefinition("front_handspring", "Front Handspring",
        RotationAxis.LATERAL, Direction.FORWARD, 1.0,
        typical_height_m=0.7),
    "handstand": TrickDefinition("handstand", "Handstand",
        RotationAxis.LATERAL, Direction.BACKWARD, 1.0,
        body_shape=BodyShape.LAYOUT, typical_height_m=0.9),

    # --- Roll tricks ---
    "dive_roll": TrickDefinition("dive_roll", "Dive Roll",
        RotationAxis.LATERAL, Direction.FORWARD, 1.0,
        typical_height_m=0.7),
    "backward_roll": TrickDefinition("backward_roll", "Backward Roll",
        RotationAxis.LATERAL, Direction.BACKWARD, 1.0,
        typical_height_m=0.7),

    # --- Wall tricks ---
    "wall_spin": TrickDefinition("wall_spin", "Wall Spin",
        RotationAxis.LONGITUDINAL, Direction.BACKWARD, 1.0,
        entry=EntryType.WALL, typical_height_m=0.7),
    "wall_full": TrickDefinition("wall_full", "Wall Full",
        RotationAxis.LATERAL, Direction.BACKWARD, 1.0,
        twist_count=1.0, entry=EntryType.WALL, typical_height_m=0.7),
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
