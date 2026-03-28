"""Gravity-referenced rotation tracking for trick recognition.

Replaces the old swing-twist frame-by-frame decomposition with a continuous
tracking approach that correctly handles multi-rotation and off-axis tricks.

FLIP = how many times the body goes inverted relative to gravity.
       Tracked by monitoring the body's Y-axis (head direction) relative to world Y (up).
       Each 360° of tilt change = 1 complete flip.

TWIST = how much the body rotates around its own spine axis.
        Tracked via frame-to-frame axis projection, corrected for Berry phase
        (geometric phase from the body_y spiral on the unit sphere).
        Each 360° of axial rotation = 1 full twist.

OFF-AXIS ANGLE = the tilt of the rotation axis from vertical.
                 0° = pure twist (spinning upright like a top)
                 90° = pure flip (rotating perpendicular to gravity)
                 ~45° = cork (off-axis rotation)

GVHMR compensation:
  GVHMR is trained on AMASS which has no acrobatic motions. This causes:
  - Tilt underestimation at extreme angles (max ~158° instead of 180°)
  - Frame-to-frame rotation jitter
  The tracker compensates via:
  - Quaternion smoothing of rotations before tracking
  - Berry phase correction to fix twist overcounting during simultaneous flip+twist
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation


def smooth_rotations(global_orient: np.ndarray, sigma: float = 1.5) -> np.ndarray:
    """Smooth GVHMR rotation trajectory via quaternion-space Gaussian filter.

    GVHMR produces frame-to-frame jitter that inflates total rotation by ~10-20%.
    Smoothing in quaternion space (with sign consistency) reduces this while
    preserving the overall rotation trajectory.

    Args:
        global_orient: (T, 3) axis-angle rotations.
        sigma: Gaussian kernel width in frames. 1.5 is mild smoothing.

    Returns:
        (T, 3) smoothed axis-angle rotations.
    """
    if sigma <= 0:
        return global_orient

    T = global_orient.shape[0]
    quats = np.zeros((T, 4))
    for t in range(T):
        quats[t] = Rotation.from_rotvec(global_orient[t]).as_quat()

    # Ensure consistent quaternion sign (q and -q are the same rotation)
    for t in range(1, T):
        if np.dot(quats[t], quats[t - 1]) < 0:
            quats[t] = -quats[t]

    # Smooth each component
    smoothed = np.zeros_like(quats)
    for c in range(4):
        smoothed[:, c] = gaussian_filter1d(quats[:, c], sigma=sigma)

    # Renormalize
    norms = np.linalg.norm(smoothed, axis=1, keepdims=True)
    smoothed /= norms

    # Back to axis-angle
    result = np.zeros_like(global_orient)
    for t in range(T):
        result[t] = Rotation.from_quat(smoothed[t]).as_rotvec()
    return result


def _compute_geometric_phase(body_y: np.ndarray, start: int, end: int) -> float:
    """Compute Berry phase (solid angle) swept by body_y on the unit sphere.

    When body_y traces a spiral during simultaneous flip+twist, the
    frame-to-frame axis projection accumulates a geometric phase (holonomy)
    that overcounts twist. This function computes that phase so it can be
    subtracted.

    Uses the spherical triangle formula:
      Ω_i = 2·atan2(n·(v_i × v_{i+1}), 1 + n·v_i + n·v_{i+1} + v_i·v_{i+1})
    where n = north pole [0,1,0].

    Args:
        body_y: (T, 3) body Y direction at each frame.
        start, end: frame range for the trick segment.

    Returns:
        Geometric phase in degrees.
    """
    n = np.array([0.0, 1.0, 0.0])
    total = 0.0

    for t in range(start + 1, end + 1):
        v0 = body_y[t - 1]
        v1 = body_y[t]

        triple = np.dot(n, np.cross(v0, v1))
        denom = 1.0 + np.dot(n, v0) + np.dot(n, v1) + np.dot(v0, v1)

        if abs(denom) > 1e-10:
            total += 2.0 * np.arctan2(triple, denom)

    return np.degrees(total)


def track_rotation(global_orient: np.ndarray) -> dict:
    """Track flip and twist rotation continuously from SMPL global_orient.

    Args:
        global_orient: (T, 3) axis-angle rotation per frame from GVHMR.

    Returns:
        Dict with per-frame arrays:
        - tilt_angle: (T,) body tilt from upright in degrees (0=upright, 180=inverted)
        - tilt_cumulative: (T,) cumulative tilt change in degrees (tracks flip progress)
        - twist_cumulative: (T,) cumulative twist in degrees (raw axis projection)
        - twist_angle: (T,) alias for twist_cumulative
        - body_y: (T, 3) body's up-direction in world frame
        - rotation_rate: (T,) total rotation speed in deg/frame
        - inversion_crossings: int count of 90° tilt crossings
    """
    T = global_orient.shape[0]

    tilt_angle = np.zeros(T)
    tilt_cumulative = np.zeros(T)
    twist_cumulative = np.zeros(T)
    body_y_arr = np.zeros((T, 3))
    rotation_rate = np.zeros(T)
    inversion_crossings = 0

    world_y = np.array([0.0, 1.0, 0.0])

    for t in range(T):
        R_body = Rotation.from_rotvec(global_orient[t])
        body_y = R_body.apply([0.0, 1.0, 0.0])
        body_y_arr[t] = body_y

        cos_tilt = np.clip(np.dot(body_y, world_y), -1.0, 1.0)
        tilt_angle[t] = np.degrees(np.arccos(cos_tilt))

        if t > 0:
            R_prev = Rotation.from_rotvec(global_orient[t - 1])
            R_delta = R_body * R_prev.inv()
            delta_rotvec = R_delta.as_rotvec()
            delta_deg = np.degrees(np.linalg.norm(delta_rotvec))

            if delta_deg > 0.01:
                delta_axis = delta_rotvec / np.radians(delta_deg)
                body_y_prev = R_prev.apply([0.0, 1.0, 0.0])
                twist_component = np.dot(delta_axis, body_y_prev) * delta_deg
                twist_cumulative[t] = twist_cumulative[t - 1] + twist_component
            else:
                twist_cumulative[t] = twist_cumulative[t - 1]

            dt = tilt_angle[t] - tilt_angle[t - 1]
            tilt_cumulative[t] = tilt_cumulative[t - 1] + abs(dt)
            rotation_rate[t] = delta_deg

            # Inversion crossing detection
            if (tilt_angle[t - 1] < 90 and tilt_angle[t] >= 90) or \
               (tilt_angle[t - 1] >= 90 and tilt_angle[t] < 90):
                inversion_crossings += 1

    return {
        "tilt_angle": tilt_angle,
        "tilt_cumulative": tilt_cumulative,
        "twist_angle": twist_cumulative.copy(),
        "twist_cumulative": twist_cumulative,
        "body_y": body_y_arr,
        "rotation_rate": rotation_rate,
        "inversion_crossings": inversion_crossings,
    }


def extract_trick_physics(
    tracking: dict,
    start: int,
    end: int,
    global_orient: np.ndarray | None = None,
    body_pose: np.ndarray | None = None,
    transl: np.ndarray | None = None,
) -> dict:
    """Extract physics signature for a single trick segment.

    Applies Berry phase correction to fix twist overcounting during
    simultaneous flip+twist (caused by geometric phase accumulation
    in the frame-by-frame axis projection).

    Args:
        tracking: Output from track_rotation()
        start, end: Frame indices for this trick
        global_orient: (T, 3) for direction detection
        body_pose: (T, 21/23, 3) for body shape detection
        transl: (T, 3) for entry type detection

    Returns:
        Dict with trick physics: flip_count, twist_count, axis, direction, body_shape, entry
    """
    tilt = tracking["tilt_angle"][start:end + 1]
    tilt_cum = tracking["tilt_cumulative"]
    twist_cum = tracking["twist_cumulative"]
    body_y = tracking["body_y"]

    # ── FLIP COUNT ──
    total_tilt_change = tilt_cum[end] - tilt_cum[start]
    flip_count = total_tilt_change / 360.0

    # ── QUALITY-BASED FLIP COMPENSATION ──
    # On wide-angle footage, GVHMR underestimates tilt (flip).
    # Use tilt-rate (not total rotation rate which includes twist) to detect this.
    seg_tilt_full = tracking["tilt_angle"][start:end + 1]
    tilt_deltas = np.abs(np.diff(seg_tilt_full))
    tilt_rate_sum = float(np.sum(tilt_deltas))
    # tilt_rate_sum should equal total_tilt_change (both are abs sums of tilt deltas)
    # They can diverge if smoothing was applied at different stages

    # ── RAW TWIST ──
    raw_twist = twist_cum[end] - twist_cum[start]
    raw_twist_abs = abs(raw_twist)

    # ── BERRY PHASE CORRECTION ──
    # The axis-projection method overcounts twist during simultaneous flip+twist
    # because body_y traces a spiral on the unit sphere, accumulating geometric phase.
    # The correction = solid angle swept by body_y (computed via spherical triangles).
    # Use magnitude subtraction: if geo_phase > raw_twist, real twist is ~0.
    geo_phase = _compute_geometric_phase(body_y, start, end)
    corrected_twist = max(0.0, raw_twist_abs - abs(geo_phase))
    twist_count = corrected_twist / 360.0

    # ── WENT INVERTED? ──
    max_tilt = float(np.max(tilt))
    went_inverted = max_tilt > 120

    # ── ROTATION AXIS ──
    # Use rotation rate to find peak frames (actual trick vs standing)
    seg_rate = tracking["rotation_rate"][start:end + 1]
    rate_threshold = max(np.median(seg_rate), 5.0)
    peak_mask = seg_rate > rate_threshold
    peak_tilt = tilt[peak_mask] if np.any(peak_mask) else tilt

    if flip_count < 0.2 and twist_count > 0.5:
        axis = "longitudinal"
        off_axis_angle = 0.0
    elif went_inverted and twist_count < 0.3:
        axis, off_axis_angle = _detect_flip_axis(
            tilt, global_orient, start, end
        )
    elif went_inverted and twist_count > 0.3:
        # Distinguish off-axis (cork) from lateral flip with twist (full/double full).
        # Cork: body maintains a tilted angle (~40-70°) throughout the rotation.
        # Full: body goes fully inverted (tilt peaks near 160-180°) then returns.
        # Use peak-frame tilt statistics (ignoring standing phases).
        peak_mean_tilt = float(np.mean(peak_tilt))
        peak_median_tilt = float(np.median(peak_tilt))

        # Cork indicator: median tilt during rotation is moderate (30-80°)
        # AND max tilt is below 155° (GVHMR corks don't go as inverted as laterals)
        is_cork = (30 < peak_median_tilt < 80 and max_tilt < 155)

        if is_cork:
            axis = "off_axis"
            off_axis_angle = peak_mean_tilt
        else:
            axis = "lateral"
            off_axis_angle = 90.0
    else:
        axis = "lateral"
        off_axis_angle = 90.0

    # ── DIRECTION ──
    direction = _detect_direction(tilt, global_orient, start, end, went_inverted)

    # ── BODY SHAPE ──
    body_shape = _detect_body_shape(tilt, body_pose, start)

    # ── ENTRY TYPE ──
    entry = _detect_entry(transl, start, end)

    # ── SNAP TO HALF-INTEGER ──
    # GVHMR underestimates tilt by ~10% (max ~160° instead of 180°), causing
    # flip counts to read ~0.9 instead of 1.0. Snap to nearest 0.5 if within
    # tolerance. This is appropriate because tricks always have integer or
    # half-integer flips/twists (0.5, 1.0, 1.5, 2.0, etc.).
    snap_tolerance = 0.20  # Within 20% of a half-integer -> snap
    flip_snapped = _snap_to_half(flip_count, snap_tolerance)
    twist_snapped = _snap_to_half(twist_count, snap_tolerance)

    return {
        "flip_count": round(flip_snapped, 1),
        "twist_count": round(twist_snapped, 1),
        "flip_count_raw": round(flip_count, 2),
        "twist_count_raw": round(twist_count, 2),
        "flip_deg": round(total_tilt_change, 1),
        "twist_deg": round(corrected_twist, 1),
        "twist_deg_raw": round(raw_twist_abs, 1),
        "geometric_phase_deg": round(abs(geo_phase), 1),
        "axis": axis,
        "off_axis_angle": round(off_axis_angle, 1),
        "direction": direction,
        "body_shape": body_shape,
        "entry": entry,
        "max_tilt": round(max_tilt, 1),
        "went_inverted": went_inverted,
    }


def _snap_to_half(value: float, tolerance: float = 0.15) -> float:
    """Snap a value to the nearest 0.5 if within tolerance.

    Tricks always have integer or half-integer flip/twist counts.
    GVHMR tilt underestimation causes ~10% error that this corrects.
    """
    nearest_half = round(value * 2) / 2  # Round to nearest 0.5
    if abs(value - nearest_half) <= tolerance:
        return nearest_half
    return value


# ── Private helpers ──────────────────────────────────────────────────


def _detect_flip_axis(
    tilt: np.ndarray,
    global_orient: np.ndarray | None,
    start: int,
    end: int,
) -> tuple[str, float]:
    """Detect whether a pure flip is lateral (front/back) or sagittal (side)."""
    if global_orient is None:
        return "lateral", 90.0

    R_start = Rotation.from_rotvec(global_orient[start])

    target_tilt = 45
    best_frame = start
    best_diff = 999.0
    for i in range(len(tilt)):
        diff = abs(tilt[i] - target_tilt)
        if diff < best_diff:
            best_diff = diff
            best_frame = start + i
    R_mid = Rotation.from_rotvec(global_orient[min(best_frame, end)])

    by_start = R_start.apply([0.0, 1.0, 0.0])
    by_mid = R_mid.apply([0.0, 1.0, 0.0])
    tilt_axis = np.cross(by_start, by_mid)
    tilt_norm = np.linalg.norm(tilt_axis)

    if tilt_norm > 0.1:
        tilt_axis /= tilt_norm
        tilt_in_body = R_start.inv().apply(tilt_axis)
        x_component = abs(tilt_in_body[0])
        z_component = abs(tilt_in_body[2])
        axis = "lateral" if x_component > z_component else "sagittal"
    else:
        axis = "lateral"

    return axis, 90.0


def _detect_direction(
    tilt: np.ndarray,
    global_orient: np.ndarray | None,
    start: int,
    end: int,
    went_inverted: bool,
) -> str:
    """Detect flip direction (forward/backward).

    Uses the body's facing direction (Z-axis) at an early tilt phase (30-50°)
    compared to the start. This avoids the 90°-crossing method which fails
    when twist rotates body_y before the crossing point.

    - Backward flip: body leans back → body_z goes UP (Y increases)
    - Forward flip: body leans forward → body_z goes DOWN (Y decreases)
    """
    if global_orient is None or not went_inverted:
        return "backward"

    R_start = Rotation.from_rotvec(global_orient[start])
    body_z_start = R_start.apply([0.0, 0.0, 1.0])

    # Find the most upright frame near the start as reference
    ref_frame = start
    min_tilt = tilt[0]
    for i in range(min(len(tilt), int(len(tilt) * 0.3))):
        if tilt[i] < min_tilt:
            min_tilt = tilt[i]
            ref_frame = start + i

    R_ref = Rotation.from_rotvec(global_orient[ref_frame])
    body_y_ref = R_ref.apply([0.0, 1.0, 0.0])
    body_z_ref = R_ref.apply([0.0, 0.0, 1.0])

    # Check which direction body_y (head) moves relative to body_z (facing).
    # Backward flip: head moves OPPOSITE to facing → dot(body_y_delta, body_z_ref) < 0
    # Forward flip: head moves SAME as facing → dot(body_y_delta, body_z_ref) > 0
    # This is robust to twist because it uses the INITIAL facing direction
    # and only needs ~10° of tilt change to determine direction.
    # Start searching AFTER the reference frame (not before, which is the approach).
    # Use the cross product of body_y vectors to find the flip axis, then check
    # its alignment with body_x to determine direction. This is robust to the
    # body's absolute facing direction.
    search_start = ref_frame - start + 1
    for i in range(search_start, len(tilt)):
        frame = start + i
        if frame > end:
            break
        tilt_increase = tilt[i] - min_tilt
        if tilt_increase > 25 and tilt[i] < 90:
            R_f = Rotation.from_rotvec(global_orient[frame])
            body_y_f = R_f.apply([0.0, 1.0, 0.0])

            # Flip axis = cross(body_y_ref, body_y_tilted)
            flip_axis = np.cross(body_y_ref, body_y_f)
            flip_norm = np.linalg.norm(flip_axis)
            if flip_norm < 0.01:
                continue
            flip_axis /= flip_norm

            # Project flip axis into body frame at reference
            flip_in_body = R_ref.inv().apply(flip_axis)

            # In body frame: flip axis along +X = forward, along -X = backward
            # (head goes in the direction of cross product, which for +X means forward)
            if abs(flip_in_body[0]) > abs(flip_in_body[2]):
                # Lateral flip
                return "forward" if flip_in_body[0] > 0 else "backward"
            else:
                # Sagittal flip (side) — use Z component
                return "forward" if flip_in_body[2] > 0 else "backward"

    return "backward"


def _detect_body_shape(
    tilt: np.ndarray,
    body_pose: np.ndarray | None,
    start: int,
) -> str:
    """Classify body shape from SMPL joint angles during peak tilt."""
    if body_pose is None:
        return "tuck"

    tilt_threshold = np.percentile(tilt, 70)
    peak_mask = tilt > max(tilt_threshold, 30)
    if not np.any(peak_mask):
        return "tuck"

    peak_indices = np.where(peak_mask)[0] + start
    knee_angles = []
    hip_angles = []
    for idx in peak_indices[::2]:
        if idx < body_pose.shape[0]:
            l_knee = np.degrees(np.linalg.norm(body_pose[idx, 4]))
            r_knee = np.degrees(np.linalg.norm(body_pose[idx, 5]))
            l_hip = np.degrees(np.linalg.norm(body_pose[idx, 1]))
            r_hip = np.degrees(np.linalg.norm(body_pose[idx, 2]))
            knee_angles.append(max(l_knee, r_knee))
            hip_angles.append(max(l_hip, r_hip))

    if not knee_angles:
        return "tuck"

    avg_knee = np.mean(knee_angles)
    avg_hip = np.mean(hip_angles)

    if avg_knee > 45 and avg_hip > 30:
        return "tuck"
    elif avg_knee < 35 and avg_hip > 30:
        return "pike"
    elif avg_knee < 45 and avg_hip < 25:
        return "layout"
    else:
        bend = (avg_knee + avg_hip) / 2
        return "tuck" if bend > 40 else "layout"


def _detect_entry(transl: np.ndarray | None, start: int, end: int) -> str:
    """Detect trick entry type (standing/running/wall)."""
    if transl is None or start < 5:
        return "standing"

    pre_trick = transl[max(0, start - 10):start]
    if len(pre_trick) > 3:
        horiz_vel = np.linalg.norm(np.diff(pre_trick[:, [0, 2]], axis=0), axis=1)
        if np.mean(horiz_vel) > 0.05:
            return "running"

    com_y_seg = transl[start:end + 1, 1]
    com_y_pre = transl[max(0, start - 15):start, 1]
    if len(com_y_pre) > 3:
        height_change = com_y_seg[0] - np.mean(com_y_pre)
        if height_change > 0.3:
            return "wall"

    return "standing"
