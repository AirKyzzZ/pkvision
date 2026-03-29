"""Motion Signature Matching — trick recognition from 2D keypoint trajectories.

Bypasses SMPL/3D reconstruction entirely. Uses ViTPose 2D keypoint trajectories
(17 COCO joints over time) to match tricks against reference clips.

Pipeline:
  1. Extract 2D keypoint trajectory from a video segment
  2. Normalize (center, scale, speed)
  3. Compare against reference trajectories using DTW
  4. Combine with physics hints (rotation rate, airtime) for confidence

Only needs 1 reference clip per trick — no training data required.

Usage:
    db = SignatureDatabase()
    db.add_reference("backflip", vitpose_data, fps=30)
    matches = db.match(query_segment, fps=30, top_k=3)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.distance import cdist


# COCO-17 keypoint indices
NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 1, 2, 3, 4
L_SHOULDER, R_SHOULDER = 5, 6
L_ELBOW, R_ELBOW = 7, 8
L_WRIST, R_WRIST = 9, 10
L_HIP, R_HIP = 11, 12
L_KNEE, R_KNEE = 13, 14
L_ANKLE, R_ANKLE = 15, 16

# Key joints for trick recognition (skip face keypoints — too noisy)
BODY_JOINTS = [L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW, L_WRIST, R_WRIST,
               L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANKLE, R_ANKLE]

# Joint pairs for computing relative angles (view-invariant features)
LIMB_PAIRS = [
    (L_SHOULDER, L_ELBOW), (L_ELBOW, L_WRIST),  # left arm
    (R_SHOULDER, R_ELBOW), (R_ELBOW, R_WRIST),  # right arm
    (L_HIP, L_KNEE), (L_KNEE, L_ANKLE),          # left leg
    (R_HIP, R_KNEE), (R_KNEE, R_ANKLE),          # right leg
    (L_SHOULDER, L_HIP), (R_SHOULDER, R_HIP),    # torso sides
    (L_SHOULDER, R_SHOULDER),                      # shoulder width
    (L_HIP, R_HIP),                                # hip width
]


@dataclass
class MotionSignature:
    """Normalized motion signature for a trick."""
    name: str
    # Normalized trajectory: (T_norm, N_features)
    trajectory: np.ndarray
    # Raw features for physics hints
    duration_s: float = 0.0
    max_height_change: float = 0.0  # normalized
    rotation_speed: float = 0.0  # avg angular change of body axis per frame
    went_inverted: bool = False
    num_inversions: int = 0


@dataclass
class SignatureMatch:
    """Result of matching a query against the database."""
    name: str
    distance: float
    confidence: float
    physics_bonus: float = 0.0


class SignatureDatabase:
    """Database of reference motion signatures for trick matching."""

    def __init__(self, norm_length: int = 60):
        """
        Args:
            norm_length: All trajectories are resampled to this length for comparison.
                         60 frames is enough to capture the motion pattern.
        """
        self.norm_length = norm_length
        self.references: list[MotionSignature] = []

    def add_reference(
        self,
        name: str,
        vitpose: np.ndarray,
        fps: float = 30.0,
        trim_start: float = 0.0,
        trim_end: float = 0.0,
    ):
        """Add a reference trick clip to the database.

        Args:
            name: Trick name (e.g., "backflip", "double_cork")
            vitpose: (T, 17, 3) ViTPose keypoints [x, y, confidence]
            fps: Video frame rate
            trim_start: Seconds to trim from start
            trim_end: Seconds to trim from end
        """
        # Trim
        start_f = int(trim_start * fps)
        end_f = len(vitpose) - int(trim_end * fps)
        if end_f <= start_f:
            end_f = len(vitpose)
        kps = vitpose[start_f:end_f].copy()

        # Extract features and normalize
        trajectory = self._extract_features(kps)
        trajectory_norm = self._normalize_time(trajectory)

        # Physics hints
        duration = len(kps) / fps
        body_axis = self._compute_body_axis(kps)
        height_change = self._compute_height_change(kps)
        rot_speed = self._compute_rotation_speed(body_axis)
        inverted, n_inv = self._detect_inversion(body_axis)

        sig = MotionSignature(
            name=name,
            trajectory=trajectory_norm,
            duration_s=duration,
            max_height_change=height_change,
            rotation_speed=rot_speed,
            went_inverted=inverted,
            num_inversions=n_inv,
        )
        self.references.append(sig)

    def match(
        self,
        vitpose_segment: np.ndarray,
        fps: float = 30.0,
        top_k: int = 5,
    ) -> list[SignatureMatch]:
        """Match a query segment against all references.

        Args:
            vitpose_segment: (T, 17, 3) keypoints for the query segment
            fps: Frame rate
            top_k: Return top K matches

        Returns:
            List of SignatureMatch sorted by confidence (highest first)
        """
        if len(self.references) == 0:
            return []

        # Extract and normalize query features
        query_feat = self._extract_features(vitpose_segment)
        query_norm = self._normalize_time(query_feat)

        # Query physics hints
        body_axis = self._compute_body_axis(vitpose_segment)
        q_duration = len(vitpose_segment) / fps
        q_height = self._compute_height_change(vitpose_segment)
        q_rot_speed = self._compute_rotation_speed(body_axis)
        q_inverted, q_n_inv = self._detect_inversion(body_axis)

        # First pass: compute all DTW distances for normalization
        raw_dists = []
        for ref in self.references:
            d = self._trajectory_distance(query_norm, ref.trajectory)
            raw_dists.append(d)

        # Normalize distances to [0, 1] using min-max relative to this query
        min_dist = min(raw_dists) if raw_dists else 0
        max_dist = max(raw_dists) if raw_dists else 1
        dist_range = max(max_dist - min_dist, 0.01)

        matches = []
        for idx, ref in enumerate(self.references):
            traj_dist = raw_dists[idx]
            # Normalize: best match = 0.0, worst = 1.0
            norm_dist = (traj_dist - min_dist) / dist_range

            # Physics similarity bonus
            physics_bonus = 0.0

            # Duration similarity
            dur_ratio = min(q_duration, ref.duration_s) / max(q_duration, ref.duration_s, 0.1)
            physics_bonus += 0.1 * dur_ratio

            # Inversion match
            if q_inverted == ref.went_inverted:
                physics_bonus += 0.15
            if q_n_inv == ref.num_inversions:
                physics_bonus += 0.1

            # Rotation speed similarity
            speed_ratio = min(q_rot_speed, ref.rotation_speed) / max(q_rot_speed, ref.rotation_speed, 0.01)
            physics_bonus += 0.1 * speed_ratio

            # Height change similarity
            height_ratio = min(q_height, ref.max_height_change) / max(q_height, ref.max_height_change, 0.01)
            physics_bonus += 0.05 * height_ratio

            # Combined confidence: trajectory similarity (primary) + physics hints (secondary)
            traj_similarity = max(0.0, 1.0 - norm_dist)
            confidence = traj_similarity * 0.7 + physics_bonus * 0.6
            confidence = min(1.0, max(0.0, confidence))

            matches.append(SignatureMatch(
                name=ref.name,
                distance=round(traj_dist, 4),
                confidence=round(confidence, 3),
                physics_bonus=round(physics_bonus, 3),
            ))

        matches.sort(key=lambda m: -m.confidence)
        return matches[:top_k]

    # ── Feature Extraction ─────────────────────────────────────

    def _extract_features(self, kps: np.ndarray) -> np.ndarray:
        """Extract VIEW-INVARIANT features from keypoint sequence.

        All features are based on inter-joint DISTANCES and ANGLES relative
        to the body's own axis — NOT positions relative to the image frame.
        This means a backflip looks the same whether filmed from the side,
        front, or at an angle.

        Features per frame (26 total):
        - 10 inter-joint distances (normalized by torso length)
        - 12 limb angles relative to body axis (not image vertical)
        - 1 body compactness (how tucked)
        - 1 body extension (how stretched)
        - 1 symmetry (left-right similarity)
        - 1 body axis angle relative to image vertical (only useful metric that IS view-dependent but captures inversion)

        Returns: (T, 26) array
        """
        T = len(kps)
        features = []

        for t in range(T):
            frame_kps = kps[t, :, :2]  # (17, 2) xy

            # Compute torso length for normalization
            shoulder_mid = (frame_kps[L_SHOULDER] + frame_kps[R_SHOULDER]) / 2
            hip_mid = (frame_kps[L_HIP] + frame_kps[R_HIP]) / 2
            torso_len = max(np.linalg.norm(shoulder_mid - hip_mid), 1.0)

            # Body axis vector (hip -> shoulder direction)
            body_axis_vec = shoulder_mid - hip_mid
            body_axis_norm = np.linalg.norm(body_axis_vec)
            if body_axis_norm > 0:
                body_axis_vec = body_axis_vec / body_axis_norm

            # Feature 1: Inter-joint distances (10 distances, normalized by torso)
            dist_pairs = [
                (L_WRIST, L_HIP), (R_WRIST, R_HIP),     # hands to hips (tuck indicator)
                (L_ANKLE, L_HIP), (R_ANKLE, R_HIP),      # feet to hips
                (L_WRIST, L_ANKLE), (R_WRIST, R_ANKLE),  # hand-foot distance (tuck/pike)
                (L_WRIST, R_WRIST),                        # hand spread
                (L_ANKLE, R_ANKLE),                        # foot spread
                (NOSE, L_HIP), (NOSE, R_HIP),             # head to hips
            ]
            distances = []
            for j1, j2 in dist_pairs:
                d = np.linalg.norm(frame_kps[j1] - frame_kps[j2]) / torso_len
                distances.append(d)

            # Feature 2: Limb angles relative to BODY AXIS (not image vertical)
            # This makes them view-invariant
            limb_angles = []
            for j1, j2 in LIMB_PAIRS:
                limb_vec = frame_kps[j2] - frame_kps[j1]
                limb_norm = np.linalg.norm(limb_vec)
                if limb_norm > 0 and body_axis_norm > 0:
                    limb_vec = limb_vec / limb_norm
                    # Angle between limb and body axis
                    cos_a = np.clip(np.dot(limb_vec, body_axis_vec), -1, 1)
                    angle = np.arccos(cos_a) / np.pi  # [0, 1]
                else:
                    angle = 0.5  # neutral
                limb_angles.append(angle)

            # Feature 3: Body compactness (avg distance of extremities to center)
            center = (shoulder_mid + hip_mid) / 2
            extremity_dists = [
                np.linalg.norm(frame_kps[j] - center) / torso_len
                for j in [L_WRIST, R_WRIST, L_ANKLE, R_ANKLE]
            ]
            compactness = np.mean(extremity_dists)

            # Feature 4: Body extension (max distance between any two extremities)
            ext_joints = [frame_kps[j] for j in [L_WRIST, R_WRIST, L_ANKLE, R_ANKLE, NOSE]]
            max_ext = 0
            for a in range(len(ext_joints)):
                for b in range(a+1, len(ext_joints)):
                    d = np.linalg.norm(ext_joints[a] - ext_joints[b]) / torso_len
                    max_ext = max(max_ext, d)

            # Feature 5: Left-right symmetry (how similar are left and right sides)
            l_arm = np.linalg.norm(frame_kps[L_WRIST] - frame_kps[L_SHOULDER]) / torso_len
            r_arm = np.linalg.norm(frame_kps[R_WRIST] - frame_kps[R_SHOULDER]) / torso_len
            l_leg = np.linalg.norm(frame_kps[L_ANKLE] - frame_kps[L_HIP]) / torso_len
            r_leg = np.linalg.norm(frame_kps[R_ANKLE] - frame_kps[R_HIP]) / torso_len
            symmetry = 1.0 - (abs(l_arm - r_arm) + abs(l_leg - r_leg)) / 2.0

            # Feature 6: Body axis angle relative to vertical (captures inversion)
            # This is the ONE view-dependent feature we keep because inversion
            # looks the same from any horizontal camera angle
            body_angle = np.arctan2(body_axis_vec[0], -body_axis_vec[1]) / np.pi

            frame_feat = np.array(
                distances + limb_angles + [compactness, max_ext, symmetry, body_angle],
                dtype=np.float32
            )
            features.append(frame_feat)

        return np.array(features, dtype=np.float32)

    def _normalize_time(self, features: np.ndarray) -> np.ndarray:
        """Resample feature sequence to fixed length using interpolation."""
        T, N = features.shape
        if T == self.norm_length:
            return features

        x_old = np.linspace(0, 1, T)
        x_new = np.linspace(0, 1, self.norm_length)

        result = np.zeros((self.norm_length, N), dtype=np.float32)
        for f in range(N):
            interp_fn = interp1d(x_old, features[:, f], kind='linear', fill_value='extrapolate')
            result[:, f] = interp_fn(x_new)

        return result

    def _compute_body_axis(self, kps: np.ndarray) -> np.ndarray:
        """Compute body axis angle over time (shoulder-to-hip direction).

        Returns: (T,) array of angles in degrees (0 = upright, 180 = inverted)
        """
        T = len(kps)
        angles = np.zeros(T)
        for t in range(T):
            shoulder_mid = (kps[t, L_SHOULDER, :2] + kps[t, R_SHOULDER, :2]) / 2
            hip_mid = (kps[t, L_HIP, :2] + kps[t, R_HIP, :2]) / 2
            # In image coords: y increases downward
            # Upright: shoulders ABOVE hips (shoulder_y < hip_y)
            dx = shoulder_mid[0] - hip_mid[0]
            dy = shoulder_mid[1] - hip_mid[1]
            # Angle from "upright" (dy < 0 when upright in image coords)
            angle = np.degrees(np.arctan2(dx, -dy))  # 0 = upright, 180 = inverted
            angles[t] = abs(angle)
        return angles

    def _compute_height_change(self, kps: np.ndarray) -> float:
        """Compute normalized height change (how much the person moves vertically)."""
        hip_y = (kps[:, L_HIP, 1] + kps[:, R_HIP, 1]) / 2
        shoulder_y = (kps[:, L_SHOULDER, 1] + kps[:, R_SHOULDER, 1]) / 2
        torso_len = np.median(np.abs(shoulder_y - hip_y))
        if torso_len < 1:
            torso_len = 1
        height_range = (np.max(hip_y) - np.min(hip_y)) / torso_len
        return float(height_range)

    def _compute_rotation_speed(self, body_axis: np.ndarray) -> float:
        """Average angular speed of body axis in degrees/frame."""
        if len(body_axis) < 2:
            return 0.0
        deltas = np.abs(np.diff(body_axis))
        return float(np.mean(deltas))

    def _detect_inversion(self, body_axis: np.ndarray) -> tuple[bool, int]:
        """Detect if body went inverted and count inversions.

        Returns: (went_inverted, num_inversions)
        """
        inverted_frames = body_axis > 100  # > 100 degrees from upright
        went_inverted = bool(np.any(inverted_frames))

        # Count crossings of the 90-degree threshold
        crossings = 0
        was_inverted = False
        for a in body_axis:
            is_inv = a > 90
            if is_inv and not was_inverted:
                crossings += 1
            was_inverted = is_inv

        return went_inverted, crossings

    # ── Distance Computation ───────────────────────────────────

    def _trajectory_distance(self, query: np.ndarray, ref: np.ndarray) -> float:
        """Compute distance between two trajectories using DTW.

        DTW (Dynamic Time Warping) handles timing differences naturally:
        a trick performed faster/slower still matches correctly because
        DTW finds the optimal alignment between the two sequences.
        """
        try:
            from fastdtw import fastdtw
            from scipy.spatial.distance import euclidean
            distance, _ = fastdtw(query, ref, radius=10, dist=euclidean)
            # Normalize by sequence length
            distance = distance / max(len(query), len(ref))
        except ImportError:
            # Fallback: MSE + velocity correlation
            mse = float(np.mean((query - ref) ** 2))
            q_vel = np.diff(query, axis=0).flatten()
            r_vel = np.diff(ref, axis=0).flatten()
            norm_q = np.linalg.norm(q_vel)
            norm_r = np.linalg.norm(r_vel)
            if norm_q > 0 and norm_r > 0:
                vel_sim = float(np.dot(q_vel, r_vel) / (norm_q * norm_r))
            else:
                vel_sim = 0.0
            distance = mse * 0.6 + (1.0 - vel_sim) * 0.4

        return float(distance)
