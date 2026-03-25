"""DTW detection strategy — classifies tricks by comparing feature sequences to references.

No training required. Add 1-4 reference .npy files per trick and it works immediately.
This is the cornerstone of the few-shot classification approach: new trick = add reference files.

References are stored in data/references/{trick_id}/*.npy, each a (n_frames, 60) feature
array produced by FeatureSequence.to_array().

Usage:
    strategy = DTWStrategy(references_dir="data/references")
    detection = strategy.evaluate(trick_config, analyzed_frames)
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from core.models import FrameAnalysis, TrickConfig, TrickDetection
from core.pose.features import ANGLE_NAMES, _ANGLE_MAX, _VELOCITY_CLIP

logger = logging.getLogger(__name__)

# Try fast C implementation, fall back to pure numpy
try:
    from dtaidistance import dtw_ndim

    _HAS_DTAIDISTANCE = True
except ImportError:
    _HAS_DTAIDISTANCE = False
    logger.warning("dtaidistance not installed — using numpy DTW fallback (slower)")


# ── DTW Implementations ─────────────────────────────────────────────

_N_ANGLES = len(ANGLE_NAMES)


def _normalize_features(arr: np.ndarray) -> np.ndarray:
    """Normalize a raw (n_frames, 60) feature array for DTW comparison.

    Applies the same normalization as FeatureSequence.to_array(normalize=True),
    then replaces NaN with 0 (DTW requires finite values).
    """
    result = arr.copy()
    # Angles: [0, 180] → [0, 1]
    result[:, :_N_ANGLES] /= _ANGLE_MAX
    # Velocities: clip to ±1000 then scale to [-1, 1]
    result[:, _N_ANGLES : 2 * _N_ANGLES] = (
        np.clip(result[:, _N_ANGLES : 2 * _N_ANGLES], -_VELOCITY_CLIP, _VELOCITY_CLIP)
        / _VELOCITY_CLIP
    )
    # Replace NaN with 0 (neutral value in normalized space)
    np.nan_to_num(result, copy=False, nan=0.0)
    return result


def _dtw_distance(
    s1: np.ndarray,
    s2: np.ndarray,
    window: int | None = None,
) -> float:
    """Compute DTW distance between two multivariate sequences.

    Uses dtaidistance C implementation when available, otherwise pure numpy.

    Args:
        s1: (n, d) array — first sequence.
        s2: (m, d) array — second sequence.
        window: Sakoe-Chiba band width. None = no constraint.
    """
    if _HAS_DTAIDISTANCE:
        return float(dtw_ndim.distance(s1, s2, window=window))
    return _dtw_distance_numpy(s1, s2, window)


def _dtw_distance_numpy(
    s1: np.ndarray,
    s2: np.ndarray,
    window: int | None = None,
) -> float:
    """Pure numpy DTW fallback — O(n*m) with optional Sakoe-Chiba band."""
    n, m = len(s1), len(s2)
    cost = np.full((n + 1, m + 1), np.inf, dtype=np.float64)
    cost[0, 0] = 0.0

    for i in range(1, n + 1):
        j_lo = max(1, i - window) if window else 1
        j_hi = min(m, i + window) if window else m
        for j in range(j_lo, j_hi + 1):
            d = float(np.linalg.norm(s1[i - 1] - s2[j - 1]))
            cost[i, j] = d + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])

    return float(cost[n, m])


# ── DTW Strategy ─────────────────────────────────────────────────────


class DTWStrategy:
    """Trick detection via Dynamic Time Warping against reference sequences.

    No training required. Just add .npy reference feature files per trick.
    Falls back gracefully — even 1 reference clip per trick works.

    References are stored as raw feature arrays (from FeatureSequence.to_array())
    and normalized at load time for consistent comparison.
    """

    def __init__(
        self,
        references_dir: Path | str = Path("data/references"),
        min_confidence: float = 0.3,
        distance_scale: float = 1.5,
        window_fraction: float = 0.2,
    ):
        """
        Args:
            references_dir: Directory containing {trick_id}/*.npy reference files.
            min_confidence: Minimum confidence to report a detection.
            distance_scale: Controls confidence falloff. Larger = more lenient.
                Average per-step DTW distance equal to distance_scale → confidence = 0.
            window_fraction: Sakoe-Chiba band as fraction of max sequence length.
                0.2 = warping path can deviate by ±20%. Set to 1.0 to disable.
        """
        self.references_dir = Path(references_dir)
        self.min_confidence = min_confidence
        self.distance_scale = distance_scale
        self.window_fraction = window_fraction

        # Normalized reference arrays per trick
        self._references: dict[str, list[np.ndarray]] = {}
        self._load_references()

        # Feature extraction cache (avoids re-extracting for each trick)
        self._cached_frames_id: int | None = None
        self._cached_features: np.ndarray | None = None

    @property
    def loaded_tricks(self) -> list[str]:
        """Trick IDs that have at least one reference loaded."""
        return list(self._references.keys())

    def _load_references(self) -> None:
        """Load and normalize all reference .npy files from the references directory."""
        if not self.references_dir.exists():
            logger.info("References directory %s does not exist — DTW has no references", self.references_dir)
            return

        for trick_dir in sorted(self.references_dir.iterdir()):
            if not trick_dir.is_dir():
                continue

            trick_id = trick_dir.name
            refs: list[np.ndarray] = []

            for npy_file in sorted(trick_dir.glob("*.npy")):
                arr = np.load(npy_file).astype(np.float32)
                if arr.ndim == 2 and arr.shape[1] >= 60:
                    refs.append(_normalize_features(arr))
                else:
                    logger.warning(
                        "Skipping %s: expected shape (n, >=60), got %s",
                        npy_file, arr.shape,
                    )

            if refs:
                self._references[trick_id] = refs
                logger.debug("Loaded %d references for %s", len(refs), trick_id)

    def add_references(self, trick_id: str, feature_arrays: list[np.ndarray]) -> None:
        """Add reference feature arrays programmatically (for testing or runtime use).

        Args:
            trick_id: Trick identifier.
            feature_arrays: List of raw (n_frames, 60) feature arrays.
        """
        normalized = [_normalize_features(arr.astype(np.float32)) for arr in feature_arrays]
        self._references.setdefault(trick_id, []).extend(normalized)

    def evaluate(
        self,
        trick: TrickConfig,
        frames: list[FrameAnalysis],
    ) -> TrickDetection | None:
        """Evaluate whether the given trick matches the frame sequence via DTW.

        Returns None if:
        - No references exist for this trick
        - No frames provided
        - Best DTW confidence is below min_confidence
        """
        if trick.trick_id not in self._references:
            return None

        if not frames:
            return None

        input_features = self._extract_features(frames)
        if input_features is None:
            return None

        # Compute DTW distance against each reference, keep the best
        refs = self._references[trick.trick_id]
        best_confidence = 0.0

        for ref in refs:
            confidence = self._compute_confidence(input_features, ref)
            best_confidence = max(best_confidence, confidence)

        if best_confidence < self.min_confidence:
            return None

        return TrickDetection(
            trick_id=trick.trick_id,
            trick_name=trick.get_name(),
            confidence=best_confidence,
            start_frame=frames[0].frame_idx,
            end_frame=frames[-1].frame_idx,
            start_time_ms=frames[0].timestamp_ms,
            end_time_ms=frames[-1].timestamp_ms,
            strategy_used="dtw",
        )

    def _compute_confidence(
        self,
        input_seq: np.ndarray,
        reference: np.ndarray,
    ) -> float:
        """Compute confidence score from DTW distance between two sequences."""
        max_len = max(len(input_seq), len(reference))
        window = max(1, int(max_len * self.window_fraction))

        distance = _dtw_distance(input_seq, reference, window=window)

        # Normalize by path length (approximate as max of the two lengths)
        avg_distance = distance / max_len
        confidence = max(0.0, 1.0 - avg_distance / self.distance_scale)

        return confidence

    def _extract_features(self, frames: list[FrameAnalysis]) -> np.ndarray | None:
        """Extract and normalize features from FrameAnalysis objects.

        Caches results — safe to call repeatedly with the same frames list
        (e.g., when TrickClassifier iterates over all tricks).
        """
        frames_id = id(frames)
        if self._cached_frames_id == frames_id and self._cached_features is not None:
            return self._cached_features

        from core.pose.features import extract_features_from_frames

        seq = extract_features_from_frames(frames)
        if seq.n_frames == 0:
            return None

        features = _normalize_features(seq.to_array())
        self._cached_frames_id = frames_id
        self._cached_features = features
        return features


# ── Utility Functions ────────────────────────────────────────────────


def save_reference(
    feature_array: np.ndarray,
    trick_id: str,
    references_dir: Path | str = Path("data/references"),
    name: str | None = None,
) -> Path:
    """Save a feature array as a DTW reference file.

    Args:
        feature_array: Raw (n_frames, 60) array from FeatureSequence.to_array().
        trick_id: Trick identifier (becomes subdirectory name).
        references_dir: Base directory for references.
        name: Optional filename (without .npy). Auto-generates if None.

    Returns:
        Path to the saved .npy file.
    """
    ref_dir = Path(references_dir) / trick_id
    ref_dir.mkdir(parents=True, exist_ok=True)

    if name is None:
        existing = list(ref_dir.glob("*.npy"))
        name = f"ref_{len(existing) + 1:03d}"

    path = ref_dir / f"{name}.npy"
    np.save(path, feature_array.astype(np.float32))
    return path
