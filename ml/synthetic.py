"""Synthetic skeleton data generator for PkVision.

Generates thousands of synthetic keypoint sequences from a few real examples
by applying physically-plausible transformations:
- Speed variation (faster/slower trick execution)
- Camera angle rotation (simulating different viewing angles)
- Body proportion scaling (different body types)
- Gaussian noise (imperfect technique)
- Height/amplitude scaling (higher/lower jumps)
- Temporal jittering (slight timing variations)
- Horizontal mirroring (left/right flip)

The key insight: YOLO pose estimation outputs 17 keypoints (x, y) per frame.
The classification model (ST-GCN / VideoMAE) only sees these vectors, not pixels.
So we can generate training data by transforming skeleton sequences directly.

From 3 real examples → 3,000 synthetic variations per trick.

Usage:
    from ml.synthetic import SyntheticGenerator
    gen = SyntheticGenerator()
    gen.load_real_examples("back_flip", [array1, array2, array3])
    synthetic_data = gen.generate("back_flip", n=1000)
"""

from __future__ import annotations

import math

import numpy as np


# COCO 17-keypoint skeleton connections for physical constraints
SKELETON_PAIRS = [
    (5, 6),   # shoulders
    (5, 7),   # left shoulder → left elbow
    (7, 9),   # left elbow → left wrist
    (6, 8),   # right shoulder → right elbow
    (8, 10),  # right elbow → right wrist
    (11, 12), # hips
    (5, 11),  # left shoulder → left hip
    (6, 12),  # right shoulder → right hip
    (11, 13), # left hip → left knee
    (13, 15), # left knee → left ankle
    (12, 14), # right hip → right knee
    (14, 16), # right knee → right ankle
]

# Left/right pairs for mirroring
MIRROR_PAIRS = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]


class SyntheticGenerator:
    """Generate synthetic keypoint sequences from real examples."""

    def __init__(self, target_frames: int = 64, seed: int | None = None):
        self.target_frames = target_frames
        self.rng = np.random.default_rng(seed)
        self.real_examples: dict[str, list[np.ndarray]] = {}

    def load_real_examples(self, trick_id: str, sequences: list[np.ndarray]) -> None:
        """Load real keypoint sequences for a trick.

        Each sequence should be shape (3, T, 17) — channels (x, y, conf), frames, joints.
        """
        self.real_examples[trick_id] = sequences

    def load_from_npy(self, trick_id: str, paths: list[str]) -> None:
        """Load real examples from .npy files."""
        sequences = []
        for path in paths:
            data = np.load(path).astype(np.float32)
            # Ensure shape is (3, T, 17)
            if data.ndim == 3 and data.shape[0] != 3:
                data = data.transpose(2, 0, 1)
            sequences.append(data)
        self.real_examples[trick_id] = sequences

    def generate(self, trick_id: str, n: int = 1000) -> list[np.ndarray]:
        """Generate n synthetic sequences for a trick.

        Returns list of (3, target_frames, 17) arrays.
        """
        if trick_id not in self.real_examples:
            raise ValueError(f"No real examples loaded for '{trick_id}'")

        real = self.real_examples[trick_id]
        synthetic = []

        for i in range(n):
            # Pick a random real example as base
            base = real[self.rng.integers(len(real))].copy()

            # Apply random combination of transformations
            augmented = self._augment(base)

            # Normalize to target frame count
            augmented = self._resize_temporal(augmented, self.target_frames)

            synthetic.append(augmented)

        return synthetic

    def _augment(self, data: np.ndarray) -> np.ndarray:
        """Apply a random combination of transformations."""
        result = data.copy()

        # 1. Speed variation (0.7x to 1.4x)
        if self.rng.random() < 0.8:
            speed = self.rng.uniform(0.7, 1.4)
            result = self._speed_variation(result, speed)

        # 2. Camera angle rotation (-30 to +30 degrees)
        if self.rng.random() < 0.7:
            angle = self.rng.uniform(-30, 30)
            result = self._rotate_2d(result, angle)

        # 3. Body proportion scaling
        if self.rng.random() < 0.6:
            scale_x = self.rng.uniform(0.85, 1.15)
            scale_y = self.rng.uniform(0.85, 1.15)
            result = self._scale_proportions(result, scale_x, scale_y)

        # 4. Translation (shift position)
        if self.rng.random() < 0.7:
            dx = self.rng.uniform(-0.15, 0.15)
            dy = self.rng.uniform(-0.15, 0.15)
            result = self._translate(result, dx, dy)

        # 5. Height/amplitude scaling (higher/lower jump)
        if self.rng.random() < 0.5:
            amplitude = self.rng.uniform(0.8, 1.2)
            result = self._scale_amplitude(result, amplitude)

        # 6. Gaussian noise (technique imperfection)
        if self.rng.random() < 0.8:
            noise_std = self.rng.uniform(0.005, 0.02)
            result = self._add_noise(result, noise_std)

        # 7. Temporal jitter (slight timing variations)
        if self.rng.random() < 0.5:
            result = self._temporal_jitter(result)

        # 8. Horizontal mirror
        if self.rng.random() < 0.5:
            result = self._mirror(result)

        # 9. Confidence variation
        if self.rng.random() < 0.6:
            result = self._vary_confidence(result)

        return result

    def _speed_variation(self, data: np.ndarray, factor: float) -> np.ndarray:
        """Resample temporal dimension by speed factor."""
        C, T, V = data.shape
        new_T = max(4, int(T / factor))
        indices = np.linspace(0, T - 1, new_T)
        result = np.zeros((C, new_T, V), dtype=data.dtype)
        for c in range(C):
            for v in range(V):
                result[c, :, v] = np.interp(indices, np.arange(T), data[c, :, v])
        return result

    def _rotate_2d(self, data: np.ndarray, angle_deg: float) -> np.ndarray:
        """Rotate all keypoints around the center of mass."""
        result = data.copy()
        angle = math.radians(angle_deg)
        cos_a, sin_a = math.cos(angle), math.sin(angle)

        # Compute center of mass per frame
        cx = data[0].mean(axis=1, keepdims=True)  # (T, 1)
        cy = data[1].mean(axis=1, keepdims=True)

        # Translate to origin, rotate, translate back
        x = data[0] - cx
        y = data[1] - cy
        result[0] = x * cos_a - y * sin_a + cx
        result[1] = x * sin_a + y * cos_a + cy

        return np.clip(result, 0, 1)

    def _scale_proportions(self, data: np.ndarray, sx: float, sy: float) -> np.ndarray:
        """Scale body proportions from center of mass."""
        result = data.copy()
        cx = data[0].mean(axis=1, keepdims=True)
        cy = data[1].mean(axis=1, keepdims=True)
        result[0] = (data[0] - cx) * sx + cx
        result[1] = (data[1] - cy) * sy + cy
        return np.clip(result, 0, 1)

    def _translate(self, data: np.ndarray, dx: float, dy: float) -> np.ndarray:
        """Shift all keypoints."""
        result = data.copy()
        result[0] = np.clip(data[0] + dx, 0, 1)
        result[1] = np.clip(data[1] + dy, 0, 1)
        return result

    def _scale_amplitude(self, data: np.ndarray, factor: float) -> np.ndarray:
        """Scale the vertical movement amplitude (jump height)."""
        result = data.copy()
        # Find the mean y position across all frames
        mean_y = data[1].mean()
        # Scale deviation from mean
        result[1] = mean_y + (data[1] - mean_y) * factor
        return np.clip(result, 0, 1)

    def _add_noise(self, data: np.ndarray, std: float) -> np.ndarray:
        """Add Gaussian noise to x, y coordinates (not confidence)."""
        result = data.copy()
        noise = self.rng.normal(0, std, size=(2, data.shape[1], data.shape[2])).astype(data.dtype)
        result[:2] += noise
        return np.clip(result, 0, 1)

    def _temporal_jitter(self, data: np.ndarray) -> np.ndarray:
        """Apply slight random temporal shifts to individual joints."""
        C, T, V = data.shape
        if T < 5:
            return data
        result = data.copy()
        for v in range(V):
            shift = self.rng.integers(-2, 3)  # -2 to +2 frames
            if shift != 0:
                for c in range(2):  # Only x, y
                    shifted = np.roll(data[c, :, v], shift)
                    # Fix edges
                    if shift > 0:
                        shifted[:shift] = shifted[shift]
                    elif shift < 0:
                        shifted[shift:] = shifted[shift - 1]
                    result[c, :, v] = shifted
        return result

    def _mirror(self, data: np.ndarray) -> np.ndarray:
        """Horizontal flip: mirror x coordinates and swap left/right joints."""
        result = data.copy()
        result[0] = 1.0 - data[0]
        for left, right in MIRROR_PAIRS:
            result[:, :, left], result[:, :, right] = (
                result[:, :, right].copy(),
                result[:, :, left].copy(),
            )
        return result

    def _vary_confidence(self, data: np.ndarray) -> np.ndarray:
        """Randomly vary keypoint confidence scores."""
        result = data.copy()
        # Multiply confidence by random factor
        conf_factor = self.rng.uniform(0.7, 1.0, size=(1, data.shape[1], data.shape[2]))
        result[2] = np.clip(data[2] * conf_factor, 0, 1)
        # Randomly drop some keypoints (set confidence to 0)
        drop_mask = self.rng.random(size=(data.shape[1], data.shape[2])) < 0.05
        result[2][drop_mask] = 0.0
        return result

    def _resize_temporal(self, data: np.ndarray, target: int) -> np.ndarray:
        """Resize to target frame count."""
        C, T, V = data.shape
        if T == target:
            return data
        indices = np.linspace(0, T - 1, target)
        result = np.zeros((C, target, V), dtype=data.dtype)
        for c in range(C):
            for v in range(V):
                result[c, :, v] = np.interp(indices, np.arange(T), data[c, :, v])
        return result


def generate_synthetic_dataset(
    real_data_dir: str,
    labels_path: str,
    output_dir: str,
    samples_per_trick: int = 500,
    target_frames: int = 64,
    seed: int = 42,
) -> dict:
    """Generate a full synthetic dataset from real extracted keypoints.

    Args:
        real_data_dir: Directory with .npy keypoint files (from extract_poses.py)
        labels_path: Path to labels.json mapping files to trick IDs
        output_dir: Where to save synthetic .npy files
        samples_per_trick: How many synthetic samples per trick
        target_frames: Frame count for each sample

    Returns:
        Manifest dict with samples and classes.
    """
    import json
    from pathlib import Path

    real_dir = Path(real_data_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(labels_path) as f:
        labels = json.load(f)

    # Handle both formats
    if isinstance(labels, dict):
        labels_list = labels.get("labels", [])
    else:
        labels_list = labels

    # Group real keypoints by trick
    trick_files: dict[str, list[Path]] = {}
    for entry in labels_list:
        trick_id = entry.get("trick_id", "")
        filename = Path(entry["file"]).stem + ".npy"
        kp_path = real_dir / filename
        if kp_path.exists():
            trick_files.setdefault(trick_id, []).append(kp_path)

    print(f"Real examples per trick:")
    for tid, paths in sorted(trick_files.items()):
        print(f"  {tid}: {len(paths)} real → {samples_per_trick} synthetic")

    # Generate
    gen = SyntheticGenerator(target_frames=target_frames, seed=seed)

    for tid, paths in trick_files.items():
        sequences = []
        for p in paths:
            data = np.load(p).astype(np.float32)
            # Ensure (3, T, 17)
            if data.ndim == 3 and data.shape[0] != 3:
                if data.shape[2] == 3:
                    data = data.transpose(2, 0, 1)
                elif data.shape[1] == 3:
                    pass  # Already correct
            if data.shape[0] == 3 and data.shape[2] == 17:
                sequences.append(data)
        if sequences:
            gen.load_real_examples(tid, sequences)

    manifest_samples = []
    classes = sorted(trick_files.keys())

    for tid in classes:
        if tid not in gen.real_examples:
            continue

        print(f"  Generating {samples_per_trick} synthetic {tid}...", end=" ", flush=True)
        synthetics = gen.generate(tid, n=samples_per_trick)

        for i, synth in enumerate(synthetics):
            fname = f"synth_{tid}_{i:04d}.npy"
            np.save(out_dir / fname, synth)
            manifest_samples.append({
                "file": fname,
                "original": f"synthetic_{tid}",
                "class": tid,
            })

        print(f"OK")

    # Save manifest
    manifest = {
        "classes": classes,
        "samples": manifest_samples,
        "class_counts": {c: sum(1 for s in manifest_samples if s["class"] == c) for c in classes},
    }

    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    total = len(manifest_samples)
    print(f"\nGenerated {total} synthetic samples across {len(classes)} classes")
    print(f"Manifest: {manifest_path}")

    return manifest
