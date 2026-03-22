"""Data augmentation for skeleton-based action recognition."""

from __future__ import annotations

import numpy as np


def mirror_skeleton(data: np.ndarray) -> np.ndarray:
    """Horizontal flip: swap left/right keypoints and mirror x coordinates.

    data: (C, T, V) where C=3 (x, y, conf), V=17 COCO joints
    """
    result = data.copy()

    # Mirror x coordinates (channel 0)
    result[0] = 1.0 - result[0]

    # Swap left/right joint pairs
    # COCO pairs: (1,2), (3,4), (5,6), (7,8), (9,10), (11,12), (13,14), (15,16)
    swap_pairs = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]
    for left, right in swap_pairs:
        result[:, :, left], result[:, :, right] = (
            result[:, :, right].copy(),
            result[:, :, left].copy(),
        )

    return result


def speed_variation(data: np.ndarray, factor: float) -> np.ndarray:
    """Resample the temporal dimension by a speed factor.

    factor < 1.0 = slower (more frames), factor > 1.0 = faster (fewer frames).
    data: (C, T, V)
    """
    C, T, V = data.shape
    new_T = max(1, int(T / factor))

    indices = np.linspace(0, T - 1, new_T)
    result = np.zeros((C, new_T, V), dtype=data.dtype)

    for c in range(C):
        for v in range(V):
            result[c, :, v] = np.interp(indices, np.arange(T), data[c, :, v])

    return result


def random_noise(data: np.ndarray, std: float = 0.01) -> np.ndarray:
    """Add Gaussian noise to x and y coordinates (not confidence).

    data: (C, T, V)
    """
    result = data.copy()
    noise = np.random.normal(0, std, size=result[:2].shape).astype(result.dtype)
    result[:2] += noise
    result[:2] = np.clip(result[:2], 0.0, 1.0)
    return result


def random_crop(data: np.ndarray, target_frames: int) -> np.ndarray:
    """Randomly crop a temporal window of target_frames length.

    data: (C, T, V)
    """
    C, T, V = data.shape
    if T <= target_frames:
        # Pad with zeros if too short
        padded = np.zeros((C, target_frames, V), dtype=data.dtype)
        padded[:, :T, :] = data
        return padded

    start = np.random.randint(0, T - target_frames)
    return data[:, start : start + target_frames, :]


def augment_sample(
    data: np.ndarray,
    target_frames: int = 64,
    do_mirror: bool = True,
    speed_range: tuple[float, float] = (0.8, 1.2),
    noise_std: float = 0.01,
) -> np.ndarray:
    """Apply a random combination of augmentations to a sample.

    data: (C, T, V)
    Returns: (C, target_frames, V)
    """
    result = data.copy()

    # Random mirror
    if do_mirror and np.random.random() < 0.5:
        result = mirror_skeleton(result)

    # Random speed variation
    factor = np.random.uniform(*speed_range)
    result = speed_variation(result, factor)

    # Random noise
    if noise_std > 0:
        result = random_noise(result, noise_std)

    # Crop/pad to target length
    result = random_crop(result, target_frames)

    return result
