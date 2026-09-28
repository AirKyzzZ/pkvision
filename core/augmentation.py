"""Domain augmentation for competition footage simulation.

Transforms clean parkourtheory clips to look like competition footage:
- Shrink athlete in frame (simulate wide-angle camera)
- Add synthetic backgrounds (stadiums, gyms, outdoor)
- Camera shake / jitter
- Compression artifacts (JPEG quality reduction)
- Motion blur
- Lighting / exposure variation
- Crowd overlay noise

Usage:
    from core.augmentation import CompetitionAugmenter
    aug = CompetitionAugmenter(intensity=0.5)
    augmented_frames = aug(frames)  # list of numpy arrays (H, W, 3)
"""

from __future__ import annotations

import random

import cv2
import numpy as np


class CompetitionAugmenter:
    """Simulates competition footage conditions on clean training clips.

    Args:
        intensity: 0.0 = no augmentation, 1.0 = maximum competition simulation.
                   Controls probability and strength of each transform.
    """

    def __init__(self, intensity: float = 0.5):
        self.intensity = intensity

    def __call__(self, frames: np.ndarray) -> np.ndarray:
        """Augment a clip of shape (T, H, W, 3) uint8."""
        if random.random() > self.intensity:
            return frames  # skip augmentation sometimes

        frames = frames.copy()

        # Each transform applied independently with probability ~ intensity
        if random.random() < self.intensity * 0.6:
            frames = self._shrink_athlete(frames)

        if random.random() < self.intensity * 0.5:
            frames = self._camera_shake(frames)

        if random.random() < self.intensity * 0.4:
            frames = self._compression_artifacts(frames)

        if random.random() < self.intensity * 0.3:
            frames = self._motion_blur(frames)

        if random.random() < self.intensity * 0.5:
            frames = self._lighting_variation(frames)

        if random.random() < self.intensity * 0.3:
            frames = self._add_noise(frames)

        return frames

    def _shrink_athlete(self, frames: np.ndarray) -> np.ndarray:
        """Shrink the athlete within the frame, simulating a distant camera."""
        T, H, W, C = frames.shape
        scale = random.uniform(0.3, 0.7)  # athlete fills 30-70% of frame
        new_h, new_w = int(H * scale), int(W * scale)

        # Random position within frame
        max_y = H - new_h
        max_x = W - new_w
        y_off = random.randint(0, max(0, max_y))
        x_off = random.randint(0, max(0, max_x))

        result = np.zeros_like(frames)
        # Fill background with edge-blurred version of frame
        for t in range(T):
            bg = cv2.GaussianBlur(frames[t], (31, 31), 15)
            # Tint background slightly (gray/brown like a gym floor)
            tint = np.array([random.randint(100, 180)] * 3, dtype=np.uint8)
            bg = cv2.addWeighted(bg, 0.3, np.full_like(bg, tint), 0.7, 0)
            result[t] = bg
            small = cv2.resize(frames[t], (new_w, new_h))
            result[t, y_off:y_off + new_h, x_off:x_off + new_w] = small

        return result

    def _camera_shake(self, frames: np.ndarray) -> np.ndarray:
        """Add camera shake/jitter between frames."""
        T, H, W, C = frames.shape
        max_shift = int(H * 0.03 * self.intensity)
        if max_shift < 1:
            return frames

        result = np.zeros_like(frames)
        # Generate smooth camera path
        dx = np.cumsum(np.random.randn(T) * max_shift * 0.3)
        dy = np.cumsum(np.random.randn(T) * max_shift * 0.3)
        # Smooth the path
        kernel = np.ones(5) / 5
        dx = np.convolve(dx, kernel, mode="same").astype(int)
        dy = np.convolve(dy, kernel, mode="same").astype(int)

        for t in range(T):
            M = np.float32([[1, 0, dx[t]], [0, 1, dy[t]]])
            result[t] = cv2.warpAffine(frames[t], M, (W, H), borderMode=cv2.BORDER_REFLECT)

        return result

    def _compression_artifacts(self, frames: np.ndarray) -> np.ndarray:
        """Simulate JPEG compression artifacts."""
        quality = random.randint(15, 50)
        result = np.zeros_like(frames)
        for t in range(frames.shape[0]):
            _, encoded = cv2.imencode(".jpg", frames[t], [cv2.IMWRITE_JPEG_QUALITY, quality])
            result[t] = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            # Convert back to RGB if needed
            if frames[t].shape[-1] == 3:
                result[t] = cv2.cvtColor(cv2.cvtColor(result[t], cv2.COLOR_BGR2RGB), cv2.COLOR_RGB2BGR)
        return result

    def _motion_blur(self, frames: np.ndarray) -> np.ndarray:
        """Add directional motion blur."""
        ksize = random.choice([3, 5, 7])
        angle = random.uniform(0, 180)

        # Create motion blur kernel
        kernel = np.zeros((ksize, ksize))
        kernel[ksize // 2, :] = 1.0
        M = cv2.getRotationMatrix2D((ksize / 2, ksize / 2), angle, 1.0)
        kernel = cv2.warpAffine(kernel, M, (ksize, ksize))
        kernel = kernel / kernel.sum()

        result = np.zeros_like(frames)
        for t in range(frames.shape[0]):
            result[t] = cv2.filter2D(frames[t], -1, kernel)
        return result

    def _lighting_variation(self, frames: np.ndarray) -> np.ndarray:
        """Simulate varying lighting conditions (outdoor/indoor)."""
        T = frames.shape[0]
        result = frames.astype(np.float32)

        # Global brightness shift
        brightness = random.uniform(-40, 40)
        contrast = random.uniform(0.7, 1.3)
        result = result * contrast + brightness

        # Color temperature shift (warm/cool)
        if random.random() < 0.5:
            # Warm (outdoor sunlight)
            result[:, :, :, 0] *= random.uniform(1.0, 1.15)  # more red
            result[:, :, :, 2] *= random.uniform(0.85, 1.0)  # less blue
        else:
            # Cool (indoor fluorescent)
            result[:, :, :, 0] *= random.uniform(0.85, 1.0)
            result[:, :, :, 2] *= random.uniform(1.0, 1.15)

        # Temporal brightness variation (flickering lights)
        if random.random() < 0.3:
            flicker = 1.0 + np.random.randn(T) * 0.03
            for t in range(T):
                result[t] *= flicker[t]

        return np.clip(result, 0, 255).astype(np.uint8)

    def _add_noise(self, frames: np.ndarray) -> np.ndarray:
        """Add sensor noise (simulates phone camera in low light)."""
        noise_level = random.uniform(5, 20) * self.intensity
        noise = np.random.randn(*frames.shape) * noise_level
        result = np.clip(frames.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        return result
