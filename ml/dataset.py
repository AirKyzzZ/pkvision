"""PyTorch Dataset for skeleton-based trick classification."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from ml.augment import augment_sample


class TrickDataset(Dataset):
    """Dataset of keypoint sequences extracted from labeled parkour clips.

    Each sample is a (C, T, V) tensor where:
    - C = 3 (x, y, confidence)
    - T = target_frames (padded/cropped)
    - V = 17 (COCO keypoints)

    Labels are integer class indices into trick_classes.
    """

    def __init__(
        self,
        keypoints_dir: Path | str,
        labels_path: Path | str,
        trick_classes: list[str],
        target_frames: int = 64,
        augment: bool = False,
    ):
        self.keypoints_dir = Path(keypoints_dir)
        self.trick_classes = trick_classes
        self.target_frames = target_frames
        self.augment = augment

        # Load labels
        with open(labels_path) as f:
            self.labels = json.load(f)

        # Build samples: (keypoints_file, class_index)
        self.samples: list[tuple[Path, int]] = []
        self._build_sample_list()

    def _build_sample_list(self) -> None:
        for entry in self.labels:
            trick_id = entry.get("trick_id", "")
            if trick_id not in self.trick_classes:
                continue

            class_idx = self.trick_classes.index(trick_id)

            # Look for extracted keypoints file
            filename = Path(entry["file"]).stem
            kp_path = self.keypoints_dir / f"{filename}.npy"

            if kp_path.exists():
                self.samples.append((kp_path, class_idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        kp_path, class_idx = self.samples[idx]

        # Load keypoint sequence: (C, T, V)
        data = np.load(kp_path).astype(np.float32)

        # Ensure shape is (3, T, 17)
        if data.ndim == 2:
            # If flat (T, 17*3), reshape
            T = data.shape[0]
            data = data.reshape(T, 3, 17).transpose(1, 0, 2)
        elif data.ndim == 3 and data.shape[0] != 3:
            # If (T, 17, 3), transpose to (3, T, 17)
            data = data.transpose(2, 0, 1)

        if self.augment:
            data = augment_sample(data, target_frames=self.target_frames)
        else:
            data = self._pad_or_crop(data)

        return torch.FloatTensor(data), class_idx

    def _pad_or_crop(self, data: np.ndarray) -> np.ndarray:
        """Pad or crop to target_frames."""
        C, T, V = data.shape

        if T >= self.target_frames:
            # Center crop
            start = (T - self.target_frames) // 2
            return data[:, start : start + self.target_frames, :]

        # Pad with zeros
        padded = np.zeros((C, self.target_frames, V), dtype=data.dtype)
        padded[:, :T, :] = data
        return padded
