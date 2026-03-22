"""Tests for the ST-GCN model and training pipeline."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ml.augment import augment_sample, mirror_skeleton, random_crop, speed_variation
from ml.stgcn.layers import STGCNBlock, build_coco_adjacency, normalize_adjacency
from ml.stgcn.model import STGCN


class TestAdjacencyMatrix:
    def test_shape(self):
        adj = build_coco_adjacency()
        assert adj.shape == (17, 17)

    def test_symmetric(self):
        adj = build_coco_adjacency()
        np.testing.assert_array_equal(adj, adj.T)

    def test_self_loops(self):
        adj = build_coco_adjacency()
        assert all(adj[i, i] == 1.0 for i in range(17))

    def test_normalization_preserves_shape(self):
        adj = build_coco_adjacency()
        normed = normalize_adjacency(adj)
        assert normed.shape == (17, 17)


class TestSTGCNModel:
    def test_forward_shape(self):
        model = STGCN(num_classes=10, in_channels=3, num_joints=17)
        x = torch.randn(2, 3, 64, 17)  # batch=2, channels=3, frames=64, joints=17
        out = model(x)
        assert out.shape == (2, 10)

    def test_forward_different_frames(self):
        model = STGCN(num_classes=5, in_channels=3, num_joints=17)
        x = torch.randn(1, 3, 32, 17)
        out = model(x)
        assert out.shape == (1, 5)

    def test_single_sample(self):
        model = STGCN(num_classes=3, in_channels=3, num_joints=17)
        x = torch.randn(1, 3, 64, 17)
        out = model(x)
        assert out.shape == (1, 3)
        # Softmax should sum to ~1
        probs = torch.softmax(out, dim=1)
        assert abs(probs.sum().item() - 1.0) < 1e-5


class TestSTGCNBlock:
    def test_same_channels(self):
        adj = torch.FloatTensor(normalize_adjacency(build_coco_adjacency()))
        block = STGCNBlock(64, 64, adj)
        x = torch.randn(2, 64, 32, 17)
        out = block(x)
        assert out.shape == (2, 64, 32, 17)

    def test_channel_change(self):
        adj = torch.FloatTensor(normalize_adjacency(build_coco_adjacency()))
        block = STGCNBlock(32, 64, adj)
        x = torch.randn(2, 32, 32, 17)
        out = block(x)
        assert out.shape == (2, 64, 32, 17)


class TestAugmentation:
    def test_mirror(self):
        data = np.random.rand(3, 30, 17).astype(np.float32)
        mirrored = mirror_skeleton(data)
        assert mirrored.shape == data.shape
        # Nose (index 0) has no swap pair, so its x should be exactly flipped
        np.testing.assert_allclose(mirrored[0, :, 0], 1.0 - data[0, :, 0], atol=1e-6)
        # Left/right pairs should be swapped: e.g. left_eye(1) ↔ right_eye(2)
        np.testing.assert_allclose(mirrored[0, :, 1], 1.0 - data[0, :, 2], atol=1e-6)
        np.testing.assert_allclose(mirrored[0, :, 2], 1.0 - data[0, :, 1], atol=1e-6)

    def test_speed_variation_faster(self):
        data = np.random.rand(3, 60, 17).astype(np.float32)
        faster = speed_variation(data, factor=2.0)
        assert faster.shape[1] == 30  # Half the frames

    def test_speed_variation_slower(self):
        data = np.random.rand(3, 30, 17).astype(np.float32)
        slower = speed_variation(data, factor=0.5)
        assert slower.shape[1] == 60  # Double the frames

    def test_random_crop_shorter(self):
        data = np.random.rand(3, 100, 17).astype(np.float32)
        cropped = random_crop(data, target_frames=64)
        assert cropped.shape == (3, 64, 17)

    def test_random_crop_pad(self):
        data = np.random.rand(3, 20, 17).astype(np.float32)
        padded = random_crop(data, target_frames=64)
        assert padded.shape == (3, 64, 17)
        # Original data should be preserved
        np.testing.assert_array_equal(padded[:, :20, :], data)

    def test_augment_sample_output_shape(self):
        data = np.random.rand(3, 50, 17).astype(np.float32)
        augmented = augment_sample(data, target_frames=64)
        assert augmented.shape == (3, 64, 17)
