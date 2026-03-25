"""Tests for TCN model architecture and inference strategy."""

from __future__ import annotations

import math

import numpy as np
import pytest

from core.models import FrameAnalysis, TrickConfig, TrickPhase
from core.pose.angles import get_joint_angles
from core.pose.features import FEATURES_PER_FRAME

torch = pytest.importorskip("torch")


TARGET_FRAMES = 64


def _make_frames(n: int = 20) -> list[FrameAnalysis]:
    base = np.array([
        [320, 100], [310, 90], [330, 90], [300, 95], [340, 95],
        [280, 200], [360, 200], [260, 300], [380, 300],
        [240, 400], [400, 400], [290, 400], [350, 400],
        [290, 540], [350, 540], [290, 680], [350, 680],
    ], dtype=np.float32)
    confs = np.full(17, 0.95, dtype=np.float32)
    rng = np.random.default_rng(0)
    frames = []
    for i in range(n):
        kps = base.copy() + rng.normal(0, 1, base.shape).astype(np.float32)
        angles = get_joint_angles(kps, confs)
        frames.append(FrameAnalysis(
            frame_idx=i,
            timestamp_ms=i * 33.33,
            keypoints=kps,
            keypoint_confidences=confs,
            angles=angles,
        ))
    return frames


def _make_trick_config(trick_id: str = "test") -> TrickConfig:
    return TrickConfig(
        trick_id=trick_id, category="flip", difficulty=3.0,
        phases=[TrickPhase(name="exec", duration_range_ms=(200, 1000))],
        names={"en": "Test"},
    )


class TestTrickTCN:
    def test_forward_shape(self):
        from ml.tcn.model import TrickTCN
        model = TrickTCN(num_classes=5, n_features=FEATURES_PER_FRAME)
        x = torch.randn(4, FEATURES_PER_FRAME, TARGET_FRAMES)
        out = model(x)
        assert out.shape == (4, 5)

    def test_predict_proba_sums_to_one(self):
        from ml.tcn.model import TrickTCN
        model = TrickTCN(num_classes=3, n_features=FEATURES_PER_FRAME)
        x = torch.randn(2, FEATURES_PER_FRAME, TARGET_FRAMES)
        probs = model.predict_proba(x)
        assert probs.shape == (2, 3)
        sums = probs.sum(dim=1)
        torch.testing.assert_close(sums, torch.ones(2), atol=1e-5, rtol=0)

    def test_single_sample(self):
        from ml.tcn.model import TrickTCN
        model = TrickTCN(num_classes=10, n_features=FEATURES_PER_FRAME)
        model.train(False)
        x = torch.randn(1, FEATURES_PER_FRAME, TARGET_FRAMES)
        out = model(x)
        assert out.shape == (1, 10)

    def test_variable_time_length(self):
        from ml.tcn.model import TrickTCN
        model = TrickTCN(num_classes=5, n_features=FEATURES_PER_FRAME)
        model.train(False)
        for T in [32, 64, 128]:
            x = torch.randn(1, FEATURES_PER_FRAME, T)
            out = model(x)
            assert out.shape == (1, 5), f"Failed for T={T}"

    def test_custom_architecture(self):
        from ml.tcn.model import TrickTCN
        model = TrickTCN(
            num_classes=28,
            n_features=FEATURES_PER_FRAME,
            hidden_channels=(64, 64, 64),
            kernel_size=5,
            dropout=0.3,
        )
        model.train(False)
        x = torch.randn(1, FEATURES_PER_FRAME, TARGET_FRAMES)
        out = model(x)
        assert out.shape == (1, 28)

    def test_gradients_flow(self):
        from ml.tcn.model import TrickTCN
        model = TrickTCN(num_classes=5, n_features=FEATURES_PER_FRAME)
        x = torch.randn(2, FEATURES_PER_FRAME, TARGET_FRAMES, requires_grad=True)
        out = model(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape


class TestTCNStrategy:
    def _make_checkpoint(self, tmp_path, class_names=None):
        from ml.tcn.model import TrickTCN
        if class_names is None:
            class_names = ["no_trick", "back_flip", "front_flip"]
        model = TrickTCN(num_classes=len(class_names), n_features=FEATURES_PER_FRAME)
        metadata = {
            "class_names": class_names,
            "num_classes": len(class_names),
            "n_features": FEATURES_PER_FRAME,
            "target_frames": TARGET_FRAMES,
            "best_val_acc": 0.9,
            "epochs": 10,
            "model_type": "tcn",
        }
        path = tmp_path / "test_tcn.pt"
        torch.save({"model_state_dict": model.state_dict(), "metadata": metadata}, path)
        return path

    def test_load_checkpoint(self, tmp_path):
        from ml.tcn.inference import TCNStrategy
        path = self._make_checkpoint(tmp_path)
        strategy = TCNStrategy(checkpoint_path=path)
        assert strategy.is_loaded()
        assert strategy.class_names == ["no_trick", "back_flip", "front_flip"]

    def test_no_model_returns_none(self):
        from ml.tcn.inference import TCNStrategy
        strategy = TCNStrategy()
        assert not strategy.is_loaded()
        frames = _make_frames()
        trick = _make_trick_config("back_flip")
        assert strategy.evaluate(trick, frames) is None

    def test_evaluate_returns_detection(self, tmp_path):
        from ml.tcn.inference import TCNStrategy
        path = self._make_checkpoint(tmp_path, ["no_trick", "test"])
        strategy = TCNStrategy(checkpoint_path=path, min_confidence=0.0)
        frames = _make_frames()
        trick = _make_trick_config("test")
        detection = strategy.evaluate(trick, frames)
        if detection is not None:
            assert detection.trick_id == "test"
            assert detection.strategy_used == "tcn"
            assert 0.0 <= detection.confidence <= 1.0

    def test_predict_all(self, tmp_path):
        from ml.tcn.inference import TCNStrategy
        path = self._make_checkpoint(tmp_path, ["no_trick", "a", "b"])
        strategy = TCNStrategy(checkpoint_path=path)
        frames = _make_frames()
        probs = strategy.predict_all(frames)
        assert set(probs.keys()) == {"no_trick", "a", "b"}
        assert abs(sum(probs.values()) - 1.0) < 0.01

    def test_empty_frames(self, tmp_path):
        from ml.tcn.inference import TCNStrategy
        path = self._make_checkpoint(tmp_path)
        strategy = TCNStrategy(checkpoint_path=path)
        trick = _make_trick_config("back_flip")
        assert strategy.evaluate(trick, []) is None

    def test_missing_checkpoint(self, tmp_path):
        from ml.tcn.inference import TCNStrategy
        strategy = TCNStrategy(checkpoint_path=tmp_path / "nonexistent.pt")
        assert not strategy.is_loaded()
