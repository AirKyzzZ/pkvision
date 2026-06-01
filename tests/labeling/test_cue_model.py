import numpy as np
from core.labeling.cue_model import featurize, base_seq, CUE_CLASSES


def test_featurize_shape():
    seq = np.zeros((48, 17, 3), np.float32)
    feat = featurize(seq)
    assert feat.shape == (48, 88)


def test_base_seq_resamples():
    arr = np.random.rand(13, 17, 3).astype(np.float32)
    out = base_seq(arr, 48)
    assert out.shape == (48, 17, 3)


def test_cue_classes_present():
    for c in ("context", "direction", "flip", "twist", "axis"):
        assert c in CUE_CLASSES
