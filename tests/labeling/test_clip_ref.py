import pytest
import numpy as np
from core.labeling.clip_ref import ClipRef


def test_frames_clip_reads_npy(tmp_path):
    arr = np.zeros((8, 64, 64, 3), np.uint8)
    p = tmp_path / "back_full.npy"
    np.save(p, arr)
    ref = ClipRef.from_path(p)
    assert ref.slug == "back_full"
    frames = ref.get_frames(max_frames=4)
    assert frames.shape == (4, 64, 64, 3)


def test_skeleton_attaches_and_returns(tmp_path):
    ref = ClipRef(slug="x", video_path=None, frames_path=tmp_path / "x.npy",
                  skeleton=np.zeros((10, 17, 3), np.float32))
    assert ref.get_skeleton().shape == (10, 17, 3)


def test_get_frames_raises_when_no_source():
    ref = ClipRef(slug="empty", video_path=None, frames_path=None)
    with pytest.raises(ValueError, match="no frames_path or video_path"):
        ref.get_frames()
