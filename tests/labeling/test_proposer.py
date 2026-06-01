import numpy as np
from core.labeling.clip_ref import ClipRef
from core.labeling.proposer import Proposal, LocalModelProposer


class _StubModel:
    """Returns fixed cues so we test proposer wiring, not the net."""
    def predict(self, skeleton):
        return {"flip": 1.0, "flip_conf": 0.9, "direction": "backward",
                "direction_conf": 0.8, "context": "acrobatics", "context_conf": 0.7}


def test_local_proposer_emits_proposal():
    ref = ClipRef("back_full", None, None, skeleton=np.zeros((10, 17, 3), np.float32))
    prop = LocalModelProposer(model=_StubModel()).propose(ref)
    assert isinstance(prop, Proposal)
    assert prop.cues["flip"] == 1.0
    assert "flip_conf" not in prop.cues          # confidences stripped from cues
    assert 0.0 <= prop.confidence <= 1.0
    assert prop.source == "local"


def test_local_proposer_raises_on_missing_skeleton():
    ref = ClipRef("no_skel", None, None, skeleton=None)
    import pytest
    with pytest.raises(ValueError, match="no skeleton"):
        LocalModelProposer(model=_StubModel()).propose(ref)


def test_local_proposer_with_real_cue_model():
    from core.labeling.cue_model import CueModel, CUE_CLASSES
    ref = ClipRef("x", None, None, skeleton=np.zeros((20, 17, 3), np.float32))
    prop = LocalModelProposer(model=CueModel(CUE_CLASSES, 48)).propose(ref)
    assert isinstance(prop, Proposal) and prop.source == "local"
