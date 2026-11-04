import numpy as np

from scripts.probe_rule_counter import apply_affine, decision_12, invert_affine, reference_set


def test_reference_excludes_name_contradicted_labels():
    ref = reference_set()
    assert len(ref) == 47
    assert "flyaway_quadruple_full" not in ref and "front_full" not in ref


def test_decision_12_is_three_per_stratum():
    groups = decision_12(reference_set())
    assert sorted(groups) == ["twist0.5", "twist0_flip", "twist1", "twist2+"]
    assert all(len(v) == 3 for v in groups.values())


def test_affine_roundtrip():
    theta = np.radians(37)
    m = np.array([[np.cos(theta), np.sin(theta), 12.0], [-np.sin(theta), np.cos(theta), -4.0]])
    pts = np.array([[3.0, 9.0], [100.0, -5.0]])
    assert np.allclose(apply_affine(apply_affine(pts, m), invert_affine(m)), pts)
