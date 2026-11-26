from scripts import probe_pt_label_fix as probe


def row(flip, twist, direction="backward"):
    return {"flip": flip, "twist": twist, "direction": direction, "axis": "lateral", "context": "wall"}


def test_corrected_zeroes_non_flips_and_masks_missed_twists(monkeypatch):
    monkeypatch.setattr(probe, "pt_categories", lambda: {
        "split_vault": {"vault"},
        "wall_full": {"wall", "flip", "twist"},
        "back_flip": {"flip"},
    })
    labels = {"split_vault": row(1, 0), "wall_full": row(1, 0), "back_flip": row(1, 0), "unknown_move": row(2, 1)}

    fixed, changes = probe.corrected(labels)

    assert fixed["split_vault"] == {**row(0, 0, "none"), "axis": None}
    assert fixed["wall_full"]["twist"] is None and fixed["wall_full"]["flip"] == 1
    assert fixed["back_flip"] == row(1, 0)
    assert fixed["unknown_move"] == row(2, 1)
    assert dict(changes) == {"flip_set_to_0": 1, "direction_set_to_none": 1, "twist_masked": 1, "unmatched": 1}
    assert labels["split_vault"]["flip"] == 1


def test_slugify_matches_clip_slugs():
    assert probe.slugify("Inward Webster Rudi") == "inward_webster_rudi"
    assert probe.slugify("360 Dive-Roll (Left)") == "360_dive_roll_left"
