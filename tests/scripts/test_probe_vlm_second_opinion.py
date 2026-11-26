from scripts import probe_vlm_second_opinion as probe


def label(flip, direction):
    return {"flip": flip, "twist": 1, "direction": direction, "context": "wall", "axis": None}


def vote(flip, direction, unsure=False):
    return {"flip": flip, "twist": 0, "direction": direction, "unsure": unsure}


def test_consensus_masks_only_contradicted_flip_and_direction_labels():
    labels = {"agree": label(1, "backward"), "flip_off": label(1, "backward"), "none": label(0, "none"),
              "unsure": label(2, "side"), "unseen": label(1, "forward")}
    votes = {"agree": vote(1.0, "backward"), "flip_off": vote(2.0, "backward"), "none": vote(0, None),
             "unsure": vote(1.0, "forward", unsure=True)}

    fixed, stats = probe.consensus(labels, votes)

    assert fixed["agree"] == label(1, "backward")
    assert fixed["flip_off"] == label(None, "backward")
    assert fixed["none"] == label(0, "none")
    assert fixed["unsure"] == label(2, "side") and fixed["unseen"] == label(1, "forward")
    assert all(row["twist"] == 1 for row in fixed.values())
    assert stats == {"flip_agree": 2, "direction_agree": 3, "flip_masked": 1, "flip_no_opinion": 2, "direction_no_opinion": 2}
    assert labels["flip_off"]["flip"] == 1
