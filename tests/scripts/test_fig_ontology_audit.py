from scripts.fig_ontology_audit import find_alias_duplicates


def test_finds_self_referential_alias_duplicates():
    fig = {"categories": {"acrobatics": {"tricks": [
        {"name": "Double Backflip 360", "score": 5.0, "flip": 2, "twist": 1,
         "aliases": ["Cork-in Backflip"]},
        {"name": "Cork-in Backflip", "score": 5.0, "flip": 2, "twist": 1},
        {"name": "Backflip", "score": 1.5, "flip": 1, "twist": 0},
    ]}}}
    dups = find_alias_duplicates(fig)
    assert ("Cork-in Backflip", "Double Backflip 360") in [
        (d["duplicate"], d["canonical"]) for d in dups]
    assert all(d["duplicate"] != "Backflip" for d in dups)


def test_normalized_name_collision_reports_all_matches():
    from scripts.fig_ontology_audit import find_alias_duplicates
    fig = {"categories": {"acro": {"tricks": [
        {"name": "Double-Backflip", "score": 5.0},
        {"name": "Double Backflip", "score": 5.0},
        {"name": "X", "score": 1.0, "aliases": ["double backflip"]},
    ]}}}
    dups = find_alias_duplicates(fig)
    reported = sorted(d["duplicate"] for d in dups if d["canonical"] == "X")
    assert reported == ["Double Backflip", "Double-Backflip"]  # BOTH, not last-wins
