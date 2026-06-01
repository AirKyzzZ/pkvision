from core.labeling.store import VerifiedStore, VerifiedRecord


def test_append_and_read_roundtrip(tmp_path):
    store = VerifiedStore(tmp_path / "verified.jsonl")
    rec = VerifiedRecord(slug="back_full", trick="Backflip 360",
                         cues={"flip": 1.0, "twist": 1.0, "direction": "backward"},
                         d_score=2.0, proposer_source="local", action="confirm",
                         verified_at="2026-06-01T10:00:00Z")
    store.append(rec)
    rows = store.read_all()
    assert len(rows) == 1
    assert rows[0].slug == "back_full"
    assert rows[0].cues["twist"] == 1.0


def test_verified_slugs_set(tmp_path):
    store = VerifiedStore(tmp_path / "v.jsonl")
    store.append(VerifiedRecord("a", "T", {}, 1.0, "local", "confirm", "t"))
    store.append(VerifiedRecord("b", "T", {}, 1.0, "local", "confirm", "t"))
    assert store.verified_slugs() == {"a", "b"}
