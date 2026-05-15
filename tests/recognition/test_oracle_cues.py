import json
import pytest
from core.recognition.oracle_cues import OracleCueBook, OracleCueError

FIG_PATH = "data/fig_tricks_2025.json"

@pytest.fixture(scope="module")
def book():
    return OracleCueBook(FIG_PATH)

def test_known_trick_returns_ontology_attributes(book):
    raw = json.load(open(FIG_PATH))
    cat, trick = next(
        (c, t) for c, v in raw["categories"].items() for t in v["tricks"]
    )
    cues = book.cues_for(trick["name"])
    assert cues["context"] == cat
    assert cues["flip"] == trick["flip"]
    assert cues["twist"] == trick["twist"]
    for k in ("direction", "axis", "entry", "takeoff", "hand_contact", "kick"):
        if k in trick:
            assert cues[k] == trick[k]
        else:
            assert k not in cues

def test_alias_resolves_to_canonical(book):
    raw = json.load(open(FIG_PATH))
    aliased = next(
        (t for v in raw["categories"].values() for t in v["tricks"]
         if t.get("aliases")),
        None,
    )
    if aliased is None:
        pytest.skip("no aliased trick in ontology")
    cues_by_alias = book.cues_for(aliased["aliases"][0])
    cues_by_name = book.cues_for(aliased["name"])
    assert cues_by_alias == cues_by_name

def test_unknown_trick_raises(book):
    with pytest.raises(OracleCueError):
        book.cues_for("definitely not a real fig trick xyz")

def test_d_score_lookup(book):
    raw = json.load(open(FIG_PATH))
    cat, trick = next(
        (c, t) for c, v in raw["categories"].items() for t in v["tricks"]
    )
    assert book.d_score_for(trick["name"]) == trick["score"]
