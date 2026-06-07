import json
from pathlib import Path

from core.recognition.trick_name_parser import parse_trick_name

GOLDEN = json.loads((Path(__file__).parent / "golden_parses.json").read_text())


def test_golden_parses_stable():
    for slug, expected in GOLDEN.items():
        assert parse_trick_name(slug).cues == expected, slug
