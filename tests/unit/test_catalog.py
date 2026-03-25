"""Tests for trick catalog loading and schema validation."""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest

from core.models import TrickConfig, TrickPrimitives

CATALOG_DIR = Path(__file__).parent.parent.parent / "data" / "tricks" / "catalog"
SCHEMA_PATH = Path(__file__).parent.parent.parent / "data" / "tricks" / "schema.json"

EXPECTED_TRICKS = {
    "kong_vault",
    "double_kong",
    "front_flip",
    "back_flip",
    "side_flip",
    "webster",
    "gainer",
    "double_front",
    "flip_360",
    "triple_cork",
}

EXPECTED_DIFFICULTIES = {
    "kong_vault": 2.0,
    "double_kong": 3.0,
    "front_flip": 3.5,
    "back_flip": 3.5,
    "side_flip": 3.0,
    "webster": 4.0,
    "gainer": 4.5,
    "double_front": 6.0,
    "flip_360": 5.0,
    "triple_cork": 8.0,
}


@pytest.fixture
def schema() -> dict:
    with open(SCHEMA_PATH) as f:
        return json.load(f)


@pytest.fixture
def en_tricks() -> list[dict]:
    tricks = []
    en_dir = CATALOG_DIR / "en"
    for path in sorted(en_dir.glob("*.json")):
        with open(path) as f:
            tricks.append(json.load(f))
    return tricks


def test_schema_file_exists():
    assert SCHEMA_PATH.exists(), "schema.json must exist"


def test_catalog_directories_exist():
    assert (CATALOG_DIR / "en").exists(), "en/ catalog must exist"
    assert (CATALOG_DIR / "fr").exists(), "fr/ catalog must exist"


def test_all_10_tricks_present():
    en_dir = CATALOG_DIR / "en"
    trick_files = {p.stem for p in en_dir.glob("*.json")}
    assert trick_files == EXPECTED_TRICKS, f"Missing tricks: {EXPECTED_TRICKS - trick_files}"


def test_fr_catalog_matches_en():
    en_files = {p.name for p in (CATALOG_DIR / "en").glob("*.json")}
    fr_files = {p.name for p in (CATALOG_DIR / "fr").glob("*.json")}
    assert en_files == fr_files, f"FR catalog missing: {en_files - fr_files}"


def test_all_tricks_validate_against_schema(en_tricks: list[dict], schema: dict):
    for trick in en_tricks:
        try:
            jsonschema.validate(trick, schema)
        except jsonschema.ValidationError as e:
            pytest.fail(f"Trick '{trick.get('trick_id', '?')}' failed schema validation: {e.message}")


def test_all_tricks_load_as_pydantic(en_tricks: list[dict]):
    for trick_data in en_tricks:
        config = TrickConfig(**trick_data)
        assert config.trick_id in EXPECTED_TRICKS
        assert 0.0 <= config.difficulty <= 10.0
        assert len(config.phases) >= 1
        assert "en" in config.names


def test_difficulty_ratings(en_tricks: list[dict]):
    for trick_data in en_tricks:
        config = TrickConfig(**trick_data)
        expected = EXPECTED_DIFFICULTIES.get(config.trick_id)
        assert expected is not None, f"Unknown trick: {config.trick_id}"
        assert config.difficulty == expected, (
            f"{config.trick_id}: expected difficulty {expected}, got {config.difficulty}"
        )


def test_all_tricks_have_bilingual_names(en_tricks: list[dict]):
    for trick_data in en_tricks:
        config = TrickConfig(**trick_data)
        assert "en" in config.names, f"{config.trick_id} missing English name"
        assert "fr" in config.names, f"{config.trick_id} missing French name"


def test_temporal_model_tricks_are_complex():
    """Tricks using temporal_model should be difficulty >= 5.0."""
    en_dir = CATALOG_DIR / "en"
    for path in en_dir.glob("*.json"):
        with open(path) as f:
            data = json.load(f)
        if data["detection_method"] == "temporal_model":
            assert data["difficulty"] >= 5.0, (
                f"{data['trick_id']}: temporal_model trick should have difficulty >= 5.0"
            )


def test_all_tricks_have_angle_rules(en_tricks: list[dict]):
    """Every trick should have at least some angle rules, even temporal_model ones."""
    for trick_data in en_tricks:
        config = TrickConfig(**trick_data)
        total_rules = sum(len(phase.angle_rules) for phase in config.phases)
        assert total_rules > 0, f"{config.trick_id} has no angle rules in any phase"


VALID_ROTATION_SAGITTAL = {"forward", "backward", "none"}
VALID_ROTATION_LATERAL = {"left", "right", "none"}
VALID_TAKEOFF = {"two_feet", "one_foot", "hands", "wall", "edge"}
VALID_LANDING = {"two_feet", "one_foot", "hands", "other"}

EXPECTED_PRIMITIVES = {
    "kong_vault": {
        "rotation_sagittal": "forward",
        "rotation_sagittal_count": 0,
        "rotation_lateral": "none",
        "twist_count": 0,
        "takeoff": "two_feet",
        "landing": "two_feet",
        "obstacle_interaction": True,
    },
    "double_kong": {
        "rotation_sagittal": "forward",
        "rotation_sagittal_count": 0,
        "rotation_lateral": "none",
        "twist_count": 0,
        "takeoff": "two_feet",
        "landing": "two_feet",
        "obstacle_interaction": True,
    },
    "front_flip": {
        "rotation_sagittal": "forward",
        "rotation_sagittal_count": 1,
        "rotation_lateral": "none",
        "twist_count": 0,
        "takeoff": "two_feet",
        "landing": "two_feet",
        "obstacle_interaction": False,
    },
    "back_flip": {
        "rotation_sagittal": "backward",
        "rotation_sagittal_count": 1,
        "rotation_lateral": "none",
        "twist_count": 0,
        "takeoff": "two_feet",
        "landing": "two_feet",
        "obstacle_interaction": False,
    },
    "side_flip": {
        "rotation_sagittal": "none",
        "rotation_sagittal_count": 0,
        "rotation_lateral": "left",
        "twist_count": 0,
        "takeoff": "two_feet",
        "landing": "two_feet",
        "obstacle_interaction": False,
    },
    "webster": {
        "rotation_sagittal": "forward",
        "rotation_sagittal_count": 1,
        "rotation_lateral": "none",
        "twist_count": 0,
        "takeoff": "one_foot",
        "landing": "two_feet",
        "obstacle_interaction": False,
    },
    "gainer": {
        "rotation_sagittal": "backward",
        "rotation_sagittal_count": 1,
        "rotation_lateral": "none",
        "twist_count": 0,
        "takeoff": "one_foot",
        "landing": "two_feet",
        "obstacle_interaction": False,
    },
    "double_front": {
        "rotation_sagittal": "forward",
        "rotation_sagittal_count": 2,
        "rotation_lateral": "none",
        "twist_count": 0,
        "takeoff": "two_feet",
        "landing": "two_feet",
        "obstacle_interaction": False,
    },
    "flip_360": {
        "rotation_sagittal": "backward",
        "rotation_sagittal_count": 1,
        "rotation_lateral": "none",
        "twist_count": 1,
        "takeoff": "two_feet",
        "landing": "two_feet",
        "obstacle_interaction": False,
    },
    "triple_cork": {
        "rotation_sagittal": "backward",
        "rotation_sagittal_count": 1,
        "rotation_lateral": "left",
        "twist_count": 3,
        "takeoff": "one_foot",
        "landing": "two_feet",
        "obstacle_interaction": False,
    },
}


def test_all_tricks_have_primitives(en_tricks: list[dict]):
    """Every trick must have a primitives field with all required keys."""
    for trick_data in en_tricks:
        trick_id = trick_data["trick_id"]
        assert "primitives" in trick_data, f"{trick_id} missing primitives field"
        prims = trick_data["primitives"]
        for key in [
            "rotation_sagittal",
            "rotation_sagittal_count",
            "rotation_lateral",
            "twist_count",
            "takeoff",
            "landing",
            "obstacle_interaction",
        ]:
            assert key in prims, f"{trick_id} primitives missing key: {key}"


def test_primitives_have_valid_values(en_tricks: list[dict]):
    """All primitive enum values must be within the allowed sets."""
    for trick_data in en_tricks:
        trick_id = trick_data["trick_id"]
        prims = trick_data["primitives"]
        assert prims["rotation_sagittal"] in VALID_ROTATION_SAGITTAL, (
            f"{trick_id}: invalid rotation_sagittal '{prims['rotation_sagittal']}'"
        )
        assert prims["rotation_lateral"] in VALID_ROTATION_LATERAL, (
            f"{trick_id}: invalid rotation_lateral '{prims['rotation_lateral']}'"
        )
        assert prims["takeoff"] in VALID_TAKEOFF, (
            f"{trick_id}: invalid takeoff '{prims['takeoff']}'"
        )
        assert prims["landing"] in VALID_LANDING, (
            f"{trick_id}: invalid landing '{prims['landing']}'"
        )
        assert isinstance(prims["rotation_sagittal_count"], int) and prims["rotation_sagittal_count"] >= 0
        assert isinstance(prims["twist_count"], int) and prims["twist_count"] >= 0
        assert isinstance(prims["obstacle_interaction"], bool)


def test_primitives_match_expected_values(en_tricks: list[dict]):
    """Each trick's primitives must match the expected values exactly."""
    for trick_data in en_tricks:
        trick_id = trick_data["trick_id"]
        expected = EXPECTED_PRIMITIVES.get(trick_id)
        assert expected is not None, f"No expected primitives for {trick_id}"
        prims = trick_data["primitives"]
        for key, val in expected.items():
            assert prims[key] == val, (
                f"{trick_id}.primitives.{key}: expected {val}, got {prims[key]}"
            )


def test_primitives_load_as_pydantic(en_tricks: list[dict]):
    """Primitives should parse into TrickPrimitives via TrickConfig."""
    for trick_data in en_tricks:
        config = TrickConfig(**trick_data)
        assert config.primitives is not None, f"{config.trick_id} primitives is None"
        assert isinstance(config.primitives, TrickPrimitives)


def test_fr_tricks_have_same_primitives_as_en():
    """French catalog primitives must match English catalog primitives."""
    for trick_id in EXPECTED_TRICKS:
        en_path = CATALOG_DIR / "en" / f"{trick_id}.json"
        fr_path = CATALOG_DIR / "fr" / f"{trick_id}.json"
        with open(en_path) as f:
            en_prims = json.load(f).get("primitives")
        with open(fr_path) as f:
            fr_prims = json.load(f).get("primitives")
        assert en_prims == fr_prims, (
            f"{trick_id}: EN and FR primitives differ"
        )


def test_get_name_method():
    trick = TrickConfig(
        trick_id="test",
        category="flip",
        difficulty=1.0,
        detection_method="angle_threshold",
        phases=[{"name": "execution", "duration_range_ms": [100, 500]}],
        names={"en": "Test Trick", "fr": "Trick Test"},
    )
    assert trick.get_name("en") == "Test Trick"
    assert trick.get_name("fr") == "Trick Test"
    assert trick.get_name("de") == "Test Trick"  # fallback to English
