import csv

from scripts import pt_partner_data as pt
from scripts.pt_partner_data import as_gold, build_sheet, compare, corpus_slug, slugify, validate


def move(name, type_="Flip", url_slug=None, **extra):
    return {"name": name, "type": type_, "url": f"https://parkourtheory.com/move/{url_slug or slugify(name)}", **extra}


def test_slugify_matches_corpus_file_names():
    assert slugify("One-Step Palm Gainer Full") == "one_step_palm_gainer_full"
    assert slugify(None) == ""


def test_corpus_slug_falls_back_to_name_on_site_and_alias():
    corpus = {"wall_full", "parkour_roll"}
    assert corpus_slug(move("Wall Full", url_slug="wall_full_2"), corpus) == "wall_full"
    assert corpus_slug(move("Shoulder Roll", alias="Parkour Roll"), corpus) == "parkour_roll"
    assert corpus_slug(move("Kong Vault", "Vault"), corpus) is None


def test_sheet_puts_corpus_twists_then_flips_first_and_leaves_fill_columns_empty():
    moves = [move("Kong Vault", "Vault"), move("Back Full", "Flip/Twist"), move("Dash Vault", "Vault"),
             move("Wall Flip", "Wall/Flip"), {"name": "Broken", "error": "404"}]
    rows = build_sheet(moves, {"back_full", "dash_vault", "wall_flip"})
    assert [(r["name"], r["priority"]) for r in rows] == [
        ("Back Full", 1), ("Wall Flip", 2), ("Dash Vault", 3), ("Kong Vault", 4)]
    assert rows[0]["pkvision_clip"] == "back_full" and rows[3]["pkvision_clip"] == ""
    assert all(r[c] == "" for r in rows for c in pt.FILL_COLUMNS)


def test_validate_accepts_blanks_question_marks_and_combined_hands():
    assert validate({"flips": "1.25", "twists": "?", "flip_dir": "Back", "hands": "wall+obstacle",
                     "move_start_s": "0.4", "move_end_s": "1.1"}) == []
    assert validate({}) == []


def test_validate_reports_each_bad_value():
    errors = validate({"flips": "1.3", "twists": "-1", "axis": "cork", "hands": "wall+feet",
                       "move_start_s": "2", "move_end_s": "1"})
    assert len(errors) == 5
    assert any(e.startswith("flips=1.3") for e in errors) and any(e.startswith("axis='cork'") for e in errors)
    assert validate({"twists": "one"}) == ["twists='one' is not a number"]


def test_as_gold_maps_to_manifest_fields():
    assert as_gold({"flips": "1", "twists": "0.5", "flip_dir": "back", "axis": "off", "shape": "straight"}) == {
        "flip": 1.0, "twist": 0.5, "direction": "backward", "rotation_axis": "off_axis", "body_shape": "layout",
        "label_provenance": "parkourtheory"}
    assert as_gold({})["flip"] is None and as_gold({"shape": "any"})["body_shape"] is None


def test_compare_counts_agreement_only_where_both_sides_answer():
    gold = {"a": {"flip": 1.0, "twist": 0.0, "direction": "backward"},
            "b": {"flip": 1.0, "twist": 1.0, "direction": "backward"},
            "c": {"flip": 2.0, "twist": None, "direction": "forward"}}
    rows = [{"pkvision_clip": "a", "flips": "1", "twists": "0", "flip_dir": "back"},
            {"pkvision_clip": "b", "flips": "1", "twists": "2", "flip_dir": "?"},
            {"pkvision_clip": "c", "flips": "1", "twists": "1", "flip_dir": "front"},
            {"pkvision_clip": "", "flips": "3"}]
    result = compare(rows, gold)
    assert result["flip"]["compared"] == 3 and result["flip"]["agree"] == 2
    assert result["flip"]["top_disagreements"] == [[2.0, 1.0, 1]]
    assert result["twist"] == {"compared": 2, "agree": 1, "top_disagreements": [[1.0, 2.0, 1]]}
    assert result["direction"]["compared"] == 2 and result["direction"]["agree"] == 2


def test_check_prints_line_numbers_not_names_and_writes_labels(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(pt, "LABELS", tmp_path / "labels.json")
    monkeypatch.setattr(pt, "ROOT", tmp_path)
    monkeypatch.setattr(pt, "load_gold", lambda: {"back_full": {"flip": 1.0, "twist": 1.0, "direction": "backward"}})
    sheet = tmp_path / "returned.csv"
    columns = pt.CONTEXT_COLUMNS + pt.FILL_COLUMNS

    def write(rows):
        with sheet.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            writer.writerows([{c: r.get(c, "") for c in columns} for r in rows])

    write([{"name": "Back Full", "pkvision_clip": "back_full", "flips": "1.1"}])
    assert pt.main(["check", str(sheet)]) == 1
    out = capsys.readouterr().out
    assert "line 2: flips=1.1" in out and "Back Full" not in out

    write([{"name": "Back Full", "pkvision_clip": "back_full", "flips": "1", "twists": "1", "flip_dir": "back"},
           {"name": "Kong Vault"}])
    assert pt.main(["check", str(sheet), "--write"]) == 0
    assert "2 rows, 1 filled, 0 with errors" in capsys.readouterr().out
    assert '"back_full"' in (tmp_path / "labels.json").read_text(encoding="utf-8")
