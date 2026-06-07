"""Gate 1: parser precision + abstention on FIG 149 + gold 99."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.recognition.trick_name_parser import parse_trick_name

FIG = ROOT / "data" / "fig_tricks_2025.json"
GOLD = ROOT / "data" / "p0_blind_set" / "verified.csv"
NUMERIC = {"flip", "twist"}
CUES = ("flip", "twist", "direction", "axis", "context")

# FIG labels contexts/directions in its own vocab; the parser + manifest use another.
# Normalize FIG -> manifest vocab so the comparison is apples-to-apples.
FIG_CONTEXT_MAP = {"bar_or_rail": "swing", "ground": "acrobatics", "obstacle": "pk_basics", "wall": "wall"}
FIG_DIR_MAP = {"left": "side", "right": "side", "sideways": "side"}


def _eq(cue, a, b) -> bool:
    if cue in NUMERIC:
        return abs(float(a) - float(b)) < 0.26
    return str(a) == str(b)


def score_cue(cue: str, preds: list, golds: list) -> dict:
    fired = correct = abstained = 0
    for p, g in zip(preds, golds):
        if cue not in g or g[cue] in (None, "none", ""):
            continue
        if cue not in p:
            abstained += 1
            continue
        fired += 1
        if _eq(cue, p[cue], g[cue]):
            correct += 1
    prec = correct / fired if fired else 0.0
    return {"fired": fired, "correct": correct, "abstained": abstained, "precision": prec}


def load_fig_rows() -> list:
    fig = json.loads(FIG.read_text())
    rows = []
    for cat in fig["categories"].values():
        ctx = FIG_CONTEXT_MAP.get(cat.get("context"), cat.get("context"))
        for t in cat["tricks"]:
            rows.append({"slug": t["name"], "cues": {
                "flip": t.get("flip"), "twist": t.get("twist"),
                "direction": FIG_DIR_MAP.get(t.get("direction"), t.get("direction")),
                "axis": t.get("axis"), "context": ctx}})
    return rows


def load_gold_rows() -> list:
    rows = []
    with GOLD.open() as f:
        for r in csv.DictReader(f):
            if r.get("keep") != "1":
                continue
            slug = Path(r["clip_path"]).stem
            cues = {}
            for part in r["disambig_group"].split("|"):
                if "=" in part:
                    k, v = part.split("=", 1)
                    cues[{"dir": "direction"}.get(k, k)] = v
                elif part:
                    cues["context"] = part
            rows.append({"slug": slug, "cues": cues})
    return rows


def _report(name: str, rows: list) -> None:
    preds = [parse_trick_name(r["slug"]).cues for r in rows]
    golds = [r["cues"] for r in rows]
    print(f"\n=== {name} (n={len(rows)}) ===")
    print(f"{'cue':<11}{'fired':<7}{'correct':<9}{'abstain':<9}{'precision':<10}")
    for cue in CUES:
        s = score_cue(cue, preds, golds)
        flag = "  <-- below 0.90" if s["fired"] and s["precision"] < 0.90 else ""
        print(f"{cue:<11}{s['fired']:<7}{s['correct']:<9}{s['abstained']:<9}{s['precision']:<10.3f}{flag}")


def main() -> None:
    _report("FIG 149", load_fig_rows())
    _report("GOLD 99", load_gold_rows())


if __name__ == "__main__":
    main()
