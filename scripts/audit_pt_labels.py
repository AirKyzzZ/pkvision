from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.recognition.trick_name_parser import parse_trick_name

MANIFEST = ROOT / "data/v5_attribute_training/manifest_baseline_noeval.json"
GOLD = ROOT / "data/v5_attribute_training/eval_gold_concordant.json"
OUT = ROOT / "docs/science-superpowers/probes/2026-11-26-pt-label-audit.json"


def half_step(value) -> float:
    return round(float(value or 0) * 2) / 2


def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def graph_categories(path: Path) -> dict[str, set[str]]:
    nodes = json.loads(path.read_text(encoding="utf-8"))["nodes"]
    return {slugify(n["name"]): set(n.get("categoryIds") or []) for n in nodes if n["kind"] == "move"}


def against_parser(clips: list[dict], cue: str) -> dict:
    relation = Counter()
    for c in clips:
        parsed = parse_trick_name(c["slug"].replace("_", " ")).cues
        if cue not in parsed:
            relation["parser_abstains"] += 1
        else:
            relation["agree" if half_step(parsed[cue]) == half_step(c.get(cue)) else "disagree"] += 1
    return dict(relation)


def against_categories(rows: list[tuple[float, set[str]]], category: str) -> dict:
    table = Counter((min(value, 2.0), category in cats) for value, cats in rows)
    return {str(v): {"pt_yes": table[(v, True)], "pt_no": table[(v, False)]} for v in sorted({k[0] for k in table})}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", type=Path, required=True)
    args = ap.parse_args()
    categories = graph_categories(args.graph)
    clips = json.loads(MANIFEST.read_text(encoding="utf-8"))["clips"]
    gold = json.loads(GOLD.read_text(encoding="utf-8"))
    matched = [c for c in clips if c["slug"] in categories]
    report = {
        "graph": args.graph.name,
        "train_clips": len(clips),
        "train_matched": len(matched),
        "manifest_flip": dict(sorted(Counter(half_step(c.get("flip")) for c in clips).items())),
        "manifest_twist": dict(sorted(Counter(half_step(c.get("twist")) for c in clips).items())),
        "manifest_direction": dict(sorted(Counter(str(c.get("direction") or "none") for c in clips).items())),
        "manifest_vs_parser": {cue: against_parser(clips, cue) for cue in ("flip", "twist")},
        "train_flip_vs_pt": against_categories([(half_step(c.get("flip")), categories[c["slug"]]) for c in matched], "category:flip"),
        "train_twist_vs_pt": against_categories([(half_step(c.get("twist")), categories[c["slug"]]) for c in matched], "category:twist"),
        "gold_clips": len(gold),
        "gold_has_flip_vs_pt": against_categories(
            [(float(v["flip"] > 0), categories[s]) for s, v in gold.items() if "flip" in v and s in categories], "category:flip"),
        "gold_has_twist_vs_pt": against_categories(
            [(float(v["twist"] > 0), categories[s]) for s, v in gold.items() if "twist" in v and s in categories], "category:twist"),
    }
    OUT.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
