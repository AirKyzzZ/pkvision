"""Draft token->cue associations (with support + purity) from FIG + unified_tricks."""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CUE_KEYS = ("direction", "rotation_axis", "body_shape", "entry", "family")


def _tokens(name: str) -> list:
    return [t for t in re.split(r"[_\-\s]+", name.strip().lower()) if t and not t.isdigit()]


def token_associations(rows: list) -> dict:
    assoc: dict = defaultdict(lambda: {k: defaultdict(int) for k in CUE_KEYS} | {"_support": 0})
    for r in rows:
        phys = r.get("physics", {}) or {}
        for tok in set(_tokens(r["name"])):
            assoc[tok]["_support"] += 1
            for k in CUE_KEYS:
                v = phys.get(k)
                if v is not None:
                    assoc[tok][k][str(v)] += 1
    return {t: {k: (dict(v) if k != "_support" else v) for k, v in d.items()} for t, d in assoc.items()}


def _load_rows() -> list:
    rows = json.loads((ROOT / "data" / "unified_tricks.json").read_text())
    return [{"name": t.get("canonical_name", t["name"]), "physics": t.get("physics", {})} for t in rows]


def main() -> None:
    rows = _load_rows()
    assoc = token_associations(rows)
    ranked = sorted(assoc.items(), key=lambda kv: -kv[1]["_support"])
    out = ROOT / "data" / "name_grammar" / "lexicon_draft.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(ranked), indent=2))
    print(f"mined {len(assoc)} tokens -> {out}")
    for tok, d in ranked[:25]:
        best = {k: max(v, key=v.get) for k, v in d.items() if k != "_support" and v}
        print(f"  {tok:<14} support={d['_support']:<4} {best}")


if __name__ == "__main__":
    main()
