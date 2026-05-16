"""Audit data/fig_tricks_2025.json for trick entries that are aliases of
another trick (data duplicates that create false recognition errors).
Read-only: reports; does not mutate the ontology.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.recognition.oracle_cues import _norm  # noqa: E402


def find_alias_duplicates(fig: dict) -> list[dict]:
    tricks = [t for o in fig["categories"].values() for t in o["tricks"]]
    by_norm = {_norm(t["name"]): t for t in tricks}
    dups: list[dict] = []
    for t in tricks:
        for alias in t.get("aliases", []) or []:
            an = _norm(alias)
            if an in by_norm and an != _norm(t["name"]):
                dups.append({"duplicate": by_norm[an]["name"],
                             "canonical": t["name"]})
    return dups


def main() -> None:
    fig = json.loads(Path("data/fig_tricks_2025.json").read_text())
    dups = find_alias_duplicates(fig)
    print(f"{len(dups)} alias-duplicate trick entries:")
    for d in dups:
        print(f"  - {d['duplicate']!r} is an alias of {d['canonical']!r}")


if __name__ == "__main__":
    main()
