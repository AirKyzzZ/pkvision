"""Append-only, auditable store of human-verified labels (data/labeling/verified.jsonl)."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class VerifiedRecord:
    slug: str
    trick: str | None
    cues: dict
    d_score: float | None
    proposer_source: str          # local / vlm
    action: str                   # confirm / correct_trick / override_cue
    verified_at: str              # ISO-8601 UTC


class VerifiedStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, rec: VerifiedRecord) -> None:
        with self.path.open("a") as f:
            f.write(json.dumps(asdict(rec)) + "\n")

    def read_all(self) -> list[VerifiedRecord]:
        if not self.path.exists():
            return []
        return [VerifiedRecord(**json.loads(line))
                for line in self.path.read_text().splitlines() if line.strip()]

    def verified_slugs(self) -> set[str]:
        return {r.slug for r in self.read_all()}
