"""Compositional parkour trick-name -> cue parser. High-precision or abstain."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_LEXICON_PATH = _ROOT / "data" / "name_grammar" / "lexicon.json"

ABSTAIN_THRESHOLD = 0.5

MULTIWORD = {
    ("one", "and", "a", "half"): "one_and_a_half",
    ("double", "full"): "double_full",
    ("triple", "full"): "triple_full",
}


@dataclass
class ParsedCues:
    cues: dict = field(default_factory=dict)
    confidence: dict = field(default_factory=dict)
    trace: list = field(default_factory=list)
    unparsed_tokens: list = field(default_factory=list)


@lru_cache(maxsize=1)
def _load_lexicon() -> dict:
    if not _LEXICON_PATH.exists():
        return {"moves": {}, "twist_words": {}, "flip_words": {},
                "numeric": {}, "phase_boundaries": []}
    return json.loads(_LEXICON_PATH.read_text())


def tokenize(name: str) -> list:
    raw = [t for t in re.split(r"[_\-\s]+", name.strip().lower()) if t]
    out, i = [], 0
    while i < len(raw):
        matched = False
        for seq, atom in MULTIWORD.items():
            n = len(seq)
            if tuple(raw[i:i + n]) == seq:
                out.append(atom)
                i += n
                matched = True
                break
        if not matched:
            out.append(raw[i])
            i += 1
    return out


def parse_trick_name(name: str) -> ParsedCues:
    if not isinstance(name, str):
        raise TypeError(f"trick name must be str, got {type(name).__name__}")
    tokens = tokenize(name)
    lex = _load_lexicon()
    known = set(lex["moves"]) | set(lex["twist_words"]) | set(lex["flip_words"]) \
        | set(lex["numeric"]) | set(lex["phase_boundaries"])
    unparsed = [t for t in tokens if t not in known]
    return ParsedCues(cues={}, confidence={}, trace=[], unparsed_tokens=unparsed)
