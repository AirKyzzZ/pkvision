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
                "numeric": {"deg": {}, "flip_families": []}, "phase_boundaries": []}
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


@dataclass
class _Contrib:
    cue: str
    value: object
    conf: float
    rule: str


def segment_phases(tokens: list) -> list:
    lex = _load_lexicon()
    bounds = set(lex["phase_boundaries"])
    phases, cur = [], []
    for t in tokens:
        if t in bounds:
            phases.append(cur)
            cur = []
        else:
            cur.append(t)
    phases.append(cur)
    return phases


def _twist_contribs(tokens: list, lex: dict) -> list:
    tw = lex["twist_words"]
    return [_Contrib("twist", float(tw[t]), 0.8, f"twist_word:{t}") for t in tokens if t in tw]


FLIP_DIRECTION_NOUNS = {"front", "back", "side"}


def _is_flip_noun(tok, lex) -> bool:
    if tok is None:
        return False
    return (tok in set(lex["numeric"]["flip_families"])
            or tok in FLIP_DIRECTION_NOUNS
            or (tok in lex["moves"] and "flip" in lex["moves"][tok]))


def _flip_contribs(tokens: list, lex: dict) -> list:
    fw, moves = lex["flip_words"], lex["moves"]
    out, suppress = [], set()
    for i, t in enumerate(tokens):
        if t in fw:
            nxt = tokens[i + 1] if i + 1 < len(tokens) else None
            if not _is_flip_noun(nxt, lex):
                continue
            out.append(_Contrib("flip", float(fw[t]), 0.75, f"flip_count:{t}"))
            if nxt in moves and "flip" in moves[nxt]:
                suppress.add(i + 1)
    for i, t in enumerate(tokens):
        if i in suppress:
            continue
        m = moves.get(t)
        if m and "flip" in m:
            out.append(_Contrib("flip", float(m["flip"]), float(m["conf"]), f"move_flip:{t}"))
    return out


def _categorical_contribs(tokens: list, lex: dict) -> list:
    moves = lex["moves"]
    out = []
    for t in tokens:
        m = moves.get(t)
        if not m:
            continue
        for cue in ("direction", "axis", "context", "body_shape", "entry"):
            if cue in m:
                out.append(_Contrib(cue, m[cue], float(m["conf"]), f"move_{cue}:{t}"))
        if "twist" in m:
            out.append(_Contrib("twist", float(m["twist"]), float(m["conf"]), f"move_twist:{t}"))
    return out


def _numeric_contribs(tokens: list, lex: dict) -> list:
    deg = lex["numeric"]["deg"]
    fam = set(lex["numeric"]["flip_families"])
    moves = lex["moves"]
    nums = [t for t in tokens if t in deg]
    if not nums:
        return []
    is_flip = any(t in fam for t in tokens)
    is_turn = any(moves.get(t, {}).get("context") == "pk_basics" for t in tokens)
    out = []
    for t in nums:
        if is_flip and not is_turn:
            out.append(_Contrib("flip", float(deg[t]), 0.7, f"numeric_flip:{t}"))
        elif is_turn and not is_flip:
            out.append(_Contrib("twist", float(deg[t]), 0.7, f"numeric_twist:{t}"))
    return out


def _aggregate(contribs: list, unparsed: list) -> ParsedCues:
    by_cue: dict = {}
    for c in contribs:
        by_cue.setdefault(c.cue, []).append(c)
    cues, conf, trace = {}, {}, []
    NUMERIC = {"flip", "twist"}
    for cue, cs in by_cue.items():
        if cue in NUMERIC:
            val = float(sum(c.value for c in cs))
            cconf = min(c.conf for c in cs)
        else:
            vals = {c.value for c in cs}
            if len(vals) > 1:
                trace.append(f"conflict:{cue}:{sorted(map(str, vals))}->abstain")
                continue
            best = max(cs, key=lambda c: c.conf)
            val, cconf = best.value, best.conf
        trace.extend(c.rule for c in cs)
        if cconf >= ABSTAIN_THRESHOLD:
            cues[cue] = val
            conf[cue] = cconf
    return ParsedCues(cues=cues, confidence=conf, trace=trace, unparsed_tokens=unparsed)


def parse_trick_name(name: str) -> ParsedCues:
    if not isinstance(name, str):
        raise TypeError(f"trick name must be str, got {type(name).__name__}")
    tokens = tokenize(name)
    lex = _load_lexicon()
    known = set(lex["moves"]) | set(lex["twist_words"]) | set(lex["flip_words"]) \
        | set(lex["numeric"]["deg"]) | set(lex["phase_boundaries"]) | set(lex["numeric"]["flip_families"])
    unparsed = [t for t in tokens if t not in known]
    phases = segment_phases(tokens)
    contribs: list = []
    for ph in phases:
        contribs += _twist_contribs(ph, lex)
        contribs += _flip_contribs(ph, lex)
    contribs += _numeric_contribs(tokens, lex)
    contribs += _categorical_contribs(tokens, lex)
    return _aggregate(contribs, unparsed)
