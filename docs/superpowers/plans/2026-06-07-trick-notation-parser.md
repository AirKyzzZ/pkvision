# Parkour Trick-Notation Parser Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Parse compositional parkour trick names into cue labels (flip/twist/direction/axis/context/body_shape) with calibrated confidence, use them under a layered authority to fill/correct/flag the noisy auto-labels, route changes through human verification, and prove a core-mF1 lift over the 0.390 baseline.

**Architecture:** A pure 5-stage parser (`core/recognition/trick_name_parser.py`) reads a curated JSON lexicon (`data/name_grammar/lexicon.json`, drafted by a miner). A validator gates precision against FIG 149 + gold 99. A layered merge in `build_attribute_dataset.py` writes a versioned manifest with per-cue provenance plus a changes diff, which feeds the existing verify UI. Retraining `train_cue_model.py` on the cleaned manifest measures the lift.

**Tech Stack:** Python 3, numpy, pytest, PyTorch (existing CueModel). No new dependencies.

**Spec:** `docs/superpowers/specs/2026-06-07-trick-notation-parser-design.md`

---

## File structure

- Create `core/recognition/trick_name_parser.py` — `ParsedCues` dataclass + `parse_trick_name()` + 5 stages. Reads the lexicon. One responsibility: name → cues.
- Create `data/name_grammar/lexicon.json` — curated rule table (moves, twist/flip words, numeric routing, phase boundaries).
- Create `scripts/mine_name_grammar.py` — drafts `lexicon_draft.json` from FIG + unified_tricks with purity stats.
- Create `scripts/validate_name_parser.py` — Gate 1: precision/abstention on FIG 149 + gold 99.
- Modify `scripts/build_attribute_dataset.py` — layered merge → `attribute_manifest_v2.json` + `changes.json` + per-cue provenance.
- Modify `scripts/make_proposals.py` — add a parser proposal source + disagreement priority flag.
- Reuse `scripts/train_cue_model.py` — Gate 2 retrain on the v2 manifest (add `--manifest`).
- Tests: `tests/recognition/test_trick_name_parser.py`, `tests/recognition/test_trick_name_parser_golden.py`, `tests/labeling/test_layered_merge.py`.

Conventions: tests under `tests/<area>/`, run with `python3 -m pytest`. No comments unless a non-obvious invariant needs one. Commit after each task (conventional commits, subject only).

---

## Task 1: Parser scaffolding + abstention contract

**Files:**
- Create: `core/recognition/__init__.py` (if missing — check first; do not overwrite)
- Create: `core/recognition/trick_name_parser.py`
- Test: `tests/recognition/__init__.py` (empty), `tests/recognition/test_trick_name_parser.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/recognition/test_trick_name_parser.py
from core.recognition.trick_name_parser import ParsedCues, parse_trick_name, tokenize


def test_returns_parsedcues_shape():
    out = parse_trick_name("frontflip")
    assert isinstance(out, ParsedCues)
    assert isinstance(out.cues, dict)
    assert isinstance(out.confidence, dict)
    assert isinstance(out.trace, list)
    assert isinstance(out.unparsed_tokens, list)


def test_non_string_raises():
    import pytest
    with pytest.raises(TypeError):
        parse_trick_name(123)


def test_gibberish_abstains_completely():
    out = parse_trick_name("qwxz_zzzz")
    assert out.cues == {}
    assert "qwxz" in out.unparsed_tokens


def test_tokenize_folds_multiword_and_splits():
    assert tokenize("back_one_and_a_half_full") == ["back", "one_and_a_half", "full"]
    assert tokenize("Dash Vault") == ["dash", "vault"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/recognition/test_trick_name_parser.py -q`
Expected: FAIL with `ModuleNotFoundError` / `ImportError` (module not yet created).

- [ ] **Step 3: Write minimal implementation**

```python
# core/recognition/trick_name_parser.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/recognition/test_trick_name_parser.py -q`
Expected: PASS (4 tests). Note: `core/recognition/__init__.py` already exists in this repo; only create `tests/recognition/__init__.py`.

- [ ] **Step 5: Commit**

```bash
git add core/recognition/trick_name_parser.py tests/recognition/__init__.py tests/recognition/test_trick_name_parser.py
git commit -m "feat(parser): trick-name parser scaffolding + abstention contract"
```

---

## Task 2: Miner — draft the lexicon from FIG + unified_tricks

**Files:**
- Create: `scripts/mine_name_grammar.py`
- Test: `tests/recognition/test_mine_name_grammar.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/recognition/test_mine_name_grammar.py
from scripts.mine_name_grammar import token_associations


def test_token_associations_purity():
    rows = [
        {"name": "gainer flip", "physics": {"direction": "backward"}},
        {"name": "gainer full", "physics": {"direction": "backward"}},
        {"name": "front gainer", "physics": {"direction": "forward"}},
    ]
    assoc = token_associations(rows)
    g = assoc["gainer"]["direction"]
    assert g["backward"] == 2 and g["forward"] == 1
    assert assoc["gainer"]["_support"] == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/recognition/test_mine_name_grammar.py -q`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/mine_name_grammar.py
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
```

- [ ] **Step 4: Run test to verify it passes, then generate the draft**

Run: `python3 -m pytest tests/recognition/test_mine_name_grammar.py -q`
Expected: PASS.
Then run: `python3 scripts/mine_name_grammar.py`
Expected: prints `mined <N> tokens` and a top-25 table; writes `data/name_grammar/lexicon_draft.json`.

- [ ] **Step 5: Commit**

```bash
git add scripts/mine_name_grammar.py tests/recognition/test_mine_name_grammar.py data/name_grammar/lexicon_draft.json
git commit -m "feat(parser): lexicon miner (token->cue purity from unified_tricks)"
```

---

## Task 3: Curated lexicon + loader test

**Files:**
- Create: `data/name_grammar/lexicon.json`
- Test: extend `tests/recognition/test_trick_name_parser.py`

This is the curated starter lexicon. It encodes the validated high-precision anchors and the compositional operators. The miner draft (Task 2) is the reference for extending the long tail; promote a token only when its purity is high. The user is the domain authority on ambiguous calls.

- [ ] **Step 1: Write the lexicon file**

```json
{
  "moves": {
    "gainer":    {"direction": "backward", "flip": 1, "family": "flip", "conf": 0.9},
    "cork":      {"axis": "off_axis", "flip": 1, "family": "flip", "conf": 0.9},
    "corkscrew": {"axis": "off_axis", "flip": 1, "family": "flip", "conf": 0.9},
    "raiz":      {"direction": "backward", "axis": "off_axis", "conf": 0.7},
    "layout":    {"body_shape": "layout", "conf": 0.9},
    "tuck":      {"body_shape": "tuck", "conf": 0.9},
    "pike":      {"body_shape": "pike", "conf": 0.9},
    "arabian":   {"direction": "forward", "twist": 0.5, "family": "flip", "conf": 0.85},
    "frontflip": {"direction": "forward", "flip": 1, "family": "flip", "conf": 0.9},
    "backflip":  {"direction": "backward", "flip": 1, "family": "flip", "conf": 0.9},
    "sideflip":  {"direction": "side", "flip": 1, "family": "flip", "conf": 0.85},
    "front":     {"direction": "forward", "conf": 0.6},
    "back":      {"direction": "backward", "conf": 0.6},
    "side":      {"direction": "side", "conf": 0.6},
    "cat":       {"context": "pk_basics", "conf": 0.8},
    "kong":      {"context": "pk_basics", "conf": 0.8},
    "dash":      {"context": "pk_basics", "conf": 0.8},
    "vault":     {"context": "pk_basics", "conf": 0.9},
    "precision": {"context": "pk_basics", "conf": 0.9},
    "wall":      {"context": "wall", "conf": 0.9},
    "palm":      {"context": "wall", "conf": 0.7},
    "pimp":      {"context": "wall", "conf": 0.7},
    "flyaway":   {"context": "swing", "conf": 0.85},
    "castaway":  {"context": "swing", "conf": 0.8},
    "giant":     {"context": "swing", "conf": 0.85},
    "gainer_flip": {"direction": "backward", "flip": 1, "family": "flip", "conf": 0.9}
  },
  "twist_words": {"full": 1.0, "double_full": 2.0, "triple_full": 3.0, "half": 0.5},
  "flip_words":  {"double": 2.0, "triple": 3.0, "quad": 4.0, "one_and_a_half": 1.5},
  "numeric": {
    "flip_families": ["flip", "dive", "roll", "handspring", "tinsica"],
    "deg": {"180": 0.5, "360": 1.0, "540": 1.5, "720": 2.0, "900": 2.5, "1080": 3.0, "1260": 3.5, "1440": 4.0}
  },
  "phase_boundaries": ["in", "out", "unwind", "down"]
}
```

- [ ] **Step 2: Write the failing test**

```python
# append to tests/recognition/test_trick_name_parser.py
def test_lexicon_loads_anchor_moves():
    from core.recognition.trick_name_parser import _load_lexicon
    _load_lexicon.cache_clear()
    lex = _load_lexicon()
    assert lex["moves"]["gainer"]["direction"] == "backward"
    assert lex["twist_words"]["full"] == 1.0
    assert "in" in lex["phase_boundaries"]
```

- [ ] **Step 3: Run test to verify it passes**

Run: `python3 -m pytest tests/recognition/test_trick_name_parser.py::test_lexicon_loads_anchor_moves -q`
Expected: PASS (the loader from Task 1 reads the new file).

- [ ] **Step 4: Commit**

```bash
git add data/name_grammar/lexicon.json tests/recognition/test_trick_name_parser.py
git commit -m "feat(parser): curated starter lexicon (anchors + operators)"
```

---

## Task 4: Stage — phase segmentation + flip/twist from phases

**Files:**
- Modify: `core/recognition/trick_name_parser.py`
- Test: extend `tests/recognition/test_trick_name_parser.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/recognition/test_trick_name_parser.py
def test_segment_phases_splits_on_boundaries():
    from core.recognition.trick_name_parser import segment_phases
    assert segment_phases(["back", "full", "in", "full", "out"]) == [["back", "full"], ["full"], []]


def test_twist_sums_across_phases():
    out = parse_trick_name("back_full_in_full_out")
    assert out.cues["twist"] == 2.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/recognition/test_trick_name_parser.py -q`
Expected: FAIL (`segment_phases` undefined; `twist` not emitted yet).

- [ ] **Step 3: Implement segmentation + a contributions/aggregate skeleton**

Replace the body of `parse_trick_name` and add helpers:

```python
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
    return _aggregate(contribs, unparsed)
```

Note: the Task 1 `known` set referenced `lex["numeric"]` as a flat dict; the lexicon now nests `numeric.deg` + `numeric.flip_families`. This task updates `known` accordingly — keep them in sync.

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/recognition/test_trick_name_parser.py -q`
Expected: PASS (all tests, including `test_twist_sums_across_phases`).

- [ ] **Step 5: Commit**

```bash
git add core/recognition/trick_name_parser.py tests/recognition/test_trick_name_parser.py
git commit -m "feat(parser): phase segmentation + twist summation"
```

---

## Task 5: Stage — modifier adjacency (twist vs flip)

**Files:**
- Modify: `core/recognition/trick_name_parser.py`
- Test: extend `tests/recognition/test_trick_name_parser.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/recognition/test_trick_name_parser.py
def test_double_full_is_two_twists():
    out = parse_trick_name("back_double_full")
    assert out.cues["twist"] == 2.0


def test_double_back_is_two_flips():
    out = parse_trick_name("double_backflip")
    assert out.cues["flip"] == 2.0


def test_one_and_a_half_is_flip():
    out = parse_trick_name("one_and_a_half_frontflip")
    assert out.cues["flip"] == 1.5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/recognition/test_trick_name_parser.py -q`
Expected: FAIL (`double_full` token folds to a twist_word=2.0 already → first test may pass; flip-word handling for `double`/`one_and_a_half` not implemented → 2nd/3rd FAIL).

- [ ] **Step 3: Implement flip-word + move-flip contributions with adjacency**

Add helpers and wire them into the per-phase loop:

```python
def _flip_contribs(tokens: list, lex: dict) -> list:
    fw = lex["flip_words"]
    moves = lex["moves"]
    out = []
    for i, t in enumerate(tokens):
        if t in fw:
            nxt = tokens[i + 1] if i + 1 < len(tokens) else None
            if nxt in lex["twist_words"]:
                continue
            out.append(_Contrib("flip", float(fw[t]), 0.75, f"flip_word:{t}"))
    for t in tokens:
        m = moves.get(t)
        if m and "flip" in m:
            out.append(_Contrib("flip", float(m["flip"]), float(m["conf"]), f"move_flip:{t}"))
    return out
```

In `parse_trick_name`, inside the phase loop add:

```python
        contribs += _flip_contribs(ph, lex)
```

Update `_twist_contribs` so a `double_full`/`triple_full` atom (already folded) is counted once (it is, since folding produced a single token in `twist_words`). Ensure `double` immediately followed by `full` does NOT also fire `_flip_contribs` — handled by the `nxt in twist_words` guard above. For `back_double_full`, tokenize yields `["back","double_full"]` (folded), so `double_full`=2.0 twist; `double` never appears alone. Confirm the fold covers `double_full` and `triple_full` (it does, via `MULTIWORD`).

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/recognition/test_trick_name_parser.py -q`
Expected: PASS (all). `double_backflip` → `double`(flip 2.0); `one_and_a_half_frontflip` → `one_and_a_half`(flip 1.5) + `frontflip`(flip 1.0)? That sums to 2.5 — WRONG.

Fix before committing: a flip-count word adjacent to a flip move should *set* the count, not add to the move's intrinsic +1. Adjust `_flip_contribs`: when a flip-word is immediately followed by a flip move, suppress the move's intrinsic flip and use the word's count. Implement:

```python
def _flip_contribs(tokens: list, lex: dict) -> list:
    fw, moves, tw = lex["flip_words"], lex["moves"], lex["twist_words"]
    out, suppress = [], set()
    for i, t in enumerate(tokens):
        if t in fw:
            nxt = tokens[i + 1] if i + 1 < len(tokens) else None
            if nxt in tw:
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
```

Re-run Step 4. Expected now: `one_and_a_half_frontflip` → flip 1.5 (move suppressed); `double_backflip` → flip 2.0; `back_double_full` → twist 2.0. All PASS.

- [ ] **Step 5: Commit**

```bash
git add core/recognition/trick_name_parser.py tests/recognition/test_trick_name_parser.py
git commit -m "feat(parser): modifier adjacency (twist vs flip counts)"
```

---

## Task 6: Stage — numeric rotation routing (flip vs twist by family)

**Files:**
- Modify: `core/recognition/trick_name_parser.py`
- Test: extend `tests/recognition/test_trick_name_parser.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/recognition/test_trick_name_parser.py
def test_numeric_routes_to_flip_for_somersault_family():
    out = parse_trick_name("1080_dive_roll")
    assert out.cues["flip"] == 3.0


def test_numeric_routes_to_twist_for_turning_family():
    out = parse_trick_name("180_cat")
    assert out.cues["twist"] == 0.5


def test_numeric_abstains_when_family_ambiguous():
    out = parse_trick_name("360")
    assert "flip" not in out.cues and "twist" not in out.cues
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/recognition/test_trick_name_parser.py -q`
Expected: FAIL (numeric tokens not yet producing contributions).

- [ ] **Step 3: Implement numeric routing**

Add a numeric contribution helper operating on the *whole* token list (family is global to the name, not per-phase):

```python
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
```

In `parse_trick_name`, after the phase loop, add (once, on all tokens):

```python
    contribs += _numeric_contribs(tokens, lex)
```

`180_cat` → `cat` is pk_basics (turn) → twist 0.5. `1080_dive_roll` → `dive`,`roll` in flip_families → flip 3.0. `360` alone → neither → abstain.

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/recognition/test_trick_name_parser.py -q`
Expected: PASS (all).

- [ ] **Step 5: Commit**

```bash
git add core/recognition/trick_name_parser.py tests/recognition/test_trick_name_parser.py
git commit -m "feat(parser): numeric rotation routing by move family"
```

---

## Task 7: Stage — categorical cues (direction/axis/context/body_shape) + conflict abstain

**Files:**
- Modify: `core/recognition/trick_name_parser.py`
- Test: extend `tests/recognition/test_trick_name_parser.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/recognition/test_trick_name_parser.py
def test_categorical_cues_from_moves():
    out = parse_trick_name("wall_gainer")
    assert out.cues["context"] == "wall"
    assert out.cues["direction"] == "backward"
    assert out.cues["flip"] == 1.0


def test_layout_sets_body_shape():
    out = parse_trick_name("back_layout_full")
    assert out.cues["body_shape"] == "layout"
    assert out.cues["twist"] == 1.0


def test_conflicting_context_abstains():
    out = parse_trick_name("wall_vault_flyaway")
    assert "context" not in out.cues
    assert any("conflict:context" in t for t in out.trace)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/recognition/test_trick_name_parser.py -q`
Expected: FAIL (categorical move cues not emitted yet).

- [ ] **Step 3: Implement categorical contributions**

```python
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
```

In `parse_trick_name`, after numeric contribs:

```python
    contribs += _categorical_contribs(tokens, lex)
```

`wall_vault_flyaway` produces context contributions wall(0.9), pk_basics(0.9), swing(0.85) → two+ distinct values → `_aggregate` records `conflict:context` and abstains. (The existing `_aggregate` conflict branch already handles this.)

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/recognition/test_trick_name_parser.py -q`
Expected: PASS (all).

- [ ] **Step 5: Commit**

```bash
git add core/recognition/trick_name_parser.py tests/recognition/test_trick_name_parser.py
git commit -m "feat(parser): categorical cues + conflict abstention"
```

---

## Task 8: Golden snapshot regression test

**Files:**
- Create: `tests/recognition/test_trick_name_parser_golden.py`
- Create: `tests/recognition/golden_parses.json`

- [ ] **Step 1: Generate candidate golden parses**

Run this one-off to produce a snapshot over representative real slugs, then hand-verify each line is correct before saving:

```bash
python3 -c "
import json, sys; sys.path.insert(0,'.')
from core.recognition.trick_name_parser import parse_trick_name
slugs=['gainer_full','back_double_full','1080_dive_roll','180_cat','wall_gainer','back_layout_full','arabian','cork_double_full','frontflip','dive_half_back','kong_gainer','back_full_in_full_out','double_backflip','one_and_a_half_frontflip','butterfly_twist','flyaway_full','castaway','palm_flip','precision','side_flip']
print(json.dumps({s: parse_trick_name(s).cues for s in slugs}, indent=2))
" > tests/recognition/golden_parses.json
```

Review `golden_parses.json` against the spec semantics. Correct any wrong entry by fixing the lexicon (not the snapshot) and regenerating. Only commit once every line is right.

- [ ] **Step 2: Write the snapshot test**

```python
# tests/recognition/test_trick_name_parser_golden.py
import json
from pathlib import Path
from core.recognition.trick_name_parser import parse_trick_name

GOLDEN = json.loads((Path(__file__).parent / "golden_parses.json").read_text())


def test_golden_parses_stable():
    for slug, expected in GOLDEN.items():
        assert parse_trick_name(slug).cues == expected, slug
```

- [ ] **Step 3: Run test to verify it passes**

Run: `python3 -m pytest tests/recognition/test_trick_name_parser_golden.py -q`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add tests/recognition/test_trick_name_parser_golden.py tests/recognition/golden_parses.json
git commit -m "test(parser): golden snapshot regression over real slugs"
```

---

## Task 9: Gate 1 — validate precision on FIG 149 + gold 99

**Files:**
- Create: `scripts/validate_name_parser.py`
- Test: `tests/recognition/test_validate_name_parser.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/recognition/test_validate_name_parser.py
from scripts.validate_name_parser import score_cue, load_gold_rows


def test_score_cue_counts_hits_and_abstains():
    pred = [{"flip": 1.0}, {}, {"flip": 2.0}]
    gold = [{"flip": 1.0}, {"flip": 1.0}, {"flip": 1.0}]
    s = score_cue("flip", pred, gold)
    assert s["fired"] == 2 and s["correct"] == 1 and s["abstained"] == 1


def test_load_gold_rows_parses_disambig_group():
    rows = load_gold_rows()
    assert len(rows) > 0
    r = rows[0]
    assert "slug" in r and "cues" in r
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/recognition/test_validate_name_parser.py -q`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement the validator**

```python
# scripts/validate_name_parser.py
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
        for t in cat["tricks"]:
            rows.append({"slug": t["name"], "cues": {
                "flip": t.get("flip"), "twist": t.get("twist"),
                "direction": t.get("direction"), "axis": t.get("axis"),
                "context": cat.get("context")}})
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
```

- [ ] **Step 4: Run test, then the gate**

Run: `python3 -m pytest tests/recognition/test_validate_name_parser.py -q`
Expected: PASS.
Then run: `python3 scripts/validate_name_parser.py`
Expected: two tables. **Gate 1 decision:** if any cue is below 0.90 precision on fired predictions, iterate on `data/name_grammar/lexicon.json` (raise conf thresholds, fix wrong move entries, add high-purity tokens from `lexicon_draft.json`) and re-run. Do NOT proceed to Task 10 until precision clears the bar (abstention is acceptable; wrong fires are not). Record the final table in the commit message.

- [ ] **Step 5: Commit**

```bash
git add scripts/validate_name_parser.py tests/recognition/test_validate_name_parser.py data/name_grammar/lexicon.json
git commit -m "feat(parser): Gate 1 validator (FIG 149 + gold 99 precision)"
```

---

## Task 10: Layered merge → manifest_v2 + provenance + changes diff

**Files:**
- Modify: `scripts/build_attribute_dataset.py`
- Test: `tests/labeling/test_layered_merge.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/labeling/test_layered_merge.py
from scripts.build_attribute_dataset import merge_layered


def test_parser_fills_missing_cue():
    base = {"flip": 1.0, "twist": 0.0, "direction": "none"}
    parsed = {"cues": {"direction": "backward"}, "confidence": {"direction": 0.9}}
    merged, prov, changes = merge_layered("back_x", base, "unified", parsed, conf_thresh=0.7)
    assert merged["direction"] == "backward"
    assert prov["direction"] == "parser"
    assert any(c["kind"] == "fill" for c in changes)


def test_parser_overrides_unified_when_confident():
    base = {"flip": 1.0, "twist": 0.0, "direction": "forward"}
    parsed = {"cues": {"direction": "backward"}, "confidence": {"direction": 0.9}}
    merged, prov, changes = merge_layered("g", base, "unified", parsed, conf_thresh=0.7)
    assert merged["direction"] == "backward"
    assert any(c["kind"] == "correct" for c in changes)


def test_fig_wins_but_logs_disagreement():
    base = {"flip": 1.0, "twist": 0.0, "direction": "forward"}
    parsed = {"cues": {"direction": "backward"}, "confidence": {"direction": 0.9}}
    merged, prov, changes = merge_layered("g", base, "fig", parsed, conf_thresh=0.7)
    assert merged["direction"] == "forward"
    assert prov["direction"] == "fig"
    assert any(c["kind"] == "fig_disagree" for c in changes)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/labeling/test_layered_merge.py -q`
Expected: FAIL (`merge_layered` undefined).

- [ ] **Step 3: Add `merge_layered` to `build_attribute_dataset.py`**

Add near the top (after imports), and import the parser:

```python
from core.recognition.trick_name_parser import parse_trick_name

MERGE_CUES = ("direction", "flip", "twist", "axis", "context", "body_shape")


def merge_layered(slug, base, base_source, parsed, conf_thresh=0.7):
    cues = parsed["cues"]
    conf = parsed["confidence"]
    merged = dict(base)
    prov = {c: base_source for c in base}
    changes = []
    for cue in MERGE_CUES:
        if cue not in cues or conf.get(cue, 0.0) < conf_thresh:
            continue
        pv = cues[cue]
        cur = base.get(cue)
        missing = cur in (None, "none", "unknown", 0, 0.0) or cue not in base
        if cur is not None and not missing and str(cur) != str(pv):
            if base_source == "fig":
                changes.append({"slug": slug, "cue": cue, "old_value": cur,
                                "old_source": "fig", "parser_value": pv,
                                "parser_conf": conf[cue], "kind": "fig_disagree"})
                continue
            merged[cue] = pv
            prov[cue] = "parser"
            changes.append({"slug": slug, "cue": cue, "old_value": cur,
                            "old_source": base_source, "parser_value": pv,
                            "parser_conf": conf[cue], "kind": "correct"})
        elif missing:
            merged[cue] = pv
            prov[cue] = "parser"
            changes.append({"slug": slug, "cue": cue, "old_value": cur,
                            "old_source": base_source, "parser_value": pv,
                            "parser_conf": conf[cue], "kind": "fill"})
    return merged, prov, changes
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/labeling/test_layered_merge.py -q`
Expected: PASS (3 tests).

- [ ] **Step 5: Wire `merge_layered` into `main()` and emit v2 outputs**

In `main()`, after each clip's `attrs` is built (before binning), apply the parser merge and collect provenance + changes. After the loop, write `attribute_manifest_v2.json` (mirroring the existing manifest write but adding `label_provenance` per clip) and `data/name_grammar/changes.json`. Add the per-clip call:

```python
        parsed = parse_trick_name(slug)
        base_source = attrs.get("source", "unmatched")
        attrs, prov, clip_changes = merge_layered(slug, attrs, base_source, vars_of(parsed))
        attrs["label_provenance"] = prov
        all_changes.extend(clip_changes)
```

where `vars_of(parsed)` adapts the dataclass: add a tiny helper `def vars_of(p): return {"cues": p.cues, "confidence": p.confidence}`. Initialize `all_changes = []` before the loop. After saving the manifest, also write:

```python
    v2 = OUTPUT_PATH.parent / "attribute_manifest_v2.json"
    with open(v2, "w") as f:
        json.dump(manifest, f, indent=2)
    changes_path = ROOT / "data" / "name_grammar" / "changes.json"
    changes_path.parent.mkdir(parents=True, exist_ok=True)
    changes_path.write_text(json.dumps(all_changes, indent=2))
    print(f"  v2 manifest: {v2}  | changes: {len(all_changes)} -> {changes_path}")
```

Run: `python3 scripts/build_attribute_dataset.py`
Expected: prints source stats plus `v2 manifest:` and a changes count. Verify `attribute_manifest_v2.json` has `label_provenance` on clips and `changes.json` lists fills/corrections/fig_disagrees.

- [ ] **Step 6: Commit**

```bash
git add scripts/build_attribute_dataset.py tests/labeling/test_layered_merge.py data/v5_attribute_training/attribute_manifest_v2.json data/name_grammar/changes.json
git commit -m "feat(parser): layered merge -> manifest_v2 + provenance + changes diff"
```

---

## Task 11: Verify hook — parser proposal source + disagreement priority

**Files:**
- Modify: `scripts/make_proposals.py`
- Test: `tests/labeling/test_parser_proposals.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/labeling/test_parser_proposals.py
from scripts.make_proposals import changed_slugs_first


def test_changed_slugs_sort_first():
    changes = [{"slug": "b"}, {"slug": "b"}, {"slug": "d"}]
    order = changed_slugs_first(["a", "b", "c", "d"], changes)
    assert order[0] in {"b", "d"}
    assert set(order) == {"a", "b", "c", "d"}
    assert order.index("b") < order.index("a")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/labeling/test_parser_proposals.py -q`
Expected: FAIL (`changed_slugs_first` undefined).

- [ ] **Step 3: Implement disagreement-first ordering in make_proposals**

Add to `scripts/make_proposals.py`:

```python
def changed_slugs_first(slugs, changes):
    from collections import Counter
    weight = Counter(c["slug"] for c in changes)
    return sorted(slugs, key=lambda s: (-weight.get(s, 0), s))
```

Wire it into `main()`: load `data/name_grammar/changes.json` if present and reorder `slugs` via `changed_slugs_first(slugs, changes)` before the proposal loop, so the verify UI surfaces the parser's corrections first. Keep the existing `--ckpt` flow unchanged; this only changes ordering.

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/labeling/test_parser_proposals.py -q`
Expected: PASS.

- [ ] **Step 5: Generate a prioritized batch + commit**

Run: `python3 scripts/make_proposals.py --limit 50 --ckpt data/models/cue_model_v2.pt`
Expected: writes proposals with changed clips first.

```bash
git add scripts/make_proposals.py tests/labeling/test_parser_proposals.py
git commit -m "feat(parser): disagreement-first ordering for the verify queue"
```

(Human verification of the surfaced clips via `python3 scripts/verify_server.py --port 8899` is the out-of-band step — verdicts append to the existing `VerifiedStore`.)

---

## Task 12: Gate 2 — retrain on cleaned manifest + report lift

**Files:**
- Modify: `scripts/train_cue_model.py`
- Test: extend `tests/labeling/` (smoke only)

- [ ] **Step 1: Add a `--manifest` flag**

In `scripts/train_cue_model.py`, replace the hardcoded `MANIFEST` use with an arg:

```python
    ap.add_argument("--manifest", default=str(MANIFEST))
```

and in `load_labels()` accept a path param sourced from `args.manifest` (thread it through `main()`), defaulting to the original manifest so existing behavior is unchanged.

- [ ] **Step 2: Run the baseline-vs-cleaned comparison**

Run (baseline, already known ≈0.390):
`python3 scripts/train_cue_model.py --out data/models/cue_model_v2_baseline.pt`
Run (cleaned):
`python3 scripts/train_cue_model.py --manifest data/v5_attribute_training/attribute_manifest_v2.json --out data/models/cue_model_v3.pt`
Expected: each prints best core-mF1 + per-cue lift. **Gate 2:** cleaned core-mF1 should exceed the 0.390 baseline. Record both numbers.

- [ ] **Step 3: Verify the checkpoint loads**

Run: `python3 -c "import torch,sys; sys.path.insert(0,'.'); from core.labeling.cue_model import CueModel, CUE_CLASSES; m=CueModel(CUE_CLASSES,t=48); m.load_state_dict(torch.load('data/models/cue_model_v3.pt', map_location='cpu', weights_only=True)); print('loads OK')"`
Expected: `loads OK`.

- [ ] **Step 4: Commit**

```bash
git add scripts/train_cue_model.py
git commit -m "feat(parser): train_cue_model --manifest for Gate 2 retrain"
```

- [ ] **Step 5: Final full-suite check**

Run: `python3 -m pytest tests/ -q`
Expected: all green. Report the Gate 1 table + Gate 2 lift (baseline vs cleaned core-mF1) + counts of labels filled/corrected/fig_disagree from `changes.json`.

---

## Self-review notes (for the executor)

- **Type consistency:** `ParsedCues` fields (`cues`, `confidence`, `trace`, `unparsed_tokens`) are used identically in Tasks 1, 4–7, 9, 10. `_Contrib(cue,value,conf,rule)` is stable from Task 4 on. The lexicon `numeric` is nested (`deg`, `flip_families`) from Task 3 — Task 4 updates the `known` set accordingly.
- **Spec coverage:** parser engine (T1,4–7) → spec §5; miner/lexicon (T2,3) → §5; Gate 1 (T9) → §7; layered merge + provenance + changes (T10) → §5/§6; verify hook (T11) → §5; Gate 2 (T12) → §7. Phase 2 (aux signal) intentionally not in this plan (spec §10, deferred).
- **Gate discipline:** Task 9 must clear precision before Task 10 changes any label; Task 12 reports the payoff. Abstention is acceptable everywhere; silent wrong fires are not.
```
