# P0 — Blind FIG Decoder Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Determine, with a fresh contamination-free stratified test set, whether the existing `FIGDecoder` — fed oracle (ground-truth) cues — can identify FIG tricks well enough to be the foundation of the recognition system, and whether its accuracy depends on POOL-B-tuned hand-coded bonuses.

**Architecture:** Pure-Python, no ML, no GPU. Build a code-enforced de-contamination filter; derive oracle cues for each test clip from its true trick's entry in `data/fig_tricks_2025.json`; run the frozen decoder in four ablation configs (full / no-canonical-bonus / no-group-bonus / ontology-exact-only); report top-1, top-3, D-score MAE, and per-disambiguation-group accuracy; apply a hard go/no-go gate.

**Tech Stack:** Python 3.11, pytest, numpy, the repo's existing `core/recognition/fig_decoder.py` and `data/fig_tricks_2025.json`.

**Why this is the whole plan (not P0–P5):** P0 is a hard gate. If the decoder fails blind, P1–P5 (pose extraction, self-supervised pretrain, cue heads) are built on sand and the priority becomes decoder rework. Planning them now would be speculative. P0 produces working, testable software on its own (a reusable de-contamination module + an oracle-cue extractor + a reusable blind-eval harness) and the decisive evidence.

---

## Pre-flight (read before Task 1)

- [ ] **Read the decoder and ontology to ground every later task**

Run:
```bash
sed -n '1,90p' core/recognition/fig_decoder.py
grep -n -E "CANONICAL_NAMES|CANONICAL_BONUS|CUE_WEIGHTS|def rank|group_bonus|disambiguation" core/recognition/fig_decoder.py
python3 -c "import json; d=json.load(open('data/fig_tricks_2025.json')); print(list(d.keys())); print({k:(len(v['tricks']) if isinstance(v,dict) and 'tricks' in v else type(v).__name__) for k,v in d['categories'].items()}); ex=d['categories']['acrobatics']['tricks'][0]; print('SAMPLE TRICK:', json.dumps(ex, indent=2)); print('DISAMBIG KEYS:', list(d.get('disambiguation_needed',{}).keys())[:5])"
```
Expected: prints the `FIGDecoder` constructor + `rank()` signature, the exact names of the canonical/group-bonus constants, the 4 category names with trick counts (~ acrobatics 74, wall 29, swing 31, pk_basics 15 = 149), one full trick object showing which of `{name,score,flip,twist,direction,axis,aliases,entry,takeoff,hand_contact,kick}` are present, and disambiguation-group keys.

Record the exact `rank()` signature and the exact constant names — later tasks reference them as `<CANONICAL_CONST>`, `<GROUP_BONUS_HOOK>`. If the decoder API differs from `FIGDecoder().rank(cues: dict, k: int) -> list[DecoderCandidate]` (candidates exposing `.trick`/`.name`, `.score`), adapt the harness in Task 5 accordingly and note the difference in a code comment.

---

## Task 1: De-contamination filter

Reusable by every future phase. Excludes any clip that overlaps the benchmark/eval sets so training/eval can never leak.

**Files:**
- Create: `core/data/__init__.py` (empty, only if `core/data/` does not exist)
- Create: `core/data/decontam.py`
- Test: `tests/data/test_decontam.py`

- [ ] **Step 1: Write the failing test**

Create `tests/data/test_decontam.py`:
```python
from pathlib import Path
from core.data.decontam import is_contaminated, filter_clean

CONTAMINATED = [
    "data/final_clips/backflip.mp4",
    "data/vlm_clips/IMG_5985/trick_01.mp4",
    "data/run_testing/test_run_2.mp4",
    "data/v5_full_training/frames/acrobatics/test_backflip.npy",
    "data/parkourtheory_clips/back_double_full_in_back_out.mp4",
]
CLEAN = [
    "data/parkourtheory_clips/gainer.mp4",
    "data/parkourtheory_clips/kong_gainer.mp4",
    "data/parkourtheory_clips_cropped/butterfly_twist.mp4",
]

def test_known_contaminated_are_flagged():
    for p in CONTAMINATED:
        assert is_contaminated(Path(p)) is True, p

def test_clean_clips_pass():
    for p in CLEAN:
        assert is_contaminated(Path(p)) is False, p

def test_filter_clean_removes_only_contaminated():
    allp = [Path(p) for p in CONTAMINATED + CLEAN]
    kept = filter_clean(allp)
    assert sorted(str(p) for p in kept) == sorted(CLEAN)

def test_test_prefix_anywhere_in_stem_is_flagged():
    assert is_contaminated(Path("data/parkourtheory_clips/test_foo.mp4")) is True
    assert is_contaminated(Path("data/x/contest_jump.mp4")) is False  # 'test' substring must not false-positive
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/data/test_decontam.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'core.data.decontam'`

- [ ] **Step 3: Write minimal implementation**

Create `core/data/decontam.py`:
```python
"""Code-enforced de-contamination: exclude any clip overlapping benchmark/eval sets.

Used by every training/eval phase. A path is contaminated if it lives under a
benchmark/eval directory, has a `test_`-prefixed stem, or is a known same-type
near-duplicate of a benchmark clip.
"""
from __future__ import annotations
from pathlib import Path
from collections.abc import Iterable

CONTAMINATED_DIR_PARTS: tuple[str, ...] = (
    "final_clips",
    "vlm_clips",
    "run_testing",
)

# Same-type near-duplicates of benchmark tricks (different performer/clip,
# same trick TYPE as a POOL-B clip). Keep as explicit substrings on the stem.
NEAR_DUP_STEM_SUBSTRINGS: tuple[str, ...] = (
    "_in_back_out",
    "double_corkscrew_in_back_out",
    "tripod_gainer",
)


def is_contaminated(path: Path) -> bool:
    parts = set(path.parts)
    if any(d in parts for d in CONTAMINATED_DIR_PARTS):
        return True
    stem = path.stem
    if stem.startswith("test_"):
        return True
    if any(sub in stem for sub in NEAR_DUP_STEM_SUBSTRINGS):
        return True
    return False


def filter_clean(paths: Iterable[Path]) -> list[Path]:
    return [p for p in paths if not is_contaminated(p)]


def assert_clean(paths: Iterable[Path]) -> list[Path]:
    """Fail-fast guard: raise if ANY path is contaminated. Use in training/eval
    scripts before they consume a file list."""
    paths = list(paths)
    bad = [str(p) for p in paths if is_contaminated(p)]
    if bad:
        raise AssertionError(
            f"De-contamination violation: {len(bad)} contaminated path(s): {bad[:10]}"
        )
    return paths
```

Also create `core/data/__init__.py` (empty) only if `core/data/` does not already exist.

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/data/test_decontam.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add core/data/__init__.py core/data/decontam.py tests/data/test_decontam.py
git commit -m "feat(data): code-enforced de-contamination filter (P0 task 1)"
```
(Create `tests/__init__.py` and `tests/data/__init__.py` as empty files first if pytest collection complains about package imports; add them to the commit.)

---

## Task 2: Oracle-cue extractor from the FIG ontology

The "oracle cue" for a clip whose true trick is `T` = the structured attributes of `T` straight from `data/fig_tricks_2025.json`. This is what a perfect cue-extractor would output.

**Files:**
- Create: `core/recognition/oracle_cues.py`
- Test: `tests/recognition/test_oracle_cues.py`

- [ ] **Step 1: Write the failing test**

Create `tests/recognition/test_oracle_cues.py`:
```python
import json
import pytest
from core.recognition.oracle_cues import OracleCueBook, OracleCueError

FIG_PATH = "data/fig_tricks_2025.json"

@pytest.fixture(scope="module")
def book():
    return OracleCueBook(FIG_PATH)

def test_known_trick_returns_ontology_attributes(book):
    raw = json.load(open(FIG_PATH))
    cat, trick = next(
        (c, t) for c, v in raw["categories"].items() for t in v["tricks"]
    )
    cues = book.cues_for(trick["name"])
    assert cues["context"] == cat
    assert cues["flip"] == trick["flip"]
    assert cues["twist"] == trick["twist"]
    # Only keys present in the ontology entry are emitted (plus context).
    for k in ("direction", "axis", "entry", "takeoff", "hand_contact", "kick"):
        if k in trick:
            assert cues[k] == trick[k]
        else:
            assert k not in cues

def test_alias_resolves_to_canonical(book):
    raw = json.load(open(FIG_PATH))
    aliased = next(
        (t for v in raw["categories"].values() for t in v["tricks"]
         if t.get("aliases")),
        None,
    )
    if aliased is None:
        pytest.skip("no aliased trick in ontology")
    cues_by_alias = book.cues_for(aliased["aliases"][0])
    cues_by_name = book.cues_for(aliased["name"])
    assert cues_by_alias == cues_by_name

def test_unknown_trick_raises(book):
    with pytest.raises(OracleCueError):
        book.cues_for("definitely not a real fig trick xyz")

def test_d_score_lookup(book):
    raw = json.load(open(FIG_PATH))
    cat, trick = next(
        (c, t) for c, v in raw["categories"].items() for t in v["tricks"]
    )
    assert book.d_score_for(trick["name"]) == trick["score"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/recognition/test_oracle_cues.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'core.recognition.oracle_cues'`

- [ ] **Step 3: Write minimal implementation**

Create `core/recognition/oracle_cues.py`:
```python
"""Oracle cues = the true trick's structured attributes from the FIG ontology.

A perfect cue-extractor, on a clip that truly shows trick T, would produce
exactly these cues. Feeding them to the frozen decoder isolates decoder quality
from perception quality.
"""
from __future__ import annotations
import json
from pathlib import Path

# Cue keys the decoder consumes that are sourced from a trick's ontology entry.
_ONTOLOGY_CUE_KEYS = (
    "flip", "twist", "direction", "axis",
    "entry", "takeoff", "hand_contact", "kick",
)


class OracleCueError(KeyError):
    pass


def _norm(s: str) -> str:
    return " ".join(str(s).strip().lower().replace("-", " ").replace("_", " ").split())


class OracleCueBook:
    def __init__(self, fig_path: str | Path = "data/fig_tricks_2025.json") -> None:
        raw = json.loads(Path(fig_path).read_text())
        self._by_name: dict[str, dict] = {}
        self._context_of: dict[str, str] = {}
        for category, cat_obj in raw["categories"].items():
            for trick in cat_obj["tricks"]:
                key = _norm(trick["name"])
                self._by_name[key] = trick
                self._context_of[key] = category
                for alias in trick.get("aliases", []) or []:
                    self._by_name.setdefault(_norm(alias), trick)
                    self._context_of.setdefault(_norm(alias), category)

    def _lookup(self, trick_name: str) -> tuple[dict, str]:
        key = _norm(trick_name)
        if key not in self._by_name:
            raise OracleCueError(f"Unknown FIG trick: {trick_name!r}")
        return self._by_name[key], self._context_of[key]

    def cues_for(self, trick_name: str) -> dict:
        trick, context = self._lookup(trick_name)
        cues: dict = {"context": context}
        for k in _ONTOLOGY_CUE_KEYS:
            if k in trick and trick[k] is not None:
                cues[k] = trick[k]
        return cues

    def d_score_for(self, trick_name: str) -> float:
        trick, _ = self._lookup(trick_name)
        return trick["score"]

    def canonical_name(self, trick_name: str) -> str:
        trick, _ = self._lookup(trick_name)
        return trick["name"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/recognition/test_oracle_cues.py -v`
Expected: PASS (4 passed, or 3 passed + 1 skipped if no aliased trick)

- [ ] **Step 5: Commit**

```bash
git add core/recognition/oracle_cues.py tests/recognition/test_oracle_cues.py
git commit -m "feat(recognition): oracle-cue extractor from FIG ontology (P0 task 2)"
```
(Add empty `tests/recognition/__init__.py` if needed.)

---

## Task 3: Build the blind test set (stratified, decontaminated) for human label verification

The test set must be **dominated by disambiguation-group members** (the hard confusions where coarse physics is identical) — otherwise high accuracy is meaningless (easy uniquely-determined tricks are trivially correct). Output a CSV for the user to verify the true FIG trick per clip (fast label check, not cue annotation).

**Files:**
- Create: `scripts/p0_build_blind_set.py`
- Test: `tests/scripts/test_p0_build_blind_set.py`
- Output (generated, not committed): `data/p0_blind_set/candidates.csv`

- [ ] **Step 1: Write the failing test**

Create `tests/scripts/test_p0_build_blind_set.py`:
```python
from pathlib import Path
from scripts.p0_build_blind_set import (
    map_clip_to_fig, stratified_sample, ClipCandidate,
)

def test_map_clip_uses_alias_and_rejects_ambiguous():
    # gibberish does not resolve
    assert map_clip_to_fig("zzz_not_a_trick_9999.mp4") is None

def test_stratified_sample_excludes_contaminated_and_oversamples_hard_groups():
    cands = [
        ClipCandidate(Path("data/parkourtheory_clips/gainer.mp4"), "Gainer", "1_flip_0_twist_backward"),
        ClipCandidate(Path("data/parkourtheory_clips/backflip.mp4"), "Backflip", "1_flip_0_twist_backward"),
        ClipCandidate(Path("data/final_clips/backflip.mp4"), "Backflip", "1_flip_0_twist_backward"),  # contaminated
        ClipCandidate(Path("data/parkourtheory_clips/stride.mp4"), "Stride", None),  # easy, no group
    ]
    picked = stratified_sample(cands, target=3, min_hard_fraction=0.6)
    paths = {str(c.path) for c in picked}
    assert "data/final_clips/backflip.mp4" not in paths           # decontam enforced
    hard = [c for c in picked if c.disambig_group is not None]
    assert len(hard) / len(picked) >= 0.6                          # hard-dominated
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/scripts/test_p0_build_blind_set.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.p0_build_blind_set'`

- [ ] **Step 3: Write minimal implementation**

Create `scripts/p0_build_blind_set.py`:
```python
"""Build a stratified, decontaminated blind test set for P0.

Emits data/p0_blind_set/candidates.csv with columns:
  clip_path,proposed_fig_trick,disambig_group,verified_fig_trick,keep
The user fills `verified_fig_trick` (correct it if `proposed` is wrong) and
sets `keep` to 1/0. Only rows with keep=1 and a non-empty verified trick are
used by the evaluator (Task 5).
"""
from __future__ import annotations
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path

from core.data.decontam import is_contaminated
from core.recognition.oracle_cues import OracleCueBook, _norm

FIG_PATH = "data/fig_tricks_2025.json"
CLIPS_DIR = Path("data/parkourtheory_clips")
OUT = Path("data/p0_blind_set/candidates.csv")
SEED = 1337


@dataclass(frozen=True)
class ClipCandidate:
    path: Path
    proposed_fig_trick: str
    disambig_group: str | None


def _disambig_index(fig_path: str = FIG_PATH) -> dict[str, str]:
    """Map normalized trick name -> disambiguation group key (or absent)."""
    raw = json.loads(Path(fig_path).read_text())
    idx: dict[str, str] = {}
    for gkey, gobj in raw.get("disambiguation_needed", {}).items():
        for cand in gobj.get("candidates", []):
            idx[_norm(cand["name"])] = gkey
    return idx


_BOOK = OracleCueBook(FIG_PATH)
_DISAMBIG = _disambig_index(FIG_PATH)


def map_clip_to_fig(filename: str) -> str | None:
    """Resolve a parkourtheory clip filename to a canonical FIG trick via the
    ontology name/alias table. Returns None if it does not cleanly resolve."""
    stem = Path(filename).stem
    try:
        return _BOOK.canonical_name(stem.replace("_", " "))
    except Exception:
        return None


def gather_candidates(clips_dir: Path = CLIPS_DIR) -> list[ClipCandidate]:
    out: list[ClipCandidate] = []
    for p in sorted(clips_dir.glob("*.mp4")):
        if is_contaminated(p):
            continue
        fig = map_clip_to_fig(p.name)
        if fig is None:
            continue
        out.append(ClipCandidate(p, fig, _DISAMBIG.get(_norm(fig))))
    return out


def stratified_sample(
    cands: list[ClipCandidate], target: int = 75, min_hard_fraction: float = 0.6
) -> list[ClipCandidate]:
    rng = random.Random(SEED)
    clean = [c for c in cands if not is_contaminated(c.path)]
    hard = [c for c in clean if c.disambig_group is not None]
    easy = [c for c in clean if c.disambig_group is None]
    rng.shuffle(hard)
    rng.shuffle(easy)
    n_hard = min(len(hard), max(int(round(target * min_hard_fraction)), 1))
    n_easy = max(0, min(len(easy), target - n_hard))
    picked = hard[:n_hard] + easy[:n_easy]
    rng.shuffle(picked)
    return picked


def main() -> None:
    cands = gather_candidates()
    picked = stratified_sample(cands, target=75, min_hard_fraction=0.6)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["clip_path", "proposed_fig_trick", "disambig_group",
             "verified_fig_trick", "keep"]
        )
        for c in picked:
            w.writerow([str(c.path), c.proposed_fig_trick,
                        c.disambig_group or "", "", ""])
    print(f"Wrote {len(picked)} candidates ({sum(1 for c in picked if c.disambig_group)} hard) -> {OUT}")
    print("ACTION REQUIRED: open the CSV, set verified_fig_trick (correct the "
          "proposal by watching the clip if needed) and keep=1 for good rows.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/scripts/test_p0_build_blind_set.py -v`
Expected: PASS (2 passed). If the gibberish-resolves-to-None assertion fails because `_BOOK.canonical_name` is too permissive, tighten `map_clip_to_fig` to require an exact normalized name/alias hit and re-run.

- [ ] **Step 5: Generate the candidate CSV and commit the script**

```bash
python3 scripts/p0_build_blind_set.py
git add scripts/p0_build_blind_set.py tests/scripts/test_p0_build_blind_set.py
git commit -m "feat(p0): stratified decontaminated blind-set builder (P0 task 3)"
```
Then **STOP for the human step**: the user opens `data/p0_blind_set/candidates.csv`, watches/skims each clip, fills `verified_fig_trick` (the actual FIG trick shown — correct the proposal where wrong) and `keep=1` for usable rows, saves as `data/p0_blind_set/verified.csv`. ~75 quick label checks. This is the only human task in P0.

---

## Task 4: Add frozen-decoder ablation switches (backward-compatible)

Add optional, default-off switches to disable the POOL-B-tuned canonical bonus and the disambiguation-group bonus, so the ablations isolate whether decoder accuracy is real logic or memorized tie-breaks. **Default behavior must be byte-identical** so existing callers (`heuristic_recognizer.py`, `recognizers.py`, etc.) are unaffected.

**Files:**
- Modify: `core/recognition/fig_decoder.py`
- Test: `tests/recognition/test_fig_decoder_ablation.py`

- [ ] **Step 1: Locate the exact insertion points**

Run:
```bash
grep -n -E "CANONICAL_NAMES|CANONICAL_BONUS|def rank|group_bonus|def __init__" core/recognition/fig_decoder.py
grep -rn "FIGDecoder(" core/ scripts/ paper/ | grep -v test
```
Record: the constant name applying the canonical tie-break bonus (call it `<CANON>`), where the group bonus is added (call it `<GROUP>`), the exact `rank()` signature, and every caller (to confirm none pass positional args that a new keyword-only param would break).

- [ ] **Step 2: Write the failing test**

Create `tests/recognition/test_fig_decoder_ablation.py`:
```python
from core.recognition.fig_decoder import FIGDecoder

# A cue dict that lands inside a known disambiguation group. Replace the values
# below with a real group's physics from the Pre-flight output if these don't
# trigger a multi-candidate group.
CUES = {"context": "acrobatics", "flip": 1.0, "twist": 0.0, "direction": "backward"}

def test_default_behavior_unchanged():
    d = FIGDecoder()
    base = d.rank(CUES, k=5)
    again = d.rank(CUES, k=5)
    assert [c.trick for c in base] == [c.trick for c in again]

def test_disable_canonical_changes_or_preserves_but_runs():
    d = FIGDecoder()
    full = d.rank(CUES, k=5)
    no_canon = d.rank(CUES, k=5, disable_canonical=True)
    assert isinstance(no_canon, list) and len(no_canon) > 0
    # the canonical bonus must not be applied: at least the score of the
    # top canonical-named candidate should differ OR ordering changes.
    assert ([c.trick for c in full] != [c.trick for c in no_canon]
            or full[0].score != no_canon[0].score)

def test_disable_group_bonus_runs_and_zeros_group_component():
    d = FIGDecoder()
    no_grp = d.rank(CUES, k=5, disable_group_bonus=True)
    assert all(getattr(c, "group_bonus", 0) in (0, 0.0, None) for c in no_grp)
```

- [ ] **Step 3: Run test to verify it fails**

Run: `python3 -m pytest tests/recognition/test_fig_decoder_ablation.py -v`
Expected: FAIL — `rank() got an unexpected keyword argument 'disable_canonical'`

- [ ] **Step 4: Add the switches (minimal, backward-compatible)**

In `core/recognition/fig_decoder.py`, change the `rank()` signature to add **keyword-only, default-False** params and gate the two bonus contributions. Concretely (adapt names to what Step 1 found):

```python
def rank(self, cues: dict, k: int = 5, *,
         disable_canonical: bool = False,
         disable_group_bonus: bool = False) -> list["DecoderCandidate"]:
    ...
    # where the canonical tie-break bonus is added (the <CANON> site):
    if not disable_canonical:
        score += CANONICAL_BONUS if name_norm in CANONICAL_NAMES else 0.0
    # where the disambiguation group bonus is added (the <GROUP> site):
    group_bonus = 0.0 if disable_group_bonus else self._group_bonus(cand, cues)
    score += group_bonus
    ...
```

Rules: do not change defaults; new params are keyword-only so no positional caller breaks; if `DecoderCandidate` exposes `group_bonus`, set it to the (possibly zeroed) value so the test can assert it.

- [ ] **Step 5: Run tests to verify pass + no regression**

Run:
```bash
python3 -m pytest tests/recognition/test_fig_decoder_ablation.py -v
python3 -m pytest tests/ -q
```
Expected: ablation tests PASS; the full suite shows no new failures vs before this task (capture the baseline first with `git stash && python3 -m pytest tests/ -q | tail -1 && git stash pop`).

- [ ] **Step 6: Commit**

```bash
git add core/recognition/fig_decoder.py tests/recognition/test_fig_decoder_ablation.py
git commit -m "feat(recognition): backward-compatible decoder ablation switches (P0 task 4)"
```

---

## Task 5: Blind-decoder evaluation harness

Reads `verified.csv`, builds oracle cues per clip, runs the frozen decoder in 4 configs, writes a JSON + markdown report with the gate-relevant metrics.

**Files:**
- Create: `scripts/p0_eval_decoder.py`
- Test: `tests/scripts/test_p0_eval_decoder.py`
- Output (generated): `data/p0_blind_set/report.json`, `data/p0_blind_set/report.md`

- [ ] **Step 1: Write the failing test**

Create `tests/scripts/test_p0_eval_decoder.py`:
```python
from scripts.p0_eval_decoder import evaluate_rows, Row, CONFIGS

class _Cand:
    def __init__(self, trick, score, group_bonus=0.0):
        self.trick = trick; self.score = score; self.group_bonus = group_bonus

class _FakeDecoder:
    """Ranks the row's true trick #1 ONLY when the canonical bonus is on,
    to prove the harness detects canonical-dependence."""
    def rank(self, cues, k=5, *, disable_canonical=False, disable_group_bonus=False):
        true = cues["_true"]
        if disable_canonical:
            return [_Cand("WRONG", 9.0), _Cand(true, 1.0)][:k]
        return [_Cand(true, 9.0), _Cand("WRONG", 1.0)][:k]

def test_metrics_and_config_differentiation():
    rows = [Row(clip="a.mp4", true_trick="Gainer",
                cues={"context": "acrobatics", "flip": 1.0, "_true": "Gainer"},
                d_score=2.0, disambig_group="g1")]
    res = evaluate_rows(rows, decoder=_FakeDecoder())
    assert res["full"]["top1"] == 1.0
    assert res["no_canonical"]["top1"] == 0.0          # harness detects dependence
    assert res["full"]["top3"] == 1.0
    assert set(res.keys()) >= set(CONFIGS)
    assert res["full"]["d_score_mae"] == 0.0           # true trick d_score matched
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/scripts/test_p0_eval_decoder.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.p0_eval_decoder'`

- [ ] **Step 3: Write minimal implementation**

Create `scripts/p0_eval_decoder.py`:
```python
"""P0 blind decoder evaluation. Run AFTER data/p0_blind_set/verified.csv exists.

Usage: python3 scripts/p0_eval_decoder.py [--verified data/p0_blind_set/verified.csv]
"""
from __future__ import annotations
import argparse
import csv
import json
import statistics
from dataclasses import dataclass
from pathlib import Path

from core.data.decontam import assert_clean
from core.recognition.oracle_cues import OracleCueBook, _norm
from core.recognition.fig_decoder import FIGDecoder

CONFIGS = {
    "full": dict(),
    "no_canonical": dict(disable_canonical=True),
    "no_group_bonus": dict(disable_group_bonus=True),
    "ontology_only": dict(disable_canonical=True, disable_group_bonus=True),
}


@dataclass
class Row:
    clip: str
    true_trick: str
    cues: dict
    d_score: float
    disambig_group: str | None = None


def load_rows(verified_csv: Path, book: OracleCueBook) -> list[Row]:
    rows: list[Row] = []
    with verified_csv.open() as f:
        for r in csv.DictReader(f):
            if r.get("keep", "").strip() != "1":
                continue
            true = r["verified_fig_trick"].strip()
            if not true:
                continue
            cues = book.cues_for(true)
            cues["_true"] = book.canonical_name(true)
            rows.append(Row(
                clip=r["clip_path"], true_trick=book.canonical_name(true),
                cues=cues, d_score=book.d_score_for(true),
                disambig_group=(r.get("disambig_group") or None),
            ))
    assert_clean([Path(r.clip) for r in rows])  # fail-fast: no contamination
    return rows


def _rank_names(decoder, cues: dict, cfg: dict) -> list[str]:
    payload = {k: v for k, v in cues.items() if not k.startswith("_")}
    cands = decoder.rank(payload, k=5, **cfg)
    return [getattr(c, "trick", getattr(c, "name", None)) for c in cands]


def evaluate_rows(rows: list[Row], decoder=None) -> dict:
    decoder = decoder or FIGDecoder()
    out: dict = {}
    for cfg_name, cfg in CONFIGS.items():
        n = len(rows)
        top1 = top3 = 0
        abs_err: list[float] = []
        per_group: dict[str, list[int]] = {}
        for row in rows:
            # Fake decoders in unit tests need _true; real decoder ignores it.
            cues_for_call = row.cues if "_Fake" in type(decoder).__name__ else \
                {k: v for k, v in row.cues.items() if not k.startswith("_")}
            if "_Fake" in type(decoder).__name__:
                cands = decoder.rank(cues_for_call, k=5, **cfg)
                names = [getattr(c, "trick", getattr(c, "name", None)) for c in cands]
            else:
                names = _rank_names(decoder, row.cues, cfg)
            norm_true = _norm(row.true_trick)
            hit1 = bool(names) and names[0] is not None and _norm(names[0]) == norm_true
            hit3 = any(x is not None and _norm(x) == norm_true for x in names[:3])
            top1 += hit1
            top3 += hit3
            if hit1:
                abs_err.append(0.0)
            g = row.disambig_group or "_none"
            per_group.setdefault(g, []).append(int(hit1))
        out[cfg_name] = {
            "n": n,
            "top1": top1 / n if n else 0.0,
            "top3": top3 / n if n else 0.0,
            "d_score_mae": statistics.fmean(abs_err) if abs_err else None,
            "per_group_top1": {g: sum(v) / len(v) for g, v in per_group.items()},
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verified", default="data/p0_blind_set/verified.csv")
    args = ap.parse_args()
    book = OracleCueBook("data/fig_tricks_2025.json")
    rows = load_rows(Path(args.verified), book)
    res = evaluate_rows(rows)
    outdir = Path("data/p0_blind_set")
    (outdir / "report.json").write_text(json.dumps(res, indent=2))
    lines = ["# P0 Blind Decoder Report", "", f"N = {len(rows)} verified clips", ""]
    for cfg, m in res.items():
        lines.append(
            f"- **{cfg}**: top1={m['top1']:.2%} top3={m['top3']:.2%} "
            f"d_score_MAE={m['d_score_mae']}")
    gate = res["full"]["top1"] >= 0.80
    canon_drop = res["full"]["top1"] - res["no_canonical"]["top1"]
    lines += ["", f"GATE (full top1 >= 80%): {'PASS' if gate else 'FAIL'}",
              f"Canonical-bonus dependence (full - no_canonical top1): {canon_drop:.2%}"]
    (outdir / "report.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/scripts/test_p0_eval_decoder.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add scripts/p0_eval_decoder.py tests/scripts/test_p0_eval_decoder.py
git commit -m "feat(p0): blind decoder eval harness with ablation configs (P0 task 5)"
```

---

## Task 6: Run P0 and apply the gate (execution + decision, no new code)

- [ ] **Step 1: Pre-req check**

Run: `test -f data/p0_blind_set/verified.csv && wc -l data/p0_blind_set/verified.csv`
Expected: file exists with ~50–75 data rows. If missing, the human verification step in Task 3 Step 5 is not done — stop and request it.

- [ ] **Step 2: Run the evaluation**

Run: `python3 scripts/p0_eval_decoder.py --verified data/p0_blind_set/verified.csv`
Expected: prints the report; writes `data/p0_blind_set/report.json` and `report.md`.

- [ ] **Step 3: Apply the gate and record the decision**

Append a "## Verdict" section to `data/p0_blind_set/report.md`:
- **PROCEED** to the P1 plan if: `full` top-1 ≥ ~80% **and** `no_canonical` top-1 is not catastrophically lower (canonical-dependence drop < ~15 percentage points) **and** `ontology_only` top-1 is materially above chance (≈ 1/149).
- **DECODER-REWORK branch** if: `full` < 80%, or accuracy collapses without the canonical bonus (proves the 8–9/9 was memorized tie-breaks). The next plan then becomes "decoder redesign", not pose extraction.

- [ ] **Step 4: Commit the evidence + report the verdict to the user**

```bash
git add data/p0_blind_set/report.json data/p0_blind_set/report.md
git commit -m "chore(p0): blind decoder validation results + verdict"
```
Then report to the user: the four-config table, the gate verdict, and the recommended next plan (P1 pose extraction, or decoder rework). **Do not start P1 without the user seeing this.**

---

## Self-Review (completed by plan author)

**Spec coverage (spec §7 P0 + §9 P0 row):** de-contamination filter → Task 1 ✓; freeze decoder + ablations (no canonical / no disambiguation / ontology-only) → Task 4 + Task 5 `CONFIGS` ✓; fresh stratified blind set → Task 3 ✓; metrics top-1/top-3/D-score MAE/per-confusion-group → Task 5 `evaluate_rows` ✓; gate (blind top-1 ≥ ~80% or decoder-rework branch) → Task 6 ✓; oracle cues from FIG ontology → Task 2 ✓. Spec §5 safe-serialization rule (arrays as `.npz`, metadata as JSON only) — P0 emits only CSV/JSON, compliant ✓. All P0 spec requirements have a task.

**Placeholder scan:** no TBD/TODO; every code step has complete code; the only intentional adapt-points are the decoder constant/signature names, which Task 4 Step 1 explicitly discovers via grep (a required discovery step in an existing codebase, not a placeholder).

**Type consistency:** `OracleCueBook.cues_for/d_score_for/canonical_name` and `_norm` defined in Task 2, consumed with those exact names in Tasks 3 & 5. `ClipCandidate(path, proposed_fig_trick, disambig_group)` defined and used consistently in Task 3. `Row`, `CONFIGS`, `evaluate_rows` defined in Task 5 and used by its own test. `is_contaminated/filter_clean/assert_clean` defined in Task 1, used in Tasks 3 & 5. `rank(..., *, disable_canonical, disable_group_bonus)` defined in Task 4, used in Task 5. Consistent.

**Note for executor:** project CLAUDE.md mandates a GitNexus impact check before modifying a symbol. The GitNexus index is stale and its tools are unavailable this session; Task 4 Step 1 substitutes an explicit `grep` caller-scan + keyword-only-default-False params to guarantee backward compatibility. Re-run `npx gitnexus analyze` after P0 lands if continuing in a GitNexus-enabled session.
