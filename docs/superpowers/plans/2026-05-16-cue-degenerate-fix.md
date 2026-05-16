# Cue-Degenerate Families Fix — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Determine how much of P0's "73.1% trick top-1" ceiling is actually a *scoring* problem vs. D-score-irrelevant trick-naming confusion, then make targeted, honest ontology improvements only where they improve real scoring — and lock the final cue vocabulary P1–P3's pose model must predict.

**Architecture:** Pure-Python, no GPU/Colab. Reuse the merged P0 harness (`scripts/p0_eval_decoder.py`) and blind set. Add D-score-equivalence-aware metrics; categorize each failing collision group; fix ontology data-quality (alias-duplicates) and add a `movement` cue dimension for the pk_basics vault family only where the FIG Code of Points genuinely distinguishes tricks; re-validate via the P0 harness keeping it non-circular.

**Tech Stack:** Python 3.14, pytest, the in-repo `core/recognition/fig_decoder.py`, `core/recognition/oracle_cues.py`, `scripts/p0_eval_decoder.py`, `data/fig_tricks_2025.json`.

**Scope note:** This is the cheap, no-infra, decisive next unit. P1 (heavy offline pose extraction + audit) is Colab-gated and gets its own plan once this lands and the `colab` MCP is wired.

**Grounding facts (verified against `data/p0_blind_set/report.json` + `data/fig_tricks_2025.json`):**
- Failing groups (full top1 <0.6): `acro|1|0.5|back|off_axis` Raiz/Cork/Dark Arabian/Kroc (Cork↔Dark Arabian, Kroc↔Krok/Reverse Cork are aliases); `acro|2|1|back|lat` Double Backflip 360 ≡ Cork-in Backflip (**listed as separate entries but aliases of each other — data bug**); `pk_basics|0|0|None|None` **13 vault tricks** (Stride/Plyo/Tic Tac/Side/Pop/Kong/Reverse/Kash/Dong/Double Kong Vault/Wallrun/Climb Up/Dyno — all with zero distinguishing fields) at 0.11; several swing/wall groups with partial/no distinguishers.
- Distinguisher coverage in ontology: `entry` 21/149, `takeoff` 15/149, `hand_contact` 9/149, `kick` 2/149. Decoder already weights entry/takeoff/hand_contact at 1.5 each but the ontology barely populates them.
- `fig_tricks_2025.json` top-level keys: `metadata, scoring, categories, disambiguation_needed`; `scoring` has `e_score, d_score`.

---

## Task 1: D-score equivalence classes + scoring-aware metrics

The real product metric is D-score, not trick name. Build the equivalence machinery and re-score P0 *without changing the decoder*.

**Files:**
- Create: `core/recognition/dscore_equiv.py`
- Test: `tests/recognition/test_dscore_equiv.py`

- [ ] **Step 1: Write the failing test**

Create `tests/recognition/test_dscore_equiv.py`:
```python
from core.recognition.dscore_equiv import DScoreBook, same_dscore

def test_known_trick_dscore_and_equivalence():
    b = DScoreBook("data/fig_tricks_2025.json")
    # Backflip and Gainer have different D; a trick equals itself.
    bf = b.d_score_for("Backflip")
    assert isinstance(bf, float)
    assert b.same_dscore("Backflip", "Backflip") is True

def test_same_dscore_groups_real_cluster():
    b = DScoreBook("data/fig_tricks_2025.json")
    # Cork / Kroc / Raiz sit in one physics cluster; verify the API answers
    # consistently (equal-D pairs return True, unequal return False).
    for a, c in [("Cork", "Cork"), ("Backflip", "Backflip")]:
        assert b.same_dscore(a, c) is True
    assert b.same_dscore("Stride", "Swing Double Gainer 1080 (Miller)") is False  # 0.1 vs 7.7

def test_unknown_raises():
    b = DScoreBook("data/fig_tricks_2025.json")
    import pytest
    with pytest.raises(KeyError):
        b.d_score_for("not a real trick zzz")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/recognition/test_dscore_equiv.py -v`
Expected: FAIL `ModuleNotFoundError: No module named 'core.recognition.dscore_equiv'`

- [ ] **Step 3: Write minimal implementation**

Create `core/recognition/dscore_equiv.py`:
```python
"""D-score lookup + equivalence. For an auto-SCORING system, confusing two
tricks with the same D-score is a zero-cost error. These helpers let the eval
report scoring-aware accuracy alongside raw trick top-1.
"""
from __future__ import annotations
import json
from pathlib import Path

from core.recognition.oracle_cues import _norm


class DScoreBook:
    def __init__(self, fig_path: str | Path = "data/fig_tricks_2025.json") -> None:
        raw = json.loads(Path(fig_path).read_text())
        self._d: dict[str, float] = {}
        for obj in raw["categories"].values():
            for t in obj["tricks"]:
                self._d[_norm(t["name"])] = float(t["score"])
                for a in t.get("aliases", []) or []:
                    self._d.setdefault(_norm(a), float(t["score"]))

    def d_score_for(self, trick: str) -> float:
        k = _norm(trick)
        if k not in self._d:
            raise KeyError(f"Unknown trick: {trick!r}")
        return self._d[k]

    def same_dscore(self, a: str, b: str, tol: float = 1e-9) -> bool:
        return abs(self.d_score_for(a) - self.d_score_for(b)) <= tol


def same_dscore(a: str, b: str, fig_path: str = "data/fig_tricks_2025.json") -> bool:
    return DScoreBook(fig_path).same_dscore(a, b)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/recognition/test_dscore_equiv.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add core/recognition/dscore_equiv.py tests/recognition/test_dscore_equiv.py
git commit -m "feat(recognition): D-score lookup + equivalence helper"
```

---

## Task 2: Scoring-aware re-evaluation of P0 (no decoder change)

Add a scoring-aware metric to the eval harness and re-run on the existing blind set. This is the decisive measurement: it likely shows the *scoring* ceiling is far above 73% trick top-1.

**Files:**
- Modify: `scripts/p0_eval_decoder.py` (add `dscore_correct` metric: top-1 prediction whose D-score equals the true trick's D-score counts as scoring-correct)
- Test: `tests/scripts/test_p0_eval_decoder.py` (extend)

- [ ] **Step 1: Locate the metric site**

Run: `grep -n "top1\|d_score_mae\|per_group_top1\|def evaluate_rows\|for row in rows" scripts/p0_eval_decoder.py`
Record the exact loop and metric-assembly lines.

- [ ] **Step 2: Write the failing test (extend existing file)**

Append to `tests/scripts/test_p0_eval_decoder.py`:
```python
def test_dscore_correct_metric_counts_equal_dscore_as_correct():
    from scripts.p0_eval_decoder import evaluate_rows, Row

    class _C:
        def __init__(s, fig_name, d):
            s.fig_name = fig_name; s.d_score = d; s.score = 1.0
            s.group_bonus = 0.0; s.breakdown = {}

    class _D:  # predicts WRONG name but SAME d_score as the true trick
        def rank(s, cues, k=5, candidate_filter=None, *,
                 disable_canonical=False, disable_group_bonus=False):
            return [_C("WrongName", 2.0), _C(cues["__true__"], 2.0)][:k]

    rows = [Row(clip="a", true_trick="Cork", cues={"__true__": "Cork"},
                d_score=2.0, disambig_group="g")]
    res = evaluate_rows(rows, decoder=_D())
    assert res["full"]["top1"] == 0.0            # raw name wrong
    assert res["full"]["dscore_correct"] == 1.0  # but scoring-correct
```
(The fake passes `cues["__true__"]`; ensure `evaluate_rows` strips keys starting with `__` before calling a real decoder — add that filter if not present.)

- [ ] **Step 3: Run test to verify it fails**

Run: `python3 -m pytest tests/scripts/test_p0_eval_decoder.py::test_dscore_correct_metric_counts_equal_dscore_as_correct -v`
Expected: FAIL (`KeyError: 'dscore_correct'`)

- [ ] **Step 4: Implement**

In `scripts/p0_eval_decoder.py`: import `from core.recognition.dscore_equiv import DScoreBook`; construct one `DScoreBook` in `evaluate_rows`; for each row, after computing `names`, compute `dscore_hit = bool(names) and names[0] is not None and abs(dbook.d_score_for(names[0]) - row.d_score) <= 1e-9` (guard `KeyError` for unmapped predicted names → treat as not scoring-correct). Add `"dscore_correct": dscore_hits / n` to each config's dict. In `evaluate_rows`, before calling the decoder, filter cue keys: `payload = {k: v for k, v in row.cues.items() if not k.startswith("__")}` and pass `payload` (so the `__true__` test hook never reaches a real decoder; real runs are unaffected since real cues have no `__`-keys).

- [ ] **Step 5: Run tests to verify pass**

Run: `python3 -m pytest tests/scripts/test_p0_eval_decoder.py -q`
Expected: all pass (existing + new).

- [ ] **Step 6: Re-run P0 with the new metric and record**

Run: `python3 scripts/p0_eval_decoder.py --verified data/p0_blind_set/verified.csv`
Expected: report now includes `dscore_correct` per config. Commit the regenerated report.

```bash
git add scripts/p0_eval_decoder.py tests/scripts/test_p0_eval_decoder.py data/p0_blind_set/report.json data/p0_blind_set/report.md
git commit -m "feat(p0): scoring-aware (D-score-equivalent) accuracy metric + re-eval"
```

- [ ] **Step 7: DECISION GATE (no code) — report to the controller/user**

Read `dscore_correct` (full) from the regenerated report.
- If `dscore_correct` is high (e.g. ≥ ~90%): the decoder is already near-sufficient for *scoring*; the trick-ID ceiling is largely D-score-irrelevant. **Recommend: deprioritize deep ontology work; proceed to P1 (pose extraction) — the real bottleneck is cue *extraction*, not decoding.** Skip Tasks 3–4 (or do only the cheap data-quality fix Task 3a).
- If `dscore_correct` is materially below trick top-1 expectations (real scoring errors remain in the failing groups): proceed to Tasks 3–4.

---

## Task 3: Ontology data-quality fixes (cheap, always worth doing)

### Task 3a: De-duplicate alias-duplicate trick entries

`Double Backflip 360` and `Cork-in Backflip` are listed as separate trick objects but are aliases of each other (and similar cases). These create false "errors".

**Files:**
- Create: `scripts/fig_ontology_audit.py`
- Test: `tests/scripts/test_fig_ontology_audit.py`

- [ ] **Step 1: Write the failing test**

Create `tests/scripts/test_fig_ontology_audit.py`:
```python
from scripts.fig_ontology_audit import find_alias_duplicates

def test_finds_self_referential_alias_duplicates():
    fig = {"categories": {"acrobatics": {"tricks": [
        {"name": "Double Backflip 360", "score": 5.0, "flip": 2, "twist": 1,
         "aliases": ["Cork-in Backflip"]},
        {"name": "Cork-in Backflip", "score": 5.0, "flip": 2, "twist": 1},
        {"name": "Backflip", "score": 1.5, "flip": 1, "twist": 0},
    ]}}}
    dups = find_alias_duplicates(fig)
    assert ("Cork-in Backflip", "Double Backflip 360") in [
        (d["duplicate"], d["canonical"]) for d in dups]
    assert all(d["duplicate"] != "Backflip" for d in dups)
```

- [ ] **Step 2: Run to verify fail**

Run: `python3 -m pytest tests/scripts/test_fig_ontology_audit.py -v`
Expected: FAIL `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

Create `scripts/fig_ontology_audit.py`:
```python
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
```

- [ ] **Step 4: Run to verify pass**

Run: `python3 -m pytest tests/scripts/test_fig_ontology_audit.py -v`
Expected: PASS.

- [ ] **Step 5: Run the audit, record findings (no ontology mutation yet)**

Run: `python3 scripts/fig_ontology_audit.py`
Capture the list. **Ontology mutation (removing/merging duplicate entries) is a separate, human-reviewed step** — editing `data/fig_tricks_2025.json` (the authoritative scoring table) must be deliberate; the audit output + FIG Code of Points is the reference. Commit only the audit tool:
```bash
git add scripts/fig_ontology_audit.py tests/scripts/test_fig_ontology_audit.py
git commit -m "feat(fig): alias-duplicate ontology audit tool"
```

- [ ] **Step 6: Apply de-dup to a working copy + re-evaluate impact**

Produce `data/fig_tricks_2025.audit_dedup.json` (a copy with confirmed alias-duplicate entries removed — keep the canonical, drop the pure-alias entry). Run the P0 harness with `--fig` pointed at it *if the harness supports an override; if not, this step is read-only analysis only*: quantify how many P0 "failures" were alias-duplicate artifacts. Record in the report. Do NOT overwrite the authoritative `data/fig_tricks_2025.json` without explicit user confirmation (it is the scoring source of truth).

---

## Task 4 (CONDITIONAL on Task 2 gate): `movement` cue for the pk_basics vault family

Only if Task 2's gate showed pk_basics confusion causes real scoring error. The 13 pk_basics tricks are genuinely distinct movements the FIG CoP defines; the ontology lacks any cue for them.

**Files:**
- Modify: `core/recognition/fig_decoder.py` (add `movement` to `CUE_WEIGHTS` + `_cue_agreement`, keyword-safe, default-inert)
- Modify: `core/recognition/oracle_cues.py` (expose a populated `movement` cue if present in the ontology entry)
- Modify: `data/fig_tricks_2025.json` (populate `movement` for the 13 pk_basics tricks per FIG CoP — human/CoP-referenced)
- Test: `tests/recognition/test_movement_cue.py`

- [ ] **Step 1: Confirm the gate**

Re-read Task 2 Step 7 decision. If the gate said "deprioritize", STOP — do not do this task; record the rationale and proceed to P1 planning instead.

- [ ] **Step 2: Write the failing test**

Create `tests/recognition/test_movement_cue.py`:
```python
from core.recognition.fig_decoder import FIGDecoder

def test_movement_cue_separates_two_pk_basics_when_provided():
    d = FIGDecoder()
    base = {"context": "pk_basics", "flip": 0.0, "twist": 0.0}
    # Without a movement cue, the two are indistinguishable (same top set).
    a = d.rank({**base}, k=3)
    # With a movement cue, the matching trick must outrank the other.
    b = d.rank({**base, "movement": "kong_vault"}, k=3)
    assert isinstance(b, list) and b
    assert [c.fig_name for c in a] != [c.fig_name for c in b] \
        or b[0].score != a[0].score

def test_default_unchanged_when_movement_absent():
    d = FIGDecoder()
    cues = {"context": "acrobatics", "flip": 1.0, "twist": 0.0,
            "direction": "backward"}
    assert [c.fig_name for c in d.rank(cues, k=5)] == \
           [c.fig_name for c in d.rank(cues, k=5)]
```

- [ ] **Step 3: Run to verify fail**

Run: `python3 -m pytest tests/recognition/test_movement_cue.py -v`
Expected: FAIL (movement cue has no effect yet / not in vocabulary).

- [ ] **Step 4: Implement decoder + oracle support (minimal)**

In `core/recognition/fig_decoder.py`: add `"movement": 1.5` to `CUE_WEIGHTS`; in `_cue_agreement`, treat `movement` as a string-categorical cue (same branch as entry/takeoff: exact match +1.0, mismatch -0.5, FIG value `None` → return `None` so absent-in-ontology tricks are unaffected). No other logic changes — verify default behaviour byte-identical when `movement` absent from cues AND from the FIG entry (regression: existing `tests/recognition/test_fig_decoder_ablation.py` still green).
In `core/recognition/oracle_cues.py`: add `"movement"` to `_ONTOLOGY_CUE_KEYS` so it is emitted when present (the existing omit-None logic already handles absence).

- [ ] **Step 5: Populate `movement` for the 13 pk_basics tricks**

Edit `data/fig_tricks_2025.json`: add a `"movement"` field to each of Stride, Plyo, Tic Tac, Side Vault, Pop Vault, Wallrun, Kong Vault, Reverse Vault, Kash Vault, Climb Up, Dyno, Dong Vault, Double Kong Vault, using FIG Code of Points vault/movement nomenclature (e.g. `stride`, `plyo`, `tic_tac`, `side_vault`, `pop_vault`, `wallrun`, `kong_vault`, `reverse_vault`, `kash_vault`, `climb_up`, `dyno`, `dong_vault`, `double_kong_vault`). This is a deliberate, versioned ontology edit; the `movement` value must match the FIG CoP definition for each trick. Bump `metadata` version note in the JSON.

- [ ] **Step 6: Run tests + regression**

Run:
```
python3 -m pytest tests/recognition/test_movement_cue.py -v
python3 -m pytest tests/recognition tests/data tests/scripts -q
```
Expected: movement tests pass; full P0 scope green (no regression — the ablation/default tests must still pass, proving non-pk_basics behaviour unchanged).

- [ ] **Step 7: Re-validate via P0 harness + commit**

Run: `python3 scripts/p0_eval_decoder.py --verified data/p0_blind_set/verified.csv`
Expected: `pk_basics|flip=0|twist=0` group top-1 materially up (target ≥0.6), overall top-1 and `dscore_correct` up, NO regression on previously-passing groups, `no_canonical`/`ontology_only` still strong (non-circular preserved).
```bash
git add core/recognition/fig_decoder.py core/recognition/oracle_cues.py data/fig_tricks_2025.json tests/recognition/test_movement_cue.py data/p0_blind_set/report.json data/p0_blind_set/report.md
git commit -m "feat(fig): movement cue for pk_basics vault family; re-validated on P0 blind set"
```

---

## Task 5: Final cue contract for P1–P3 (honest extractability flags)

**Files:**
- Create: `docs/superpowers/specs/2026-05-16-cue-contract.md`

- [ ] **Step 1: Write the cue-contract doc**

Document the FINAL cue vocabulary the P3 pose model must predict to feed the validated decoder: `context, flip, twist, direction, axis, entry, takeoff, hand_contact, kick` (+ `movement` if Task 4 ran). For each cue, state: decoder weight, whether reliably extractable from a single 2D camera, and the known ceiling. Explicitly flag (per P0 + research memory): `twist ≥ 1.5` and fine `movement`/obstacle-interaction are likely NOT single-camera-extractable — P1–P3 must not be expected to solve these; they motivate the deferred multi-camera path. Reference `data/p0_blind_set/report.md` numbers.

- [ ] **Step 2: Commit**

```bash
git add docs/superpowers/specs/2026-05-16-cue-contract.md
git commit -m "docs: P1-P3 cue contract + single-camera extractability flags"
```

---

## Self-Review (completed by plan author)

**Spec coverage:** This plan implements the "scoped cue-degenerate fix" the P0 verdict mandated (project_p0_result memory) and feeds the A′ spec's P3 cue-model requirement (cue contract, Task 5). It does NOT cover P1 pose extraction (Colab-gated; explicitly deferred to its own plan — consistent with the spec's gated-phase structure and the scope rule that each plan be independently executable).

**Placeholder scan:** No TBD/TODO. Task 4 is explicitly CONDITIONAL with a hard gate (Task 2 Step 7) — not a placeholder but a real decision branch (mirrors P0's gate pattern). Ontology mutations (Task 3 dedup, Task 4 `movement`) are flagged as deliberate human/CoP-referenced edits to the authoritative scoring table, never silent — this is a defined review step, not a vague instruction. The exact 13 pk_basics trick names and `movement` values are enumerated.

**Type consistency:** `DScoreBook.d_score_for/same_dscore` and module `same_dscore` defined in Task 1, used in Task 2. `find_alias_duplicates(fig)->list[dict]` with keys `duplicate`/`canonical` defined and asserted consistently in Task 3. `_norm` reused from `oracle_cues` (consistent with existing P0 code). `movement` cue name consistent across decoder/oracle/ontology/test in Task 4. Eval `evaluate_rows`/`Row` signatures match the merged P0 harness (verified against the in-repo file structure).

**Non-circularity guard:** Task 2 and Task 4 re-validate via the P0 harness and explicitly require `no_canonical`/`ontology_only` to stay strong, so improvements cannot come from re-introducing POOL-B-style memorization.
