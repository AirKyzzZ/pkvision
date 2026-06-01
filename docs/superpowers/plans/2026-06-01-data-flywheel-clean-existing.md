# Data Flywheel — Foundation + Free Verification Loop (Plan 1 of 2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the propose→verify→store foundation and run a $0 verification loop on real clips, so a human can confirm/correct machine-proposed `{trick, cues}` and accumulate clean labels.

**Architecture:** A `ClipRef` abstracts the three clip formats (parkourtheory `.mp4`+skeleton, v5/gold `.npy` frames). A `Proposer` turns a clip into a `{trick, cues, confidence}` proposal; the free `LocalModelProposer` uses the extracted v2 cue model + the existing `FIGMatcher`. `make_proposals.py` caches proposals; `verify_server.py` (modeled on `label_server.py`) shows them for one-click confirm / cue-override and appends clean records to `VerifiedStore` (`data/labeling/verified.jsonl`).

**Tech Stack:** Python 3.14, PyTorch 2.11 (MPS), numpy, OpenCV, stdlib `http.server`, pytest. Reuses `core/vlm/`, `core/recognition/fig_decoder.py`, `core/recognition/oracle_cues.py`, `scripts/diagnose_skeleton_cues_v2.py`.

**Scope:** Spec phases P-A + P-B only (`docs/superpowers/specs/2026-06-01-data-flywheel-clean-existing-design.md`). The selector, VLM probe-gate, and retrain (P-C…P-F) are **Plan 2**, written after P-B proves the loop.

---

## File structure

- Create `core/labeling/__init__.py` — package marker.
- Create `core/labeling/clip_ref.py` — `ClipRef` over video/frames; `get_frames()`, `get_skeleton()`, `slug`.
- Create `core/labeling/cue_model.py` — extracted v2 model: `featurize`, `base_seq`, `Encoder`, `CueModel`, `CueClasses`, `predict_cues()`.
- Create `core/labeling/proposer.py` — `Proposal` dataclass, `Proposer` ABC, `LocalModelProposer`.
- Create `core/labeling/store.py` — `VerifiedStore` (append/read jsonl), `VerifiedRecord`.
- Create `scripts/make_proposals.py` — generate + cache proposals for a clip list.
- Create `scripts/verify_server.py` — hybrid confirm/override web UI → `VerifiedStore` (modeled on `scripts/label_server.py`).
- Create `tests/labeling/{test_clip_ref,test_cue_model,test_proposer,test_store}.py`.
- Data (gitignored): `data/labeling/proposals/<proposer>/<slug>.json`, `data/labeling/verified.jsonl`.

---

## Task 1: `ClipRef` — one handle for video + frame clips

**Files:**
- Create: `core/labeling/__init__.py` (empty)
- Create: `core/labeling/clip_ref.py`
- Test: `tests/labeling/test_clip_ref.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/labeling/test_clip_ref.py
import numpy as np
from core.labeling.clip_ref import ClipRef


def test_frames_clip_reads_npy(tmp_path):
    arr = np.zeros((8, 64, 64, 3), np.uint8)
    p = tmp_path / "back_full.npy"
    np.save(p, arr)
    ref = ClipRef.from_path(p)
    assert ref.slug == "back_full"
    frames = ref.get_frames(max_frames=4)
    assert frames.shape == (4, 64, 64, 3)


def test_skeleton_attaches_and_returns(tmp_path):
    ref = ClipRef(slug="x", video_path=None, frames_path=tmp_path / "x.npy",
                  skeleton=np.zeros((10, 17, 3), np.float32))
    assert ref.get_skeleton().shape == (10, 17, 3)
```

- [ ] **Step 2: Run test, verify it fails**

Run: `python3 -m pytest tests/labeling/test_clip_ref.py -v`
Expected: FAIL (`ModuleNotFoundError: core.labeling.clip_ref`).

- [ ] **Step 3: Implement**

```python
# core/labeling/clip_ref.py
"""One handle over the project's clip formats: .mp4 video (parkourtheory) and
.npy RGB frame arrays (v5/gold/comp). Skeletons (T,17,3) attach optionally."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class ClipRef:
    slug: str
    video_path: Path | None
    frames_path: Path | None
    skeleton: np.ndarray | None = None

    @classmethod
    def from_path(cls, path: str | Path) -> "ClipRef":
        path = Path(path)
        is_video = path.suffix.lower() in {".mp4", ".mov", ".webm"}
        return cls(
            slug=path.stem,
            video_path=path if is_video else None,
            frames_path=path if not is_video else None,
        )

    def get_frames(self, max_frames: int = 8) -> np.ndarray:
        """Uniformly sampled RGB frames (N,H,W,3) uint8, for VLM + UI preview."""
        if self.frames_path is not None:
            arr = np.load(self.frames_path)
        else:
            import cv2

            cap = cv2.VideoCapture(str(self.video_path))
            got = []
            while True:
                ok, f = cap.read()
                if not ok:
                    break
                got.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
            cap.release()
            arr = np.stack(got) if got else np.zeros((0, 1, 1, 3), np.uint8)
        if len(arr) <= max_frames or len(arr) == 0:
            return arr
        idx = np.linspace(0, len(arr) - 1, max_frames).astype(int)
        return arr[idx]

    def get_skeleton(self) -> np.ndarray | None:
        return self.skeleton
```

- [ ] **Step 4: Run test, verify it passes**

Run: `python3 -m pytest tests/labeling/test_clip_ref.py -v`
Expected: PASS (both tests).

- [ ] **Step 5: Commit**

```bash
git add core/labeling/__init__.py core/labeling/clip_ref.py tests/labeling/test_clip_ref.py
git commit -m "feat(labeling): ClipRef over video + frame clips"
```

---

## Task 2: Extract the v2 cue model into `core/labeling/cue_model.py`

The proven v2 model logic currently lives only inside the one-off `scripts/diagnose_skeleton_cues_v2.py`. Extract it verbatim (same `base_seq`, `featurize`, `Encoder`, `CueModel`, `LR_SWAP`, the cue class sets) into an importable module, plus a `predict_cues(skeleton) -> dict` that the proposer calls. Do **not** change the math — copy the functions as-is so behaviour matches the diagnostic.

**Files:**
- Create: `core/labeling/cue_model.py`
- Test: `tests/labeling/test_cue_model.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/labeling/test_cue_model.py
import numpy as np
from core.labeling.cue_model import featurize, base_seq, CUE_CLASSES


def test_featurize_shape():
    seq = np.zeros((48, 17, 3), np.float32)
    feat = featurize(seq)
    assert feat.shape == (48, 88)


def test_base_seq_resamples():
    arr = np.random.rand(13, 17, 3).astype(np.float32)
    out = base_seq(arr, 48)
    assert out.shape == (48, 17, 3)


def test_cue_classes_present():
    for c in ("context", "direction", "flip", "twist", "axis"):
        assert c in CUE_CLASSES
```

- [ ] **Step 2: Run test, verify it fails**

Run: `python3 -m pytest tests/labeling/test_cue_model.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement — copy from `scripts/diagnose_skeleton_cues_v2.py`**

Copy these symbols **unchanged** from `scripts/diagnose_skeleton_cues_v2.py` into `core/labeling/cue_model.py`: `LR_SWAP`, `D`, `base_seq`, `augment`, `featurize`, the `Encoder` and `CueModel` `nn.Module` classes (lift them out of `main()` to module scope, taking `classes` + `t` as constructor args). Then add the fixed cue vocabulary and a predictor:

```python
# appended to core/labeling/cue_model.py
import torch

CUE_CLASSES = {
    "context": ["acrobatics", "pk_basics", "swing", "wall"],
    "direction": ["backward", "forward", "none", "side"],
    "flip": [0, 1, 2, 3],
    "twist": [0, 1, 2, 3],
    "axis": ["lateral", "longitudinal", "off_axis", "sagittal"],
}
TW_NUM = {0: 0.0, 1: 0.5, 2: 1.0}  # bin 3 (>=1.5) -> abstain per cue contract


def predict_cues(model: "CueModel", skeleton: np.ndarray, t: int = 48) -> dict:
    """skeleton (T,17,3) -> {cue: value, '<cue>_conf': float}. Twist bin 3 abstains."""
    model.eval()
    x = torch.tensor(featurize(base_seq(skeleton, t))[None])
    with torch.no_grad():
        out = model(x.to(next(model.parameters()).device))
    cues: dict = {}
    for cue, logits in out.items():
        probs = torch.softmax(logits[0], -1)
        idx = int(probs.argmax())
        conf = float(probs[idx])
        val = CUE_CLASSES[cue][idx] if cue in CUE_CLASSES else idx
        if cue == "twist":
            if idx in TW_NUM:
                cues["twist"] = TW_NUM[idx]
        elif cue == "flip":
            cues["flip"] = float(val)
        elif val not in (None, "none"):
            cues[cue] = val
        cues[f"{cue}_conf"] = conf
    return cues
```

Note: `CueModel.__init__` must accept the `classes` dict so heads size correctly; default it to `CUE_CLASSES`.

- [ ] **Step 4: Run test, verify it passes**

Run: `python3 -m pytest tests/labeling/test_cue_model.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add core/labeling/cue_model.py tests/labeling/test_cue_model.py
git commit -m "refactor(labeling): extract v2 cue model into core/labeling/cue_model.py"
```

---

## Task 3: `VerifiedStore` — append-only clean labels

**Files:**
- Create: `core/labeling/store.py`
- Test: `tests/labeling/test_store.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/labeling/test_store.py
from core.labeling.store import VerifiedStore, VerifiedRecord


def test_append_and_read_roundtrip(tmp_path):
    store = VerifiedStore(tmp_path / "verified.jsonl")
    rec = VerifiedRecord(slug="back_full", trick="Backflip 360",
                         cues={"flip": 1.0, "twist": 1.0, "direction": "backward"},
                         d_score=2.0, proposer_source="local", action="confirm",
                         verified_at="2026-06-01T10:00:00Z")
    store.append(rec)
    rows = store.read_all()
    assert len(rows) == 1
    assert rows[0].slug == "back_full"
    assert rows[0].cues["twist"] == 1.0


def test_verified_slugs_set(tmp_path):
    store = VerifiedStore(tmp_path / "v.jsonl")
    store.append(VerifiedRecord("a", "T", {}, 1.0, "local", "confirm", "t"))
    store.append(VerifiedRecord("b", "T", {}, 1.0, "local", "confirm", "t"))
    assert store.verified_slugs() == {"a", "b"}
```

- [ ] **Step 2: Run test, verify it fails**

Run: `python3 -m pytest tests/labeling/test_store.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement**

```python
# core/labeling/store.py
"""Append-only, auditable store of human-verified labels (data/labeling/verified.jsonl)."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
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
```

- [ ] **Step 4: Run test, verify it passes**

Run: `python3 -m pytest tests/labeling/test_store.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add core/labeling/store.py tests/labeling/test_store.py
git commit -m "feat(labeling): append-only VerifiedStore"
```

---

## Task 4: `Proposer` interface + `LocalModelProposer`

**Files:**
- Create: `core/labeling/proposer.py`
- Test: `tests/labeling/test_proposer.py`

- [ ] **Step 1: Write the failing test** (uses a stub model so it needs no checkpoint)

```python
# tests/labeling/test_proposer.py
import numpy as np
from core.labeling.clip_ref import ClipRef
from core.labeling.proposer import Proposal, LocalModelProposer


class _StubModel:
    """Returns fixed cues so we test the proposer wiring, not the net."""
    def predict(self, skeleton):
        return {"flip": 1.0, "flip_conf": 0.9, "direction": "backward",
                "direction_conf": 0.8, "context": "acrobatics", "context_conf": 0.7}


def test_local_proposer_emits_proposal():
    ref = ClipRef("back_full", None, None, skeleton=np.zeros((10, 17, 3), np.float32))
    prop = LocalModelProposer(model=_StubModel()).propose(ref)
    assert isinstance(prop, Proposal)
    assert prop.cues["flip"] == 1.0
    assert 0.0 <= prop.confidence <= 1.0
    assert prop.source == "local"
```

- [ ] **Step 2: Run test, verify it fails**

Run: `python3 -m pytest tests/labeling/test_proposer.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement** (`FIGMatcher` maps cues→trick+D-score; confidence = mean of per-cue confs)

```python
# core/labeling/proposer.py
"""clip -> {trick, cues, confidence} proposal. LocalModelProposer is free ($0)."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from core.labeling.clip_ref import ClipRef
from core.recognition.fig_matcher import FIGMatcher  # NOTE: confirm module path at impl time


@dataclass
class Proposal:
    slug: str
    trick: str | None
    cues: dict
    d_score: float | None
    confidence: float
    source: str


class Proposer(ABC):
    @abstractmethod
    def propose(self, clip: ClipRef) -> Proposal: ...


class LocalModelProposer(Proposer):
    def __init__(self, model, matcher: FIGMatcher | None = None):
        self.model = model
        self.matcher = matcher or FIGMatcher()

    def propose(self, clip: ClipRef) -> Proposal:
        skel = clip.get_skeleton()
        cues = self.model.predict(skel)
        confs = [v for k, v in cues.items() if k.endswith("_conf")]
        confidence = float(sum(confs) / len(confs)) if confs else 0.0
        match = self.matcher.match(
            trick_name=cues.get("context", ""),
            flip_count=cues.get("flip", 0.0),
            twist_count=cues.get("twist", 0.0),
            direction=cues.get("direction"),
        )
        clean = {k: v for k, v in cues.items() if not k.endswith("_conf")}
        return Proposal(clip.slug, match.fig_name if match else None, clean,
                        match.d_score if match else None, confidence, "local")
```

NOTE for the implementer: `fig_matcher.py` currently lives at `core/vlm/fig_matcher.py` (verify import path; adjust the import above to match). Wrap `model.predict` so a real `CueModel` is adapted via a thin lambda passing through `core.labeling.cue_model.predict_cues`.

- [ ] **Step 4: Run test, verify it passes**

Run: `python3 -m pytest tests/labeling/test_proposer.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add core/labeling/proposer.py tests/labeling/test_proposer.py
git commit -m "feat(labeling): Proposer interface + free LocalModelProposer"
```

---

## Task 5: `make_proposals.py` — generate + cache proposals

**Files:**
- Create: `scripts/make_proposals.py`

- [ ] **Step 1: Implement** (loads skeletons, builds `ClipRef`s, runs a proposer, caches one JSON per clip)

```python
# scripts/make_proposals.py
"""Generate + cache {trick,cues,conf} proposals for a clip list.

Usage: python3 scripts/make_proposals.py --limit 50 --proposer local
Caches to data/labeling/proposals/<proposer>/<slug>.json (never recomputed).
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.labeling.clip_ref import ClipRef
from core.labeling.cue_model import CueModel, CUE_CLASSES, predict_cues
from core.labeling.proposer import LocalModelProposer

POSE_GLOB = str(ROOT / "data" / "keypoints" / "**" / "shard_*.npz")
CLIPS = ROOT / "data" / "parkourtheory_clips_cropped"


def load_skeletons() -> dict:
    skel = {}
    for sh in sorted(glob.glob(POSE_GLOB, recursive=True)):
        with np.load(sh) as z:
            for s in z.files:
                skel[s] = z[s].astype(np.float32)
    return skel


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--proposer", choices=["local"], default="local")
    ap.add_argument("--ckpt", default="")  # optional cue_model checkpoint; stub if empty
    args = ap.parse_args()

    out = ROOT / "data" / "labeling" / "proposals" / args.proposer
    out.mkdir(parents=True, exist_ok=True)
    skel = load_skeletons()

    import torch
    model = CueModel(CUE_CLASSES, t=48)
    if args.ckpt:
        model.load_state_dict(torch.load(args.ckpt, map_location="cpu", weights_only=True))

    class _Adapter:
        def predict(self, s):
            return predict_cues(model, s)

    proposer = LocalModelProposer(model=_Adapter())
    slugs = sorted(skel)[: args.limit]
    for s in slugs:
        ref = ClipRef(s, video_path=CLIPS / f"{s}.mp4", frames_path=None, skeleton=skel[s])
        prop = proposer.propose(ref)
        (out / f"{s}.json").write_text(json.dumps(asdict(prop), indent=2))
    print(f"wrote {len(slugs)} proposals to {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-run** (untrained model is fine — we are validating wiring, not accuracy)

Run: `python3 scripts/make_proposals.py --limit 5 --proposer local`
Expected: prints `wrote 5 proposals`, and `data/labeling/proposals/local/*.json` each contain `slug,trick,cues,d_score,confidence,source`.

- [ ] **Step 3: Commit**

```bash
git add scripts/make_proposals.py
git commit -m "feat(labeling): make_proposals — generate + cache proposals"
```

---

## Task 6: `verify_server.py` — hybrid confirm/override UI → VerifiedStore

Model this on `scripts/label_server.py` (stdlib `http.server`, the same HTML/JS card + keyboard pattern). **Read `scripts/label_server.py` fully first**, then build a parallel, purpose-built server (leave `label_server.py` untouched).

**Files:**
- Create: `scripts/verify_server.py`

Server contract (endpoints):
- `GET /api/pending` → JSON list of `{slug, preview_png_url, proposal:{trick,cues,confidence}}` built from the proposals cache, **excluding** slugs already in `VerifiedStore.verified_slugs()`.
- `GET /api/preview?slug=…` → a PNG strip of `ClipRef(...).get_frames()` (reuse the cv2 montage approach from `label_server.py`).
- `POST /api/verify` with `{slug, action, trick, cues}` → builds a `VerifiedRecord` (`verified_at` = UTC now passed in by the handler), `VerifiedStore.append(...)`, returns `{ok:true}`.

UI behaviour (hybrid): card shows preview + proposed trick + per-cue chips pre-selected to the proposal. **Enter** = confirm-as-is (`action="confirm"`); editing a chip then Enter = `action="override_cue"`; typing a different trick = `action="correct_trick"` (cues refill from `FIGMatcher`/ontology). Keep the existing keyboard map.

- [ ] **Step 1: Implement** `scripts/verify_server.py` per the contract above (reuse `label_server.py`'s `HTTPServer`/handler skeleton, HTML template, and montage helper; swap data source to the proposals cache + `VerifiedStore`; add the `/api/verify` POST).

- [ ] **Step 2: Manual smoke test**

Run: `python3 scripts/make_proposals.py --limit 30 --proposer local` then `python3 scripts/verify_server.py --port 8899`.
Open `http://localhost:8899`. Expected: 30 cards load with previews + proposed cues; confirming one writes a line to `data/labeling/verified.jsonl` and removes it from `/api/pending` on reload.

- [ ] **Step 3: Commit**

```bash
git add scripts/verify_server.py
git commit -m "feat(labeling): verify_server hybrid confirm/override UI"
```

---

## Task 7: P-B milestone — free validation batch (runbook)

No new code. Prove the loop end-to-end at $0.

- [ ] **Step 1:** `python3 scripts/make_proposals.py --limit 50 --proposer local`
- [ ] **Step 2:** `python3 scripts/verify_server.py` and verify ~30–50 clips (confirm / correct trick / override a cue at least once each, to exercise all three actions).
- [ ] **Step 3:** Confirm `data/labeling/verified.jsonl` has one record per verified clip with the right `action` tags. Run: `python3 -c "from core.labeling.store import VerifiedStore; print(len(VerifiedStore('data/labeling/verified.jsonl').read_all()))"`
- [ ] **Step 4:** Note observed **verification throughput** (clips/min) — this sizes whether the VLM proposer (Plan 2, P-D) is worth its ~$1 probe.
- [ ] **Step 5: Commit the seed labels**

```bash
git add data/labeling/verified.jsonl
git commit -m "data(labeling): first free-loop verified seed labels"
```

**Gate to Plan 2:** the loop runs, records are clean and correctly tagged, and throughput is acceptable. Then write Plan 2 (Selector → VLM probe-gate → first real round → retrain + measure lift).

---

## Self-review notes

- **Spec coverage:** P-A → Tasks 1–4 (clip_ref, cue_model, store, proposer); P-B → Tasks 5–7 (make_proposals, verify UI, free batch). Selector / probe / retrain are explicitly deferred to Plan 2 per the spec's phase split. ✓
- **Decontamination:** the held-out clean eval split is a Plan-2 concern (introduced with retrain in P-F); not needed for P-A/P-B. Flagged here so it is not forgotten.
- **Open impl detail:** confirm `FIGMatcher` import path (`core/vlm/fig_matcher.py`) in Task 4; `CueModel.__init__` must take the `classes` dict (Task 2) for `make_proposals` (Task 5) to construct it head-correct.
