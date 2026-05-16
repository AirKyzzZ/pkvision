"""Build a stratified, decontaminated blind test set for P0.

P0 feeds ORACLE cues (from each clip's true FIG trick) to the frozen decoder,
so only (clip_id, true_fig_trick) pairs are needed -- not pixels.

"Hard" = the trick shares its physics signature
(category, flip, twist, direction, axis) with >=1 other FIG trick (25 groups /
106 of 149 tricks). Candidates: (1) the 99 FIG-grounded entries in the
attribute manifest (reliable fig_name; identity = npy_path); (2) parkourtheory
clips resolved by filename. Unioned, decontaminated, deduped to one clip per
distinct true trick (oracle cues are per-trick; duplicates add no signal).

Emits data/p0_blind_set/candidates.csv. The user fills verified_fig_trick +
keep=1 and saves data/p0_blind_set/verified.csv. fig_grounded rows are
high-confidence; filename rows need closer scrutiny.
"""
from __future__ import annotations
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root on path when run directly

from core.data.decontam import is_contaminated
from core.recognition.oracle_cues import OracleCueBook, _norm

FIG_PATH = "data/fig_tricks_2025.json"
MANIFEST_PATH = "data/v5_attribute_training/attribute_manifest.json"
CLIPS_DIR = Path("data/parkourtheory_clips")
OUT = Path("data/p0_blind_set/candidates.csv")
SEED = 1337


@dataclass(frozen=True)
class ClipCandidate:
    path: Path
    proposed_fig_trick: str
    disambig_group: str | None
    source: str = "filename"


def physics_collision_groups(fig_path: str = FIG_PATH) -> dict[str, str]:
    """Normalized FIG trick name -> stable physics-signature group key, only
    for tricks whose (category, flip, twist, direction, axis) signature is
    shared by >=2 tricks. Unique-signature tricks are omitted (easy)."""
    raw = json.loads(Path(fig_path).read_text())
    by_sig: dict[tuple, list[str]] = {}
    for category, obj in raw["categories"].items():
        for t in obj["tricks"]:
            sig = (category, t.get("flip"), t.get("twist"),
                   t.get("direction"), t.get("axis"))
            by_sig.setdefault(sig, []).append(t["name"])
    groups: dict[str, str] = {}
    for (cat, flip, twist, direction, axis), names in by_sig.items():
        if len(names) < 2:
            continue
        key = f"{cat}|flip={flip}|twist={twist}|dir={direction}|axis={axis}"
        for n in names:
            groups[_norm(n)] = key
    return groups


_BOOK = OracleCueBook(FIG_PATH)
_GROUPS = physics_collision_groups(FIG_PATH)


def _group_for(trick_name: str) -> str | None:
    return _GROUPS.get(_norm(trick_name))


def map_clip_to_fig(filename: str) -> str | None:
    stem = Path(filename).stem
    try:
        return _BOOK.canonical_name(stem.replace("_", " "))
    except Exception:
        return None


def _manifest_items(manifest_path: str = MANIFEST_PATH) -> list[dict]:
    raw = json.loads(Path(manifest_path).read_text())
    if isinstance(raw, list):
        return [x for x in raw if isinstance(x, dict)]
    for v in raw.values():
        if isinstance(v, list) and v and isinstance(v[0], dict):
            return v
    raise ValueError(f"Could not locate clip list in {manifest_path}")


def load_fig_grounded_candidates(
    manifest_path: str = MANIFEST_PATH,
) -> list[ClipCandidate]:
    out: list[ClipCandidate] = []
    for it in _manifest_items(manifest_path):
        fig = str(it.get("fig_name") or "").strip()
        if not fig:
            continue
        raw_path = str(it.get("npy_path") or it.get("slug") or "").strip()
        if not raw_path:
            continue
        p = Path(raw_path)
        if is_contaminated(p):
            continue
        try:
            canon = _BOOK.canonical_name(fig)
        except Exception:
            canon = fig
        out.append(ClipCandidate(p, canon, _group_for(canon), "fig_grounded"))
    return out


def gather_candidates(
    clips_dir: Path = CLIPS_DIR, manifest_path: str = MANIFEST_PATH
) -> list[ClipCandidate]:
    cands: list[ClipCandidate] = list(load_fig_grounded_candidates(manifest_path))
    if clips_dir.exists():
        for p in sorted(clips_dir.glob("*.mp4")):
            if is_contaminated(p):
                continue
            fig = map_clip_to_fig(p.name)
            if fig is None:
                continue
            cands.append(ClipCandidate(p, fig, _group_for(fig), "filename"))
    # one candidate per distinct true trick; prefer fig_grounded source
    best: dict[str, ClipCandidate] = {}
    for c in cands:
        k = _norm(c.proposed_fig_trick)
        cur = best.get(k)
        if cur is None or (cur.source != "fig_grounded"
                           and c.source == "fig_grounded"):
            best[k] = c
    return list(best.values())


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
        w.writerow(["clip_path", "source", "proposed_fig_trick",
                    "disambig_group", "verified_fig_trick", "keep"])
        for c in picked:
            w.writerow([str(c.path), c.source, c.proposed_fig_trick,
                        c.disambig_group or "", "", ""])
    if not picked:
        print("No candidates produced!")
        return
    n_hard = sum(1 for c in picked if c.disambig_group)
    print(f"Wrote {len(picked)} candidates "
          f"({n_hard} hard, {n_hard / len(picked):.0%}) -> {OUT}")
    print("ACTION REQUIRED: fill verified_fig_trick + keep=1, save as "
          "data/p0_blind_set/verified.csv")


if __name__ == "__main__":
    main()
