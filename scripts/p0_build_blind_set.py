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
        if not isinstance(gobj, dict):
            continue
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
