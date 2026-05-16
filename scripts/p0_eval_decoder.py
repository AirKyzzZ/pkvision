"""P0 blind decoder evaluation. Run AFTER data/p0_blind_set/verified.csv exists.

Feeds ORACLE cues (true trick's FIG attributes) to the FROZEN decoder in 4
ablation configs and reports top-1 / top-3 / D-score MAE / per-group top-1.
Task 6 applies the go/no-go gate to this report.

Usage: python3 scripts/p0_eval_decoder.py [--verified data/p0_blind_set/verified.csv]
"""
from __future__ import annotations
import argparse
import csv
import json
import statistics
from dataclasses import dataclass
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root on path when run directly

from core.data.decontam import assert_clean
from core.recognition.oracle_cues import OracleCueBook, OracleCueError, _norm
from core.recognition.fig_decoder import FIGDecoder

CONFIGS: dict[str, dict] = {
    "full": {},
    "no_canonical": {"disable_canonical": True},
    "no_group_bonus": {"disable_group_bonus": True},
    "ontology_only": {"disable_canonical": True, "disable_group_bonus": True},
}


@dataclass
class Row:
    clip: str
    true_trick: str
    cues: dict
    d_score: float
    disambig_group: str | None = None


def load_rows(verified_csv: Path, book: OracleCueBook) -> tuple[list[Row], list[tuple[str, str]]]:
    """Returns (usable rows, skipped[(clip, unresolved_name)]). Rows with
    keep!=1 or empty verified trick are ignored. A verified trick the ontology
    can't resolve is skipped+recorded (one human typo must not nuke the gate).
    """
    rows: list[Row] = []
    skipped: list[tuple[str, str]] = []
    with verified_csv.open() as f:
        for r in csv.DictReader(f):
            if (r.get("keep") or "").strip() != "1":
                continue
            true = (r.get("verified_fig_trick") or "").strip()
            if not true:
                continue
            try:
                canon = book.canonical_name(true)
                cues = book.cues_for(true)
                d = book.d_score_for(true)
            except OracleCueError:
                skipped.append((r.get("clip_path", "?"), true))
                continue
            rows.append(Row(
                clip=r.get("clip_path", "?"),
                true_trick=canon,
                cues=cues,
                d_score=d,
                disambig_group=(r.get("disambig_group") or None),
            ))
    assert_clean([Path(x.clip) for x in rows])  # fail-fast: no contamination
    return rows, skipped


def evaluate_rows(rows: list[Row], decoder=None) -> dict:
    decoder = decoder or FIGDecoder()
    out: dict = {}
    for cfg_name, cfg in CONFIGS.items():
        n = len(rows)
        top1 = top3 = 0
        dscore_hits = 0
        abs_err: list[float] = []
        per_group: dict[str, list[int]] = {}
        for row in rows:
            cands = decoder.rank(dict(row.cues), k=5, **cfg)
            names = [getattr(c, "fig_name", None) for c in cands]
            nt = _norm(row.true_trick)
            hit1 = bool(names) and names[0] is not None and _norm(names[0]) == nt
            hit3 = any(x is not None and _norm(x) == nt for x in names[:3])
            top1 += int(hit1)
            top3 += int(hit3)
            if cands:
                pred_d = float(cands[0].d_score)
                abs_err.append(abs(pred_d - float(row.d_score)))
                dscore_hit = abs(pred_d - float(row.d_score)) <= 1e-9
                dscore_hits += int(dscore_hit)
            g = row.disambig_group or "_none"
            per_group.setdefault(g, []).append(int(hit1))
        out[cfg_name] = {
            "n": n,
            "top1": (top1 / n) if n else 0.0,
            "top3": (top3 / n) if n else 0.0,
            "d_score_mae": statistics.fmean(abs_err) if abs_err else None,
            "per_group_top1": {g: sum(v) / len(v) for g, v in per_group.items()},
            "dscore_correct": (dscore_hits / n) if n else 0.0,
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verified", default="data/p0_blind_set/verified.csv")
    args = ap.parse_args()
    vpath = Path(args.verified)
    if not vpath.exists():
        raise SystemExit(f"Missing {vpath} — run scripts/p0_build_blind_set.py "
                         f"then complete the human verification step first.")
    book = OracleCueBook("data/fig_tricks_2025.json")
    rows, skipped = load_rows(vpath, book)
    if not rows:
        raise SystemExit("No usable verified rows (keep=1 + resolvable trick).")
    res = evaluate_rows(rows, FIGDecoder())
    outdir = Path("data/p0_blind_set")
    (outdir / "report.json").write_text(json.dumps(
        {"n_rows": len(rows), "n_skipped": len(skipped),
         "skipped": skipped, "configs": res}, indent=2))
    lines = ["# P0 Blind Decoder Report", "",
             f"N = {len(rows)} verified rows; skipped (unresolved verified "
             f"trick) = {len(skipped)}", ""]
    for cfg, m in res.items():
        lines.append(
            f"- **{cfg}**: top1={m['top1']:.1%} top3={m['top3']:.1%} "
            f"dscore_correct={m['dscore_correct']:.1%} "
            f"d_score_MAE={m['d_score_mae']}")
    gate = res["full"]["top1"] >= 0.80
    canon_drop = res["full"]["top1"] - res["no_canonical"]["top1"]
    lines += ["",
              f"GATE (full top1 >= 80%): {'PASS' if gate else 'FAIL'}",
              f"Canonical-bonus dependence (full - no_canonical top1): "
              f"{canon_drop:.1%}"]
    if skipped:
        lines += ["", "Skipped rows (verified trick not in FIG ontology — "
                  "fix these names in verified.csv):"]
        lines += [f"  - {c}: {n!r}" for c, n in skipped]
    (outdir / "report.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
