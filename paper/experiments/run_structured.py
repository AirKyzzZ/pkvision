#!/usr/bin/env python3
"""Structured benchmark harness — recognizer-agnostic.

Loads `ground_truth.csv`, runs a chosen `Recognizer` (heuristic, VLM, or
future structured models) through every selected clip, and writes a fresh
jsonl file so new runs never mix with old VLM results.

Usage
-----
    # Single-trick benchmark with the heuristic baseline.
    python paper/experiments/run_structured.py \\
        --recognizer heuristic \\
        --pool POOL-B \\
        --out paper/experiments/results_v2.jsonl

    # Re-run a VLM through the new harness (still produces top-k).
    python paper/experiments/run_structured.py \\
        --recognizer vlm:gemini-2.5-flash --pool POOL-B

    # Dry-run.
    python paper/experiments/run_structured.py --recognizer heuristic --dry-run
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from paper.experiments.recognizers import build_recognizer, RecognizerOutput  # noqa: E402


def load_ground_truth(path: Path, pools: list[str]) -> list[dict]:
    rows: list[dict] = []
    with path.open() as f:
        for raw in csv.DictReader(f):
            if not raw["fig_name"].strip():
                continue
            if raw["pool"].strip() not in pools:
                continue
            clip_path = REPO / raw["clip_path"].strip()
            if not clip_path.exists():
                print(f"[warn] missing clip {clip_path}", file=sys.stderr)
                continue
            rows.append({
                "clip_path": clip_path,
                "pool": raw["pool"].strip(),
                "fig_name": raw["fig_name"].strip(),
                "d_score": _f(raw.get("d_score")),
                "flip": _f(raw.get("flip_count")),
                "twist": _f(raw.get("twist_count")),
                "direction": raw.get("direction", "").strip(),
                "takeoff": raw.get("takeoff", "").strip(),
            })
    return rows


def _f(v: str | None) -> float:
    if not v:
        return 0.0
    try:
        return float(v)
    except ValueError:
        return 0.0


def record(gt: dict, out: RecognizerOutput, topk: int) -> dict:
    top_names = out.topk_names(topk)
    gt_name = gt["fig_name"]
    return {
        "clip_id": gt["clip_path"].stem,
        "clip_path": str(gt["clip_path"].relative_to(REPO)),
        "pool": gt["pool"],
        "ground_truth": {
            "fig_name": gt_name,
            "d_score": gt["d_score"],
            "attributes": {
                "flip": gt["flip"],
                "twist": gt["twist"],
                "direction": gt["direction"],
                "takeoff": gt["takeoff"],
            },
        },
        "recognizer": out.recognizer,
        "mode": out.mode,
        "top1": out.top1,
        "topk": top_names,
        "top1_correct": _matches_any(gt_name, [out.top1]),
        "topk_correct": _matches_any(gt_name, top_names),
        "candidates": [asdict(c) for c in out.candidates[:topk]],
        "cues": out.cues,
        "raw_text": out.raw_text,
        "latency_s": round(out.latency_s, 2),
        "input_tokens": out.input_tokens,
        "output_tokens": out.output_tokens,
        "error": out.error,
        "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }


def _matches_any(gt_name: str, candidates: list[str]) -> bool:
    """POOL-A truths are comma-separated lists; POOL-B is single names.

    For POOL-B this is exact-match. For POOL-A we consider a "hit" if any
    candidate matches any of the comma-separated ground-truth names (weak,
    but the POOL-A full-run task is handled by the ordering pipeline, not
    by single-clip top-k).
    """
    gt_names = [n.strip() for n in gt_name.split(",")]
    for cand in candidates:
        if not cand:
            continue
        for gt in gt_names:
            if cand.lower() == gt.lower():
                return True
    return False


def summarize(records: list[dict]) -> dict:
    total = len(records)
    if total == 0:
        return {}
    top1 = sum(1 for r in records if r["top1_correct"])
    topk = sum(1 for r in records if r["topk_correct"])
    return {
        "n_clips": total,
        "top1_accuracy": round(top1 / total, 3),
        "topk_accuracy": round(topk / total, 3),
        "errors": sum(1 for r in records if r["error"]),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--recognizer",
        required=True,
        help="Recognizer spec. Examples: 'heuristic', 'vlm:gemini-2.5-flash', 'vlm:claude'.",
    )
    ap.add_argument("--pool", action="append", default=None,
                    help="Repeatable. Default: POOL-B.")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--gt", type=Path,
                    default=REPO / "paper" / "experiments" / "ground_truth.csv")
    ap.add_argument("--out", type=Path,
                    default=REPO / "paper" / "experiments" / "results_v2.jsonl")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    pools = args.pool or ["POOL-B"]
    gt_rows = load_ground_truth(args.gt, pools)
    print(f"Loaded {len(gt_rows)} ground-truth rows ({pools}) "
          f"from {args.gt.relative_to(REPO)}")

    if args.dry_run:
        for gt in gt_rows:
            print(f"  would run {args.recognizer} on "
                  f"{gt['clip_path'].relative_to(REPO)} -> GT={gt['fig_name']}")
        return

    rec = build_recognizer(args.recognizer)
    print(f"Recognizer: {rec.name}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    with args.out.open("a") as sink:
        for gt in gt_rows:
            print(f"\n[{rec.name}] {gt['clip_path'].name}  (GT: {gt['fig_name']})")
            try:
                out = rec.recognize(gt["clip_path"])
            except Exception as exc:  # noqa: BLE001
                print(f"  ! recognizer raised: {exc}")
                out = RecognizerOutput(recognizer=rec.name, error=str(exc))
            if out.error:
                print(f"  ! {out.error[:120]}")
            top = out.top1 or "<empty>"
            mark = "✓" if _matches_any(gt["fig_name"], [top]) else " "
            print(f"  {mark} top1={top}  topk={out.topk_names(3)}  ({out.latency_s:.1f}s)")
            rec_row = record(gt, out, args.topk)
            records.append(rec_row)
            sink.write(json.dumps(rec_row) + "\n")
            sink.flush()

    summary = summarize(records)
    if summary:
        print(f"\n── Summary ({rec.name}) ──")
        print(f"  clips:        {summary['n_clips']}")
        print(f"  top-1 acc:    {summary['top1_accuracy']:.1%}")
        print(f"  top-{args.topk} acc:    {summary['topk_accuracy']:.1%}")
        print(f"  errors:       {summary['errors']}")
    try:
        rel_out = args.out.relative_to(REPO)
    except ValueError:
        rel_out = args.out
    print(f"\nAppended to {rel_out}")


if __name__ == "__main__":
    main()
