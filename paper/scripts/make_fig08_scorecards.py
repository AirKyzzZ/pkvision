#!/usr/bin/env python3
"""Render Figure 8: qualitative scorecards for IMG_5985 and IMG_4243.

Shows side-by-side ground-truth vs best-VLM scorecards. Because POOL-B in
this preprint is single-trick only, this figure is assembled from the
per-trick predictions where a full-run clip was segmented into trick_0N.mp4
pieces (IMG_5985 -> vlm_clips/IMG_5985/trick_{01,02,04}.mp4).

For IMG_4243 / test_run_2 the underlying per-trick clips may not exist
yet, in which case the corresponding subplot shows a "pending full-run
benchmark" message. The figure is still generated so the LaTeX build
never fails.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

RESULTS = REPO / "paper" / "experiments" / "results.jsonl"
GT_CSV = REPO / "paper" / "experiments" / "ground_truth.csv"
OUT = REPO / "paper" / "figures" / "fig08_scorecards.pdf"

# Map a POOL-A full-run file to the POOL-B clip prefix used for its segments.
RUN_TO_SEGMENT_PREFIX = {
    "data/run_testing/IMG_5985.mov": "data/vlm_clips/IMG_5985/",
    "data/run_testing/test_run_2.mp4": "data/vlm_clips/test_run_2_zoomed/",
}


def load_gt_pool_a() -> list[dict]:
    out: list[dict] = []
    with GT_CSV.open() as f:
        for r in csv.DictReader(f):
            if r["pool"] == "POOL-A":
                out.append(r)
    return out


def load_results() -> list[dict]:
    if not RESULTS.exists():
        return []
    return [json.loads(l) for l in RESULTS.read_text().splitlines() if l.strip()]


def preferred_model(rows: list[dict]) -> str | None:
    """Pick the model with the most coverage on the supplied rows."""
    from collections import Counter
    c = Counter(r["model"] for r in rows)
    return c.most_common(1)[0][0] if c else None


def build_card(run_row: dict, results: list[dict]) -> tuple[list[dict], list[dict] | None, str]:
    """Return (gt_card, vlm_card or None, model_label)."""
    run_path = run_row["clip_path"]
    prefix = RUN_TO_SEGMENT_PREFIX.get(run_path)

    # GT side --- parse the comma-separated list.
    gt_names = [x.strip() for x in run_row["fig_name"].split(",") if x.strip()]
    # Lookup each GT name's D-score from fig_tricks_2025.json.
    from json import load as jload
    with (REPO / "data" / "fig_tricks_2025.json").open() as f:
        fig = jload(f)
    fig_lookup: dict[str, float] = {}
    for cat in fig["categories"].values():
        for t in cat["tricks"]:
            fig_lookup[t["name"].lower()] = float(t.get("score", 0) or 0)
            for a in t.get("aliases", []) or []:
                fig_lookup[a.lower()] = float(t.get("score", 0) or 0)
    gt_card = [
        {"name": n, "d": fig_lookup.get(n.lower(), 0.0)} for n in gt_names
    ]

    if prefix is None:
        return gt_card, None, "not yet benchmarked"

    # Find matching VLM predictions on the per-segment clips.
    seg_rows = [r for r in results if r["clip_path"].startswith(prefix)]
    if not seg_rows:
        return gt_card, None, "no per-segment results yet"

    model = preferred_model(seg_rows)
    model_rows = sorted(
        [r for r in seg_rows if r["model"] == model],
        key=lambda r: r["clip_path"],
    )
    vlm_card = []
    for r in model_rows:
        fm = r.get("fig_match") or {}
        name = fm.get("name") or r.get("parsed_trick") or "?"
        d = float(fm.get("d_score") or 0)
        vlm_card.append({"name": name, "d": d})
    return gt_card, vlm_card, model.split("/")[-1]


def draw_card(ax, title: str, items: list[dict] | None, color: str,
              note: str = "") -> None:
    ax.set_axis_off()
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.add_patch(FancyBboxPatch(
        (0.02, 0.04), 0.96, 0.92,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        linewidth=1, edgecolor=color, facecolor="#f6f9fc",
    ))
    ax.text(0.5, 0.92, title, ha="center", va="top",
            fontsize=9, fontweight="bold", color=color)

    if not items:
        ax.text(0.5, 0.5, note or "(pending)", ha="center", va="center",
                fontsize=8, color="#999")
        return

    y = 0.80
    step = max(0.09, 0.70 / max(len(items), 1))
    for i, item in enumerate(items):
        ax.text(0.06, y, f"{i+1}.", fontsize=7.5, color="#333")
        ax.text(0.14, y, item["name"][:28], fontsize=7.5, color="#000")
        ax.text(0.94, y, f"D={item['d']:.1f}", ha="right", fontsize=7.5,
                color="#000", family="monospace")
        y -= step
    top3 = sorted([i["d"] for i in items], reverse=True)[:3]
    total = sum(top3)
    ax.plot([0.08, 0.92], [0.13, 0.13], color=color, lw=0.8)
    ax.text(0.06, 0.08, "Top-3 D-sum", fontsize=7.5, color=color, fontweight="bold")
    ax.text(0.94, 0.08, f"D={total:.1f}", ha="right", fontsize=7.5,
            color=color, fontweight="bold", family="monospace")


def main() -> None:
    gts = load_gt_pool_a()
    results = load_results()

    # Focus on IMG_5985 and test_run_2 (the ones with per-segment clips).
    runs = [g for g in gts if g["clip_path"] in RUN_TO_SEGMENT_PREFIX]
    if not runs:
        runs = gts[:2]

    fig, axes = plt.subplots(len(runs), 2, figsize=(7.0, 1.7 * len(runs)))
    if len(runs) == 1:
        axes = [axes]

    for i, run in enumerate(runs):
        gt_card, vlm_card, model_label = build_card(run, results)
        name = Path(run["clip_path"]).name
        draw_card(axes[i][0], f"{name} --- ground truth", gt_card, "#17456e")
        draw_card(
            axes[i][1],
            f"{name} --- VLM ({model_label})",
            vlm_card, "#c7530f",
            note="(per-segment benchmark\nnot yet run on this file)",
        )

    fig.suptitle("Qualitative scorecards: ground truth vs.\\ VLM predictions",
                 fontsize=10, y=0.995)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=160, bbox_inches="tight")
    print(f"Wrote {OUT.relative_to(REPO)} ({len(runs)} runs)")


if __name__ == "__main__":
    main()
