#!/usr/bin/env python3
"""Render Figure 1: teaser --- competition clip -> VLM -> FIG scorecard.

Layout (full width, single column-pair if used as figure*):
  Left  : 4 keyframes from IMG_5985.mov (gainer full -> krok -> double cork)
  Right : a small card showing the auto-generated FIG scorecard

Reads results.jsonl if present for the actual model output; otherwise uses the
ground-truth labels as a preview.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

VIDEO = REPO / "data" / "run_testing" / "IMG_5985.mov"
RESULTS = REPO / "paper" / "experiments" / "results.jsonl"
GT_CSV = REPO / "paper" / "experiments" / "ground_truth.csv"
OUT = REPO / "paper" / "figures" / "fig01_teaser.pdf"


def extract_key_frames(video: Path, num: int = 4) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # Pick frames at the three known inversions + one intro frame.
    # IMG_5985: tricks at ~1.9-4.3 / 5.6-7.9 / 9.8-11.9s @ 30fps.
    idxs = [10, 90, 200, 330]
    if len(idxs) != num:
        idxs = np.linspace(0, total - 1, num, dtype=int).tolist()
    out: list[np.ndarray] = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(min(i, total - 1)))
        ok, fr = cap.read()
        if ok:
            out.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
    cap.release()
    return out


def load_img5985_scorecard() -> list[dict]:
    """Prefer Gemini Flash predictions on IMG_5985 single clips; fall back to GT."""
    gt_tricks = ["Gainer 360", "Kroc", "Double Cork"]
    gt_d = [2.8, 2.4, 3.3]
    card = [
        {"label": f"#{i+1}", "name": n, "d": d}
        for i, (n, d) in enumerate(zip(gt_tricks, gt_d))
    ]
    if not RESULTS.exists():
        return card

    rows = [json.loads(l) for l in RESULTS.read_text().splitlines() if l.strip()]
    # Use Gemini Flash predictions on IMG_5985 trick clips if available.
    wanted_stems = ["trick_01", "trick_02", "trick_04"]  # segments that are real tricks
    preds = {}
    for r in rows:
        if r["model"] != "gemini/gemini-2.5-flash":
            continue
        if "IMG_5985" not in r["clip_path"]:
            continue
        stem = Path(r["clip_path"]).stem
        preds[stem] = r

    if all(s in preds for s in wanted_stems):
        card = []
        for i, stem in enumerate(wanted_stems):
            r = preds[stem]
            fm = r.get("fig_match") or {}
            card.append({
                "label": f"#{i+1}",
                "name": fm.get("name") or r.get("parsed_trick") or "?",
                "d": float(fm.get("d_score") or 0),
            })
    return card


def main() -> None:
    frames = extract_key_frames(VIDEO)
    card = load_img5985_scorecard()
    total_d = sum(c["d"] for c in card)

    fig = plt.figure(figsize=(7.0, 2.3))
    gs = fig.add_gridspec(1, 5, width_ratios=[1, 1, 1, 1, 1.4], wspace=0.08)

    for i, frame in enumerate(frames[:4]):
        ax = fig.add_subplot(gs[0, i])
        # Center-crop to 4:3
        h, w = frame.shape[:2]
        target_w = int(h * 3 / 4)
        if target_w < w:
            x0 = (w - target_w) // 2
            frame = frame[:, x0:x0 + target_w]
        ax.imshow(frame)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor("#999")
            spine.set_linewidth(0.5)
        ax.set_title(f"$t_{{{i+1}}}$", fontsize=8, pad=2, color="#555")

    # Scorecard on the right
    ax = fig.add_subplot(gs[0, 4])
    ax.set_axis_off()
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    # Rounded box background
    box = FancyBboxPatch(
        (0.02, 0.04), 0.96, 0.92,
        boxstyle="round,pad=0.02,rounding_size=0.04",
        linewidth=1, edgecolor="#367fb0", facecolor="#f4f8fc",
    )
    ax.add_patch(box)

    ax.text(0.5, 0.90, "PkVision scorecard", ha="center", va="top",
            fontsize=9, color="#17456e", fontweight="bold")

    y = 0.75
    for c in card:
        ax.text(0.07, y, c["label"], ha="left", va="top", fontsize=8, color="#17456e")
        ax.text(0.19, y, c["name"], ha="left", va="top", fontsize=8, color="#000")
        ax.text(0.93, y, f"D={c['d']:.1f}", ha="right", va="top",
                fontsize=8, color="#000", family="monospace")
        y -= 0.17

    ax.plot([0.08, 0.92], [y + 0.05, y + 0.05], color="#367fb0", lw=0.8)
    ax.text(0.07, y - 0.02, "Total", ha="left", va="top",
            fontsize=8, color="#17456e", fontweight="bold")
    ax.text(0.93, y - 0.02, f"D={total_d:.1f}", ha="right", va="top",
            fontsize=8, color="#17456e", fontweight="bold",
            family="monospace")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    print(f"Wrote {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
