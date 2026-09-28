#!/usr/bin/env python3
"""Render Figure 9: failure gallery --- three miscalled clips with diagnosis.

Picks the first 3 (clip, model) pairs in results.jsonl where the fig_match
disagrees with the ground truth, and composes a 3-panel figure showing:
  top:    4 uniformly-spaced frames from the clip
  middle: ground-truth FIG name, model prediction, match level
  bottom: one-line diagnosis
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

RESULTS = REPO / "paper" / "experiments" / "results.jsonl"
OUT = REPO / "paper" / "figures" / "fig09_failure_gallery.pdf"


def pick_failures(n: int = 3) -> list[dict]:
    if not RESULTS.exists():
        return []
    rows = [json.loads(l) for l in RESULTS.read_text().splitlines() if l.strip()]
    failures: list[dict] = []
    seen_clips: set[str] = set()
    for r in rows:
        if r.get("error"):
            continue
        fm = r.get("fig_match") or {}
        pred = fm.get("name") or r.get("parsed_trick") or ""
        gt = r["ground_truth"]["fig_name"]
        if pred.lower() != gt.lower() and r["clip_id"] not in seen_clips:
            failures.append(r)
            seen_clips.add(r["clip_id"])
            if len(failures) >= n:
                break
    return failures


def extract_frames(video: Path, n: int = 4) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []
    idxs = np.linspace(max(1, int(0.1 * total)), total - 2, n, dtype=int)
    out: list[np.ndarray] = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, fr = cap.read()
        if ok:
            out.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
    cap.release()
    return out


DIAGNOSIS_HEURISTICS = {
    ("Double Cork", "Frontflip"):
        "Twist-axis ambiguity: the VLM collapsed off-axis rotation to a plain front flip.",
    ("Double Cork", "Cork"):
        "Count error: the VLM saw cork-like rotation but missed the second flip.",
    ("Kroc", "Cork"):
        "Direction confusion: Kroc is a reverse cork; the matcher fell back to plain Cork.",
    ("Backflip 720", "Backflip 360"):
        "Twist under-counting by one full rotation.",
    ("Backflip 720", "Backflip"):
        "Twist entirely missed: the VLM reported a plain backflip.",
    ("Wall Inward Frontflip", "Frontflip"):
        "Wall context missed: a wall-contact inward flip was scored as a ground frontflip.",
    ("Gainer 360", "Gainer"):
        "Twist under-counting: the VLM saw the gainer but missed the 360.",
    ("Gainer 360", "Backflip"):
        "Takeoff confusion: one-foot running takeoff mis-read as two-foot standing.",
}


def diagnose(gt: str, pred: str) -> str:
    key = (gt, pred)
    if key in DIAGNOSIS_HEURISTICS:
        return DIAGNOSIS_HEURISTICS[key]
    return f"Category mismatch: {gt} $\\rightarrow$ {pred}."


def main() -> None:
    failures = pick_failures(3)
    if not failures:
        print("No failures in results.jsonl yet.")
        fig, ax = plt.subplots(figsize=(7.0, 2.0))
        ax.text(0.5, 0.5,
                "(No failures to show yet --- paper/experiments/results.jsonl is empty or all correct.)",
                ha="center", va="center", fontsize=10, color="#666")
        ax.set_axis_off()
        OUT.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(OUT, dpi=160, bbox_inches="tight")
        print(f"Wrote placeholder {OUT.relative_to(REPO)}")
        return

    n = len(failures)
    fig = plt.figure(figsize=(7.0, 1.1 * n + 1.3))
    gs = fig.add_gridspec(n, 5, wspace=0.05, hspace=0.35)

    for row_idx, r in enumerate(failures):
        clip = REPO / r["clip_path"]
        frames = extract_frames(clip, 4)
        for i, fr in enumerate(frames[:4]):
            ax = fig.add_subplot(gs[row_idx, i])
            h, w = fr.shape[:2]
            tw = int(h * 3 / 4)
            if tw < w:
                x0 = (w - tw) // 2
                fr = fr[:, x0:x0 + tw]
            ax.imshow(fr)
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_edgecolor("#aaa"); s.set_linewidth(0.5)
            if i == 0:
                ax.set_ylabel(f"#{row_idx+1}", fontsize=9, rotation=0,
                              labelpad=12, color="#c44")

        text_ax = fig.add_subplot(gs[row_idx, 4])
        text_ax.set_axis_off()
        text_ax.set_xlim(0, 1); text_ax.set_ylim(0, 1)
        gt = r["ground_truth"]["fig_name"]
        fm = r.get("fig_match") or {}
        pred = fm.get("name") or r.get("parsed_trick") or "?"
        level = fm.get("level", "---")
        model = r["model"].split("/")[-1]
        text_ax.text(0.0, 0.90, f"model: {model}", fontsize=7.5, color="#666")
        text_ax.text(0.0, 0.72, f"GT:   {gt}", fontsize=8.5, color="#17456e", fontweight="bold")
        text_ax.text(0.0, 0.54, f"pred: {pred}", fontsize=8.5, color="#c44", fontweight="bold")
        text_ax.text(0.0, 0.36, f"match: {level}", fontsize=7.5, color="#666")
        text_ax.text(0.0, 0.16, diagnose(gt, pred), fontsize=7.0, color="#333",
                     wrap=True)

    fig.suptitle("Failure gallery: three mis-called tricks",
                 fontsize=10, y=0.995)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=160, bbox_inches="tight")
    print(f"Wrote {OUT.relative_to(REPO)} ({n} failures shown)")


if __name__ == "__main__":
    main()
