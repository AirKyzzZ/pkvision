#!/usr/bin/env python3
"""Render Figure 3: inversion-based segmentation timeline on IMG_5985.

Reads the real video through the production `core/video.py` +
`core/segmentation.py` pipeline and plots:

  - nose and hip Y coordinates (top panel)
  - combined inversion score s_t (middle panel)
  - detected trick segments shaded (bottom panel)

Output: paper/figures/fig03_segmentation.pdf
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from core.video import read_video, detect_and_track, smooth_boxes  # noqa: E402
from core.segmentation import segment_tricks  # noqa: E402

VIDEO = REPO / "data" / "run_testing" / "IMG_5985.mov"
OUT = REPO / "paper" / "figures" / "fig03_segmentation.pdf"


def main() -> None:
    from ultralytics import YOLO
    yolo = YOLO(str(REPO / "yolo11n-pose.pt")) if (REPO / "yolo11n-pose.pt").exists() else YOLO("yolo11n-pose.pt")

    print(f"Reading {VIDEO.name}...")
    frames, fps = read_video(VIDEO)
    print(f"  {len(frames)} frames @ {fps:.0f}fps")

    print("Running YOLO-pose...")
    tr = detect_and_track(frames, yolo)
    tr.boxes = smooth_boxes(tr.boxes)

    print("Segmenting...")
    segs = segment_tricks(tr.head_y, tr.hip_y, tr.body_angle, fps)
    print(f"  {len(segs)} segments: "
          + ", ".join(f"{s.start_frame/fps:.1f}-{s.end_frame/fps:.1f}s" for s in segs))

    T = len(frames)
    t = np.arange(T) / fps

    # Rebuild the inversion signal the same way segment_tricks does, for plotting.
    head_y = tr.head_y
    hip_y = tr.hip_y
    inv = np.zeros(T)
    for i in range(T):
        if not np.isnan(head_y[i]) and not np.isnan(hip_y[i]):
            inv[i] = max(0, head_y[i] - hip_y[i])
    if inv.max() > 0:
        pos = inv[inv > 0]
        if len(pos) > 0:
            inv = inv / np.percentile(pos, 90)
    k = max(int(fps * 0.1), 3)
    kernel = np.ones(k) / k
    inv_smooth = np.convolve(np.clip(inv, 0, 2), kernel, mode="same")

    ang = np.copy(tr.body_angle)
    valid = ~np.isnan(ang)
    if valid.sum() >= 2:
        idxs = np.where(valid)[0]
        ang = np.interp(np.arange(T), idxs, ang[idxs])
    else:
        ang = np.zeros(T)
    ang_vel = np.abs(np.diff(ang, prepend=ang[0]))
    ang_vel = np.where(ang_vel > np.pi, 2 * np.pi - ang_vel, ang_vel)
    ang_vel_smooth = np.convolve(ang_vel, kernel, mode="same")
    if ang_vel_smooth.max() > 0:
        pos = ang_vel_smooth[ang_vel_smooth > 0]
        if len(pos) > 0:
            ang_vel_norm = ang_vel_smooth / np.percentile(pos, 90)
        else:
            ang_vel_norm = ang_vel_smooth
    else:
        ang_vel_norm = ang_vel_smooth
    score = 0.7 * inv_smooth + 0.3 * ang_vel_norm
    score = np.convolve(score, kernel, mode="same")

    # --- Plot ---
    fig, axes = plt.subplots(
        3, 1, figsize=(7.0, 4.2), sharex=True,
        gridspec_kw={"height_ratios": [1.1, 1.0, 0.4]},
    )
    ax1, ax2, ax3 = axes

    ax1.plot(t, head_y, color="#e07a1f", label="nose $y$", lw=1.2)
    ax1.plot(t, hip_y,  color="#376fb0", label="hip $y$",  lw=1.2)
    ax1.invert_yaxis()   # image coords: down is positive
    ax1.set_ylabel("Image $y$ (px)")
    ax1.legend(frameon=False, fontsize=8, loc="upper right")
    ax1.grid(alpha=0.15)

    ax2.plot(t, score, color="#2e8b57", lw=1.3, label="combined score $s_t$")
    ax2.axhline(0.5, color="#888", lw=0.8, ls="--")
    ax2.set_ylabel("$s_t$ (a.u.)")
    ax2.set_ylim(0, max(1.2, score.max() * 1.05))
    ax2.legend(frameon=False, fontsize=8, loc="upper right")
    ax2.grid(alpha=0.15)

    # Segment shading on all three panels
    colors = ["#e57373", "#64b5f6", "#81c784", "#ba68c8", "#ffd54f"]
    for idx, seg in enumerate(segs):
        a = seg.start_frame / fps
        b = seg.end_frame / fps
        c = colors[idx % len(colors)]
        for ax in (ax1, ax2):
            ax.axvspan(a, b, color=c, alpha=0.15)
        ax3.add_patch(plt.Rectangle((a, 0.2), b - a, 0.6, color=c, alpha=0.7))
        ax3.text((a + b) / 2, 0.5, f"#{idx+1}",
                 ha="center", va="center", fontsize=8, color="#222")

    ax3.set_xlim(0, T / fps)
    ax3.set_ylim(0, 1)
    ax3.set_yticks([])
    ax3.set_xlabel("Time (s)")
    ax3.set_title("Detected trick segments", fontsize=9, loc="left", pad=2)
    ax3.grid(alpha=0.15, axis="x")

    for ax in (ax1, ax2, ax3):
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    fig.suptitle("Inversion-based segmentation on \\texttt{IMG\\_5985.mov}",
                 fontsize=10, y=0.995)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=160, bbox_inches="tight")
    print(f"Wrote {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
