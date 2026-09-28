#!/usr/bin/env python3
"""P1 apex-confidence audit: RTMPose vs YOLO11x-pose head-to-head.

A' spec section 6, first sub-gate. Runs both pose estimators on the 25-clip
hard set and measures keypoint confidence through the apex of inversion --
the moment fast rotation is most likely to break 2D pose estimation.

If even the heavy estimator's confidence collapses at the apex, that is a
decisive negative finding for the skeleton-cue approach, surfaced before any
model is trained.

Output: data/keypoints/pose_audit_report.json + a printed comparison table.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import uniform_filter1d

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DEFAULT_CLIPS_DIR = ROOT / "data" / "parkourtheory_clips_cropped"
DEFAULT_REPORT_PATH = ROOT / "data" / "keypoints" / "pose_audit_report.json"

# COCO-17. Extremities (wrists 9-10, ankles 15-16) are the fast-moving joints
# most prone to collapse during rotation.
COCO_NAMES = [
    "nose", "l_eye", "r_eye", "l_ear", "r_ear",
    "l_shoulder", "r_shoulder", "l_elbow", "r_elbow",
    "l_wrist", "r_wrist", "l_hip", "r_hip",
    "l_knee", "r_knee", "l_ankle", "r_ankle",
]
EXTREMITIES = [9, 10, 15, 16]
APEX_WINDOW_S = 0.15
CONF_VALID = 0.3

AUDIT_CLIPS = [
    # triple twist -- fastest rotation
    "corkscrew_counter_triple_full", "flyaway_triple_full",
    "gainer_triple_full", "running_gainer_triple_flash_kick",
    # double flip -- sustained inversion
    "double_back_flip", "castaway_double_back", "caster_wall_double_back",
    "hang_castaway_double_back", "swing_castaway_double_back",
    # double full -- flip + twist together
    "back_double_full", "flyaway_double_full", "kong_gainer_double_full",
    "ew_gainer_double_full", "castaway_double_full", "elbow_flyaway_double_full",
    # compound sequence -- segmentation + apex stress
    "flyaway_full_in_double_back_out", "flyaway_in_double_full_out",
    "corkscrew_arabian_1_12_dive_roll", "dive_half_back_unwind_double_full_down",
    # off-axis corkscrew -- spin axis near camera axis
    "cork_zero", "double_corkscrew", "corkscrew_counter_arabian",
    "trapdoor_wall_corkscrew",
    # high-rotation / wall
    "1080_dive_roll", "one_hand_tsukahara_double_full_dismount",
]


@dataclass
class EstimatorResult:
    estimator: str
    n_frames: int
    valid_frame_ratio: float       # frames with >=10 joints above CONF_VALID
    mean_conf_all: float
    mean_conf_apex: float
    apex_drop: float               # mean_conf_all - mean_conf_apex
    apex_frame: int | None
    extremity_conf_apex: float     # mean conf of wrists+ankles at apex
    per_joint_apex: list[float]


def _largest_person(keypoints: np.ndarray, scores: np.ndarray) -> int:
    """Index of the person with the widest keypoint spread (the athlete)."""
    spans = []
    for p in range(len(keypoints)):
        valid = scores[p] > CONF_VALID
        if valid.sum() < 4:
            spans.append(0.0)
            continue
        pts = keypoints[p][valid]
        spans.append(float(np.ptp(pts[:, 0]) * np.ptp(pts[:, 1])))
    return int(np.argmax(spans)) if spans else 0


def run_yolo(video_path: Path, model) -> tuple[np.ndarray, np.ndarray, float]:
    """YOLO11x-pose -> (kp (T,17,2), conf (T,17), fps)."""
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    kps, confs = [], []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        res = model(frame, conf=0.25, verbose=False)
        kp = np.zeros((17, 2))
        cf = np.zeros(17)
        if res and res[0].keypoints is not None and len(res[0].keypoints) > 0:
            kp_data = res[0].keypoints.xy.cpu().numpy()
            cf_data = res[0].keypoints.conf
            cf_data = cf_data.cpu().numpy() if cf_data is not None else None
            if cf_data is not None and len(kp_data) > 0:
                best = _largest_person(kp_data, cf_data)
                if kp_data[best].shape[0] >= 17:
                    kp = kp_data[best][:17]
                    cf = cf_data[best][:17]
        kps.append(kp)
        confs.append(cf)
    cap.release()
    return np.array(kps), np.array(confs), fps


def run_rtmpose(video_path: Path, body) -> tuple[np.ndarray, np.ndarray, float]:
    """RTMPose (rtmlib, performance tier) -> (kp (T,17,2), conf (T,17), fps)."""
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    kps, confs = [], []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        kp = np.zeros((17, 2))
        cf = np.zeros(17)
        keypoints, scores = body(frame)
        if keypoints is not None and len(keypoints) > 0:
            best = _largest_person(keypoints, scores)
            if keypoints[best].shape[0] >= 17:
                kp = keypoints[best][:17]
                cf = scores[best][:17]
        kps.append(kp)
        confs.append(cf)
    cap.release()
    return np.array(kps), np.array(confs), fps


def find_apex(kp: np.ndarray, conf: np.ndarray) -> int | None:
    """Apex frame = peak inversion (head below hip midpoint, image Y-down)."""
    head_valid = conf[:, 0] > CONF_VALID
    hip_valid = (conf[:, 11] > CONF_VALID) & (conf[:, 12] > CONF_VALID)
    both = head_valid & hip_valid
    if both.sum() < 4:
        return None
    head_y = kp[:, 0, 1]
    hip_y = (kp[:, 11, 1] + kp[:, 12, 1]) / 2
    signal = np.where(both, head_y - hip_y, np.nan)
    idx = np.where(both)[0]
    signal = np.interp(np.arange(len(signal)), idx, signal[idx])
    signal = uniform_filter1d(signal, size=5)
    return int(np.argmax(signal))


def audit(estimator: str, kp: np.ndarray, conf: np.ndarray, fps: float) -> EstimatorResult:
    T = len(kp)
    valid_per_frame = (conf > CONF_VALID).sum(axis=1)
    valid_frame_ratio = float((valid_per_frame >= 10).mean())
    mean_conf_all = float(conf.mean())

    apex = find_apex(kp, conf)
    if apex is None:
        return EstimatorResult(
            estimator, T, valid_frame_ratio, mean_conf_all,
            0.0, mean_conf_all, None, 0.0, [0.0] * 17,
        )

    half = max(1, int(fps * APEX_WINDOW_S))
    lo, hi = max(0, apex - half), min(T, apex + half + 1)
    apex_conf = conf[lo:hi]
    mean_conf_apex = float(apex_conf.mean())
    per_joint_apex = [float(apex_conf[:, j].mean()) for j in range(17)]
    extremity_conf_apex = float(apex_conf[:, EXTREMITIES].mean())

    return EstimatorResult(
        estimator, T, valid_frame_ratio, mean_conf_all,
        mean_conf_apex, mean_conf_all - mean_conf_apex,
        apex, extremity_conf_apex, per_joint_apex,
    )


def _resolve_device(requested: str) -> str:
    """On Windows, expose torch's bundled CUDA/cuDNN DLLs to onnxruntime-gpu.

    onnxruntime-gpu only registers CUDAExecutionProvider if it can load the
    CUDA runtime; torch's cu12x build ships compatible DLLs in torch/lib.
    """
    if requested == "cpu":
        return "cpu"
    if sys.platform == "win32":
        try:
            import torch
            torch_lib = Path(torch.__file__).parent / "lib"
            if torch_lib.is_dir():
                os.add_dll_directory(str(torch_lib))
        except Exception as e:
            print(f"[warn] could not expose torch CUDA DLLs: {e}")
    import onnxruntime as ort
    providers = ort.get_available_providers()
    print(f"onnxruntime providers: {providers}")
    if "CUDAExecutionProvider" not in providers:
        print("[warn] CUDAExecutionProvider unavailable -> RTMPose runs on CPU")
        return "cpu"
    return "cuda"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clips-dir", type=Path, default=DEFAULT_CLIPS_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = parser.parse_args()

    device = _resolve_device(args.device)

    from ultralytics import YOLO
    from rtmlib import Body

    print(f"Loading estimators (rtmpose device={device})...", flush=True)
    yolo = YOLO("yolo11x-pose.pt")
    body = Body(mode="performance", backend="onnxruntime", device=device)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict] = {}

    for i, name in enumerate(AUDIT_CLIPS, 1):
        path = args.clips_dir / f"{name}.mp4"
        if not path.exists():
            print(f"[{i}/{len(AUDIT_CLIPS)}] MISSING {name}")
            continue
        t0 = time.time()
        y_kp, y_cf, fps = run_yolo(path, yolo)
        r_kp, r_cf, _ = run_rtmpose(path, body)
        y_res = audit("yolo11x", y_kp, y_cf, fps)
        r_res = audit("rtmpose", r_kp, r_cf, fps)
        results[name] = {"yolo11x": asdict(y_res), "rtmpose": asdict(r_res)}
        print(
            f"[{i}/{len(AUDIT_CLIPS)}] {name:<42} "
            f"yolo apex={y_res.mean_conf_apex:.2f}(drop {y_res.apex_drop:+.2f}) | "
            f"rtm apex={r_res.mean_conf_apex:.2f}(drop {r_res.apex_drop:+.2f}) "
            f"[{time.time() - t0:.1f}s]"
        )

    summary = _summarize(results)
    args.out.write_text(json.dumps({"summary": summary, "clips": results}, indent=2))
    print(f"\nReport written to {args.out}")
    _print_summary(summary)


def _summarize(results: dict) -> dict:
    out = {}
    for est in ("yolo11x", "rtmpose"):
        rows = [r[est] for r in results.values()]
        out[est] = {
            "mean_conf_apex": round(float(np.mean([r["mean_conf_apex"] for r in rows])), 4),
            "mean_apex_drop": round(float(np.mean([r["apex_drop"] for r in rows])), 4),
            "mean_extremity_conf_apex": round(
                float(np.mean([r["extremity_conf_apex"] for r in rows])), 4),
            "mean_valid_frame_ratio": round(
                float(np.mean([r["valid_frame_ratio"] for r in rows])), 4),
            "clips_apex_below_0.4": sum(r["mean_conf_apex"] < 0.4 for r in rows),
        }
    return out


def _print_summary(summary: dict) -> None:
    print("\n=== APEX-CONFIDENCE AUDIT SUMMARY ===")
    for est, s in summary.items():
        print(f"\n{est}:")
        for k, v in s.items():
            print(f"  {k:<28} {v}")
    win = ("rtmpose" if summary["rtmpose"]["mean_conf_apex"]
           > summary["yolo11x"]["mean_conf_apex"] else "yolo11x")
    print(f"\nHigher apex confidence: {win}")


if __name__ == "__main__":
    main()
