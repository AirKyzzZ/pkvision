#!/usr/bin/env python3
"""Diagnostic: can a skeleton model predict the FIG decoder cues?

Joins the RTMPose-x skeleton shards (data/keypoints/parkourtheory_pose) with the
AUTOMATIC attribute labels (data/v5_attribute_training/attribute_manifest.json,
derived from trick names — zero manual labelling) by clip slug, trains a small
temporal Transformer with one head per cue, and reports TRAIN vs VAL accuracy vs
the majority-class baseline.

Reading the result:
  - val >> baseline            -> the cue is learnable from skeletons.
  - train >> val ~= baseline   -> OVERFIT: signal may exist but model/labels/size
                                  don't generalise (SSL pretrain is the remedy).
  - train ~= val ~= baseline   -> labels don't correlate with skeletons (label
                                  noise) OR the signal isn't in one 2D view.
The label distribution + spot-checks below test the auto-labels directly.

Runs locally on CPU/MPS. No GPU spend.
"""
from __future__ import annotations

import argparse
import glob
import json
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
POSE_GLOB = str(ROOT / "data" / "keypoints" / "**" / "shard_*.npz")
MANIFEST = ROOT / "data" / "v5_attribute_training" / "attribute_manifest.json"

CUES = ("context", "direction", "flip", "twist")


def load_skeletons() -> dict[str, np.ndarray]:
    skel: dict[str, np.ndarray] = {}
    shards = sorted(glob.glob(POSE_GLOB, recursive=True))
    for shard in shards:
        with np.load(shard) as z:
            for slug in z.files:
                skel[slug] = z[slug].astype(np.float32)  # (T,17,3)
    print(f"skeletons: {len(skel)} clips from {len(shards)} shards")
    return skel


def _flip_bin(v) -> int:
    return min(int(round(float(v))), 3)  # {0,1,2,3+}


def _twist_bin(v) -> int:
    v = float(v)
    if v < 0.25:
        return 0
    if v < 0.75:
        return 1  # 0.5
    if v < 1.25:
        return 2  # 1.0
    return 3  # >=1.5 (single-camera-unrecoverable bucket)


def load_labels() -> dict[str, dict]:
    clips = json.loads(MANIFEST.read_text())["clips"]
    out: dict[str, dict] = {}
    for c in clips:
        out[c["slug"]] = {
            "context": str(c.get("context") or "none"),
            "direction": str(c.get("direction") or "none"),
            "flip": _flip_bin(c.get("flip", 0)),
            "twist": _twist_bin(c.get("twist", 0)),
            "_flip_raw": c.get("flip"),
            "_twist_raw": c.get("twist"),
        }
    print(f"labels: {len(out)} clips")
    return out


def normalize(arr: np.ndarray, t_out: int) -> np.ndarray:
    """(T,17,3) -> (t_out, 51): hip-centred, torso-scaled xy + confidence."""
    if arr.shape[0] == 0:
        return np.zeros((t_out, 51), np.float32)
    xy = arr[:, :, :2].astype(np.float32).copy()
    conf = arr[:, :, 2:3].astype(np.float32)
    both_hips = (arr[:, 11, 2] > 0.3) & (arr[:, 12, 2] > 0.3)
    hip_mid = (xy[:, 11] + xy[:, 12]) / 2.0
    center = hip_mid[both_hips].mean(0) if both_hips.any() else xy.reshape(-1, 2).mean(0)
    xy -= center[None, None, :]
    shoulder_mid = (xy[:, 5] + xy[:, 6]) / 2.0
    torso = np.linalg.norm(shoulder_mid - (xy[:, 11] + xy[:, 12]) / 2.0, axis=1)
    scale = np.median(torso[torso > 1e-3]) if (torso > 1e-3).any() else 1.0
    xy /= scale + 1e-6
    feat = np.concatenate([xy, conf], axis=2).reshape(arr.shape[0], 51)
    idx = np.linspace(0, arr.shape[0] - 1, t_out).astype(int)
    return feat[idx].astype(np.float32)


def label_audit(slugs, labels, classes):
    print("\n=== label distributions (full join) ===")
    for c in CUES:
        ctr = Counter(labels[s][c] for s in slugs)
        dist = {classes[c][i] if c in ("context", "direction") else i: n
                for i, n in sorted(((classes[c].index(k) if c in ("context", "direction") else k, v)
                                     for k, v in ctr.items()))}
        print(f"  {c:<10} {dict(sorted(ctr.items()))}")
    print("\n=== spot-check auto-labels on recognisable slugs ===")
    tokens = ("front", "back", "side", "double", "triple", "full", "360", "540", "gainer", "cork")
    shown = 0
    for s in slugs:
        if shown >= 14:
            break
        if any(t in s for t in tokens):
            L = labels[s]
            print(f"  {s[:36]:<36} ctx={L['context']:<10} dir={L['direction']:<8} "
                  f"flip={L['flip']}(raw {L['_flip_raw']}) twist={L['twist']}(raw {L['_twist_raw']})")
            shown += 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--t-out", type=int, default=48)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = (
        "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"device={device}")

    skel = load_skeletons()
    labels = load_labels()
    slugs = sorted(set(skel) & set(labels))
    print(f"joined: {len(slugs)} clips with both skeleton + labels")

    classes = {
        "context": sorted({labels[s]["context"] for s in slugs}),
        "direction": sorted({labels[s]["direction"] for s in slugs}),
        "flip": [0, 1, 2, 3],
        "twist": [0, 1, 2, 3],
    }
    idx_of = {c: {v: i for i, v in enumerate(classes[c])} for c in CUES}

    label_audit(slugs, labels, classes)

    X = np.stack([normalize(skel[s], args.t_out) for s in slugs])
    Y = {c: np.array([idx_of[c][labels[s][c]] for s in slugs]) for c in CUES}

    n = len(slugs)
    perm = np.random.permutation(n)
    n_val = max(1, int(0.2 * n))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    print(f"\ntrain={len(tr_idx)} val={len(val_idx)}")

    Xt = torch.tensor(X)
    Yt = {c: torch.tensor(Y[c], dtype=torch.long) for c in CUES}
    tr = TensorDataset(Xt[tr_idx], *[Yt[c][tr_idx] for c in CUES])
    va = TensorDataset(Xt[val_idx], *[Yt[c][val_idx] for c in CUES])
    tr_dl = DataLoader(tr, batch_size=args.batch, shuffle=True)
    tr_eval_dl = DataLoader(tr, batch_size=256)
    va_dl = DataLoader(va, batch_size=256)

    class CueNet(nn.Module):
        def __init__(self, d=96, nhead=4, layers=3, t=args.t_out):
            super().__init__()
            self.proj = nn.Linear(51, d)
            self.pos = nn.Parameter(torch.randn(1, t, d) * 0.02)
            enc = nn.TransformerEncoderLayer(d, nhead, 2 * d, dropout=0.1, batch_first=True)
            self.enc = nn.TransformerEncoder(enc, layers)
            self.heads = nn.ModuleDict({c: nn.Linear(d, len(classes[c])) for c in CUES})

        def forward(self, x):
            h = self.enc(self.proj(x) + self.pos).mean(1)
            return {c: head(h) for c, head in self.heads.items()}

    model = CueNet().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss()

    def evaluate(dl) -> dict[str, float]:
        model.eval()
        hits = {c: 0 for c in CUES}
        tot = 0
        with torch.no_grad():
            for xb, *yb in dl:
                out = model(xb.to(device))
                tot += xb.shape[0]
                for c, y in zip(CUES, yb):
                    hits[c] += (out[c].argmax(1).cpu() == y).sum().item()
        return {c: hits[c] / tot for c in CUES}

    best_val = {c: 0.0 for c in CUES}
    for ep in range(args.epochs):
        model.train()
        for xb, *yb in tr_dl:
            xb = xb.to(device)
            out = model(xb)
            loss = sum(lossf(out[c], y.to(device)) for c, y in zip(CUES, yb))
            opt.zero_grad()
            loss.backward()
            opt.step()
        acc = evaluate(va_dl)
        for c in CUES:
            best_val[c] = max(best_val[c], acc[c])
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"ep{ep+1:>3} val " + " ".join(f"{c}={acc[c]:.3f}" for c in CUES))

    tr_acc = evaluate(tr_eval_dl)
    base = {c: np.bincount(Y[c][val_idx], minlength=len(classes[c])).max() / len(val_idx) for c in CUES}

    print("\n=== DIAGNOSTIC: skeleton -> cue (TRAIN can-fit vs VAL generalise) ===")
    print(f"{'cue':<10}{'baseline':<10}{'train':<9}{'val_best':<10}{'val-base':<9}{'verdict'}")
    for c in CUES:
        vb = best_val[c] - base[c]
        if tr_acc[c] - base[c] < 0.08:
            verdict = "labels/signal absent (train can't fit)"
        elif vb < 0.05:
            verdict = "OVERFIT (fits train, not val)"
        else:
            verdict = "LEARNABLE"
        print(f"{c:<10}{base[c]:<10.3f}{tr_acc[c]:<9.3f}{best_val[c]:<10.3f}{vb:+.3f}    {verdict}")


if __name__ == "__main__":
    main()
