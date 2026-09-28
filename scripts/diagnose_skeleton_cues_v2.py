#!/usr/bin/env python3
"""Skeleton -> cue diagnostic v2: every cheap generalization lever stacked.

v1 showed the model fits train (94%+) but only `context` generalizes — classic
overfit on small/imbalanced/auto-labelled data. v2 adds, all at once:
  * motion features (joint velocity + torso-angle sin/cos + angular velocity)
  * on-the-fly skeleton augmentation (mirror / rotate / scale / jitter / t-crop)
  * class-weighted loss + macro-F1 (minority-class learning becomes visible)
  * multi-task aux heads (axis/entry/body_shape) regularizing a shared encoder
  * heavier dropout/weight-decay + early stopping on best val macro-F1

Honest read stays the same: val macro-F1 >> baseline-F1 == the cue is learnable.
Local CPU/MPS, no GPU spend. The remaining heavyweight lever (GFP self-supervised
pretraining) is deliberately NOT here — measure the cheap levers first.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)  # so FIGDecoder's relative data paths (data/fig_tricks_2025.json) resolve
POSE_GLOB = str(ROOT / "data" / "keypoints" / "**" / "shard_*.npz")
MANIFEST = ROOT / "data" / "v5_attribute_training" / "attribute_manifest.json"

CORE = ("context", "direction", "flip", "twist", "axis")
AUX = ("entry", "body_shape")
ALL_CUES = CORE + AUX
LR_SWAP = np.array([0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15])  # COCO-17 mirror


def load_skeletons() -> dict[str, np.ndarray]:
    skel = {}
    shards = sorted(glob.glob(POSE_GLOB, recursive=True))
    for shard in shards:
        with np.load(shard) as z:
            for s in z.files:
                skel[s] = z[s].astype(np.float32)
    print(f"skeletons: {len(skel)} clips / {len(shards)} shards")
    return skel


def _flip_bin(v) -> int:
    return min(int(round(float(v))), 3)


def _twist_bin(v) -> int:
    v = float(v)
    return 0 if v < 0.25 else 1 if v < 0.75 else 2 if v < 1.25 else 3


def load_labels() -> dict[str, dict]:
    clips = json.loads(MANIFEST.read_text())["clips"]
    out = {}
    for c in clips:
        out[c["slug"]] = {
            "context": str(c.get("context") or "none"),
            "direction": str(c.get("direction") or "none"),
            "flip": _flip_bin(c.get("flip", 0)),
            "twist": _twist_bin(c.get("twist", 0)),
            "axis": str(c["rotation_axis"]) if c.get("rotation_axis") else None,
            "entry": str(c.get("entry") or "unknown"),
            "body_shape": str(c.get("body_shape") or "unknown"),
            "source": c.get("source", "unified"),
            "fig_name": c.get("fig_name"),
        }
    print(f"labels: {len(out)} clips")
    return out


def base_seq(arr: np.ndarray, t: int) -> np.ndarray:
    """(T,17,3) -> (t,17,3): hip-centred, torso-scaled, resampled. Keeps conf."""
    if arr.shape[0] == 0:
        return np.zeros((t, 17, 3), np.float32)
    xy = arr[:, :, :2].astype(np.float32).copy()
    conf = arr[:, :, 2:3].astype(np.float32)
    hips_ok = (arr[:, 11, 2] > 0.3) & (arr[:, 12, 2] > 0.3)
    hip = (xy[:, 11] + xy[:, 12]) / 2.0
    center = hip[hips_ok].mean(0) if hips_ok.any() else xy.reshape(-1, 2).mean(0)
    xy -= center[None, None, :]
    sh = (xy[:, 5] + xy[:, 6]) / 2.0
    torso = np.linalg.norm(sh - (xy[:, 11] + xy[:, 12]) / 2.0, axis=1)
    scale = np.median(torso[torso > 1e-3]) if (torso > 1e-3).any() else 1.0
    xy /= scale + 1e-6
    seq = np.concatenate([xy, conf], axis=2)
    idx = np.linspace(0, arr.shape[0] - 1, t).astype(int)
    return seq[idx].astype(np.float32)


def augment(seq: np.ndarray) -> np.ndarray:
    t = seq.shape[0]
    xy = seq[:, :, :2].copy()
    conf = seq[:, :, 2:3].copy()
    if np.random.rand() < 0.5:  # horizontal mirror (label-safe)
        xy = xy[:, LR_SWAP, :]
        conf = conf[:, LR_SWAP, :]
        xy[:, :, 0] *= -1.0
    th = np.random.uniform(-0.26, 0.26)  # +-15 deg
    rot = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]], np.float32)
    xy = xy @ rot.T
    xy *= np.random.uniform(0.9, 1.1)
    xy += np.random.normal(0, 0.02, xy.shape).astype(np.float32)
    if np.random.rand() < 0.5:  # temporal crop -> resample
        a = np.random.randint(0, max(1, int(t * 0.15)))
        b = t - np.random.randint(0, max(1, int(t * 0.15)))
        idx = np.linspace(a, b - 1, t).astype(int)
        xy, conf = xy[idx], conf[idx]
    return np.concatenate([xy, conf], axis=2)


def featurize(seq: np.ndarray) -> np.ndarray:
    """(t,17,3) -> (t,88): pos + conf + velocity + torso angle (sin,cos,angvel)."""
    t = seq.shape[0]
    xy = seq[:, :, :2]
    conf = seq[:, :, 2]
    vel = np.zeros_like(xy)
    vel[1:] = xy[1:] - xy[:-1]
    sh = (xy[:, 5] + xy[:, 6]) / 2.0
    hp = (xy[:, 11] + xy[:, 12]) / 2.0
    ang = np.arctan2((sh - hp)[:, 1], (sh - hp)[:, 0])
    angvel = np.zeros(t, np.float32)
    angvel[1:] = np.diff(np.unwrap(ang))
    return np.concatenate([
        xy.reshape(t, 34), conf.reshape(t, 17), vel.reshape(t, 34),
        np.sin(ang)[:, None], np.cos(ang)[:, None], angvel[:, None],
    ], axis=1).astype(np.float32)


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    f1s = []
    for k in np.unique(y_true):
        tp = int(((y_pred == k) & (y_true == k)).sum())
        fp = int(((y_pred == k) & (y_true != k)).sum())
        fn = int(((y_pred != k) & (y_true == k)).sum())
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * p * r / (p + r) if p + r else 0.0)
    return float(np.mean(f1s)) if f1s else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=70)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--t", type=int, default=48)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
              else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    skel = load_skeletons()
    labels = load_labels()
    slugs = sorted(set(skel) & set(labels))
    print(f"joined: {len(slugs)}")

    classes = {c: sorted({labels[s][c] for s in slugs if labels[s][c] is not None})
               for c in ALL_CUES}
    classes["flip"] = [0, 1, 2, 3]
    classes["twist"] = [0, 1, 2, 3]
    idx_of = {c: {v: i for i, v in enumerate(classes[c])} for c in ALL_CUES}

    def enc(s, c):
        v = labels[s][c]
        return idx_of[c][v] if v is not None and v in idx_of[c] else -1

    bases = {s: base_seq(skel[s], args.t) for s in slugs}
    Y = {c: np.array([enc(s, c) for s in slugs]) for c in ALL_CUES}

    n = len(slugs)
    perm = np.random.permutation(n)
    n_val = int(0.2 * n)
    val_i, tr_i = perm[:n_val], perm[n_val:]
    tr_slugs = [slugs[i] for i in tr_i]
    val_slugs = [slugs[i] for i in val_i]
    print(f"train={len(tr_i)} val={len(val_i)}")

    class DS(Dataset):
        def __init__(self, ss, train):
            self.ss, self.train = ss, train

        def __len__(self):
            return len(self.ss)

        def __getitem__(self, i):
            s = self.ss[i]
            seq = augment(bases[s]) if self.train else bases[s]
            y = [labels[s][c] for c in ALL_CUES]
            y = [idx_of[c][labels[s][c]] if labels[s][c] is not None and labels[s][c] in idx_of[c]
                 else -1 for c in ALL_CUES]
            return featurize(seq), np.array(y, np.int64)

    tr_dl = DataLoader(DS(tr_slugs, True), batch_size=args.batch, shuffle=True)
    tr_eval = DataLoader(DS(tr_slugs, False), batch_size=256)
    va_dl = DataLoader(DS(val_slugs, False), batch_size=256)

    # inverse-frequency class weights (on train), per cue
    cw = {}
    for ci, c in enumerate(ALL_CUES):
        yt = Y[c][tr_i]
        yt = yt[yt >= 0]
        cnt = np.bincount(yt, minlength=len(classes[c])).astype(np.float64)
        w = (cnt.sum() / (len(cnt) * np.maximum(cnt, 1)))
        cw[c] = torch.tensor(w, dtype=torch.float32, device=device)

    class Net(nn.Module):
        def __init__(self, d=128, t=args.t):
            super().__init__()
            self.proj = nn.Linear(88, d)
            self.pos = nn.Parameter(torch.randn(1, t, d) * 0.02)
            enc_l = nn.TransformerEncoderLayer(d, 4, 2 * d, dropout=0.3, batch_first=True)
            self.enc = nn.TransformerEncoder(enc_l, 3)
            self.drop = nn.Dropout(0.3)
            self.heads = nn.ModuleDict({c: nn.Linear(d, len(classes[c])) for c in ALL_CUES})

        def forward(self, x):
            h = self.drop(self.enc(self.proj(x) + self.pos).mean(1))
            return {c: self.heads[c](h) for c in ALL_CUES}

    model = Net().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=4e-4, weight_decay=5e-2)
    losses = {c: nn.CrossEntropyLoss(weight=cw[c], ignore_index=-1) for c in ALL_CUES}
    w_cue = {**{c: 1.0 for c in CORE}, **{c: 0.3 for c in AUX}}

    def collect(dl):
        model.eval()
        P = {c: [] for c in ALL_CUES}
        T = {c: [] for c in ALL_CUES}
        with torch.no_grad():
            for xb, yb in dl:
                out = model(xb.to(device))
                for ci, c in enumerate(ALL_CUES):
                    P[c].append(out[c].argmax(1).cpu().numpy())
                    T[c].append(yb[:, ci].numpy())
        return ({c: np.concatenate(P[c]) for c in ALL_CUES},
                {c: np.concatenate(T[c]) for c in ALL_CUES})

    def metrics(P, T):
        m = {}
        for c in ALL_CUES:
            mask = T[c] >= 0
            yt, yp = T[c][mask], P[c][mask]
            acc = (yt == yp).mean() if len(yt) else 0.0
            m[c] = (acc, macro_f1(yt, yp))
        return m

    best, best_state, wait = -1.0, None, 0
    for ep in range(args.epochs):
        model.train()
        for xb, yb in tr_dl:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            loss = sum(w_cue[c] * losses[c](out[c], yb[:, ci]) for ci, c in enumerate(ALL_CUES))
            opt.zero_grad()
            loss.backward()
            opt.step()
        vm = metrics(*collect(va_dl))
        score = float(np.mean([vm[c][1] for c in CORE]))  # mean core macro-F1
        if score > best:
            best, best_state, wait = score, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            wait += 1
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"ep{ep+1:>3} core-mF1={score:.3f} " + " ".join(f"{c}={vm[c][1]:.2f}" for c in CORE))
        if wait >= args.patience:
            print(f"early stop @ ep{ep+1}")
            break

    model.load_state_dict(best_state)
    trm = metrics(*collect(tr_eval))
    vam = metrics(*collect(va_dl))

    def base_f1(c, split_i):
        yt = Y[c][split_i]
        yt = yt[yt >= 0]
        maj = np.bincount(yt).argmax()
        return macro_f1(yt, np.full_like(yt, maj))

    print("\n=== v2 DIAGNOSTIC (macro-F1; beats baseline-F1 == learnable) ===")
    print(f"{'cue':<10}{'baseF1':<9}{'trainF1':<9}{'valF1':<9}{'valAcc':<9}{'lift':<8}")
    for c in CORE:
        bf = base_f1(c, val_i)
        lift = vam[c][1] - bf
        flag = "  <-- learns" if lift > 0.05 else ""
        print(f"{c:<10}{bf:<9.3f}{trm[c][1]:<9.3f}{vam[c][1]:<9.3f}{vam[c][0]:<9.3f}{lift:+.3f}{flag}")
    print("aux:", " ".join(f"{c}=val_mF1 {vam[c][1]:.2f}" for c in AUX))

    # ---- end-to-end: predicted cues -> frozen FIGDecoder -> trick top-1 ----
    from core.recognition.fig_decoder import FIGDecoder
    from core.recognition.oracle_cues import _norm
    decoder = FIGDecoder()
    inv = {c: {i: v for v, i in idx_of[c].items()} for c in ALL_CUES}
    tw_num = {0: 0.0, 1: 0.5, 2: 1.0}  # bin 3 (>=1.5) -> abstain per cue contract

    def cue_dict(ctx, dr, fb, tb, ax, en, bs):
        d = {"flip": float(fb)}
        if ctx:
            d["context"] = ctx
        if dr and dr != "none":
            d["direction"] = dr
        if tb in tw_num:
            d["twist"] = tw_num[tb]
        if ax and ax != "none":
            d["axis"] = ax
        if en and en != "unknown":
            d["entry"] = en
        if bs and bs != "unknown":
            d["body_shape"] = bs
        return d

    def top1(cues):
        cands = decoder.rank(dict(cues), k=1)
        return getattr(cands[0], "fig_name", None) if cands else None

    model.eval()
    pidx = {c: [] for c in ALL_CUES}
    with torch.no_grad():
        for xb, _yb in va_dl:
            out = model(xb.to(device))
            for c in ALL_CUES:
                pidx[c].append(out[c].argmax(1).cpu().numpy())
    pidx = {c: np.concatenate(pidx[c]) for c in ALL_CUES}  # aligned to val_slugs (no shuffle)

    a1 = fig_hit = fig_oracle = fig_n = 0
    for j, s in enumerate(val_slugs):
        p = {c: inv[c].get(int(pidx[c][j])) for c in ALL_CUES}
        L = labels[s]
        pred_t = top1(cue_dict(p["context"], p["direction"], p["flip"], p["twist"], p["axis"], p["entry"], p["body_shape"]))
        true_t = top1(cue_dict(L["context"], L["direction"], L["flip"], L["twist"], L["axis"], L["entry"], L["body_shape"]))
        if pred_t and true_t and _norm(pred_t) == _norm(true_t):
            a1 += 1
        if L["source"] == "fig" and L["fig_name"]:
            fig_n += 1
            if pred_t and _norm(pred_t) == _norm(L["fig_name"]):
                fig_hit += 1
            if true_t and _norm(true_t) == _norm(L["fig_name"]):
                fig_oracle += 1
    nv = len(val_slugs)
    print("\n=== END-TO-END: predicted cues -> frozen FIGDecoder (P0-validated) ===")
    print(f"cue-preservation top-1 (model pick == oracle-cue pick): {a1}/{nv} = {a1/nv:.1%}")
    if fig_n:
        print(f"absolute top-1 on FIG-gold val (n={fig_n}): model {fig_hit}/{fig_n}={fig_hit/fig_n:.0%}  "
              f"| oracle-cue ceiling {fig_oracle}/{fig_n}={fig_oracle/fig_n:.0%}")
    print("(73% = oracle top-1 on the P0 blind set: the ceiling for perfect cues)")


if __name__ == "__main__":
    main()
