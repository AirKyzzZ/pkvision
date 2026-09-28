#!/usr/bin/env python3
"""Direct skeleton -> D-score regressor vs the cue-pipeline (head-to-head on MAE).

The cue-pipeline is brittle (needs many cues jointly right). The product goal is
DIFFICULTY scoring, so this tests whether a model trained to predict D-score
*directly* from the skeleton beats the cue-pipeline's D-score — bypassing the
cue bottleneck. Same encoder, same split, both from scratch:
    DIRECT       : skeleton -> mean-pool -> scalar D-score (L1 regression)
    CUE-PIPELINE : skeleton -> cue heads -> frozen FIGDecoder -> D-score (= v2)
Target D-score = decoder.rank(true cues)[0].d_score (all clips); gold d_score
checked on the FIG-99. Baseline = predict the train-mean D-score.

Note: the direct model gives a difficulty number only (no trick name); the
cue-pipeline gives both. Local CPU/MPS, $0.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
POSE_GLOB = str(ROOT / "data" / "keypoints" / "**" / "shard_*.npz")
MANIFEST = ROOT / "data" / "v5_attribute_training" / "attribute_manifest.json"
CORE = ("context", "direction", "flip", "twist", "axis")
AUX = ("entry", "body_shape")
ALL_CUES = CORE + AUX
LR_SWAP = np.array([0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15])
D = 128


def load_skeletons():
    skel = {}
    for sh in sorted(glob.glob(POSE_GLOB, recursive=True)):
        with np.load(sh) as z:
            for s in z.files:
                skel[s] = z[s].astype(np.float32)
    return skel


def _flip_bin(v):
    return min(int(round(float(v))), 3)


def _twist_bin(v):
    v = float(v)
    return 0 if v < 0.25 else 1 if v < 0.75 else 2 if v < 1.25 else 3


def load_labels():
    out = {}
    for c in json.loads(MANIFEST.read_text())["clips"]:
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
            "d_score": c.get("d_score"),
        }
    return out


def base_seq(arr, t):
    if arr.shape[0] == 0:
        return np.zeros((t, 17, 3), np.float32)
    xy = arr[:, :, :2].astype(np.float32).copy()
    conf = arr[:, :, 2:3].astype(np.float32)
    ok = (arr[:, 11, 2] > 0.3) & (arr[:, 12, 2] > 0.3)
    hip = (xy[:, 11] + xy[:, 12]) / 2.0
    ctr = hip[ok].mean(0) if ok.any() else xy.reshape(-1, 2).mean(0)
    xy -= ctr[None, None, :]
    sh = (xy[:, 5] + xy[:, 6]) / 2.0
    tor = np.linalg.norm(sh - (xy[:, 11] + xy[:, 12]) / 2.0, axis=1)
    sc = np.median(tor[tor > 1e-3]) if (tor > 1e-3).any() else 1.0
    xy /= sc + 1e-6
    seq = np.concatenate([xy, conf], 2)
    idx = np.linspace(0, arr.shape[0] - 1, t).astype(int)
    return seq[idx].astype(np.float32)


def augment(seq):
    t = seq.shape[0]
    xy = seq[:, :, :2].copy()
    conf = seq[:, :, 2:3].copy()
    if np.random.rand() < 0.5:
        xy = xy[:, LR_SWAP, :]
        conf = conf[:, LR_SWAP, :]
        xy[:, :, 0] *= -1
    th = np.random.uniform(-0.26, 0.26)
    rot = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]], np.float32)
    xy = xy @ rot.T
    xy *= np.random.uniform(0.9, 1.1)
    xy += np.random.normal(0, 0.02, xy.shape).astype(np.float32)
    if np.random.rand() < 0.5:
        a = np.random.randint(0, max(1, int(t * 0.15)))
        b = t - np.random.randint(0, max(1, int(t * 0.15)))
        idx = np.linspace(a, b - 1, t).astype(int)
        xy, conf = xy[idx], conf[idx]
    return np.concatenate([xy, conf], 2)


def featurize(seq):
    t = seq.shape[0]
    xy = seq[:, :, :2]
    conf = seq[:, :, 2]
    vel = np.zeros_like(xy)
    vel[1:] = xy[1:] - xy[:-1]
    sh = (xy[:, 5] + xy[:, 6]) / 2.0
    hp = (xy[:, 11] + xy[:, 12]) / 2.0
    d = sh - hp
    ang = np.arctan2(d[:, 1], d[:, 0])
    av = np.zeros(t, np.float32)
    av[1:] = np.diff(np.unwrap(ang))
    return np.concatenate([
        xy.reshape(t, 34), conf.reshape(t, 17), vel.reshape(t, 34),
        np.sin(ang)[:, None], np.cos(ang)[:, None], av[:, None],
    ], 1).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--ft-epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--t", type=int, default=48)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    dev = ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
           else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={dev}")

    from core.recognition.fig_decoder import FIGDecoder
    from core.recognition.oracle_cues import _norm
    decoder = FIGDecoder()

    skel = load_skeletons()
    labels = load_labels()
    slugs = sorted(set(skel) & set(labels))
    classes = {c: sorted({labels[s][c] for s in slugs if labels[s][c] is not None}) for c in ALL_CUES}
    classes["flip"] = [0, 1, 2, 3]
    classes["twist"] = [0, 1, 2, 3]
    idx_of = {c: {v: i for i, v in enumerate(classes[c])} for c in ALL_CUES}
    inv = {c: {i: v for v, i in idx_of[c].items()} for c in ALL_CUES}
    twn = {0: 0.0, 1: 0.5, 2: 1.0}

    def cue_dict(p):
        d = {"flip": float(p["flip"])}
        if p["context"]:
            d["context"] = p["context"]
        if p["direction"] and p["direction"] != "none":
            d["direction"] = p["direction"]
        if p["twist"] in twn:
            d["twist"] = twn[p["twist"]]
        if p["axis"] and p["axis"] != "none":
            d["axis"] = p["axis"]
        if p["entry"] and p["entry"] != "unknown":
            d["entry"] = p["entry"]
        if p["body_shape"] and p["body_shape"] != "unknown":
            d["body_shape"] = p["body_shape"]
        return d

    def dscore_of(cues):
        c = decoder.rank(dict(cues), k=1)
        return float(c[0].d_score) if c else None

    # target D-score per clip = decoder(true cues).d_score ; keep only resolvable
    target = {}
    for s in slugs:
        ds = dscore_of(cue_dict({k: labels[s][k] for k in ALL_CUES}))
        if ds is not None:
            target[s] = ds
    slugs = [s for s in slugs if s in target]
    print(f"clips with target D-score: {len(slugs)}")
    dvals = np.array([target[s] for s in slugs])
    print(f"D-score range [{dvals.min():.2f},{dvals.max():.2f}] mean={dvals.mean():.2f} std={dvals.std():.2f}")

    bases = {s: base_seq(skel[s], args.t) for s in slugs}
    Yc = {c: np.array([idx_of[c][labels[s][c]] if labels[s][c] is not None and labels[s][c] in idx_of[c]
                       else -1 for s in slugs]) for c in ALL_CUES}
    perm = np.random.permutation(len(slugs))
    nval = int(0.2 * len(slugs))
    val_i, tr_i = perm[:nval], perm[nval:]
    tr_slugs = [slugs[i] for i in tr_i]
    val_slugs = [slugs[i] for i in val_i]
    print(f"train={len(tr_i)} val={len(val_i)}")

    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(88, D)
            self.pos = nn.Parameter(torch.randn(1, args.t, D) * 0.02)
            layer = nn.TransformerEncoderLayer(D, 4, 2 * D, dropout=0.3, batch_first=True)
            self.enc = nn.TransformerEncoder(layer, 3)

        def forward(self, x):
            return self.enc(self.proj(x) + self.pos).mean(1)

    # ---------------- DIRECT: skeleton -> D-score ----------------
    class RegSet(Dataset):
        def __init__(self, ss, train):
            self.ss, self.train = ss, train

        def __len__(self):
            return len(self.ss)

        def __getitem__(self, i):
            s = self.ss[i]
            seq = augment(bases[s]) if self.train else bases[s]
            return featurize(seq), np.float32(target[s])

    def run_direct():
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        enc = Encoder().to(dev)
        head = nn.Linear(D, 1).to(dev)
        opt = torch.optim.AdamW(list(enc.parameters()) + list(head.parameters()), lr=4e-4, weight_decay=5e-2)
        trdl = DataLoader(RegSet(tr_slugs, True), batch_size=args.batch, shuffle=True)
        vadl = DataLoader(RegSet(val_slugs, False), batch_size=256)

        def val_mae():
            enc.eval()
            head.eval()
            err = []
            with torch.no_grad():
                for xb, yb in vadl:
                    pred = head(enc(xb.to(dev))).squeeze(1).cpu().numpy()
                    err.extend(np.abs(pred - yb.numpy()))
            return float(np.mean(err))

        best, best_mae, wait = None, 1e9, 0
        for ep in range(args.epochs):
            enc.train()
            head.train()
            for xb, yb in trdl:
                xb, yb = xb.to(dev), yb.to(dev)
                loss = nn.functional.l1_loss(head(enc(xb)).squeeze(1), yb)
                opt.zero_grad()
                loss.backward()
                opt.step()
            mae = val_mae()
            if mae < best_mae:
                best_mae, best, wait = mae, ({k: v.cpu().clone() for k, v in enc.state_dict().items()},
                                             {k: v.cpu().clone() for k, v in head.state_dict().items()}), 0
            else:
                wait += 1
            if (ep + 1) % 15 == 0 or ep == 0:
                print(f"  direct ep{ep+1:>3} val_MAE={mae:.3f}")
            if wait >= args.patience:
                break
        enc.load_state_dict(best[0])
        head.load_state_dict(best[1])
        # per-clip preds for reporting
        enc.eval()
        head.eval()
        preds = {}
        with torch.no_grad():
            for s in val_slugs:
                x = torch.tensor(featurize(bases[s])[None]).to(dev)
                preds[s] = float(head(enc(x)).item())
        return best_mae, preds

    # ---------------- CUE-PIPELINE: cues -> decoder -> D-score (= v2) ----------------
    cw = {}
    for c in ALL_CUES:
        yt = Yc[c][tr_i]
        yt = yt[yt >= 0]
        cnt = np.bincount(yt, minlength=len(classes[c])).astype(np.float64)
        cw[c] = torch.tensor(cnt.sum() / (len(cnt) * np.maximum(cnt, 1)), dtype=torch.float32, device=dev)
    w_cue = {**{c: 1.0 for c in CORE}, **{c: 0.3 for c in AUX}}

    class CueSet(Dataset):
        def __init__(self, ss, train):
            self.ss, self.train = ss, train

        def __len__(self):
            return len(self.ss)

        def __getitem__(self, i):
            s = self.ss[i]
            seq = augment(bases[s]) if self.train else bases[s]
            y = np.array([idx_of[c][labels[s][c]] if labels[s][c] is not None and labels[s][c] in idx_of[c]
                          else -1 for c in ALL_CUES], np.int64)
            return featurize(seq), y

    class CueModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = Encoder()
            self.drop = nn.Dropout(0.3)
            self.heads = nn.ModuleDict({c: nn.Linear(D, len(classes[c])) for c in ALL_CUES})

        def forward(self, x):
            h = self.drop(self.encoder(x))
            return {c: self.heads[c](h) for c in ALL_CUES}

    def run_cue():
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        model = CueModel().to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=4e-4, weight_decay=5e-2)
        lossf = {c: nn.CrossEntropyLoss(weight=cw[c], ignore_index=-1) for c in ALL_CUES}
        trdl = DataLoader(CueSet(tr_slugs, True), batch_size=args.batch, shuffle=True)
        vadl = DataLoader(CueSet(val_slugs, False), batch_size=256)

        def cue_mae():
            model.eval()
            err = []
            with torch.no_grad():
                pid = {c: [] for c in ALL_CUES}
                for xb, _y in vadl:
                    o = model(xb.to(dev))
                    for c in ALL_CUES:
                        pid[c].append(o[c].argmax(1).cpu().numpy())
                pid = {c: np.concatenate(pid[c]) for c in ALL_CUES}
            for j, s in enumerate(val_slugs):
                p = {c: inv[c].get(int(pid[c][j])) for c in ALL_CUES}
                d = dscore_of(cue_dict(p))
                if d is not None:
                    err.append(abs(d - target[s]))
            return float(np.mean(err))

        best_mae, wait = 1e9, 0
        for ep in range(args.ft_epochs):
            model.train()
            for xb, yb in trdl:
                xb, yb = xb.to(dev), yb.to(dev)
                o = model(xb)
                loss = sum(w_cue[c] * lossf[c](o[c], yb[:, ci]) for ci, c in enumerate(ALL_CUES))
                opt.zero_grad()
                loss.backward()
                opt.step()
            mae = cue_mae()
            if mae < best_mae:
                best_mae, wait = mae, 0
            else:
                wait += 1
            if wait >= args.patience:
                break
        return best_mae

    print("\n# DIRECT regressor (skeleton -> D-score)")
    direct_mae, direct_preds = run_direct()
    print("# CUE-PIPELINE (cues -> decoder -> D-score)")
    cue_mae = run_cue()

    # baselines + fig-gold check
    base_mae = float(np.mean(np.abs(dvals[val_i] - dvals[tr_i].mean())))
    fig_err_direct, fig_err_target = [], []
    for s in val_slugs:
        if labels[s]["source"] == "fig" and labels[s]["d_score"] is not None:
            fig_err_direct.append(abs(direct_preds[s] - float(labels[s]["d_score"])))
            fig_err_target.append(abs(target[s] - float(labels[s]["d_score"])))

    print("\n=== D-SCORE MAE (lower = better) ===")
    print(f"  mean-baseline      : {base_mae:.3f}")
    print(f"  CUE-PIPELINE (v2)  : {cue_mae:.3f}")
    print(f"  DIRECT regressor   : {direct_mae:.3f}")
    if fig_err_direct:
        print(f"  [FIG-gold val n={len(fig_err_direct)}] direct vs gold d_score MAE={np.mean(fig_err_direct):.3f} "
              f"| target(cue-implied) vs gold MAE={np.mean(fig_err_target):.3f}")
    print("(cue-contract oracle-cue D-score MAE on the P0 blind set was 0.225)")


if __name__ == "__main__":
    main()
