#!/usr/bin/env python3
"""GFP-style self-supervised masked-skeleton pretrain -> fine-tune cue heads (A/B).

Tests the scan's headline lever: does SSL pretraining on the 1,618 UNLABELED
skeletons sharpen the cues vs from-scratch (v2)? Pretrains the v2 encoder by
masked-frame feature reconstruction (no labels), then fine-tunes cue heads twice
on the SAME split:
    A = encoder initialised from SSL pretraining
    B = encoder from scratch  (== v2 baseline)
Reports per-cue macro-F1 + end-to-end FIGDecoder top-1 for both. If A >> B, SSL
is worth scaling to more unlabeled data (FineGym / auto-gathered) on Modal.

No manual labels. Local CPU/MPS, $0.
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


def macro_f1(yt, yp):
    f = []
    for k in np.unique(yt):
        tp = int(((yp == k) & (yt == k)).sum())
        fp = int(((yp == k) & (yt != k)).sum())
        fn = int(((yp != k) & (yt == k)).sum())
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f.append(2 * p * r / (p + r) if p + r else 0.0)
    return float(np.mean(f)) if f else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ssl-epochs", type=int, default=60)
    ap.add_argument("--ft-epochs", type=int, default=55)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--t", type=int, default=48)
    ap.add_argument("--mask", type=float, default=0.4)
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

    skel = load_skeletons()
    labels = load_labels()
    slugs = sorted(set(skel) & set(labels))
    print(f"clips={len(slugs)}")
    classes = {c: sorted({labels[s][c] for s in slugs if labels[s][c] is not None}) for c in ALL_CUES}
    classes["flip"] = [0, 1, 2, 3]
    classes["twist"] = [0, 1, 2, 3]
    idx_of = {c: {v: i for i, v in enumerate(classes[c])} for c in ALL_CUES}
    bases = {s: base_seq(skel[s], args.t) for s in slugs}
    Y = {c: np.array([idx_of[c][labels[s][c]] if labels[s][c] is not None and labels[s][c] in idx_of[c]
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
            return self.enc(self.proj(x) + self.pos)  # (B,T,D)

    class CueModel(nn.Module):
        def __init__(self, encoder):
            super().__init__()
            self.encoder = encoder
            self.drop = nn.Dropout(0.3)
            self.heads = nn.ModuleDict({c: nn.Linear(D, len(classes[c])) for c in ALL_CUES})

        def forward(self, x):
            h = self.drop(self.encoder(x).mean(1))
            return {c: self.heads[c](h) for c in ALL_CUES}

    class SSLSet(Dataset):
        def __init__(self, ss):
            self.ss = ss

        def __len__(self):
            return len(self.ss)

        def __getitem__(self, i):
            return featurize(augment(bases[self.ss[i]]))

    def pretrain():
        enc = Encoder().to(dev)
        dec = nn.Linear(D, 88).to(dev)
        mask_tok = nn.Parameter(torch.zeros(88, device=dev))
        opt = torch.optim.AdamW(list(enc.parameters()) + list(dec.parameters()) + [mask_tok],
                                lr=4e-4, weight_decay=5e-2)
        dl = DataLoader(SSLSet(slugs), batch_size=args.batch, shuffle=True)
        for ep in range(args.ssl_epochs):
            enc.train()
            tot, nb = 0.0, 0
            for xb in dl:
                xb = xb.to(dev)
                b, t, _ = xb.shape
                m = torch.rand(b, t, device=dev) < args.mask
                if not m.any():
                    continue
                xin = xb.clone()
                xin[m] = mask_tok.to(xb.dtype)
                rec = dec(enc(xin))
                loss = ((rec[m] - xb[m]) ** 2).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                tot += loss.item()
                nb += 1
            if (ep + 1) % 10 == 0 or ep == 0:
                print(f"  ssl ep{ep+1:>3} recon_mse={tot/max(nb,1):.4f}")
        return enc

    cw = {}
    for c in ALL_CUES:
        yt = Y[c][tr_i]
        yt = yt[yt >= 0]
        cnt = np.bincount(yt, minlength=len(classes[c])).astype(np.float64)
        cw[c] = torch.tensor(cnt.sum() / (len(cnt) * np.maximum(cnt, 1)), dtype=torch.float32, device=dev)
    w_cue = {**{c: 1.0 for c in CORE}, **{c: 0.3 for c in AUX}}

    class FTSet(Dataset):
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

    def finetune(encoder, tag):
        np.random.seed(args.seed)       # identical aug + shuffle + head init for A/B
        torch.manual_seed(args.seed)
        model = CueModel(encoder).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=4e-4, weight_decay=5e-2)
        lossf = {c: nn.CrossEntropyLoss(weight=cw[c], ignore_index=-1) for c in ALL_CUES}
        trdl = DataLoader(FTSet(tr_slugs, True), batch_size=args.batch, shuffle=True)
        vadl = DataLoader(FTSet(val_slugs, False), batch_size=256)

        def valf1():
            model.eval()
            P = {c: [] for c in ALL_CUES}
            T = {c: [] for c in ALL_CUES}
            with torch.no_grad():
                for xb, yb in vadl:
                    o = model(xb.to(dev))
                    for ci, c in enumerate(ALL_CUES):
                        P[c].append(o[c].argmax(1).cpu().numpy())
                        T[c].append(yb[:, ci].numpy())
            m = {}
            for c in ALL_CUES:
                yt = np.concatenate(T[c])
                yp = np.concatenate(P[c])
                msk = yt >= 0
                m[c] = macro_f1(yt[msk], yp[msk])
            return m

        best, best_state, wait = -1.0, None, 0
        for ep in range(args.ft_epochs):
            model.train()
            for xb, yb in trdl:
                xb, yb = xb.to(dev), yb.to(dev)
                o = model(xb)
                loss = sum(w_cue[c] * lossf[c](o[c], yb[:, ci]) for ci, c in enumerate(ALL_CUES))
                opt.zero_grad()
                loss.backward()
                opt.step()
            m = valf1()
            sc = float(np.mean([m[c] for c in CORE]))
            if sc > best:
                best, best_state, wait = sc, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
            else:
                wait += 1
            if wait >= args.patience:
                break
        model.load_state_dict(best_state)
        m = valf1()
        print(f"[{tag}] core-mF1={np.mean([m[c] for c in CORE]):.3f}  " +
              " ".join(f"{c}={m[c]:.2f}" for c in CORE))
        return model, m

    from core.recognition.fig_decoder import FIGDecoder
    from core.recognition.oracle_cues import _norm
    decoder = FIGDecoder()
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

    def top1(cues):
        c = decoder.rank(dict(cues), k=1)
        return getattr(c[0], "fig_name", None) if c else None

    def decode_eval(model, tag):
        model.eval()
        vadl = DataLoader(FTSet(val_slugs, False), batch_size=256)
        pid = {c: [] for c in ALL_CUES}
        with torch.no_grad():
            for xb, _y in vadl:
                o = model(xb.to(dev))
                for c in ALL_CUES:
                    pid[c].append(o[c].argmax(1).cpu().numpy())
        pid = {c: np.concatenate(pid[c]) for c in ALL_CUES}
        a1 = fh = fo = fn = 0
        for j, s in enumerate(val_slugs):
            p = {c: inv[c].get(int(pid[c][j])) for c in ALL_CUES}
            L = labels[s]
            pt = top1(cue_dict(p))
            tt = top1(cue_dict({k: L[k] for k in ALL_CUES}))
            if pt and tt and _norm(pt) == _norm(tt):
                a1 += 1
            if L["source"] == "fig" and L["fig_name"]:
                fn += 1
                if pt and _norm(pt) == _norm(L["fig_name"]):
                    fh += 1
                if tt and _norm(tt) == _norm(L["fig_name"]):
                    fo += 1
        nv = len(val_slugs)
        print(f"[{tag}] cue-preservation top1={a1}/{nv}={a1/nv:.1%}  "
              f"fig-gold model={fh}/{fn} oracle={fo}/{fn}")

    print(f"\n# SSL masked-skeleton pretrain on {len(slugs)} unlabeled clips (mask={args.mask})")
    enc_ssl = pretrain()
    print("\n# fine-tune A = SSL-pretrained encoder")
    mA, _ = finetune(enc_ssl, "A:ssl")
    print("# fine-tune B = from-scratch encoder (v2 baseline)")
    mB, _ = finetune(Encoder().to(dev), "B:scratch")
    print("\n=== A/B END-TO-END (frozen FIGDecoder) ===")
    decode_eval(mA, "A:ssl")
    decode_eval(mB, "B:scratch")


if __name__ == "__main__":
    main()
