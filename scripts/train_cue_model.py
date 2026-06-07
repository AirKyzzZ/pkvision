#!/usr/bin/env python3
"""Train + save the canonical core/labeling/cue_model.py CueModel.

Reuses the v2 diagnostic recipe (motion features, on-the-fly augmentation,
inverse-freq class weights, multi-task loss with entry/body_shape as
training-only aux heads, early-stop on mean core macro-F1) but trains the
SHIPPED CueModel so the saved 5-head state_dict loads strict into the proposer.

Usage: python3 scripts/train_cue_model.py --out data/models/cue_model_v2.pt
Local CPU/MPS, $0.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.labeling.cue_model import CUE_CLASSES, CueModel, augment, base_seq, featurize

POSE_GLOB = str(ROOT / "data" / "keypoints" / "**" / "shard_*.npz")
MANIFEST = ROOT / "data" / "v5_attribute_training" / "attribute_manifest.json"

CORE = tuple(CUE_CLASSES)  # context, direction, flip, twist, axis
AUX = ("entry", "body_shape")
D = 128


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
        }
    print(f"labels: {len(out)} clips")
    return out


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
    ap.add_argument("--out", default=str(ROOT / "data" / "models" / "cue_model_v2.pt"))
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

    aux_classes = {c: sorted({labels[s][c] for s in slugs if labels[s][c] is not None}) for c in AUX}
    idx_of = {c: {v: i for i, v in enumerate(CUE_CLASSES[c])} for c in CORE}
    idx_of.update({c: {v: i for i, v in enumerate(aux_classes[c])} for c in AUX})
    cues = CORE + AUX

    def enc(s, c):
        v = labels[s][c]
        return idx_of[c].get(v, -1) if v is not None else -1

    bases = {s: base_seq(skel[s], args.t) for s in slugs}
    Y = {c: np.array([enc(s, c) for s in slugs]) for c in cues}

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
            y = np.array([enc(s, c) for c in cues], np.int64)
            return featurize(seq), y

    tr_dl = DataLoader(DS(tr_slugs, True), batch_size=args.batch, shuffle=True)
    va_dl = DataLoader(DS(val_slugs, False), batch_size=256)

    cw = {}
    for c in cues:
        yt = Y[c][tr_i]
        yt = yt[yt >= 0]
        n_cls = len(CUE_CLASSES[c]) if c in CORE else len(aux_classes[c])
        cnt = np.bincount(yt, minlength=n_cls).astype(np.float64)
        w = cnt.sum() / (len(cnt) * np.maximum(cnt, 1))
        cw[c] = torch.tensor(w, dtype=torch.float32, device=device)

    class TrainWrapper(nn.Module):
        def __init__(self, core_model: CueModel):
            super().__init__()
            self.core = core_model
            self.aux = nn.ModuleDict({c: nn.Linear(D, len(aux_classes[c])) for c in AUX})

        def forward(self, x):
            m = self.core
            h = m.drop(m.enc(m.proj(x) + m.pos).mean(1))
            out = {c: m.heads[c](h) for c in m.heads}
            out.update({c: self.aux[c](h) for c in self.aux})
            return out

    model = TrainWrapper(CueModel(CUE_CLASSES, t=args.t)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=4e-4, weight_decay=5e-2)
    losses = {c: nn.CrossEntropyLoss(weight=cw[c], ignore_index=-1) for c in cues}
    w_cue = {**{c: 1.0 for c in CORE}, **{c: 0.3 for c in AUX}}

    def collect(dl):
        model.eval()
        P = {c: [] for c in cues}
        T = {c: [] for c in cues}
        with torch.no_grad():
            for xb, yb in dl:
                out = model(xb.to(device))
                for ci, c in enumerate(cues):
                    P[c].append(out[c].argmax(1).cpu().numpy())
                    T[c].append(yb[:, ci].numpy())
        return ({c: np.concatenate(P[c]) for c in cues},
                {c: np.concatenate(T[c]) for c in cues})

    def core_mf1(P, T):
        out = {}
        for c in CORE:
            mask = T[c] >= 0
            out[c] = macro_f1(T[c][mask], P[c][mask]) if mask.any() else 0.0
        return out

    best, best_state, wait = -1.0, None, 0
    for ep in range(args.epochs):
        model.train()
        for xb, yb in tr_dl:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            loss = sum(w_cue[c] * losses[c](out[c], yb[:, ci]) for ci, c in enumerate(cues))
            opt.zero_grad()
            loss.backward()
            opt.step()
        vm = core_mf1(*collect(va_dl))
        score = float(np.mean([vm[c] for c in CORE]))
        if score > best:
            best, best_state, wait = score, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            wait += 1
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"ep{ep+1:>3} core-mF1={score:.3f} " + " ".join(f"{c}={vm[c]:.2f}" for c in CORE))
        if wait >= args.patience:
            print(f"early stop @ ep{ep+1}")
            break

    model.load_state_dict(best_state)
    vm = core_mf1(*collect(va_dl))

    def base_f1(c):
        yt = Y[c][val_i]
        yt = yt[yt >= 0]
        maj = np.bincount(yt).argmax()
        return macro_f1(yt, np.full_like(yt, maj))

    print(f"\nbest core-mF1={best:.3f}")
    print(f"{'cue':<10}{'baseF1':<9}{'valF1':<9}{'lift':<8}")
    for c in CORE:
        bf = base_f1(c)
        print(f"{c:<10}{bf:<9.3f}{vm[c]:<9.3f}{vm[c]-bf:+.3f}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.core.state_dict(), out_path)
    print(f"\nsaved canonical CueModel state_dict -> {out_path}")

    fresh = CueModel(CUE_CLASSES, t=args.t)
    fresh.load_state_dict(torch.load(out_path, map_location="cpu", weights_only=True))
    fresh.eval()
    demo = fresh.predict(skel[val_slugs[0]])
    print(f"smoke predict ({val_slugs[0]}): "
          + " ".join(f"{k}={v}" for k, v in demo.items() if not k.endswith("_conf")))


if __name__ == "__main__":
    main()
