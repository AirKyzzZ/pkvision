"""Skeleton → cue model: extracted verbatim from scripts/diagnose_skeleton_cues_v2.py.

Do not change the math in LR_SWAP, D, base_seq, augment, featurize, or CueModel.forward —
they must stay byte-identical to the v2 diagnostic so behaviour matches exactly.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

LR_SWAP = np.array([0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15])  # COCO-17 mirror
D = 128

CUE_CLASSES = {
    "context": ["acrobatics", "pk_basics", "swing", "wall"],
    "direction": ["backward", "forward", "none", "side"],
    "flip": [0, 1, 2, 3],
    "twist": [0, 1, 2, 3],
    "axis": ["lateral", "longitudinal", "off_axis", "sagittal"],
}
TW_NUM = {0: 0.0, 1: 0.5, 2: 1.0}  # bin 3 (>=1.5) -> abstain per cue contract


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


class CueModel(nn.Module):
    def __init__(self, classes: dict = CUE_CLASSES, t: int = 48):
        super().__init__()
        self.proj = nn.Linear(88, D)
        self.pos = nn.Parameter(torch.randn(1, t, D) * 0.02)
        enc_l = nn.TransformerEncoderLayer(D, 4, 2 * D, dropout=0.3, batch_first=True)
        self.enc = nn.TransformerEncoder(enc_l, 3)
        self.drop = nn.Dropout(0.3)
        self.heads = nn.ModuleDict({c: nn.Linear(D, len(classes[c])) for c in classes})

    def forward(self, x):
        h = self.drop(self.enc(self.proj(x) + self.pos).mean(1))
        return {c: self.heads[c](h) for c in self.heads}


def predict_cues(model: "CueModel", skeleton: np.ndarray, t: int = 48) -> dict:
    """skeleton (T,17,3) -> {cue: value, '<cue>_conf': float}. Twist bin 3 abstains."""
    model.eval()
    x = torch.tensor(featurize(base_seq(skeleton, t))[None])
    with torch.no_grad():
        out = model(x.to(next(model.parameters()).device))
    cues: dict = {}
    for cue, logits in out.items():
        probs = torch.softmax(logits[0], -1)
        idx = int(probs.argmax())
        conf = float(probs[idx])
        val = CUE_CLASSES[cue][idx] if cue in CUE_CLASSES else idx
        if cue == "twist":
            if idx in TW_NUM:
                cues["twist"] = TW_NUM[idx]
        elif cue == "flip":
            cues["flip"] = float(val)
        elif val not in (None, "none"):
            cues[cue] = val
        cues[f"{cue}_conf"] = conf
    return cues
