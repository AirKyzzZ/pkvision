"""Attribute specialist — VideoMAE multi-head predictor for cue extraction.

Loads a checkpoint trained by ``scripts/train_attributes.py`` and emits the
four structured cues that the FIG decoder consumes:

    context   ∈ {acrobatics, wall, swing, pk_basics}
    direction ∈ {backward, forward, side, none}
    flip_bin  ∈ {0, 1, 2+}     →  flip numeric  {0.0, 1.0, 2.0}
    twist_bin ∈ {0, 0.5, 1, 2+} → twist numeric {0.0, 0.5, 1.0, 2.0}

The predictor is the network counterpart to the 2D-pose heuristics in
``heuristic_recognizer.py``. It is built to degrade gracefully: if the
checkpoint is missing, construction raises ``FileNotFoundError`` — callers
should catch that and fall back to pose cues.

Inference is a single forward pass over 16 uniformly-sampled frames. No
augmentation. The checkpoint carries its own ``attr_config`` so schema
changes in ``train_attributes.py`` do not silently break inference.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn as nn


HF_MODEL = "MCG-NJU/videomae-base-finetuned-kinetics"

# Map string bin labels -> numeric cue values the decoder expects.
_FLIP_NUMERIC = {"0": 0.0, "1": 1.0, "2+": 2.0, "3+": 3.0}
# `has_twist` is the coarse legacy label — map to 1.0 as a conservative guess
# (most twisting tricks in POOL-B have twist=1). Finer v2 checkpoints never
# emit this label so the fallback only matters for back-compat with the old
# `pkvision_attributes.pt` checkpoint schema.
_TWIST_NUMERIC = {"0": 0.0, "0.5": 0.5, "1": 1.0, "1.5": 1.5, "2+": 2.0, "has_twist": 1.0}


class _MultiTaskAttributeModel(nn.Module):
    """Mirror of the training-time model so state dicts load cleanly."""

    def __init__(self, backbone, hidden_size: int, attr_config: dict[str, Any]):
        super().__init__()
        self.backbone = backbone
        self.heads = nn.ModuleDict()
        for attr_name, cfg in attr_config.items():
            n_classes = len(cfg["classes"])
            self.heads[attr_name] = nn.Sequential(
                nn.Dropout(0.1),
                nn.Linear(hidden_size, n_classes),
            )

    def forward(self, pixel_values):
        outputs = self.backbone.videomae(pixel_values=pixel_values)
        cls_token = outputs.last_hidden_state[:, 0]
        return {name: head(cls_token) for name, head in self.heads.items()}


def _read_video_frames(clip_path: Path, num_frames: int = 16) -> np.ndarray:
    """Uniformly sample ``num_frames`` BGR→RGB frames from a video file."""
    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {clip_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        # Fall back to a blocking read when the container lies about length.
        frames: list[np.ndarray] = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        if not frames:
            raise RuntimeError(f"no frames in {clip_path}")
        arr = np.stack(frames)
    else:
        idxs = np.linspace(0, total - 1, min(num_frames, total), dtype=int)
        wanted = set(int(i) for i in idxs)
        by_idx: dict[int, np.ndarray] = {}
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if i in wanted:
                by_idx[i] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            i += 1
            if len(by_idx) == len(wanted):
                break
        cap.release()
        arr = np.stack([by_idx[int(j)] for j in idxs if int(j) in by_idx])

    # Pad by duplication if the clip is shorter than num_frames.
    if arr.shape[0] < num_frames:
        pad = np.repeat(arr[-1:], num_frames - arr.shape[0], axis=0)
        arr = np.concatenate([arr, pad], axis=0)
    return arr


class AttributePredictor:
    """Load an attribute checkpoint and produce cue dicts from video clips."""

    def __init__(
        self,
        ckpt_path: Path | str,
        device: str | None = None,
        num_frames: int = 16,
    ) -> None:
        from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor

        ckpt_path = Path(ckpt_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"attribute checkpoint not found: {ckpt_path}")

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = torch.device(device)
        self.num_frames = num_frames

        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        self.attr_config: dict[str, Any] = ckpt["attr_config"]
        hidden_size: int = ckpt["hidden_size"]

        self.processor = VideoMAEImageProcessor.from_pretrained(HF_MODEL)
        backbone = VideoMAEForVideoClassification.from_pretrained(HF_MODEL)
        self.model = _MultiTaskAttributeModel(backbone, hidden_size, self.attr_config)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.to(self.device).eval()

    @torch.no_grad()
    def predict(self, clip_path: Path | str) -> dict[str, Any]:
        """Return a cue dict ready for FIGDecoder.rank.

        Each cue is accompanied by a ``_conf`` sibling (softmax probability
        of the winning class), so callers can decide whether to trust the
        model or fall back to heuristics.
        """
        frames = _read_video_frames(Path(clip_path), self.num_frames)
        fl = [frames[i] for i in range(frames.shape[0])]
        inputs = self.processor(fl, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)

        logits = self.model(pixel_values)

        cues: dict[str, Any] = {}
        for attr_name, head_logits in logits.items():
            probs = torch.softmax(head_logits[0], dim=-1)
            idx = int(probs.argmax().item())
            label = self.attr_config[attr_name]["classes"][idx]
            conf = float(probs[idx].item())

            if attr_name == "context":
                cues["context"] = label
                cues["context_conf"] = conf
            elif attr_name == "direction":
                # The decoder uses `direction` in {backward, forward, side}.
                # `none` is only used for tricks with no dominant direction;
                # map it to an absent cue rather than lying.
                if label != "none":
                    cues["direction"] = label
                cues["direction_conf"] = conf
            elif attr_name == "flip_bin":
                cues["flip"] = _FLIP_NUMERIC.get(label, 1.0)
                cues["flip_conf"] = conf
            elif attr_name == "twist_bin":
                cues["twist"] = _TWIST_NUMERIC.get(label, 0.0)
                cues["twist_conf"] = conf
        return cues
