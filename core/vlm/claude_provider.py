"""Claude VLM provider — frame-based trick identification.

Claude doesn't support native video, so we extract key frames from the clip,
encode them as base64 JPEG, and send them as individual images.
"""

from __future__ import annotations

import base64
import json
import os
import re
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np

from core.vlm.base import VLMProvider, VLMResponse, VLMTrickResult
from core.vlm.prompt import build_prompt, load_fig_tricks


def _extract_frames(video_path: Path, num_frames: int = 8) -> list[np.ndarray]:
    """Extract uniformly-spaced frames from a video clip."""
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []

    indices = np.linspace(0, total - 1, num_frames, dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    cap.release()
    return frames


def _frame_to_base64(frame: np.ndarray, quality: int = 85) -> str:
    """Encode a BGR frame as base64 JPEG."""
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    _, buffer = cv2.imencode(".jpg", frame, encode_params)
    return base64.standard_b64encode(buffer).decode("utf-8")


def _parse_tricks_json(text: str) -> list[dict]:
    """Extract trick list from Claude response text."""
    cleaned = re.sub(r"```(?:json)?\s*", "", text)
    cleaned = cleaned.strip()
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", cleaned)
        if match:
            try:
                data = json.loads(match.group())
            except json.JSONDecodeError:
                return []
        else:
            return []

    if isinstance(data, dict):
        return data.get("tricks", [data])
    if isinstance(data, list):
        return data
    return []


class ClaudeProvider(VLMProvider):
    """Claude provider using frame-based image analysis.

    Extracts key frames from the video, encodes as base64 JPEG,
    and sends to Claude's vision API.
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        api_key: str | None = None,
        num_frames: int = 8,
    ):
        import anthropic

        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError("Set ANTHROPIC_API_KEY env var or pass api_key=")
        self.client = anthropic.Anthropic(api_key=key)
        self.model = model
        self.num_frames = num_frames
        self._fig_data = load_fig_tricks()

    def analyze_trick(
        self,
        video_path: Path,
        trick_list: str | None = None,
    ) -> VLMResponse:
        """Extract frames from video and send to Claude for identification."""
        video_path = Path(video_path)

        # Extract frames
        frames = _extract_frames(video_path, self.num_frames)
        if not frames:
            return VLMResponse(raw_text="Failed to extract frames")

        # Build message content: images first, then prompt
        content = []
        for i, frame in enumerate(frames):
            b64 = _frame_to_base64(frame)
            content.append({
                "type": "text",
                "text": f"Frame {i + 1}/{len(frames)}:",
            })
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": b64,
                },
            })

        # Add the analysis prompt
        prompt = build_prompt(self._fig_data)
        content.append({"type": "text", "text": prompt})

        # Call Claude
        response = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            messages=[{"role": "user", "content": content}],
        )

        raw_text = response.content[0].text if response.content else ""
        tricks_data = _parse_tricks_json(raw_text)

        tricks = []
        for t in tricks_data:
            tricks.append(VLMTrickResult(
                trick_name=t.get("trick_name", "Unknown"),
                category=t.get("category"),
                direction=t.get("direction"),
                flip_count=float(t.get("flip_count", 0)),
                twist_count=float(t.get("twist_count", 0)),
                confidence=t.get("confidence", "medium"),
                reasoning=t.get("reasoning", ""),
            ))

        return VLMResponse(
            tricks=tricks,
            model=self.model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            raw_text=raw_text,
        )
