"""OpenRouter VLM provider — access any model through one API.

Supports native video (base64 MP4) for models like Gemini, Qwen, Kimi,
and falls back to frame-based images for models that don't support video.
"""

from __future__ import annotations

import base64
import json
import os
import re
from pathlib import Path

import cv2
import numpy as np
from openai import OpenAI

from core.vlm.base import VLMProvider, VLMResponse, VLMTrickResult
from core.vlm.prompt import build_prompt, load_fig_tricks

# Models known to support native video input via OpenRouter
VIDEO_NATIVE_MODELS = {
    "google/gemini-2.5-pro",
    "google/gemini-2.5-flash",
    "google/gemini-2.5-flash-lite",
    "google/gemini-3-pro-image-preview",
    "google/gemini-3-flash-preview",
    "google/gemini-3.1-pro-preview",
    "google/gemini-3.1-flash-lite-preview",
    "qwen/qwen-vl-max",
    "qwen/qwen-vl-plus",
    "qwen/qwen2.5-vl-72b-instruct",
    "qwen/qwen2.5-vl-32b-instruct",
    "moonshotai/kimi-k2.5",
    "meta-llama/llama-4-maverick",
    "meta-llama/llama-4-scout",
}

# Default models to benchmark (good mix of quality/cost)
DEFAULT_BENCHMARK_MODELS = [
    "google/gemini-2.5-pro",
    "qwen/qwen2.5-vl-72b-instruct",
    "qwen/qwen-vl-max",
    "google/gemini-2.5-flash",
    "moonshotai/kimi-k2.5",
]


def _parse_tricks_json(text: str) -> list[dict]:
    """Extract trick list from VLM response text.

    Handles markdown fences and truncated JSON (from max_tokens cutoff).
    """
    cleaned = re.sub(r"```(?:json)?", "", text).strip()

    # Try direct parse
    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            return data.get("tricks", [data])
        if isinstance(data, list):
            return data
        return []
    except json.JSONDecodeError:
        pass

    # Try extracting a JSON object
    match = re.search(r"\{[\s\S]*\}", cleaned)
    if match:
        try:
            data = json.loads(match.group())
            if isinstance(data, dict):
                return data.get("tricks", [data])
            return []
        except json.JSONDecodeError:
            pass

    # Handle truncated JSON — find all complete trick objects
    tricks = []
    pattern = r'\{\s*"trick_name"\s*:\s*"([^"]+)"[^}]*?"category"\s*:\s*"([^"]*)"[^}]*?"direction"\s*:\s*"([^"]*)"[^}]*?"flip_count"\s*:\s*([\d.]+)[^}]*?"twist_count"\s*:\s*([\d.]+)[^}]*?"confidence"\s*:\s*"([^"]*)"[^}]*?"reasoning"\s*:\s*"([^"]*)"[^}]*?\}'
    for m in re.finditer(pattern, cleaned, re.DOTALL):
        tricks.append({
            "trick_name": m.group(1),
            "category": m.group(2),
            "direction": m.group(3),
            "flip_count": float(m.group(4)),
            "twist_count": float(m.group(5)),
            "confidence": m.group(6),
            "reasoning": m.group(7),
        })

    return tricks


def _encode_video_base64(video_path: Path) -> str:
    """Encode a video file as base64."""
    with open(video_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _extract_frames_base64(video_path: Path, num_frames: int = 8) -> list[str]:
    """Extract frames as base64 JPEG for image-only models."""
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []

    indices = np.linspace(0, total - 1, num_frames, dtype=int)
    frames_b64 = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            frames_b64.append(base64.b64encode(buffer).decode("utf-8"))
    cap.release()
    return frames_b64


class OpenRouterProvider(VLMProvider):
    """OpenRouter provider — one API for all models.

    Automatically uses native video for supported models,
    falls back to frame extraction for image-only models.
    """

    def __init__(
        self,
        model: str = "qwen/qwen3.5-omni",
        api_key: str | None = None,
        num_frames: int = 8,
    ):
        key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise ValueError("Set OPENROUTER_API_KEY env var or pass api_key=")
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=key,
        )
        self.model = model
        self.num_frames = num_frames
        self._fig_data = load_fig_tricks()
        # Only Google Gemini models reliably support native video on OpenRouter
        self._supports_video = model.startswith("google/gemini")

    def analyze_trick(
        self,
        video_path: Path,
        trick_list: str | None = None,
    ) -> VLMResponse:
        """Analyze a trick clip via OpenRouter."""
        video_path = Path(video_path)
        prompt = build_prompt(self._fig_data)

        if self._supports_video:
            content = self._build_video_content(video_path, prompt)
        else:
            content = self._build_frames_content(video_path, prompt)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": content}],
                max_tokens=16384,
            )
        except Exception as e:
            return VLMResponse(
                model=self.model,
                raw_text=f"ERROR: {e}",
            )

        raw_text = response.choices[0].message.content or "" if response.choices else ""
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

        usage = response.usage
        return VLMResponse(
            tricks=tricks,
            model=self.model,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            raw_text=raw_text,
        )

    def _build_video_content(self, video_path: Path, prompt: str) -> list[dict]:
        """Build content with native video for supported models."""
        video_b64 = _encode_video_base64(video_path)
        suffix = video_path.suffix.lower().lstrip(".")
        mime = {"mp4": "video/mp4", "mov": "video/mp4", "avi": "video/avi",
                "webm": "video/webm"}.get(suffix, "video/mp4")

        return [
            {"type": "text", "text": prompt},
            {
                "type": "video_url",
                "video_url": {"url": f"data:{mime};base64,{video_b64}"},
            },
        ]

    def _build_frames_content(self, video_path: Path, prompt: str) -> list[dict]:
        """Build content with extracted frames for image-only models."""
        frames = _extract_frames_base64(video_path, self.num_frames)
        content = []
        for i, b64 in enumerate(frames):
            content.append({"type": "text", "text": f"Frame {i + 1}/{len(frames)}:"})
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
            })
        content.append({"type": "text", "text": prompt})
        return content
