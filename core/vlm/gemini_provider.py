"""Gemini VLM provider — native video upload for trick identification."""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

from core.vlm.base import VLMProvider, VLMResponse, VLMTrickResult
from core.vlm.prompt import build_prompt, load_fig_tricks


def _parse_tricks_json(text: str) -> list[dict]:
    """Extract trick list from VLM response text (handles markdown fences)."""
    # Strip markdown code fences if present
    cleaned = re.sub(r"```(?:json)?\s*", "", text)
    cleaned = cleaned.strip()

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        # Try to find JSON object in the text
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


class GeminiProvider(VLMProvider):
    """Gemini provider with native video upload.

    Supports both API key and OAuth authentication:
    - API key: set GOOGLE_API_KEY env var
    - OAuth: pass credentials to constructor (for using Google account quota)
    """

    def __init__(
        self,
        model: str = "gemini-2.5-pro",
        api_key: str | None = None,
        credentials=None,
    ):
        from google import genai

        if credentials is not None:
            self.client = genai.Client(credentials=credentials)
        elif api_key or os.environ.get("GOOGLE_API_KEY"):
            self.client = genai.Client(api_key=api_key or os.environ["GOOGLE_API_KEY"])
        else:
            raise ValueError(
                "Set GOOGLE_API_KEY env var, pass api_key=, or pass credentials= for OAuth"
            )
        self.model = model
        self._fig_data = load_fig_tricks()

    def analyze_trick(
        self,
        video_path: Path,
        trick_list: str | None = None,
    ) -> VLMResponse:
        """Upload video to Gemini and identify tricks.

        Args:
            video_path: Path to the trick clip (MP4/MOV).
            trick_list: Optional override for the trick table in the prompt.
                        If None, uses the full FIG table.
        """
        video_path = Path(video_path)

        # Upload video via File API
        uploaded_file = self.client.files.upload(file=str(video_path))

        # Wait for processing (Gemini needs to index the video)
        while uploaded_file.state and uploaded_file.state.name == "PROCESSING":
            time.sleep(1)
            uploaded_file = self.client.files.get(name=uploaded_file.name)

        # Build prompt
        prompt = build_prompt(self._fig_data)

        # Generate
        response = self.client.models.generate_content(
            model=self.model,
            contents=[uploaded_file, prompt],
        )

        # Parse response
        raw_text = response.text or ""
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

        # Token usage
        usage = getattr(response, "usage_metadata", None)
        input_tokens = getattr(usage, "prompt_token_count", 0) or 0
        output_tokens = getattr(usage, "candidates_token_count", 0) or 0

        return VLMResponse(
            tricks=tricks,
            model=self.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            raw_text=raw_text,
        )
