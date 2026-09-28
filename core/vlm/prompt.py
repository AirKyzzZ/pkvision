"""FIG-aware prompt construction for VLM trick identification."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
FIG_PATH = ROOT / "data" / "fig_tricks_2025.json"


def load_fig_tricks() -> dict:
    """Load the full FIG tricks database."""
    with open(FIG_PATH) as f:
        return json.load(f)


def format_trick_table(fig_data: dict, category: str | None = None) -> str:
    """Format FIG tricks as a compact table for the prompt.

    Args:
        fig_data: Loaded fig_tricks_2025.json.
        category: If set, only include this category. None = all.

    Returns:
        Markdown table of tricks.
    """
    lines = ["| Name | Flips | Twists | Direction | Category |"]
    lines.append("|------|-------|--------|-----------|----------|")

    categories = fig_data.get("categories", {})
    for cat_key, cat_data in categories.items():
        if category and cat_key != category:
            continue
        label = cat_data.get("label", cat_key)
        for trick in cat_data.get("tricks", []):
            name = trick["name"]
            flips = trick.get("flip", 0)
            twists = trick.get("twist", 0)
            direction = trick.get("direction", "-") or "-"
            aliases = trick.get("aliases", [])
            alias_str = f" (aka {', '.join(aliases)})" if aliases else ""
            lines.append(
                f"| {name}{alias_str} | {flips} | {twists} | {direction} | {label} |"
            )

    return "\n".join(lines)


def format_disambiguation(fig_data: dict) -> str:
    """Format disambiguation hints for visually similar tricks."""
    disambiguation = fig_data.get("disambiguation_needed", {})
    if not disambiguation:
        return ""

    lines = ["\n## Disambiguation Guide (visually similar tricks):"]
    for key, group in disambiguation.items():
        if key.startswith("_"):
            continue
        physics = group.get("physics", {})
        desc = f"{physics.get('flip', '?')} flip, {physics.get('twist', '?')} twist, {physics.get('direction', '?')}"
        lines.append(f"\n### Same physics ({desc}):")
        for c in group.get("candidates", []):
            lines.append(f"- **{c['name']}**: distinguished by {c.get('distinguisher', '?')}")
        features = group.get("layer3_features", [])
        if features:
            lines.append(f"  Key features to look for: {', '.join(features)}")

    return "\n".join(lines)


def build_prompt(fig_data: dict, category: str | None = None) -> str:
    """Build the full VLM prompt for trick identification.

    Args:
        fig_data: Loaded FIG tricks database.
        category: Optional category filter.

    Returns:
        Complete prompt string.
    """
    trick_table = format_trick_table(fig_data, category)
    disambiguation = format_disambiguation(fig_data)

    return f"""You are an expert FIG parkour judge. Identify every trick in this competition run.

## FIG Trick Names (Code of Points 2025)
{trick_table}
{disambiguation}

## Rules
- List ALL tricks performed in chronological order. Only acrobatic tricks, vaults, and swing moves count — ignore running, jumping between obstacles, and transitions.
- Use ONLY trick names from the table above.
- Count rotations carefully: backflip = 1 flip, double backflip = 2 flips.
- Count twists: full = 1 twist (360°), half = 0.5 twist (180°).
- Distinguish gainers (forward momentum, backward flip) from backflips (standing backward takeoff).
- Pay attention to apparatus: wall contact = wall category, bar/rail = swing, open ground = acrobatics.

## Response Format
Return ONLY valid JSON, no markdown fences:
{{
  "tricks": [
    {{
      "trick_name": "exact FIG name",
      "category": "swing|wall|acrobatics|pk_basics",
      "direction": "forward|backward|side|null",
      "flip_count": 1.0,
      "twist_count": 0.0,
      "confidence": "high|medium|low",
      "reasoning": "brief explanation"
    }}
  ]
}}"""
