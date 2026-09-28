#!/usr/bin/env python3
"""Test dual-POV trick identification with focused VLM prompt."""

import sys, os, json, base64, time, re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from openai import OpenAI

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
)

# Encode both videos
videos = {}
for name in ["1", "2"]:
    path = ROOT / f"data/run_testing/double-pov/{name}.mp4"
    with open(path, "rb") as f:
        videos[name] = base64.b64encode(f.read()).decode("utf-8")

prompt = (
    "You are analyzing a parkour competition run from TWO different camera angles.\n"
    "\n"
    "VIDEO 1 = Camera angle 1\n"
    "VIDEO 2 = Camera angle 2 (same run, different perspective)\n"
    "\n"
    "## Your Task\n"
    "Watch BOTH angles carefully. For each acrobatic trick (ignore running, climbing, vaults, transitions):\n"
    "\n"
    "1. COUNT the exact number of complete backward/forward rotations (flips) — inverted = halfway\n"
    "2. COUNT the exact number of longitudinal rotations (twists) — watch chest/shoulders spinning\n"
    "3. Note the TAKEOFF: standing two-foot? running forward (gainer)? one-leg? off a wall? bar swing?\n"
    "4. Note the AXIS: straight lateral flip (backflip), off-axis/tilted (cork family), sagittal (sideflip)\n"
    "5. For off-axis tricks: which direction does the twist go relative to the flip?\n"
    "\n"
    "## Visual Cues for Similar Tricks\n"
    "- BACKFLIP vs GAINER: Both backward 1-flip. Gainer = running FORWARD then flipping backward.\n"
    "- GAINER vs GAINER FULL: Gainer = 0 twists. Gainer Full = 1 full twist (360) during the flip.\n"
    "- CORK vs KROK: Both off-axis ~0.5 twist. Cork = standard rotation. Krok = REVERSE rotation.\n"
    "- DOUBLE CORK: 2 flips + 1 twist, off-axis. Very fast, tight tuck, high in the air.\n"
    "- DOUBLE FULL (Backflip 720): 1 flip + 2 full twists. Body spins twice around long axis during one flip.\n"
    "- B-TWIST: Body stays nearly horizontal, does a full twist. Half-flip + 1 twist.\n"
    "- SIDEFLIP: Lateral rotation (body goes sideways), no twist.\n"
    "- WALL tricks: Foot/hand touches the wall during takeoff.\n"
    "\n"
    "Use BOTH camera angles to cross-check your rotation counts. If angle 1 makes it look like 1 flip\n"
    "but angle 2 reveals 2 flips, trust the angle with better visibility.\n"
    "\n"
    "## Response Format\n"
    "Return ONLY valid JSON, no markdown fences:\n"
    "{\n"
    '  "tricks": [\n'
    "    {\n"
    '      "trick_name": "descriptive name (e.g. Gainer Full, Double Cork, B-Twist)",\n'
    '      "flip_count": 1.0,\n'
    '      "twist_count": 0.0,\n'
    '      "direction": "forward/backward/side",\n'
    '      "takeoff": "standing/running-forward/one-leg/wall/bar-swing",\n'
    '      "axis": "lateral/off-axis/sagittal",\n'
    '      "confidence": "high/medium/low",\n'
    '      "reasoning": "what I saw in both angles"\n'
    "    }\n"
    "  ]\n"
    "}"
)

content = [
    {"type": "text", "text": "Camera angle 1:"},
    {"type": "video_url", "video_url": {"url": f"data:video/mp4;base64,{videos['1']}"}},
    {"type": "text", "text": "Camera angle 2:"},
    {"type": "video_url", "video_url": {"url": f"data:video/mp4;base64,{videos['2']}"}},
    {"type": "text", "text": prompt},
]

print("Sending both POVs to Gemini 3.1 Pro...")
t0 = time.time()
response = client.chat.completions.create(
    model="google/gemini-3.1-pro-preview",
    messages=[{"role": "user", "content": content}],
    max_tokens=16384,
)
elapsed = time.time() - t0
raw = response.choices[0].message.content or ""
usage = response.usage
print(f"Done in {elapsed:.1f}s | {usage.prompt_tokens}in + {usage.completion_tokens}out tokens")
print()

# Parse
cleaned = re.sub(r"```(?:json)?", "", raw).strip()
try:
    data = json.loads(cleaned)
    tricks = data.get("tricks", [])
except json.JSONDecodeError:
    tricks = []
    for m in re.finditer(r"\{[^{}]*?\"trick_name\"[^{}]*?\}", cleaned, re.DOTALL):
        try:
            tricks.append(json.loads(m.group()))
        except:
            pass

print(f"Tricks found: {len(tricks)}")
for i, t in enumerate(tricks):
    print(f"  {i+1}. {t.get('trick_name', '?')}")
    print(f"     {t.get('flip_count','?')}f {t.get('twist_count','?')}t {t.get('direction','?')}")
    print(f"     takeoff: {t.get('takeoff','?')} | axis: {t.get('axis','?')} [{t.get('confidence','?')}]")
    print(f"     {t.get('reasoning','')[:200]}")
    print()

if not tricks:
    print("Parse failed. Raw response:")
    print(raw[:1000])
