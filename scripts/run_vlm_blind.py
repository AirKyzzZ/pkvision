#!/usr/bin/env python3
"""Runner for the VLM blind probe: sends the anonymised inputs to each arm and appends votes.

  run_vlm_blind.py plan  --labels L --labels-sha256 H [--control-ids C] [--arms ...]   what would be sent; no call
  run_vlm_blind.py run   --labels L --labels-sha256 H [--control-ids C] [--arms ...] --cap-usd 5   the calls
  run_vlm_blind.py run   ... --provider fake                                             end-to-end rehearsal, $0
  run_vlm_blind.py list-models                                                           what each key can reach

Rules this enforces, from docs/science-superpowers/probes/2026-11-04-vlm-blind-prereg-DRAFT.md:
- A clip is sent forward only once the owner's label for it is final: the labels file must match
  --labels-sha256, and that hash is written into every vote row so the scorer can verify it.
- Reversed clips are sent only for the ids `score_vlm_blind.py control-ids` printed, so only at the end.
- Votes are append-only. An answered (arm, id, variant) is never requested again; a failed call may be retried.
- Spend stops at --cap-usd, counted from the usage each reply reports at the frozen prices below.
- Only files under vlm_inputs/upload leave the machine. Nothing here prints a clip name.
- No model fallback: a refusal or an error is a missing vote, never an answer from another model.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.prepare_vlm_blind import QUEUE, UPLOAD, file_id, opaque_id  # noqa: E402
from scripts.score_vlm_blind import ARMS, parse_reply, split_labels  # noqa: E402

VOTES = ROOT / "data/labeling/retest_2026-11/votes.jsonl"
GRID_SHEETS = 2

PROMPT = """This shows one person performing an acrobatic movement, in time order. Grids, when used, are numbered frames.
Track the person's body orientation frame by frame. Do not try to name the movement.
Count, for the main airborne movement only:
- flip: full head-over-heels rotations, in steps of 0.5
- twist: full rotations about the head-to-toe axis, in steps of 0.5
- direction: the flip direction at takeoff relative to the chest: "forward", "backward", "side", or null if there is no flip
- cork: true if the rotation axis is off-axis, as in a corkscrew
Set unsure to true if you cannot see enough to count.
Reply with JSON only: {"flip": number, "twist": number, "direction": string or null, "cork": boolean, "unsure": boolean}"""

# Frozen arm settings. Prices are $ per 1M tokens (input, output) from the September survey and the
# Anthropic price table; `list-models` is for checking the ids exist before the freeze.
ARM_SPECS = {
    "G-pro": {"provider": "gemini", "model": "gemini-3.1-pro-preview", "input": "video", "fps": 10,
              "resolution": "medium", "price": (2.00, 12.00)},
    "G-flash": {"provider": "gemini", "model": "gemini-3.8-flash", "input": "video", "fps": 10,
                "resolution": "medium", "price": (0.75, 3.75)},
    "G-grid": {"provider": "gemini", "model": "gemini-3.8-flash", "input": "grid", "price": (0.75, 3.75)},
    "C-grid": {"provider": "anthropic", "model": "claude-opus-5-5", "input": "grid", "effort": "medium",
               "max_tokens": 1024, "price": (4.00, 20.00)},
}
assert set(ARM_SPECS) == set(ARMS)


def cost_usd(spec: dict, usage: dict) -> float:
    price_in, price_out = spec["price"]
    return (usage.get("input_tokens", 0) * price_in + usage.get("output_tokens", 0) * price_out) / 1e6


def read_votes(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def answered(rows: list[dict]) -> set[tuple[str, str, str]]:
    return {(r["arm"], r["id"], r["variant"]) for r in rows if "twist" in r}


def plan(labels: dict, queue_order: list[str], rows: list[dict], control_ids: list[str] | None, arms: list[str]) -> list[dict]:
    """Calls still to make. Forward for every scored clip; reversed only for the given control ids."""
    truth, _ = split_labels(labels, set(queue_order))
    done = answered(rows)
    by_id = {opaque_id(s): s for s in queue_order}
    unknown = set(control_ids or []) - set(by_id)
    if unknown:
        raise ValueError(f"{len(unknown)} control ids are not queue clips")
    calls = []
    for arm in arms:
        for slug in sorted(truth, key=opaque_id):
            cid = opaque_id(slug)
            if (arm, cid, "fwd") not in done:
                calls.append({"arm": arm, "id": cid, "variant": "fwd", "file": file_id(slug, "fwd")})
        for cid in control_ids or []:
            slug = by_id[cid]
            if slug not in truth:
                raise ValueError("a control id is not a scored clip")
            if (arm, cid, "rev") not in done:
                calls.append({"arm": arm, "id": cid, "variant": "rev", "file": file_id(slug, "rev")})
    return calls


def inputs_for(spec: dict, fid: str, upload: Path | None = None) -> list[Path]:
    upload = upload or UPLOAD
    if spec["input"] == "video":
        paths = [upload / f"{fid}.mp4"]
    else:
        paths = [upload / f"{fid}_grid{n}.jpg" for n in range(1, GRID_SHEETS + 1)]
    missing = [p.name for p in paths if not p.is_file()]
    if missing:
        raise ValueError(f"missing upload files: {missing}; run prepare_vlm_blind.py")
    return paths


class FakeProvider:
    """Deterministic plumbing test. Answers depend on the file name only, never on labels."""

    def __init__(self):
        self.calls = 0

    def ask(self, spec: dict, paths: list[Path]) -> tuple[str, dict]:
        self.calls += 1
        h = int(hashlib.sha1(paths[0].name.encode()).hexdigest(), 16)
        reply = {"flip": (h % 5) / 2, "twist": ((h // 5) % 7) / 2, "direction": ["forward", "backward"][h % 2],
                 "cork": bool((h // 35) % 2), "unsure": False}
        return json.dumps(reply), {"input_tokens": 1000, "output_tokens": 40}


class GeminiProvider:
    def __init__(self):
        from google import genai  # pip install google-genai
        self.genai = genai
        self.types = genai.types
        self.client = genai.Client()  # reads GEMINI_API_KEY or GOOGLE_API_KEY

    def list_models(self) -> list[str]:
        return sorted(m.name for m in self.client.models.list())

    def ask(self, spec: dict, paths: list[Path]) -> tuple[str, dict]:
        t = self.types
        if spec["input"] == "video":
            part = t.Part.from_bytes(data=paths[0].read_bytes(), mime_type="video/mp4")
            part.video_metadata = t.VideoMetadata(fps=spec["fps"])
            parts = [part]
        else:
            parts = [t.Part.from_bytes(data=p.read_bytes(), mime_type="image/jpeg") for p in paths]
        config = t.GenerateContentConfig(response_mime_type="application/json",
                                         media_resolution=f"MEDIA_RESOLUTION_{spec.get('resolution', 'medium').upper()}")
        response = self.client.models.generate_content(model=spec["model"], contents=parts + [PROMPT], config=config)
        meta = response.usage_metadata
        usage = {"input_tokens": getattr(meta, "prompt_token_count", 0) or 0,
                 "output_tokens": getattr(meta, "candidates_token_count", 0) or 0}
        return response.text or "", usage


class AnthropicProvider:
    def __init__(self):
        import anthropic  # pip install anthropic
        self.anthropic = anthropic
        self.client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY

    def list_models(self) -> list[str]:
        return sorted(m.id for m in self.client.models.list())

    def ask(self, spec: dict, paths: list[Path]) -> tuple[str, dict]:
        images = [{"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                               "data": base64.standard_b64encode(p.read_bytes()).decode()}} for p in paths]
        response = self.client.messages.create(
            model=spec["model"], max_tokens=spec["max_tokens"], output_config={"effort": spec["effort"]},
            messages=[{"role": "user", "content": images + [{"type": "text", "text": PROMPT}]}])
        usage = {"input_tokens": response.usage.input_tokens, "output_tokens": response.usage.output_tokens}
        if response.stop_reason == "refusal":
            raise RuntimeError("refusal")  # a missing vote, deliberately with no fallback model
        text = "".join(b.text for b in response.content if b.type == "text")
        return text, usage


PROVIDERS = {"fake": FakeProvider, "gemini": GeminiProvider, "anthropic": AnthropicProvider}


def run(calls: list[dict], labels_hash: str, votes_path: Path, cap_usd: float, provider_name: str | None,
        limit: int | None = None, upload: Path | None = None) -> dict:
    """Make the calls in order, appending one row each. Stops at the cap; a failure is a row without an answer."""
    providers: dict[str, object] = {}
    spent = sum(r.get("cost_usd", 0.0) for r in read_votes(votes_path))
    summary = {"made": 0, "answered": 0, "failed": 0, "malformed": 0, "spent_usd": spent, "stopped": None}
    votes_path.parent.mkdir(parents=True, exist_ok=True)
    with votes_path.open("a") as out:
        for call in calls[:limit]:
            if spent >= cap_usd:
                summary["stopped"] = f"cap of ${cap_usd:.2f} reached"
                break
            spec = ARM_SPECS[call["arm"]]
            name = provider_name or spec["provider"]
            provider = providers.setdefault(name, PROVIDERS[name]())
            row = {"arm": call["arm"], "id": call["id"], "variant": call["variant"], "labels_sha256": labels_hash,
                   "model": spec["model"] if name != "fake" else "fake", "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}
            try:
                text, usage = provider.ask(spec, inputs_for(spec, call["file"], upload))
                row["usage"] = usage
                row["cost_usd"] = 0.0 if name == "fake" else round(cost_usd(spec, usage), 6)
                spent += row["cost_usd"]
                try:
                    row.update(parse_reply(text))
                    summary["answered"] += 1
                except ValueError:
                    row.update({"flip": None, "twist": None, "direction": None, "cork": None, "unsure": None, "raw": text[:500]})
                    summary["malformed"] += 1  # a model failure, scored as wrong
            except Exception as error:  # noqa: BLE001 - any failure is a missing vote, retried later
                row["error"] = type(error).__name__ + ": " + str(error)[:200]
                summary["failed"] += 1
            out.write(json.dumps(row) + "\n")
            out.flush()
            summary["made"] += 1
    summary["spent_usd"] = round(spent, 4)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cmd", choices=("plan", "run", "list-models"))
    parser.add_argument("--labels", type=Path, help="the owner's labels file whose clips are final")
    parser.add_argument("--labels-sha256", help="hash of --labels recorded in the freeze commit")
    parser.add_argument("--control-ids", type=Path, help="JSON list printed by score_vlm_blind.py control-ids")
    parser.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    parser.add_argument("--votes", type=Path, default=VOTES)
    parser.add_argument("--upload", type=Path, default=UPLOAD, help="the anonymised inputs folder; nothing else is ever sent")
    parser.add_argument("--cap-usd", type=float, default=5.0)
    parser.add_argument("--provider", choices=list(PROVIDERS), help="override every arm's provider; `fake` for a rehearsal")
    parser.add_argument("--limit", type=int, help="make at most this many calls, for a live smoke test")
    args = parser.parse_args()

    if args.cmd == "list-models":
        for name in ("gemini", "anthropic"):
            try:
                print(name, json.dumps(PROVIDERS[name]().list_models(), indent=1))
            except Exception as error:  # noqa: BLE001
                print(name, "unavailable:", type(error).__name__)  # never echo a key
        return

    if not args.labels or not args.labels_sha256:
        parser.error(f"{args.cmd} needs --labels and --labels-sha256")
    raw = args.labels.read_bytes()
    labels_hash = hashlib.sha256(raw).hexdigest()
    if labels_hash != args.labels_sha256.lower():
        raise ValueError("labels file does not match the hash given; freeze the labels before sending clips")
    order = [c["slug"] for c in json.loads(QUEUE.read_text())]
    control_ids = json.loads(args.control_ids.read_text()) if args.control_ids else None
    rows = read_votes(args.votes)  # earlier stages' rows are allowed; the scorer checks every row's hash
    calls = plan(json.loads(raw), order, rows, control_ids, args.arms)
    per_arm = {arm: sum(c["arm"] == arm for c in calls) for arm in args.arms}
    if args.cmd == "plan":
        print(json.dumps({"calls": len(calls), "per_arm": per_arm,
                          "forward": sum(c["variant"] == "fwd" for c in calls),
                          "reversed": sum(c["variant"] == "rev" for c in calls),
                          "already_answered": len(answered(rows)), "labels_sha256": labels_hash}, indent=1))
        return
    if args.provider != "fake" and not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
                                        or os.environ.get("ANTHROPIC_API_KEY")):
        raise ValueError("no API key in the environment; use --provider fake for a rehearsal")
    if not args.upload.is_dir():
        raise ValueError(f"no upload folder at {args.upload}; run prepare_vlm_blind.py or pass --upload")
    summary = run(calls, labels_hash, args.votes, args.cap_usd, args.provider, args.limit, args.upload)
    summary["planned"] = len(calls)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    try:
        main()
    except ValueError as error:
        sys.exit(f"stopped: {error}")
