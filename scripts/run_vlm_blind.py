#!/usr/bin/env python3
"""Runner for the VLM blind probe: sends the anonymised inputs to each arm and appends votes.

  run_vlm_blind.py plan  --labels L --labels-sha256 H [--control-ids C] [--arms ...]   what would be sent; no call
  run_vlm_blind.py run   --labels L --labels-sha256 H [--control-ids C] [--arms ...] --cap-usd 15  the calls
  run_vlm_blind.py run   ... --provider fake                                             end-to-end rehearsal, $0

Rules this enforces, from docs/science-superpowers/probes/2026-11-04-vlm-blind-prereg.md:
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
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.prepare_vlm_blind import QUEUE, UPLOAD, file_id, opaque_id  # noqa: E402
from scripts.score_vlm_blind import ALL_ARMS, ARMS, parse_reply, split_labels  # noqa: E402

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

# Frozen arm settings. Every paid arm goes through OpenRouter. Prices are $ per 1M tokens (input, output)
# from its model list on 2026-11-26, used only when a reply carries no cost.
ARM_SPECS = {
    "G-pro": {"provider": "openrouter", "model": "google/gemini-3.1-pro-preview", "input": "video", "slowdown": 10,
              "price": (2.00, 12.00)},
    "G-flash": {"provider": "openrouter", "model": "google/gemini-3.8-flash", "input": "video", "slowdown": 10,
                "price": (0.83, 4.12)},
    "G-grid": {"provider": "openrouter", "model": "google/gemini-3.8-flash", "input": "grid", "price": (0.83, 4.12)},
    "C-grid": {"provider": "openrouter", "model": "anthropic/claude-opus-5.5", "input": "grid", "effort": "medium",
               "max_tokens": 8192, "price": (4.00, 20.00)},
    # Exploratory, no decision weight: a free 4-bit model on the RTX 2060, same grids and prompt, greedy decoding.
    "L-grid": {"provider": "local", "model": "Qwen/Qwen3-VL-4B-Instruct@nf4", "input": "grid", "price": (0.0, 0.0),
               "exploratory": True},
}
assert set(ARM_SPECS) == set(ALL_ARMS)
KEYLESS_PROVIDERS = ("fake", "local")


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
        paths = [upload / (f"{fid}_slow{spec['slowdown']}.mp4" if spec.get("slowdown") else f"{fid}.mp4")]
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


class OpenRouterProvider:
    URL = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(self, post=None):
        self.key = os.environ["OPENROUTER_API_KEY"]
        self.post = post or self._post

    def _post(self, body: dict) -> dict:
        request = urllib.request.Request(self.URL, json.dumps(body).encode(),
                                         {"Authorization": "Bearer " + self.key, "Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=300) as response:
            return json.load(response)

    @staticmethod
    def part(path: Path) -> dict:
        data = base64.standard_b64encode(path.read_bytes()).decode()
        if path.suffix == ".mp4":
            return {"type": "video_url", "video_url": {"url": "data:video/mp4;base64," + data}}
        return {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + data}}

    def ask(self, spec: dict, paths: list[Path]) -> tuple[str, dict]:
        body = {"model": spec["model"],
                "messages": [{"role": "user", "content": [self.part(p) for p in paths] + [{"type": "text", "text": PROMPT}]}],
                "provider": {"data_collection": "deny", "allow_fallbacks": False}}
        if "max_tokens" in spec:
            body["max_tokens"] = spec["max_tokens"]
        if "effort" in spec:
            body["reasoning"] = {"effort": spec["effort"]}
        reply = self.post(body)
        if "error" in reply:
            raise RuntimeError(f"gateway error {reply['error'].get('code')}")
        choice = reply["choices"][0]
        if "refusal" in (choice.get("finish_reason"), choice.get("native_finish_reason")):
            raise RuntimeError("refusal")  # a missing vote, deliberately with no fallback model
        used = reply.get("usage", {})
        usage = {"input_tokens": used.get("prompt_tokens", 0), "output_tokens": used.get("completion_tokens", 0),
                 "served_by": reply.get("provider")}
        if used.get("cost") is not None:
            usage["cost_usd"] = used["cost"]
        return choice["message"].get("content") or "", usage


class LocalProvider:
    """Qwen3-VL-4B in 4-bit on the local GPU; see scripts/local_vlm.py. Imported lazily: it loads torch."""

    def __init__(self):
        from scripts.local_vlm import LocalQwenProvider

        self.inner = LocalQwenProvider()

    def ask(self, spec: dict, paths: list[Path]) -> tuple[str, dict]:
        return self.inner.ask(spec, paths)


PROVIDERS = {"fake": FakeProvider, "openrouter": OpenRouterProvider, "local": LocalProvider}


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
                row["cost_usd"] = 0.0 if name == "fake" else round(usage.get("cost_usd", cost_usd(spec, usage)), 6)
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
    parser.add_argument("cmd", choices=("plan", "run"))
    parser.add_argument("--labels", type=Path, help="the owner's labels file whose clips are final")
    parser.add_argument("--labels-sha256", help="hash of --labels recorded in the freeze commit")
    parser.add_argument("--control-ids", type=Path, help="JSON list printed by score_vlm_blind.py control-ids")
    parser.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ALL_ARMS),
                        help="defaults to the four decision arms; L-grid is the free exploratory local arm")
    parser.add_argument("--votes", type=Path, default=VOTES)
    parser.add_argument("--upload", type=Path, default=UPLOAD, help="the anonymised inputs folder; nothing else is ever sent")
    parser.add_argument("--cap-usd", type=float, default=15.0)
    parser.add_argument("--provider", choices=list(PROVIDERS), help="override every arm's provider; `fake` for a rehearsal")
    parser.add_argument("--limit", type=int, help="make at most this many calls, for a live smoke test")
    args = parser.parse_args()

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
    needs_key = args.provider not in KEYLESS_PROVIDERS and any(
        ARM_SPECS[arm]["provider"] not in KEYLESS_PROVIDERS for arm in args.arms)
    if needs_key and not os.environ.get("OPENROUTER_API_KEY"):
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
