#!/usr/bin/env python3
"""Unified VLM benchmark for the PkVision preprint.

Reads a curated ``ground_truth.csv`` (produced by
``make_ground_truth_template.py`` and filled by hand) and runs every listed
POOL-B clip through every selected model. Writes one JSON record per
(clip, model) pair to ``results.jsonl`` so all tables and figures can be
regenerated from a single source of truth.

Supported models:
    - gemini-2.5-pro / gemini-2.5-flash (direct Google API, GOOGLE_API_KEY)
    - openrouter/<slug>                  (OpenRouter, OPENROUTER_API_KEY)
    - claude                             (local ``claude -p`` CLI subprocess)
    - clip                               (local CLIP zero-shot baseline)

Usage
-----
    python paper/experiments/run_benchmark.py \
        --models gemini-2.5-flash,gemini-2.5-pro,openrouter/qwen/qwen2.5-vl-72b-instruct \
        --pool POOL-B \
        --out paper/experiments/results.jsonl

    python paper/experiments/run_benchmark.py --models claude --pool POOL-B

    # Dry run: no model calls, just print the plan.
    python paper/experiments/run_benchmark.py --models gemini-2.5-flash --dry-run
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(REPO / ".env")

from core.vlm.base import VLMResponse, VLMTrickResult  # noqa: E402
from core.vlm.fig_matcher import FIGMatcher  # noqa: E402
from core.vlm.prompt import build_prompt, load_fig_tricks  # noqa: E402


# --------------------------------------------------------------------------- #
# Ground-truth loading
# --------------------------------------------------------------------------- #

@dataclass
class GTRow:
    clip_path: Path
    pool: str
    fig_name: str
    d_score: float
    flip_count: float
    twist_count: float
    direction: str
    takeoff: str
    notes: str

    @classmethod
    def from_csv(cls, row: dict[str, str]) -> "GTRow | None":
        fig = row["fig_name"].strip()
        if not fig:
            return None
        def _f(k: str) -> float:
            v = row.get(k, "").strip()
            return float(v) if v else 0.0
        return cls(
            clip_path=REPO / row["clip_path"].strip(),
            pool=row["pool"].strip(),
            fig_name=fig,
            d_score=_f("d_score"),
            flip_count=_f("flip_count"),
            twist_count=_f("twist_count"),
            direction=row.get("direction", "").strip(),
            takeoff=row.get("takeoff", "").strip(),
            notes=row.get("notes", "").strip(),
        )


def load_ground_truth(path: Path, pools: list[str]) -> list[GTRow]:
    rows: list[GTRow] = []
    with path.open() as f:
        for raw in csv.DictReader(f):
            gt = GTRow.from_csv(raw)
            if gt is None:
                continue
            if gt.pool not in pools:
                continue
            if not gt.clip_path.exists():
                print(f"[warn] missing clip {gt.clip_path}", file=sys.stderr)
                continue
            rows.append(gt)
    return rows


# --------------------------------------------------------------------------- #
# Model runners
# --------------------------------------------------------------------------- #

@dataclass
class ModelCall:
    model: str
    mode: str
    raw_text: str
    tricks: list[VLMTrickResult] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0
    error: str = ""


def run_gemini(video: Path, model: str, max_retries: int = 3) -> ModelCall:
    from core.vlm.gemini_provider import GeminiProvider
    t0 = time.time()
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            resp = GeminiProvider(model=model).analyze_trick(video)
            return ModelCall(
                model=f"gemini/{model}",
                mode="native-video",
                raw_text=resp.raw_text,
                tricks=resp.tricks,
                input_tokens=resp.input_tokens,
                output_tokens=resp.output_tokens,
                latency_s=time.time() - t0,
            )
        except Exception as exc:
            last_exc = exc
            msg = str(exc)
            # Retry on transient Google API errors.
            if any(code in msg for code in ("503", "UNAVAILABLE", "429",
                                             "RESOURCE_EXHAUSTED", "500",
                                             "INTERNAL")):
                wait = 2 ** attempt
                time.sleep(min(wait, 8))
                continue
            break
    return ModelCall(
        model=f"gemini/{model}", mode="native-video",
        raw_text="", error=str(last_exc) if last_exc else "unknown",
        latency_s=time.time() - t0,
    )


def run_openrouter(video: Path, model_slug: str) -> ModelCall:
    from core.vlm.openrouter_provider import OpenRouterProvider
    t0 = time.time()
    try:
        provider = OpenRouterProvider(model=model_slug)
        resp = provider.analyze_trick(video)
        mode = "native-video" if provider._supports_video else "frames"
        return ModelCall(
            model=f"openrouter/{model_slug}",
            mode=mode,
            raw_text=resp.raw_text,
            tricks=resp.tricks,
            input_tokens=resp.input_tokens,
            output_tokens=resp.output_tokens,
            latency_s=time.time() - t0,
        )
    except Exception as exc:
        return ModelCall(model=f"openrouter/{model_slug}", mode="frames",
                         raw_text="", error=str(exc), latency_s=time.time() - t0)


def _extract_frames_to_dir(video: Path, out_dir: Path, num_frames: int = 8) -> list[Path]:
    """Extract `num_frames` uniformly-spaced JPEG frames using ffmpeg."""
    out_dir.mkdir(parents=True, exist_ok=True)
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(video)],
        capture_output=True, text=True,
    )
    try:
        dur = float(probe.stdout.strip())
    except ValueError:
        dur = 2.0
    ts = [dur * (i + 0.5) / num_frames for i in range(num_frames)]
    frame_paths: list[Path] = []
    for i, t in enumerate(ts):
        fp = out_dir / f"frame_{i:02d}.jpg"
        subprocess.run(
            ["ffmpeg", "-y", "-ss", f"{t:.3f}", "-i", str(video),
             "-frames:v", "1", "-q:v", "3", "-vf", "scale=768:-1", str(fp)],
            capture_output=True,
        )
        if fp.exists():
            frame_paths.append(fp)
    return frame_paths


def _build_frame_strip(video: Path, out_path: Path, num_frames: int = 8) -> Path | None:
    """Compose `num_frames` uniformly-spaced frames into a single horizontal strip.

    This matches the methodology that worked in the initial PkVision VLM
    experiments: one composite image preserves temporal sequencing visually,
    which we found critical for short rotational clips where per-frame views
    cause hallucination.
    """
    import cv2
    import numpy as np
    cap = cv2.VideoCapture(str(video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return None
    idxs = np.linspace(0, total - 1, num_frames, dtype=int)
    frames = []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if ok:
            h, w = frame.shape[:2]
            # normalize height to 360
            new_w = int(w * 360 / h)
            frame = cv2.resize(frame, (new_w, 360))
            frames.append(frame)
    cap.release()
    if not frames:
        return None
    # Label each frame with its index (top-left corner).
    for i, f in enumerate(frames):
        cv2.putText(f, f"{i+1}", (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(f, f"{i+1}", (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (0, 0, 0), 1, cv2.LINE_AA)
    strip = np.hstack(frames)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), strip, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return out_path


def run_claude_cli(video: Path, mode: str = "strip") -> ModelCall:
    """Run Claude Code CLI in print mode.

    The CLI is agentic: we give it an absolute image path (a composite
    frame-strip by default) and it uses its built-in Read tool to view it.
    No API keys consumed --- uses the user's Claude Code subscription.

    Args:
        mode: "strip" (default) = single composite image of 8 frames;
              "frames"          = 8 individual frame files.
    """
    t0 = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)

        if mode == "strip":
            strip_path = _build_frame_strip(video, tmp_dir / "strip.jpg", 8)
            if strip_path is None:
                return ModelCall(model="claude-cli", mode="strip",
                                 raw_text="", error="frame strip build failed",
                                 latency_s=time.time() - t0)
            read_targets = [strip_path]
            intro = (
                f"Read the composite image at {strip_path} using the Read tool. "
                f"It is a horizontal strip of 8 frames from a parkour clip, "
                f"labeled 1..8 in temporal order. Treat the strip as a short "
                f"sequence; the trick happens across consecutive frames.\n\n"
            )
        else:
            frames = _extract_frames_to_dir(video, tmp_dir / "frames")
            if not frames:
                return ModelCall(model="claude-cli", mode="frames",
                                 raw_text="", error="ffmpeg extracted no frames",
                                 latency_s=time.time() - t0)
            read_targets = frames
            frame_lines = "\n".join(f"- {fp}" for fp in frames)
            intro = (
                f"Read each of the following {len(frames)} JPEG image files "
                f"in chronological order using the Read tool:\n{frame_lines}\n\n"
            )

        fig_data = load_fig_tricks()
        fig_prompt = build_prompt(fig_data)
        wrapper_prompt = (
            intro
            + "After viewing, complete the task below. Output ONLY the "
              "requested JSON as your final message, with no preamble, no "
              "explanation outside the JSON, and no markdown fences.\n\n"
              "Task:\n" + fig_prompt
        )

        cmd = [
            "claude", "-p", wrapper_prompt,
            "--model", "sonnet",
            "--output-format", "json",
            "--add-dir", str(tmp_dir),
            "--permission-mode", "acceptEdits",
            "--allowedTools", "Read",
        ]
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=180,
            )
        except subprocess.TimeoutExpired:
            return ModelCall(model="claude-cli", mode="frames",
                             raw_text="", error="timeout",
                             latency_s=time.time() - t0)
        if proc.returncode != 0:
            return ModelCall(
                model="claude-cli", mode=mode, raw_text=proc.stdout,
                error=f"exit {proc.returncode}: {proc.stderr[:200]}",
                latency_s=time.time() - t0,
            )

        # Claude -p --output-format json returns a wrapper object with a
        # "result" field containing the assistant text.
        raw = proc.stdout.strip()
        assistant_text = raw
        try:
            wrapper = json.loads(raw)
            assistant_text = (
                wrapper.get("result")
                or wrapper.get("response")
                or wrapper.get("content")
                or raw
            )
            usage = wrapper.get("usage", {}) or {}
            in_tok = int(usage.get("input_tokens", 0) or 0)
            out_tok = int(usage.get("output_tokens", 0) or 0)
        except json.JSONDecodeError:
            in_tok = 0
            out_tok = 0

        # Parse the JSON the prompt asked for.
        from core.vlm.openrouter_provider import _parse_tricks_json  # reuse robust parser
        tricks_data = _parse_tricks_json(assistant_text)
        tricks = [
            VLMTrickResult(
                trick_name=t.get("trick_name", "Unknown"),
                category=t.get("category"),
                direction=t.get("direction"),
                flip_count=float(t.get("flip_count", 0) or 0),
                twist_count=float(t.get("twist_count", 0) or 0),
                confidence=t.get("confidence", "medium"),
                reasoning=t.get("reasoning", ""),
            )
            for t in tricks_data
        ]
        return ModelCall(
            model=f"claude-cli-{mode}",
            mode=mode,
            raw_text=assistant_text,
            tricks=tricks,
            input_tokens=in_tok,
            output_tokens=out_tok,
            latency_s=time.time() - t0,
        )


def run_gemini_cli(video: Path, model: str = "gemini-3.1-pro-preview",
                    mode: str = "strip") -> ModelCall:
    """Run the Gemini CLI in headless mode with a frame-strip or single frame.

    Uses the user's existing Gemini CLI subscription (no direct API key).
    Internally composes an 8-frame horizontal strip from the video, writes it
    to a temp JPEG, and asks Gemini to Read it and return FIG-grounded JSON.
    """
    t0 = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        if mode == "strip":
            strip_path = _build_frame_strip(video, tmp_dir / "strip.jpg", 8)
            if strip_path is None:
                return ModelCall(model=f"gemini-cli/{model}", mode="strip",
                                 raw_text="", error="strip build failed",
                                 latency_s=time.time() - t0)
            image_path = strip_path
            intro = (
                f"Use the Read tool to view the image at {strip_path}. It is "
                f"a horizontal strip of 8 sequential video frames (labeled "
                f"1..8) from a single parkour trick clip. Treat the frames "
                f"as a short temporal sequence.\n\n"
            )
        else:
            return ModelCall(model=f"gemini-cli/{model}", mode=mode,
                             raw_text="", error=f"unsupported mode {mode}",
                             latency_s=time.time() - t0)

        fig_data = load_fig_tricks()
        fig_prompt = build_prompt(fig_data)
        wrapper_prompt = (
            intro
            + "After viewing the image, complete the task below. Output ONLY "
              "the requested JSON as your final message, with no preamble, "
              "no explanation outside the JSON, and no markdown fences.\n\n"
              "Task:\n" + fig_prompt
        )

        cmd = [
            "gemini", "-m", model, "-y",
            "-p", wrapper_prompt,
            "-o", "json",
        ]
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=240,
                cwd=str(REPO),
            )
        except subprocess.TimeoutExpired:
            return ModelCall(model=f"gemini-cli/{model}", mode=mode,
                             raw_text="", error="timeout",
                             latency_s=time.time() - t0)
        if proc.returncode != 0:
            return ModelCall(
                model=f"gemini-cli/{model}", mode=mode, raw_text=proc.stdout,
                error=f"exit {proc.returncode}: {proc.stderr[:200]}",
                latency_s=time.time() - t0,
            )

        # Parse the Gemini CLI JSON wrapper.
        raw = proc.stdout
        idx = raw.find('{"session_id')
        if idx < 0:
            idx = raw.find("{")
        assistant_text = raw
        in_tok = 0
        out_tok = 0
        latency_ms = 0
        if idx >= 0:
            try:
                wrap = json.loads(raw[idx:])
                assistant_text = wrap.get("response", raw[idx:])
                stats = wrap.get("stats", {}).get("models", {}).get(model, {})
                in_tok = stats.get("tokens", {}).get("input", 0) or 0
                out_tok = stats.get("tokens", {}).get("candidates", 0) or 0
                latency_ms = stats.get("api", {}).get("totalLatencyMs", 0) or 0
            except json.JSONDecodeError:
                pass

        from core.vlm.openrouter_provider import _parse_tricks_json
        tricks_data = _parse_tricks_json(assistant_text)
        tricks = [
            VLMTrickResult(
                trick_name=t.get("trick_name", "Unknown"),
                category=t.get("category"),
                direction=t.get("direction"),
                flip_count=float(t.get("flip_count", 0) or 0),
                twist_count=float(t.get("twist_count", 0) or 0),
                confidence=t.get("confidence", "medium"),
                reasoning=t.get("reasoning", ""),
            )
            for t in tricks_data
        ]
        return ModelCall(
            model=f"gemini-cli/{model}",
            mode=mode,
            raw_text=assistant_text,
            tricks=tricks,
            input_tokens=in_tok,
            output_tokens=out_tok,
            latency_s=(latency_ms / 1000.0) if latency_ms else (time.time() - t0),
        )


def run_clip(video: Path, fig_names: list[str]) -> ModelCall:
    """CLIP ViT-B/32 zero-shot baseline on a single clip.

    We average frame embeddings, then score each FIG name with a simple
    caption template. No training, no prompt engineering beyond the template.
    """
    t0 = time.time()
    try:
        import cv2
        import numpy as np
        import torch
        import clip  # type: ignore
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load("ViT-B/32", device=device)

        cap = cv2.VideoCapture(str(video))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        idxs = np.linspace(0, max(total - 1, 0), 8, dtype=int)
        image_features: list[torch.Tensor] = []
        for idx in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if not ok:
                continue
            from PIL import Image
            pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            x = preprocess(pil).unsqueeze(0).to(device)
            with torch.no_grad():
                feat = model.encode_image(x)
            image_features.append(feat)
        cap.release()
        if not image_features:
            return ModelCall(model="clip-vit-b32", mode="frames", raw_text="",
                             error="no frames read", latency_s=time.time() - t0)

        img_mean = torch.cat(image_features).mean(dim=0, keepdim=True)
        img_mean = img_mean / img_mean.norm(dim=-1, keepdim=True)

        captions = [f"a video of a {name.lower()} parkour trick" for name in fig_names]
        with torch.no_grad():
            text_tok = clip.tokenize(captions).to(device)
            text_feats = model.encode_text(text_tok)
            text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
            sims = (img_mean @ text_feats.T).squeeze(0)
        best = int(sims.argmax().item())
        return ModelCall(
            model="clip-vit-b32",
            mode="frames",
            raw_text=fig_names[best],
            tricks=[VLMTrickResult(trick_name=fig_names[best], confidence="medium")],
            latency_s=time.time() - t0,
        )
    except Exception as exc:
        return ModelCall(model="clip-vit-b32", mode="frames", raw_text="",
                         error=str(exc), latency_s=time.time() - t0)


# --------------------------------------------------------------------------- #
# Dispatch
# --------------------------------------------------------------------------- #

def dispatch(model_id: str, video: Path, fig_names: list[str]) -> ModelCall:
    # Order matters: check specific CLI backends before the generic
    # "gemini-" API prefix.
    if model_id == "gemini-cli-pro":
        return run_gemini_cli(video, model="gemini-3.1-pro-preview")
    if model_id == "gemini-cli-flash":
        return run_gemini_cli(video, model="gemini-3.1-flash-lite-preview")
    if model_id.startswith("gemini-cli/"):
        return run_gemini_cli(video, model=model_id[len("gemini-cli/"):])
    if model_id.startswith("gemini-"):
        return run_gemini(video, model_id)
    if model_id.startswith("openrouter/"):
        return run_openrouter(video, model_id[len("openrouter/"):])
    if model_id == "claude" or model_id == "claude-strip":
        return run_claude_cli(video, mode="strip")
    if model_id == "claude-frames":
        return run_claude_cli(video, mode="frames")
    if model_id == "clip":
        return run_clip(video, fig_names)
    raise ValueError(f"Unknown model: {model_id!r}")


# --------------------------------------------------------------------------- #
# Main loop
# --------------------------------------------------------------------------- #

def record_from_call(
    gt: GTRow,
    call: ModelCall,
    matcher: FIGMatcher,
) -> dict[str, Any]:
    parsed = call.tricks[0] if call.tricks else None
    match = None
    if parsed:
        match = matcher.match(
            parsed.trick_name,
            flip_count=parsed.flip_count,
            twist_count=parsed.twist_count,
            direction=parsed.direction,
        )

    return {
        "clip_id": gt.clip_path.stem,
        "clip_path": str(gt.clip_path.relative_to(REPO)),
        "pool": gt.pool,
        "ground_truth": {
            "fig_name": gt.fig_name,
            "d_score": gt.d_score,
            "attributes": {
                "flip": gt.flip_count,
                "twist": gt.twist_count,
                "direction": gt.direction,
                "takeoff": gt.takeoff,
            },
        },
        "model": call.model,
        "mode": call.mode,
        "raw_output": call.raw_text,
        "parsed_trick": parsed.trick_name if parsed else "",
        "parsed_attributes": {
            "flip": parsed.flip_count if parsed else 0.0,
            "twist": parsed.twist_count if parsed else 0.0,
            "direction": parsed.direction if parsed else "",
            "confidence": parsed.confidence if parsed else "",
        },
        "fig_match": None if match is None else {
            "name": match.fig_name,
            "d_score": match.d_score,
            "category": match.category,
            "level": match.match_level,
            "confidence": match.confidence,
        },
        "input_tokens": call.input_tokens,
        "output_tokens": call.output_tokens,
        "latency_s": round(call.latency_s, 2),
        "error": call.error,
        "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--models",
        default="gemini-2.5-flash",
        help="Comma-separated model IDs (gemini-2.5-pro, "
             "openrouter/<slug>, claude, clip).",
    )
    ap.add_argument(
        "--pool",
        action="append",
        default=None,
        help="Repeatable. Ground-truth pool to include (POOL-A, POOL-B, ...). "
             "Default: POOL-B.",
    )
    ap.add_argument(
        "--gt",
        type=Path,
        default=REPO / "paper" / "experiments" / "ground_truth.csv",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=REPO / "paper" / "experiments" / "results.jsonl",
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    pools = args.pool or ["POOL-B"]
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    gt_rows = load_ground_truth(args.gt, pools)
    print(f"Loaded {len(gt_rows)} ground-truth rows from {args.gt.relative_to(REPO)}")
    print(f"Models: {models}")
    print(f"Pools:  {pools}")

    if args.dry_run:
        for gt in gt_rows:
            for m in models:
                print(f"  would run {m} on {gt.clip_path.relative_to(REPO)} -> GT={gt.fig_name}")
        return

    matcher = FIGMatcher()
    fig_names = [t.name for t in matcher.get_all_tricks()]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("a") as sink:
        for gt in gt_rows:
            for m in models:
                print(f"\n[{m}] {gt.clip_path.name}  (GT: {gt.fig_name})")
                call = dispatch(m, gt.clip_path, fig_names)
                if call.error:
                    print(f"  ! {call.error[:120]}")
                else:
                    parsed = call.tricks[0].trick_name if call.tricks else "<empty>"
                    print(f"  -> {parsed}  ({call.latency_s:.1f}s)")
                rec = record_from_call(gt, call, matcher)
                sink.write(json.dumps(rec) + "\n")
                sink.flush()

    try:
        rel_out = args.out.relative_to(REPO)
    except ValueError:
        rel_out = args.out
    print(f"\nDone. Appended to {rel_out}")


if __name__ == "__main__":
    main()
