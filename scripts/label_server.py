#!/usr/bin/env python3
"""Web UI for fast trick labeling.

Launches a local web server showing each pending trick with:
- Preview image + video context
- CLIP suggestion
- One-click buttons for common labels
- Quick keyboard shortcuts

Usage:
    python scripts/label_server.py
    python scripts/label_server.py --port 8888
"""

from __future__ import annotations

import io
import json
import shutil
import sys
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PENDING_DIR = ROOT / "data" / "active_learning" / "pending"
CONFIRMED_DIR = ROOT / "data" / "active_learning" / "confirmed"

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>PkVision — Trick Labeler</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #0a0a0a; color: #e0e0e0; }
.container { max-width: 900px; margin: 0 auto; padding: 20px; }
h1 { font-size: 1.4em; margin-bottom: 5px; color: #fff; }
.stats { color: #888; margin-bottom: 20px; font-size: 0.9em; }
.progress-bar { height: 6px; background: #333; border-radius: 3px; margin: 10px 0; }
.progress-fill { height: 100%; background: #4CAF50; border-radius: 3px; transition: width 0.3s; }
.trick-card { background: #1a1a1a; border-radius: 12px; padding: 20px; margin-bottom: 15px; }
.trick-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; }
.trick-name { font-size: 1.1em; font-weight: 600; }
.trick-time { color: #888; font-size: 0.85em; }
.preview { text-align: center; margin: 15px 0; }
.preview img { max-width: 100%; max-height: 350px; border-radius: 8px; border: 2px solid #333; }
.suggestion { background: #222; padding: 10px 15px; border-radius: 8px; margin: 10px 0; font-size: 0.9em; }
.suggestion span { color: #4CAF50; font-weight: 600; }
.label-section { margin-top: 15px; }
.label-row { display: flex; gap: 8px; margin-bottom: 8px; align-items: center; }
.label-title { width: 80px; font-size: 0.85em; color: #888; flex-shrink: 0; }
.btn { padding: 8px 16px; border: 1px solid #444; border-radius: 6px; background: #2a2a2a;
       color: #e0e0e0; cursor: pointer; font-size: 0.85em; transition: all 0.15s; }
.btn:hover { background: #3a3a3a; border-color: #666; }
.btn.selected { background: #1a5e1a; border-color: #4CAF50; color: #fff; }
.btn.skip { background: #4a2a2a; border-color: #944; }
.btn.skip:hover { background: #5a3a3a; }
.actions { display: flex; gap: 10px; margin-top: 20px; justify-content: center; }
.btn-confirm { padding: 12px 40px; background: #2e7d32; border: none; border-radius: 8px;
               color: #fff; font-size: 1em; font-weight: 600; cursor: pointer; }
.btn-confirm:hover { background: #388e3c; }
.btn-confirm:disabled { opacity: 0.4; cursor: not-allowed; }
.btn-skip-trick { padding: 12px 30px; background: #555; border: none; border-radius: 8px;
                  color: #fff; font-size: 1em; cursor: pointer; }
.btn-skip-trick:hover { background: #666; }
.done { text-align: center; padding: 60px 20px; }
.done h2 { color: #4CAF50; margin-bottom: 10px; }
.keyboard-hint { color: #555; font-size: 0.75em; text-align: center; margin-top: 15px; }
</style>
</head>
<body>
<div class="container">
  <h1>PkVision — Trick Labeler</h1>
  <div class="stats" id="stats"></div>
  <div class="progress-bar"><div class="progress-fill" id="progress"></div></div>
  <div id="card"></div>
  <div class="keyboard-hint">
    Keyboard: 1-4 direction, 5-7 flip, Q/W/E/R twist, 8/9/0/- context, Enter=confirm, S=skip, arrows=navigate
  </div>
</div>
<script>
let tricks = [];
let currentIdx = 0;
let selections = {};

async function loadTricks() {
  const res = await fetch('/api/pending');
  tricks = await res.json();
  if (tricks.length === 0) {
    document.getElementById('card').innerHTML = '<div class="done"><h2>All done!</h2><p>No more tricks to label.</p></div>';
    return;
  }
  showTrick(0);
  updateStats();
}

function updateStats() {
  const total = tricks.length;
  const done = currentIdx;
  document.getElementById('stats').textContent =
    `${done}/${total} labeled — ${total - done} remaining`;
  document.getElementById('progress').style.width = `${(done / total) * 100}%`;
}

function showTrick(idx) {
  if (idx >= tricks.length) {
    document.getElementById('card').innerHTML =
      '<div class="done"><h2>All done!</h2><p>Reload to check for new tricks.</p></div>';
    return;
  }
  currentIdx = idx;
  const t = tricks[idx];
  const sug = t.suggested_label || {};
  const sDir = sug.direction?.value || '?';
  const sFlip = sug.flip_count?.value || '?';
  const sCtx = sug.context?.value || '?';

  // Pre-select CLIP suggestions
  selections = { direction: sDir, flip_count: sFlip, twist_count: '0', context: sCtx, trick_name: '' };

  document.getElementById('card').innerHTML = `
    <div class="trick-card">
      <div class="trick-header">
        <span class="trick-name">${t._video} / ${t.name}</span>
        <span class="trick-time">${t.start_s.toFixed(1)}s — ${t.end_s.toFixed(1)}s (${t.duration_s.toFixed(1)}s)</span>
      </div>
      <div class="preview">
        <video src="/video/${encodeURIComponent(t._session)}/${t.name}.mp4" autoplay loop muted playsinline
               style="max-width:100%;max-height:350px;border-radius:8px;border:2px solid #333;"></video>
      </div>
      <div class="suggestion">CLIP suggests: <span>${sDir}</span>, <span>${sFlip} flip</span>, <span>${sCtx}</span></div>
      <div class="label-section">
        <div class="label-row">
          <span class="label-title">Direction</span>
          ${makeButtons('direction', ['backward', 'forward', 'side', 'none'], ['1','2','3','4'])}
        </div>
        <div class="label-row">
          <span class="label-title">Flip count</span>
          ${makeButtons('flip_count', ['0', '1', '2+'], ['5','6','7'])}
        </div>
        <div class="label-row">
          <span class="label-title">Twist count</span>
          ${makeButtons('twist_count', ['0', '0.5', '1', '2+'], ['q','w','e','r'])}
        </div>
        <div class="label-row">
          <span class="label-title">Context</span>
          ${makeButtons('context', ['acrobatics', 'wall', 'swing', 'pk_basics'], ['8','9','0','-'])}
        </div>
        <div class="label-row">
          <span class="label-title">Trick name</span>
          <input type="text" id="trickNameInput" placeholder="e.g. backflip, gainer full, double cork..."
                 style="flex:1;padding:8px 12px;background:#2a2a2a;border:1px solid #444;border-radius:6px;color:#e0e0e0;font-size:0.85em;"
                 oninput="selections.trick_name=this.value">
        </div>
      </div>
      <div class="actions">
        <button class="btn-skip-trick" onclick="skipTrick()">Skip (S)</button>
        <button class="btn-confirm" id="confirmBtn" onclick="confirmTrick()">Confirm (Enter)</button>
      </div>
    </div>
  `;
  updateStats();
  updateButtonStates();
  // Don't auto-focus trick name input so keyboard shortcuts still work
}

function makeButtons(attr, values, keys) {
  return values.map((v, i) => {
    const sel = selections[attr] === v ? 'selected' : '';
    const key = keys[i] || '';
    return `<button class="btn ${sel}" data-attr="${attr}" data-val="${v}" onclick="selectAttr('${attr}','${v}')">${v} ${key ? '<small>('+key+')</small>' : ''}</button>`;
  }).join('');
}

function selectAttr(attr, val) {
  selections[attr] = val;
  updateButtonStates();
}

function updateButtonStates() {
  document.querySelectorAll('.btn[data-attr]').forEach(btn => {
    const attr = btn.dataset.attr;
    const val = btn.dataset.val;
    btn.classList.toggle('selected', selections[attr] === val);
  });
}

async function confirmTrick() {
  const t = tricks[currentIdx];
  await fetch('/api/label', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
      session: t._session,
      trick_name: t.name,
      label: selections,
      status: 'confirmed'
    })
  });
  showTrick(currentIdx + 1);
}

async function skipTrick() {
  const t = tricks[currentIdx];
  await fetch('/api/label', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
      session: t._session,
      trick_name: t.name,
      status: 'skipped'
    })
  });
  showTrick(currentIdx + 1);
}

// Keyboard shortcuts (disabled when typing in trick name input)
document.addEventListener('keydown', e => {
  const inInput = document.activeElement?.tagName === 'INPUT';
  const key = e.key;
  if (key === 'Enter') { confirmTrick(); e.preventDefault(); return; }
  if (inInput) return; // don't intercept when typing trick name
  if (key === 's' || key === 'S') { skipTrick(); e.preventDefault(); }
  else if (key === 'ArrowRight') { showTrick(currentIdx + 1); }
  else if (key === 'ArrowLeft' && currentIdx > 0) { showTrick(currentIdx - 1); }
  // Direction: 1-4
  else if (key === '1') selectAttr('direction', 'backward');
  else if (key === '2') selectAttr('direction', 'forward');
  else if (key === '3') selectAttr('direction', 'side');
  else if (key === '4') selectAttr('direction', 'none');
  // Flip: 5-7
  else if (key === '5') selectAttr('flip_count', '0');
  else if (key === '6') selectAttr('flip_count', '1');
  else if (key === '7') selectAttr('flip_count', '2+');
  // Twist: q,w,e,r
  else if (key === 'q') selectAttr('twist_count', '0');
  else if (key === 'w') selectAttr('twist_count', '0.5');
  else if (key === 'e') selectAttr('twist_count', '1');
  else if (key === 'r') selectAttr('twist_count', '2+');
  // Context: 8-0,-
  else if (key === '8') selectAttr('context', 'acrobatics');
  else if (key === '9') selectAttr('context', 'wall');
  else if (key === '0') selectAttr('context', 'swing');
  else if (key === '-') selectAttr('context', 'pk_basics');
});

loadTricks();
</script>
</body>
</html>
"""


class LabelHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)

        if parsed.path == "/" or parsed.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(HTML_TEMPLATE.encode())
            return

        if parsed.path == "/api/pending":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            pending = self._get_pending()
            self.wfile.write(json.dumps(pending).encode())
            return

        if parsed.path.startswith("/preview/"):
            parts = parsed.path.split("/")
            if len(parts) >= 4:
                session = parts[2]
                filename = "/".join(parts[3:])
                filepath = PENDING_DIR / session / filename
                if filepath.exists():
                    self.send_response(200)
                    self.send_header("Content-Type", "image/jpeg")
                    self.end_headers()
                    self.wfile.write(filepath.read_bytes())
                    return

        if parsed.path.startswith("/video/"):
            parts = parsed.path.split("/")
            if len(parts) >= 4:
                session = parts[2]
                trick_name = parts[3].replace(".mp4", "")
                npy_path = PENDING_DIR / session / f"{trick_name}.npy"
                if npy_path.exists():
                    mp4_data = self._npy_to_mp4(npy_path)
                    if mp4_data:
                        self.send_response(200)
                        self.send_header("Content-Type", "video/mp4")
                        self.send_header("Content-Length", str(len(mp4_data)))
                        self.end_headers()
                        self.wfile.write(mp4_data)
                        return

        self.send_response(404)
        self.end_headers()

    def _npy_to_mp4(self, npy_path):
        """Convert .npy frames to mp4 video bytes."""
        try:
            frames = np.load(npy_path, allow_pickle=True)
            if frames.ndim != 4:
                return None
            T, H, W, C = frames.shape

            # Write to temp mp4
            tmp = npy_path.with_suffix(".tmp.mp4")
            fourcc = cv2.VideoWriter_fourcc(*"avc1")
            writer = cv2.VideoWriter(str(tmp), fourcc, 12, (W, H))
            if not writer.isOpened():
                # Fallback codec
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(tmp), fourcc, 12, (W, H))

            for t in range(T):
                bgr = cv2.cvtColor(frames[t], cv2.COLOR_RGB2BGR)
                writer.write(bgr)
                writer.write(bgr)  # double frames for slower playback
            writer.release()

            data = tmp.read_bytes()
            tmp.unlink(missing_ok=True)
            return data
        except Exception:
            return None

    def do_POST(self):
        if self.path == "/api/label":
            content_length = int(self.headers["Content-Length"])
            body = json.loads(self.rfile.read(content_length))

            session = body["session"]
            trick_name = body["trick_name"]
            status = body["status"]
            label = body.get("label")

            manifest_path = PENDING_DIR / session / "manifest.json"
            if manifest_path.exists():
                with open(manifest_path) as f:
                    manifest = json.load(f)

                for trick in manifest["tricks"]:
                    if trick["name"] == trick_name:
                        trick["status"] = status
                        if label:
                            trick["confirmed_label"] = {
                                "direction": label.get("direction", "none"),
                                "flip_count": label.get("flip_count", "0"),
                                "twist_count": label.get("twist_count", "0"),
                                "context": label.get("context", "acrobatics"),
                                "trick_name": label.get("trick_name", ""),
                            }

                        # Export if confirmed
                        if status == "confirmed":
                            self._export_clip(session, trick, manifest)
                        break

                with open(manifest_path, "w") as f:
                    json.dump(manifest, f, indent=2)

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"ok":true}')
            return

    def _get_pending(self):
        pending = []
        if not PENDING_DIR.exists():
            return pending
        for session_dir in sorted(PENDING_DIR.iterdir()):
            if not session_dir.is_dir():
                continue
            manifest_path = session_dir / "manifest.json"
            if not manifest_path.exists():
                continue
            with open(manifest_path) as f:
                manifest = json.load(f)
            for trick in manifest["tricks"]:
                if trick["status"] == "pending":
                    trick["_session"] = session_dir.name
                    trick["_video"] = manifest.get("video_name", "?")[:40]
                    pending.append(trick)
        return pending

    def _export_clip(self, session, trick, manifest):
        CONFIRMED_DIR.mkdir(parents=True, exist_ok=True)
        src = Path(trick["npy_path"])
        if not src.exists():
            return
        slug = f"{session}_{trick['name']}"
        dst = CONFIRMED_DIR / f"{slug}.npy"
        if not dst.exists():
            shutil.copy2(src, dst)

        # Update confirmed manifest
        confirmed_manifest = CONFIRMED_DIR / "confirmed_clips.json"
        existing = []
        if confirmed_manifest.exists():
            with open(confirmed_manifest) as f:
                existing = json.load(f)

        label = trick["confirmed_label"]
        entry = {
            "slug": slug,
            "npy_path": str(dst),
            "direction": label.get("direction", "none"),
            "flip_count": label.get("flip_count", "0"),
            "twist_count": label.get("twist_count", "0"),
            "context": label.get("context", "acrobatics"),
            "trick_name": label.get("trick_name", ""),
            "source": "active_learning",
            "video": manifest.get("video_name", ""),
        }
        existing_slugs = {e["slug"] for e in existing}
        if slug not in existing_slugs:
            existing.append(entry)
            with open(confirmed_manifest, "w") as f:
                json.dump(existing, f, indent=2)

    def log_message(self, format, *args):
        pass  # Suppress request logs


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    CONFIRMED_DIR.mkdir(parents=True, exist_ok=True)
    server = HTTPServer(("127.0.0.1", args.port), LabelHandler)
    print(f"\n  PkVision Labeler running at http://localhost:{args.port}")
    print(f"  Open in your browser to start labeling.")
    print(f"  Press Ctrl+C to stop.\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Server stopped.")


if __name__ == "__main__":
    main()
