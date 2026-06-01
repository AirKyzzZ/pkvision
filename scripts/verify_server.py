#!/usr/bin/env python3
"""Human verification web UI for machine-proposed clips.

For each pending proposal the operator sees a frame-strip preview, the proposed
trick name, and pre-selected cue chips. They can:
  - Press Enter to confirm as-is  (action="confirm")
  - Toggle a cue chip then Enter  (action="override_cue")
  - Type a different trick name   (action="correct_trick")

Usage:
    python3 scripts/make_proposals.py --limit 30 --proposer local
    python3 scripts/verify_server.py
    python3 scripts/verify_server.py --port 8899
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.labeling.clip_ref import ClipRef
from core.labeling.store import VerifiedRecord, VerifiedStore

PROPOSALS_DIR = ROOT / "data" / "labeling" / "proposals" / "local"
CLIPS_DIR = ROOT / "data" / "parkourtheory_clips_cropped"
STORE_PATH = ROOT / "data" / "labeling" / "verified.jsonl"

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>PkVision — Verify Proposals</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       background: #0a0a0a; color: #e0e0e0; }
.container { max-width: 960px; margin: 0 auto; padding: 20px; }
h1 { font-size: 1.4em; margin-bottom: 5px; color: #fff; }
.stats { color: #888; margin-bottom: 20px; font-size: 0.9em; }
.progress-bar { height: 6px; background: #333; border-radius: 3px; margin: 10px 0; }
.progress-fill { height: 100%; background: #4CAF50; border-radius: 3px; transition: width 0.3s; }
.card { background: #1a1a1a; border-radius: 12px; padding: 20px; margin-bottom: 15px; }
.card-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; }
.slug { font-size: 1.05em; font-weight: 600; font-family: monospace; color: #cfd; }
.conf { color: #888; font-size: 0.85em; }
.preview { text-align: center; margin: 15px 0; }
.preview img { max-width: 100%; border-radius: 8px; border: 2px solid #333; }
.preview-error { color: #888; font-style: italic; font-size: 0.85em; }
.proposal-row { background: #222; padding: 10px 15px; border-radius: 8px;
                margin: 10px 0; font-size: 0.9em; }
.proposal-row span { color: #4CAF50; font-weight: 600; }
.label-section { margin-top: 15px; }
.label-row { display: flex; gap: 8px; margin-bottom: 8px; align-items: center; flex-wrap: wrap; }
.label-title { width: 90px; font-size: 0.85em; color: #888; flex-shrink: 0; }
.btn { padding: 8px 14px; border: 1px solid #444; border-radius: 6px; background: #2a2a2a;
       color: #e0e0e0; cursor: pointer; font-size: 0.82em; transition: all 0.15s; }
.btn:hover { background: #3a3a3a; border-color: #666; }
.btn.selected { background: #1a5e1a; border-color: #4CAF50; color: #fff; }
.trick-input { flex: 1; padding: 8px 12px; background: #2a2a2a; border: 1px solid #444;
               border-radius: 6px; color: #e0e0e0; font-size: 0.85em; min-width: 200px; }
.trick-input:focus { outline: none; border-color: #4CAF50; }
.actions { display: flex; gap: 10px; margin-top: 20px; justify-content: center; }
.btn-confirm { padding: 12px 40px; background: #2e7d32; border: none; border-radius: 8px;
               color: #fff; font-size: 1em; font-weight: 600; cursor: pointer; }
.btn-confirm:hover { background: #388e3c; }
.btn-skip { padding: 12px 30px; background: #555; border: none; border-radius: 8px;
            color: #fff; font-size: 1em; cursor: pointer; }
.btn-skip:hover { background: #666; }
.done { text-align: center; padding: 60px 20px; }
.done h2 { color: #4CAF50; margin-bottom: 10px; }
.keyboard-hint { color: #555; font-size: 0.75em; text-align: center; margin-top: 15px; }
.action-tag { font-size: 0.75em; padding: 2px 8px; border-radius: 4px;
              background: #2a4a2a; color: #8f8; margin-left: 8px; }
</style>
</head>
<body>
<div class="container">
  <h1>PkVision — Verify Proposals</h1>
  <div class="stats" id="stats"></div>
  <div class="progress-bar"><div class="progress-fill" id="progress"></div></div>
  <div id="card"></div>
  <div class="keyboard-hint">
    Enter=confirm &nbsp;|&nbsp; S=skip &nbsp;|&nbsp; ← →=navigate &nbsp;|&nbsp;
    1-4=direction &nbsp;|&nbsp; 5-7=flip &nbsp;|&nbsp; Q/W/E/R=twist &nbsp;|&nbsp; 8/9/0/-=context
  </div>
</div>
<script>
let items = [];       // [{slug, proposal:{trick,cues,confidence,d_score}}]
let idx = 0;
let selections = {};  // current chip state (mirrors proposal then human edits)
let origProposal = {};

async function load() {
  const res = await fetch('/api/pending');
  items = await res.json();
  if (items.length === 0) {
    document.getElementById('card').innerHTML =
      '<div class="done"><h2>All done!</h2><p>No proposals to verify.</p></div>';
    return;
  }
  showCard(0);
  updateStats();
}

function updateStats() {
  const done = idx;
  const total = items.length;
  document.getElementById('stats').textContent =
    done + '/' + total + ' verified — ' + (total - done) + ' remaining';
  document.getElementById('progress').style.width = ((done / total) * 100) + '%';
}

function showCard(i) {
  if (i >= items.length) {
    document.getElementById('card').innerHTML =
      '<div class="done"><h2>All done!</h2><p>Reload to check for new proposals.</p></div>';
    return;
  }
  idx = i;
  const item = items[i];
  const p = item.proposal;

  origProposal = JSON.parse(JSON.stringify(p.cues || {}));
  selections = Object.assign({}, origProposal);

  const confPct = ((p.confidence || 0) * 100).toFixed(0);

  const safeSlug = escHtml(item.slug);
  const safeTrick = escHtml(p.trick || '');
  const trickDisplay = safeTrick || '<em>unmatched</em>';

  document.getElementById('card').innerHTML = `
    <div class="card">
      <div class="card-header">
        <span class="slug">${safeSlug}</span>
        <span class="conf">confidence ${confPct}%</span>
      </div>
      <div class="preview">
        <img src="/api/preview?slug=${encodeURIComponent(item.slug)}"
             onerror="this.parentElement.innerHTML='<p class=preview-error>Preview unavailable</p>'" />
      </div>
      <div class="proposal-row">
        Proposed trick: <span>${trickDisplay}</span>
      </div>
      <div class="label-section">
        <div class="label-row">
          <span class="label-title">Direction</span>
          ${chips('direction', ['forward','backward','side','none'], ['1','2','3','4'])}
        </div>
        <div class="label-row">
          <span class="label-title">Flip</span>
          ${chips('flip', ['0','1','2','3','4+'], ['5','6','7','8','9'])}
        </div>
        <div class="label-row">
          <span class="label-title">Twist</span>
          ${chips('twist', ['0','0.5','1','1.5','2+'], ['q','w','e','r','t'])}
        </div>
        <div class="label-row">
          <span class="label-title">Context</span>
          ${chips('context', ['acrobatics','wall','swing','pk_basics'], ['Q','W','E','R'])}
        </div>
        <div class="label-row">
          <span class="label-title">Trick name</span>
          <input id="trickInput" class="trick-input" type="text"
                 value="${safeTrick}"
                 placeholder="e.g. gainer full, double kong…"
                 oninput="onTrickInput(this.value)" />
          <span id="actionTag" class="action-tag">confirm</span>
        </div>
      </div>
      <div class="actions">
        <button class="btn-skip" onclick="skipCard()">Skip (S)</button>
        <button class="btn-confirm" onclick="submitCard()">Confirm (Enter)</button>
      </div>
    </div>
  `;

  updateChips();
  updateStats();
  document.getElementById('trickInput').origValue = (p.trick || '');
}

function escHtml(s) {
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')
          .replace(/"/g,'&quot;').replace(/'/g,'&#39;');
}

function chips(attr, values, keys) {
  return values.map((v, i) => {
    const key = keys[i] || '';
    return `<button class="btn" data-attr="${attr}" data-val="${v}"
             onclick="selectCue('${attr}','${v}')">${v}${key ? ' <small>('+key+')</small>' : ''}</button>`;
  }).join('');
}

function selectCue(attr, val) {
  selections[attr] = val;
  updateChips();
  refreshActionTag();
}

function updateChips() {
  document.querySelectorAll('.btn[data-attr]').forEach(btn => {
    const attr = btn.dataset.attr;
    const val = btn.dataset.val;
    const selVal = String(selections[attr] ?? '');
    btn.classList.toggle('selected', selVal === val);
  });
}

function onTrickInput(val) {
  refreshActionTag();
}

function refreshActionTag() {
  const tag = document.getElementById('actionTag');
  if (!tag) return;
  const inp = document.getElementById('trickInput');
  if (!inp) return;
  const trickChanged = inp.value.trim() !== (inp.origValue || '').trim();
  const cueChanged = Object.keys(selections).some(k => String(selections[k]) !== String(origProposal[k] ?? ''));
  if (trickChanged) {
    tag.textContent = 'correct_trick';
    tag.style.background = '#4a3a00';
    tag.style.color = '#ff8';
  } else if (cueChanged) {
    tag.textContent = 'override_cue';
    tag.style.background = '#003a4a';
    tag.style.color = '#8ff';
  } else {
    tag.textContent = 'confirm';
    tag.style.background = '#2a4a2a';
    tag.style.color = '#8f8';
  }
}

function deriveAction() {
  const inp = document.getElementById('trickInput');
  if (!inp) return 'confirm';
  const trickChanged = inp.value.trim() !== (inp.origValue || '').trim();
  const cueChanged = Object.keys(selections).some(k => String(selections[k]) !== String(origProposal[k] ?? ''));
  if (trickChanged) return 'correct_trick';
  if (cueChanged) return 'override_cue';
  return 'confirm';
}

async function submitCard() {
  const item = items[idx];
  const inp = document.getElementById('trickInput');
  const trick = inp ? inp.value.trim() : (item.proposal.trick || '');
  const action = deriveAction();

  const cues = {};
  for (const [k, v] of Object.entries(selections)) {
    const num = parseFloat(v);
    cues[k] = isNaN(num) ? v : num;
  }

  await fetch('/api/verify', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ slug: item.slug, action, trick, cues })
  });
  showCard(idx + 1);
}

async function skipCard() {
  showCard(idx + 1);
}

document.addEventListener('keydown', e => {
  const inInput = document.activeElement?.tagName === 'INPUT';
  const key = e.key;
  if (key === 'Enter') { e.preventDefault(); submitCard(); return; }
  if (inInput) return;
  if (key === 's' || key === 'S') { skipCard(); return; }
  if (key === 'ArrowRight') { showCard(idx + 1); return; }
  if (key === 'ArrowLeft' && idx > 0) { showCard(idx - 1); return; }
  // Direction
  if (key === '1') selectCue('direction','forward');
  else if (key === '2') selectCue('direction','backward');
  else if (key === '3') selectCue('direction','side');
  else if (key === '4') selectCue('direction','none');
  // Flip
  else if (key === '5') selectCue('flip','0');
  else if (key === '6') selectCue('flip','1');
  else if (key === '7') selectCue('flip','2');
  else if (key === '8') selectCue('flip','3');
  else if (key === '9') selectCue('flip','4+');
  // Twist
  else if (key === 'q') selectCue('twist','0');
  else if (key === 'w') selectCue('twist','0.5');
  else if (key === 'e') selectCue('twist','1');
  else if (key === 'r') selectCue('twist','1.5');
  else if (key === 't') selectCue('twist','2+');
  // Context
  else if (key === 'Q') selectCue('context','acrobatics');
  else if (key === 'W') selectCue('context','wall');
  else if (key === 'E') selectCue('context','swing');
  else if (key === 'R') selectCue('context','pk_basics');
});

load();
</script>
</body>
</html>
"""


def _build_montage(frames: np.ndarray, target_h: int = 240) -> bytes:
    """Horizontal strip of N frames → PNG bytes."""
    if frames is None or len(frames) == 0:
        blank = np.zeros((target_h, 320, 3), np.uint8)
        _, buf = cv2.imencode(".png", blank)
        return bytes(buf)

    strips = []
    for i in range(len(frames)):
        frame = frames[i]
        h, w = frame.shape[:2]
        scale = target_h / h
        new_w = int(w * scale)
        resized = cv2.resize(frame, (new_w, target_h))
        bgr = cv2.cvtColor(resized, cv2.COLOR_RGB2BGR)
        strips.append(bgr)

    montage = np.concatenate(strips, axis=1)
    _, buf = cv2.imencode(".png", montage)
    return bytes(buf)


class VerifyHandler(SimpleHTTPRequestHandler):
    _proposals_cache: dict | None = None

    def _load_proposals(self) -> dict[str, dict]:
        if not PROPOSALS_DIR.exists():
            return {}
        proposals = {}
        for p in sorted(PROPOSALS_DIR.glob("*.json")):
            try:
                data = json.loads(p.read_text())
                proposals[data["slug"]] = data
            except Exception:
                pass
        return proposals

    def _get_proposals(self) -> dict[str, dict]:
        return self._load_proposals()

    def do_GET(self):
        parsed = urlparse(self.path)

        if parsed.path in ("/", "/index.html"):
            self._respond(200, "text/html", HTML_TEMPLATE.encode())
            return

        if parsed.path == "/api/pending":
            all_props = self._get_proposals()
            store = VerifiedStore(STORE_PATH)
            done = store.verified_slugs()
            pending = [
                {
                    "slug": slug,
                    "proposal": {
                        "trick": p.get("trick"),
                        "cues": p.get("cues", {}),
                        "confidence": p.get("confidence", 0.0),
                        "d_score": p.get("d_score"),
                    },
                }
                for slug, p in all_props.items()
                if slug not in done
            ]
            self._respond(200, "application/json", json.dumps(pending).encode())
            return

        if parsed.path == "/api/preview":
            qs = parse_qs(parsed.query)
            slug = (qs.get("slug") or [""])[0]
            if not slug:
                self._respond(400, "text/plain", b"missing slug")
                return
            video_path = CLIPS_DIR / f"{slug}.mp4"
            ref = ClipRef(slug, video_path=video_path if video_path.exists() else None, frames_path=None)
            try:
                frames = ref.get_frames(max_frames=8)
                png = _build_montage(frames)
            except Exception:
                png = _build_montage(np.zeros((0, 1, 1, 3), np.uint8))
            self._respond(200, "image/png", png)
            return

        self._respond(404, "text/plain", b"not found")

    def do_POST(self):
        if self.path == "/api/verify":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length))

            slug = body["slug"]
            action = body["action"]
            trick = body.get("trick")
            cues = body.get("cues", {})

            all_props = self._get_proposals()
            prop = all_props.get(slug, {})
            d_score = prop.get("d_score")

            rec = VerifiedRecord(
                slug=slug,
                trick=trick,
                cues=cues,
                d_score=d_score,
                proposer_source="local",
                action=action,
                verified_at=datetime.now(timezone.utc).isoformat(),
            )
            VerifiedStore(STORE_PATH).append(rec)
            self._respond(200, "application/json", b'{"ok": true}')
            return

        self._respond(404, "text/plain", b"not found")

    def _respond(self, code: int, content_type: str, body: bytes) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        pass


def main():
    import argparse

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8899)
    args = ap.parse_args()

    server = HTTPServer(("127.0.0.1", args.port), VerifyHandler)
    print(f"\n  PkVision Verify UI  →  http://localhost:{args.port}")
    print(f"  Press Ctrl+C to stop.\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Server stopped.")


if __name__ == "__main__":
    main()
