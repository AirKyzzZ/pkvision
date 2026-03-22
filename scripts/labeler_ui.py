#!/usr/bin/env python3
"""Web-based video labeling UI for PkVision training data.

Features:
- Multiple tricks per video (each with its own start/end time)
- Extended trick list (40+ tricks)
- Progress persists across sessions (auto-saves to labels.json)
- Keyboard shortcuts for speed
- Category filters

Usage:
    python scripts/labeler_ui.py
    python scripts/labeler_ui.py --clips-dir data/clips --port 8501
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import sys
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import unquote, urlparse

sys.path.insert(0, str(Path(__file__).parent.parent))

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}

# Extended trick list with categories
ALL_TRICKS = [
    # Flips
    {"id": "back_flip", "name": "Back Flip", "cat": "flip", "key": "1"},
    {"id": "front_flip", "name": "Front Flip", "cat": "flip", "key": "2"},
    {"id": "side_flip", "name": "Side Flip", "cat": "flip", "key": "3"},
    {"id": "webster", "name": "Webster", "cat": "flip", "key": "4"},
    {"id": "gainer", "name": "Gainer", "cat": "flip", "key": "5"},
    {"id": "double_front", "name": "Double Front Flip", "cat": "flip", "key": ""},
    {"id": "double_back", "name": "Double Back Flip", "cat": "flip", "key": ""},
    {"id": "aerial", "name": "Aerial (No Hands)", "cat": "flip", "key": ""},
    {"id": "b_twist", "name": "B-Twist", "cat": "flip", "key": ""},
    {"id": "cheat_gainer", "name": "Cheat Gainer", "cat": "flip", "key": ""},
    {"id": "flash_kick", "name": "Flash Kick", "cat": "flip", "key": ""},
    {"id": "wall_flip", "name": "Wall Flip", "cat": "flip", "key": ""},
    {"id": "wall_back_flip", "name": "Wall Back Flip", "cat": "flip", "key": ""},
    # Twists
    {"id": "flip_360", "name": "360 Flip (Full)", "cat": "twist", "key": "6"},
    {"id": "triple_cork", "name": "Triple Cork", "cat": "twist", "key": ""},
    {"id": "double_cork", "name": "Double Cork", "cat": "twist", "key": ""},
    {"id": "cork", "name": "Cork", "cat": "twist", "key": ""},
    {"id": "twist_180", "name": "Half Twist (180)", "cat": "twist", "key": ""},
    {"id": "twist_360", "name": "Full Twist (360)", "cat": "twist", "key": ""},
    {"id": "twist_540", "name": "540 Twist", "cat": "twist", "key": ""},
    {"id": "twist_720", "name": "720 Twist", "cat": "twist", "key": ""},
    {"id": "raiz", "name": "Raiz", "cat": "twist", "key": ""},
    # Vaults
    {"id": "kong_vault", "name": "Kong Vault", "cat": "vault", "key": "7"},
    {"id": "double_kong", "name": "Double Kong", "cat": "vault", "key": "8"},
    {"id": "speed_vault", "name": "Speed Vault", "cat": "vault", "key": ""},
    {"id": "dash_vault", "name": "Dash Vault", "cat": "vault", "key": ""},
    {"id": "lazy_vault", "name": "Lazy Vault", "cat": "vault", "key": ""},
    {"id": "kash_vault", "name": "Kash Vault", "cat": "vault", "key": ""},
    {"id": "reverse_vault", "name": "Reverse Vault", "cat": "vault", "key": ""},
    {"id": "turn_vault", "name": "Turn Vault", "cat": "vault", "key": ""},
    {"id": "palm_spin", "name": "Palm Spin", "cat": "vault", "key": ""},
    # Precision & movement
    {"id": "precision_jump", "name": "Precision Jump", "cat": "precision", "key": "9"},
    {"id": "standing_pre", "name": "Standing Precision", "cat": "precision", "key": ""},
    {"id": "running_pre", "name": "Running Precision", "cat": "precision", "key": ""},
    {"id": "cat_leap", "name": "Cat Leap (Saut de Bras)", "cat": "precision", "key": ""},
    {"id": "tic_tac", "name": "Tic-Tac", "cat": "precision", "key": ""},
    {"id": "wall_run", "name": "Wall Run", "cat": "precision", "key": ""},
    {"id": "climb_up", "name": "Climb Up", "cat": "precision", "key": ""},
    {"id": "lache", "name": "Lache (Swing Release)", "cat": "precision", "key": ""},
    # Other
    {"id": "standing", "name": "Standing / Idle", "cat": "other", "key": "0"},
    {"id": "walking", "name": "Walking / Running", "cat": "other", "key": ""},
    {"id": "combo", "name": "Combo (multiple tricks)", "cat": "other", "key": ""},
    {"id": "other_trick", "name": "Other Trick", "cat": "other", "key": ""},
]


class LabelerHandler(SimpleHTTPRequestHandler):
    clips_dir: Path
    labels_path: Path
    custom_tricks_path: Path

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._serve_app()
        elif parsed.path == "/api/videos":
            self._serve_video_list()
        elif parsed.path == "/api/labels":
            self._serve_labels()
        elif parsed.path == "/api/tricks":
            self._serve_all_tricks()
        elif parsed.path.startswith("/video/"):
            self._serve_video(parsed.path[7:])
        else:
            self.send_error(404)

    def do_POST(self):
        parsed = urlparse(self.path)
        content_len = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(content_len)) if content_len > 0 else {}

        if parsed.path == "/api/label":
            self._save_label(body)
        elif parsed.path == "/api/labels/delete":
            self._delete_label(body)
        elif parsed.path == "/api/done":
            self._mark_done(body)
        elif parsed.path == "/api/tricks/add":
            self._add_custom_trick(body)
        else:
            self.send_error(404)

    def _serve_video(self, filename: str):
        filename = unquote(filename)
        filepath = self.clips_dir / filename
        if not filepath.exists():
            self.send_error(404)
            return
        content_type = mimetypes.guess_type(str(filepath))[0] or "video/mp4"
        file_size = filepath.stat().st_size
        range_header = self.headers.get("Range")
        if range_header:
            range_val = range_header.strip().split("=")[1]
            parts = range_val.split("-")
            start = int(parts[0])
            end = int(parts[1]) if parts[1] else file_size - 1
            length = end - start + 1
            self.send_response(206)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
            self.send_header("Content-Length", str(length))
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()
            with open(filepath, "rb") as f:
                f.seek(start)
                self.wfile.write(f.read(length))
        else:
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(file_size))
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()
            with open(filepath, "rb") as f:
                self.wfile.write(f.read())

    def _serve_video_list(self):
        videos = []
        for p in sorted(self.clips_dir.iterdir()):
            if p.suffix.lower() in VIDEO_EXTENSIONS:
                videos.append({"filename": p.name, "size_mb": round(p.stat().st_size / 1024 / 1024, 1)})
        self._json_response({"videos": videos, "total": len(videos)})

    def _serve_labels(self):
        data = self._load_labels_file()
        self._json_response(data)

    def _load_labels_file(self) -> dict:
        if self.labels_path.exists():
            with open(self.labels_path) as f:
                raw = json.load(f)
            # Support both old format (list) and new format (dict)
            if isinstance(raw, list):
                # Migrate old format
                new_data = {"labels": raw, "done_files": []}
                with open(self.labels_path, "w") as f:
                    json.dump(new_data, f, indent=2)
                return new_data
            return raw
        return {"labels": [], "done_files": []}

    def _save_labels_file(self, data: dict):
        with open(self.labels_path, "w") as f:
            json.dump(data, f, indent=2)

    def _save_label(self, body: dict):
        data = self._load_labels_file()
        # Add new label entry (multiple per file allowed)
        data["labels"].append({
            "file": body["file"],
            "trick_id": body["trick_id"],
            "start_ms": body.get("start_ms", 0),
            "end_ms": body.get("end_ms"),
        })
        self._save_labels_file(data)
        total = len(data["labels"])
        self._json_response({"status": "saved", "total_labels": total})

    def _delete_label(self, body: dict):
        data = self._load_labels_file()
        idx = body.get("index", -1)
        file = body.get("file", "")
        # Find and remove the specific label
        file_labels = [(i, l) for i, l in enumerate(data["labels"]) if l["file"] == file]
        if 0 <= idx < len(file_labels):
            real_idx = file_labels[idx][0]
            data["labels"].pop(real_idx)
        self._save_labels_file(data)
        self._json_response({"status": "deleted"})

    def _mark_done(self, body: dict):
        data = self._load_labels_file()
        filename = body.get("file", "")
        if "done_files" not in data:
            data["done_files"] = []
        if filename and filename not in data["done_files"]:
            data["done_files"].append(filename)
        self._save_labels_file(data)
        self._json_response({"status": "done", "done_count": len(data["done_files"])})

    def _add_custom_trick(self, body: dict):
        trick_id = body.get("id", "").strip().lower().replace(" ", "_").replace("-", "_")
        name = body.get("name", "").strip()
        cat = body.get("cat", "other").strip().lower()
        if not trick_id or not name:
            self._json_response({"status": "error", "message": "id and name required"}, 400)
            return
        # Load existing custom tricks
        custom = self._load_custom_tricks()
        # Check duplicates
        all_ids = [t["id"] for t in ALL_TRICKS] + [t["id"] for t in custom]
        if trick_id in all_ids:
            self._json_response({"status": "error", "message": f"'{trick_id}' already exists"}, 400)
            return
        new_trick = {"id": trick_id, "name": name, "cat": cat, "key": ""}
        custom.append(new_trick)
        self._save_custom_tricks(custom)
        self._json_response({"status": "added", "trick": new_trick, "total_custom": len(custom)})

    def _serve_all_tricks(self):
        custom = self._load_custom_tricks()
        all_t = ALL_TRICKS + custom
        self._json_response({"tricks": all_t, "total": len(all_t), "custom_count": len(custom)})

    def _load_custom_tricks(self) -> list[dict]:
        if self.custom_tricks_path.exists():
            with open(self.custom_tricks_path) as f:
                return json.load(f)
        return []

    def _save_custom_tricks(self, tricks: list[dict]):
        with open(self.custom_tricks_path, "w") as f:
            json.dump(tricks, f, indent=2)

    def _json_response(self, data: dict, status: int = 200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_app(self):
        custom = self._load_custom_tricks()
        all_tricks = ALL_TRICKS + custom
        html = get_html(json.dumps(all_tricks))
        body = html.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        pass  # Suppress logs


def get_html(tricks_json: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>PkVision Labeler</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;background:#0a0a0f;color:#e0e0e0;display:flex;flex-direction:column;height:100vh}}
header{{padding:14px 24px;background:#111118;border-bottom:1px solid #222;display:flex;align-items:center;justify-content:space-between}}
header h1{{font-size:20px;font-weight:600;color:#fff}}
header h1 em{{color:#4361ee;font-style:normal}}
.stats{{font-size:12px;color:#888;display:flex;gap:16px}}
.stats b{{color:#4361ee}}
.main{{display:flex;flex:1;overflow:hidden}}
.video-panel{{flex:1;display:flex;flex-direction:column;padding:16px;background:#0d0d14;overflow-y:auto}}
video{{width:100%;max-height:45vh;border-radius:8px;background:#000;border:1px solid #222}}
.vid-info{{margin-top:8px;font-size:12px;color:#666;display:flex;justify-content:space-between;align-items:center}}
.vid-info .fn{{color:#aaa;font-weight:500}}
.nav-btns{{display:flex;gap:8px}}
.nav-btn{{padding:5px 14px;background:#222;border:1px solid #444;border-radius:4px;color:#ccc;cursor:pointer;font-size:12px}}
.nav-btn:hover{{background:#333}}
.nav-btn:disabled{{opacity:.3;cursor:not-allowed}}
.tc{{margin-top:10px;display:flex;gap:10px;align-items:center;flex-wrap:wrap}}
.tc label{{font-size:11px;color:#888}}
.tc input{{width:70px;padding:5px;background:#1a1a24;border:1px solid #333;border-radius:4px;color:#fff;font-size:12px;text-align:center}}
.tc button{{padding:4px 10px;background:#222;border:1px solid #444;border-radius:4px;color:#ccc;cursor:pointer;font-size:11px}}
.tc button:hover{{background:#333}}
.tags-section{{margin-top:12px;background:#111118;border-radius:8px;padding:12px;border:1px solid #222}}
.tags-section h3{{font-size:12px;color:#888;margin-bottom:8px;text-transform:uppercase;letter-spacing:.5px}}
.tag-list{{display:flex;flex-wrap:wrap;gap:6px;min-height:28px}}
.tag{{display:flex;align-items:center;gap:4px;padding:4px 10px;background:#1a2a5e;border:1px solid #4361ee;border-radius:16px;font-size:12px;color:#bbd}}
.tag .time{{color:#667;font-size:10px}}
.tag .del{{cursor:pointer;color:#f55;font-size:14px;margin-left:2px;line-height:1}}
.tag .del:hover{{color:#f88}}
.no-tags{{color:#444;font-size:12px;font-style:italic}}
.add-btn{{margin-top:8px;padding:8px 16px;background:#4361ee;border:none;border-radius:6px;color:#fff;font-size:13px;font-weight:600;cursor:pointer;width:100%}}
.add-btn:hover{{background:#3451de}}
.add-btn:disabled{{background:#333;color:#666;cursor:not-allowed}}
.sidebar{{width:260px;background:#111118;border-left:1px solid #222;display:flex;flex-direction:column;overflow-y:auto}}
.cat-filter{{display:flex;gap:0;padding:8px 12px;border-bottom:1px solid #222}}
.cf-btn{{flex:1;padding:6px 4px;background:transparent;border:1px solid #333;color:#888;font-size:10px;cursor:pointer;text-align:center}}
.cf-btn:first-child{{border-radius:4px 0 0 4px}}
.cf-btn:last-child{{border-radius:0 4px 4px 0}}
.cf-btn.act{{background:#1a2a5e;border-color:#4361ee;color:#fff}}
.trick-scroll{{flex:1;overflow-y:auto;padding:8px 12px}}
.trick-btn{{display:block;width:100%;padding:8px 10px;margin-bottom:4px;background:#1a1a24;border:1px solid #2a2a3a;border-radius:5px;color:#ddd;font-size:13px;cursor:pointer;text-align:left;transition:all .1s}}
.trick-btn:hover{{background:#252535;border-color:#4361ee;color:#fff}}
.trick-btn.sel{{background:#1a2a5e;border-color:#4361ee;color:#fff}}
.trick-btn .k{{float:right;color:#555;font-family:monospace;font-size:11px}}
.bot-actions{{padding:12px;border-top:1px solid #222}}
.done-btn{{display:block;width:100%;padding:10px;background:#1a5e2a;border:none;border-radius:6px;color:#fff;font-size:13px;font-weight:600;cursor:pointer;margin-bottom:6px}}
.done-btn:hover{{background:#2a7e3a}}
.skip-btn{{display:block;width:100%;padding:8px;background:transparent;border:1px solid #333;border-radius:6px;color:#888;font-size:12px;cursor:pointer}}
.skip-btn:hover{{border-color:#555;color:#aaa}}
.toast{{position:fixed;bottom:20px;left:50%;transform:translateX(-50%);padding:8px 20px;background:#1a5e1a;color:#fff;border-radius:8px;font-size:13px;opacity:0;transition:opacity .3s;pointer-events:none;z-index:99}}
.toast.show{{opacity:1}}
.kbd{{font-size:10px;color:#444;text-align:center;margin-top:8px}}
.kbd kbd{{padding:1px 5px;background:#1a1a24;border:1px solid #333;border-radius:3px;font-family:monospace}}
.finish{{display:flex;flex-direction:column;align-items:center;justify-content:center;height:100%;width:100%;text-align:center}}
.finish h2{{font-size:24px;margin-bottom:12px;color:#4361ee}}
.finish p{{color:#888;font-size:15px;max-width:400px;line-height:1.6}}
.finish code{{display:block;margin-top:12px;font-size:14px;color:#4361ee;background:#111;padding:10px;border-radius:6px}}
.new-trick{{padding:10px 12px;border-top:1px solid #222;border-bottom:1px solid #222;background:#0d0d14}}
.new-trick summary{{font-size:12px;color:#888;cursor:pointer;text-transform:uppercase;letter-spacing:.5px}}
.new-trick summary:hover{{color:#bbb}}
.new-trick .nt-form{{margin-top:8px;display:flex;flex-direction:column;gap:6px}}
.new-trick input,.new-trick select{{padding:6px 8px;background:#1a1a24;border:1px solid #333;border-radius:4px;color:#fff;font-size:12px}}
.new-trick input:focus,.new-trick select:focus{{border-color:#4361ee;outline:none}}
.nt-add{{padding:6px;background:#4361ee;border:none;border-radius:4px;color:#fff;font-size:12px;cursor:pointer;font-weight:600}}
.nt-add:hover{{background:#3451de}}
.nt-msg{{font-size:11px;margin-top:4px}}
.nt-msg.ok{{color:#4c4}}
.nt-msg.err{{color:#f55}}
</style></head>
<body>
<header>
<h1><em>Pk</em>Vision Labeler</h1>
<div class="stats">
<span><b id="nLabeled">0</b> tags</span>
<span><b id="nDone">0</b>/<span id="nTotal">0</span> clips done</span>
<span>Clip <b id="nCur">1</b></span>
</div>
</header>
<div class="main" id="main">
<div class="video-panel">
<video id="player" controls playsinline></video>
<div class="vid-info">
<span class="fn" id="fname"></span>
<div class="nav-btns">
<button class="nav-btn" id="prevBtn">Prev</button>
<button class="nav-btn" id="nextBtn">Next (undone)</button>
</div>
</div>
<div class="tc">
<label>Start: <input type="number" id="startT" value="0" step="0.1" min="0"></label>
<button id="bss" class="tc-btn">Set current</button>
<label>End: <input type="number" id="endT" value="" step="0.1" min="0" placeholder="end"></label>
<button id="bse" class="tc-btn">Set current</button>
</div>
<div class="tags-section">
<h3>Tags for this clip</h3>
<div class="tag-list" id="tagList"><span class="no-tags">No tricks tagged yet</span></div>
<button class="add-btn" id="addTagBtn" disabled>Add selected trick</button>
</div>
<div class="kbd">
<kbd>1</kbd>-<kbd>0</kbd> select &nbsp;<kbd>A</kbd> add tag &nbsp;<kbd>D</kbd> done &amp; next &nbsp;<kbd>S</kbd> skip &nbsp;<kbd>Space</kbd> play/pause &nbsp;<kbd>&larr;</kbd><kbd>&rarr;</kbd> prev/next
</div>
</div>
<div class="sidebar">
<div class="cat-filter" id="catFilter"></div>
<details class="new-trick">
<summary>+ Add new trick</summary>
<div class="nt-form">
<input type="text" id="ntName" placeholder="Trick name (e.g. Butterfly Kick)">
<input type="text" id="ntId" placeholder="ID auto-generated" readonly>
<select id="ntCat">
<option value="flip">Flip</option>
<option value="twist">Twist</option>
<option value="vault">Vault</option>
<option value="precision">Precision</option>
<option value="other">Other</option>
</select>
<button class="nt-add" id="ntAddBtn">Add to trick list</button>
<div class="nt-msg" id="ntMsg"></div>
</div>
</details>
<div class="trick-scroll" id="trickList"></div>
<div class="bot-actions">
<button class="done-btn" id="doneBtn">Done with this clip &rarr; Next</button>
<button class="skip-btn" id="skipBtn">Skip (not useful)</button>
</div>
</div>
</div>
<div class="toast" id="toast"></div>
<script>
const TRICKS={tricks_json};
const CATS=[{{id:'all',name:'All'}},{{id:'flip',name:'Flips'}},{{id:'twist',name:'Twists'}},{{id:'vault',name:'Vaults'}},{{id:'precision',name:'Precision'}},{{id:'other',name:'Other'}}];
let videos=[],allLabels=[],doneFiles=[],ci=0,selTrick=null,curCat='all';

// Init
async function init(){{
  const[vr,lr]=await Promise.all([fetch('/api/videos').then(r=>r.json()),fetch('/api/labels').then(r=>r.json())]);
  videos=vr.videos;
  allLabels=lr.labels||[];
  doneFiles=lr.done_files||[];
  // Jump to first undone
  ci=0;
  while(ci<videos.length&&doneFiles.includes(videos[ci].filename))ci++;
  if(ci>=videos.length)ci=0;
  buildCatFilter();buildTricks();updStats();
  if(videos.length)loadVid(ci);
}}

function updStats(){{
  document.getElementById('nLabeled').textContent=allLabels.length;
  document.getElementById('nDone').textContent=doneFiles.length;
  document.getElementById('nTotal').textContent=videos.length;
  document.getElementById('nCur').textContent=ci+1;
}}

function loadVid(i){{
  if(i<0||i>=videos.length)return;
  ci=i;
  const v=videos[ci];
  document.getElementById('player').src='/video/'+encodeURIComponent(v.filename);
  document.getElementById('fname').textContent=v.filename+' ('+v.size_mb+' MB)';
  document.getElementById('startT').value='0';
  document.getElementById('endT').value='';
  selTrick=null;
  document.getElementById('addTagBtn').disabled=true;
  document.querySelectorAll('.trick-btn').forEach(function(b){{b.classList.remove('sel')}});
  document.getElementById('prevBtn').disabled=ci<=0;
  updStats();renderTags();
}}

function buildCatFilter(){{
  var cf=document.getElementById('catFilter');
  CATS.forEach(function(c){{
    var b=document.createElement('button');
    b.className='cf-btn'+(c.id===curCat?' act':'');
    b.textContent=c.name;
    b.onclick=function(){{curCat=c.id;buildTricks();
      document.querySelectorAll('.cf-btn').forEach(function(x){{x.classList.remove('act')}});
      b.classList.add('act');
    }};
    cf.appendChild(b);
  }});
}}

function buildTricks(){{
  var tl=document.getElementById('trickList');
  tl.textContent='';
  TRICKS.forEach(function(t){{
    if(curCat!=='all'&&t.cat!==curCat)return;
    var b=document.createElement('button');
    b.className='trick-btn';
    b.setAttribute('data-t',t.id);
    var sp=document.createElement('span');
    sp.textContent=t.name;
    b.appendChild(sp);
    if(t.key){{
      var k=document.createElement('span');
      k.className='k';k.textContent=t.key;
      b.appendChild(k);
    }}
    b.onclick=function(){{pickTrick(t.id)}};
    tl.appendChild(b);
  }});
}}

function pickTrick(tid){{
  selTrick=tid;
  document.getElementById('addTagBtn').disabled=false;
  document.querySelectorAll('.trick-btn').forEach(function(b){{
    if(b.getAttribute('data-t')===tid)b.classList.add('sel');
    else b.classList.remove('sel');
  }});
}}

function renderTags(){{
  var tl=document.getElementById('tagList');
  tl.textContent='';
  var fn=videos[ci].filename;
  var fl=allLabels.filter(function(l){{return l.file===fn}});
  if(fl.length===0){{
    var sp=document.createElement('span');sp.className='no-tags';
    sp.textContent='No tricks tagged yet';tl.appendChild(sp);return;
  }}
  fl.forEach(function(l,idx){{
    var tag=document.createElement('span');tag.className='tag';
    var nm=document.createElement('span');nm.textContent=l.trick_id.replace(/_/g,' ');
    tag.appendChild(nm);
    if(l.start_ms||l.end_ms){{
      var tm=document.createElement('span');tm.className='time';
      var st=(l.start_ms/1000).toFixed(1);
      var en=l.end_ms?(l.end_ms/1000).toFixed(1):'end';
      tm.textContent=' '+st+'s-'+en+'s';
      tag.appendChild(tm);
    }}
    var del=document.createElement('span');del.className='del';del.textContent='x';
    del.onclick=function(){{deleteTag(fn,idx)}};
    tag.appendChild(del);
    tl.appendChild(tag);
  }});
}}

async function addTag(){{
  if(!selTrick)return;
  var fn=videos[ci].filename;
  var sm=parseFloat(document.getElementById('startT').value||'0')*1000;
  var ev=document.getElementById('endT').value;
  var body={{file:fn,trick_id:selTrick,start_ms:sm}};
  if(ev)body.end_ms=parseFloat(ev)*1000;
  await fetch('/api/label',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify(body)}});
  allLabels.push(body);
  toast('Tagged: '+selTrick.replace(/_/g,' '));
  updStats();renderTags();
  // Reset selection
  document.getElementById('startT').value='0';
  document.getElementById('endT').value='';
}}

async function deleteTag(fn,idx){{
  await fetch('/api/labels/delete',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{file:fn,index:idx}})}});
  // Remove from local state
  var fileLabels=allLabels.filter(function(l){{return l.file===fn}});
  if(idx<fileLabels.length){{
    var target=fileLabels[idx];
    var ri=allLabels.indexOf(target);
    if(ri>=0)allLabels.splice(ri,1);
  }}
  updStats();renderTags();
}}

async function markDone(){{
  var fn=videos[ci].filename;
  await fetch('/api/done',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{file:fn}})}});
  if(!doneFiles.includes(fn))doneFiles.push(fn);
  toast('Done! Moving to next...');
  goNextUndone();
}}

function goNextUndone(){{
  var start=ci+1;
  for(var i=start;i<videos.length;i++){{
    if(!doneFiles.includes(videos[i].filename)){{loadVid(i);return;}}
  }}
  for(var i=0;i<start;i++){{
    if(!doneFiles.includes(videos[i].filename)){{loadVid(i);return;}}
  }}
  showFinish();
}}

function skipVid(){{markDone()}}

function showFinish(){{
  var m=document.getElementById('main');
  m.textContent='';
  var d=document.createElement('div');d.className='finish';
  var h=document.createElement('h2');h.textContent='All clips processed!';
  var p=document.createElement('p');p.textContent=allLabels.length+' tags across '+doneFiles.length+' clips. Run the training pipeline:';
  var c=document.createElement('code');c.textContent='bash scripts/train_backflip.sh';
  d.appendChild(h);d.appendChild(p);d.appendChild(c);m.appendChild(d);
}}

function toast(msg){{
  var el=document.getElementById('toast');el.textContent=msg;
  el.classList.add('show');setTimeout(function(){{el.classList.remove('show')}},1500);
}}

// Wire buttons
document.getElementById('bss').onclick=function(){{document.getElementById('startT').value=document.getElementById('player').currentTime.toFixed(1)}};
document.getElementById('bse').onclick=function(){{document.getElementById('endT').value=document.getElementById('player').currentTime.toFixed(1)}};
document.getElementById('addTagBtn').onclick=addTag;
document.getElementById('doneBtn').onclick=markDone;
document.getElementById('skipBtn').onclick=skipVid;
document.getElementById('prevBtn').onclick=function(){{if(ci>0)loadVid(ci-1)}};
document.getElementById('nextBtn').onclick=goNextUndone;

// Add new trick form
document.getElementById('ntName').oninput=function(){{
  var name=this.value.trim();
  var id=name.toLowerCase().replace(/[^a-z0-9]+/g,'_').replace(/^_|_$/g,'');
  document.getElementById('ntId').value=id;
}};
document.getElementById('ntAddBtn').onclick=async function(){{
  var name=document.getElementById('ntName').value.trim();
  var id=document.getElementById('ntId').value.trim();
  var cat=document.getElementById('ntCat').value;
  var msg=document.getElementById('ntMsg');
  if(!name){{msg.textContent='Enter a name';msg.className='nt-msg err';return}}
  var res=await fetch('/api/tricks/add',{{
    method:'POST',headers:{{'Content-Type':'application/json'}},
    body:JSON.stringify({{id:id,name:name,cat:cat}})
  }});
  var data=await res.json();
  if(data.status==='added'){{
    TRICKS.push(data.trick);
    buildTricks();
    msg.textContent='Added! You can now tag clips with "'+name+'"';
    msg.className='nt-msg ok';
    document.getElementById('ntName').value='';
    document.getElementById('ntId').value='';
  }}else{{
    msg.textContent=data.message||'Error adding trick';
    msg.className='nt-msg err';
  }}
}};

// Keyboard
document.addEventListener('keydown',function(e){{
  if(e.target.tagName==='INPUT')return;
  if(e.key==='a'||e.key==='A'){{addTag();return}}
  if(e.key==='d'||e.key==='D'){{markDone();return}}
  if(e.key==='s'||e.key==='S'){{skipVid();return}}
  if(e.key===' '){{e.preventDefault();var p=document.getElementById('player');p.paused?p.play():p.pause();return}}
  if(e.key==='ArrowRight'){{goNextUndone();return}}
  if(e.key==='ArrowLeft'){{if(ci>0)loadVid(ci-1);return}}
  var n=parseInt(e.key);
  if(!isNaN(n)){{
    var keyTricks=TRICKS.filter(function(t){{return t.key===String(n)}});
    if(keyTricks.length)pickTrick(keyTricks[0].id);
  }}
}});

init();
</script></body></html>"""


def main():
    parser = argparse.ArgumentParser(description="PkVision — Visual labeling UI")
    parser.add_argument("--clips-dir", default="data/clips", help="Directory with video clips")
    parser.add_argument("--port", type=int, default=8501, help="Port for the web UI")
    args = parser.parse_args()

    clips_dir = Path(args.clips_dir).resolve()
    labels_path = clips_dir / "labels.json"

    if not labels_path.exists():
        with open(labels_path, "w") as f:
            json.dump({"labels": [], "done_files": []}, f)

    LabelerHandler.clips_dir = clips_dir
    LabelerHandler.labels_path = labels_path
    LabelerHandler.custom_tricks_path = clips_dir / "custom_tricks.json"

    video_count = len([p for p in clips_dir.iterdir() if p.suffix.lower() in VIDEO_EXTENSIONS])

    # Load existing progress
    with open(labels_path) as f:
        data = json.load(f)
    if isinstance(data, list):
        existing_labels = len(data)
        done = 0
    else:
        existing_labels = len(data.get("labels", []))
        done = len(data.get("done_files", []))

    server = HTTPServer(("0.0.0.0", args.port), LabelerHandler)

    print()
    print("  PkVision Labeler v2")
    print("  ────────────────────────────────")
    print(f"  Videos: {video_count} clips")
    print(f"  Progress: {done} done, {existing_labels} tags")
    print(f"  Tricks: {len(ALL_TRICKS)} available")
    print()
    print(f"  Open: http://localhost:{args.port}")
    print()
    print("  Keys: 1-0 select | A add tag | D done & next | S skip | Space play/pause | arrows nav")
    print("  Ctrl+C to stop (progress auto-saved)")
    print()

    import webbrowser
    webbrowser.open(f"http://localhost:{args.port}")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nLabeler stopped.")
        with open(labels_path) as f:
            data = json.load(f)
        labels_list = data.get("labels", data) if isinstance(data, dict) else data
        tricks_count: dict[str, int] = {}
        for entry in labels_list:
            tid = entry["trick_id"]
            tricks_count[tid] = tricks_count.get(tid, 0) + 1
        print(f"\n{len(labels_list)} tags across {len(data.get('done_files', []))} clips:")
        for tid, count in sorted(tricks_count.items(), key=lambda x: -x[1]):
            print(f"  {tid}: {count}")


if __name__ == "__main__":
    main()
