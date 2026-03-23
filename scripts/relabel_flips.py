#!/usr/bin/env python3
"""Fast relabeling tool — shows only clips classified as 'flip' or 'somersaulting'
and lets you quickly assign specific trick names.

This is the Round 2 labeling step: converting coarse "flip" labels
into specific tricks (back_flip, front_flip, side_flip, gainer, webster, etc.)

Opens a web UI at http://localhost:8502

Usage:
    python scripts/relabel_flips.py
"""

from __future__ import annotations

import json
import mimetypes
import sys
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import unquote, urlparse

sys.path.insert(0, str(Path(__file__).parent.parent))

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}

# Specific flip tricks for relabeling
FLIP_TRICKS = [
    {"id": "back_flip", "name": "Back Flip", "key": "1"},
    {"id": "front_flip", "name": "Front Flip", "key": "2"},
    {"id": "side_flip", "name": "Side Flip", "key": "3"},
    {"id": "gainer", "name": "Gainer", "key": "4"},
    {"id": "webster", "name": "Webster", "key": "5"},
    {"id": "aerial", "name": "Aerial", "key": "6"},
    {"id": "wall_flip", "name": "Wall Flip", "key": "7"},
    {"id": "double_back", "name": "Double Back", "key": "8"},
    {"id": "double_front", "name": "Double Front", "key": "9"},
    {"id": "flash_kick", "name": "Flash Kick", "key": "0"},
    {"id": "cheat_gainer", "name": "Cheat Gainer", "key": ""},
    {"id": "back_tuck", "name": "Back Tuck", "key": ""},
    {"id": "front_tuck", "name": "Front Tuck", "key": ""},
    {"id": "back_layout", "name": "Back Layout", "key": ""},
    {"id": "front_layout", "name": "Front Layout", "key": ""},
    {"id": "back_pike", "name": "Back Pike", "key": ""},
    {"id": "front_pike", "name": "Front Pike", "key": ""},
    {"id": "b_twist", "name": "B-Twist", "key": ""},
    {"id": "cork", "name": "Cork", "key": ""},
    {"id": "double_cork", "name": "Double Cork", "key": ""},
    {"id": "raiz", "name": "Raiz", "key": ""},
    {"id": "not_a_flip", "name": "Not a flip / Skip", "key": "s"},
]


class RelabelHandler(SimpleHTTPRequestHandler):
    clips_dir: Path
    manifest_path: Path
    relabels_path: Path
    flip_clips: list

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._serve_app()
        elif parsed.path == "/api/clips":
            self._serve_clips()
        elif parsed.path == "/api/relabels":
            self._serve_relabels()
        elif parsed.path.startswith("/video/"):
            self._serve_video(parsed.path[7:])
        else:
            self.send_error(404)

    def do_POST(self):
        parsed = urlparse(self.path)
        cl = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(cl)) if cl > 0 else {}
        if parsed.path == "/api/relabel":
            self._save_relabel(body)
        elif parsed.path == "/api/tricks/add":
            self._add_trick(body)
        else:
            self.send_error(404)

    def _serve_video(self, filename):
        filename = unquote(filename)
        filepath = self.clips_dir / filename
        if not filepath.exists():
            self.send_error(404); return
        ct = mimetypes.guess_type(str(filepath))[0] or "video/mp4"
        fs = filepath.stat().st_size
        rh = self.headers.get("Range")
        if rh:
            rv = rh.strip().split("=")[1]; parts = rv.split("-")
            s = int(parts[0]); e = int(parts[1]) if parts[1] else fs - 1
            self.send_response(206)
            self.send_header("Content-Type", ct)
            self.send_header("Content-Range", f"bytes {s}-{e}/{fs}")
            self.send_header("Content-Length", str(e - s + 1))
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()
            with open(filepath, "rb") as f: f.seek(s); self.wfile.write(f.read(e - s + 1))
        else:
            self.send_response(200)
            self.send_header("Content-Type", ct)
            self.send_header("Content-Length", str(fs))
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()
            with open(filepath, "rb") as f: self.wfile.write(f.read())

    def _serve_clips(self):
        relabels = self._load_relabels()
        clips = []
        for c in self.flip_clips:
            clips.append({
                "filename": c["original"],
                "current_class": c["class"],
                "relabeled": relabels.get(c["original"]),
            })
        done = sum(1 for c in clips if c["relabeled"])
        self._json({"clips": clips, "total": len(clips), "done": done})

    def _serve_relabels(self):
        self._json(self._load_relabels())

    def _save_relabel(self, body):
        relabels = self._load_relabels()
        relabels[body["file"]] = body["trick_id"]
        with open(self.relabels_path, "w") as f:
            json.dump(relabels, f, indent=2)
        done = len(relabels)
        self._json({"status": "saved", "done": done, "total": len(self.flip_clips)})

    def _add_trick(self, body):
        trick_id = body.get("id", "").strip().lower().replace(" ", "_").replace("-", "_")
        name = body.get("name", "").strip()
        if not trick_id or not name:
            self._json({"status": "error", "message": "id and name required"}, 400)
            return
        # Check duplicates
        all_ids = [t["id"] for t in FLIP_TRICKS]
        custom = self._load_custom_tricks()
        all_ids += [t["id"] for t in custom]
        if trick_id in all_ids:
            self._json({"status": "error", "message": f"'{trick_id}' already exists"}, 400)
            return
        new_trick = {"id": trick_id, "name": name, "key": ""}
        custom.append(new_trick)
        custom_path = self.relabels_path.parent / "custom_relabel_tricks.json"
        with open(custom_path, "w") as f:
            json.dump(custom, f, indent=2)
        FLIP_TRICKS.append(new_trick)
        self._json({"status": "added", "trick": new_trick})

    def _load_custom_tricks(self) -> list:
        custom_path = self.relabels_path.parent / "custom_relabel_tricks.json"
        if custom_path.exists():
            with open(custom_path) as f:
                return json.load(f)
        return []

    def _load_relabels(self) -> dict:
        if self.relabels_path.exists():
            with open(self.relabels_path) as f:
                return json.load(f)
        return {}

    def _json(self, data, status=200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_app(self):
        self.send_response(200)
        html = get_html(json.dumps(FLIP_TRICKS)).encode()
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(html)))
        self.end_headers()
        self.wfile.write(html)

    def log_message(self, *a): pass


def get_html(tricks_json):
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>PkVision — Relabel Flips (Round 2)</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:-apple-system,sans-serif;background:#0a0a0f;color:#e0e0e0;height:100vh;display:flex;flex-direction:column}}
header{{padding:12px 20px;background:#111118;border-bottom:1px solid #222;display:flex;justify-content:space-between;align-items:center}}
header h1{{font-size:18px;color:#fff}}header h1 em{{color:#4361ee;font-style:normal}}
.stats{{font-size:12px;color:#888}}.stats b{{color:#4ade80}}
.main{{display:flex;flex:1;overflow:hidden}}
.panel{{flex:1;display:flex;flex-direction:column;align-items:center;justify-content:center;padding:16px}}
video{{max-width:100%;max-height:50vh;border-radius:8px;background:#000;border:1px solid #222}}
.info{{margin-top:8px;font-size:12px;color:#666;text-align:center}}
.info .cur{{color:#f59e0b;font-weight:600}}
.sidebar{{width:280px;background:#111118;border-left:1px solid #222;display:flex;flex-direction:column;overflow-y:auto;padding:12px}}
.sidebar h2{{font-size:13px;color:#888;margin-bottom:10px;text-transform:uppercase;letter-spacing:.5px}}
.sidebar p{{font-size:11px;color:#555;margin-bottom:12px}}
.btn{{display:block;width:100%;padding:10px;margin-bottom:6px;background:#1a1a24;border:1px solid #2a2a3a;border-radius:6px;color:#ddd;font-size:14px;cursor:pointer;text-align:left}}
.btn:hover{{background:#252535;border-color:#4361ee;color:#fff}}
.btn.sel{{background:#1a2a5e;border-color:#4361ee;color:#fff}}
.btn .k{{float:right;color:#555;font-family:monospace;font-size:11px}}
.skip{{margin-top:auto;padding-top:12px;border-top:1px solid #222}}
.skip .btn{{border-color:#533;color:#a88}}
.skip .btn:hover{{border-color:#f55;color:#faa}}
.toast{{position:fixed;bottom:20px;left:50%;transform:translateX(-50%);padding:8px 20px;background:#1a5e1a;color:#fff;border-radius:8px;font-size:13px;opacity:0;transition:opacity .3s;z-index:99}}
.toast.show{{opacity:1}}
.done{{display:flex;flex-direction:column;align-items:center;justify-content:center;height:100%;width:100%;text-align:center}}
.done h2{{font-size:24px;color:#4ade80;margin-bottom:12px}}
.done p{{color:#888;font-size:15px;max-width:400px;line-height:1.6}}
.done code{{display:block;margin-top:12px;color:#4361ee;background:#111;padding:10px;border-radius:6px;font-size:13px}}
.kbd{{font-size:10px;color:#444;text-align:center;margin-top:10px}}
.kbd kbd{{padding:1px 5px;background:#1a1a24;border:1px solid #333;border-radius:3px;font-family:monospace}}
.prog{{width:100%;height:4px;background:#222;border-radius:2px;margin-top:8px}}
.prog-bar{{height:100%;background:#4ade80;border-radius:2px;transition:width .3s}}
</style></head><body>
<header>
<h1><em>Pk</em>Vision — Round 2: Relabel Flips</h1>
<div class="stats"><b id="nDone">0</b>/<span id="nTotal">0</span> labeled &nbsp; Clip <b id="nCur">1</b></div>
</header>
<div class="prog"><div class="prog-bar" id="progBar" style="width:0%"></div></div>
<div class="main" id="main">
<div class="panel">
<video id="player" controls playsinline autoplay></video>
<div class="info">
<span id="fname"></span><br>
Currently labeled as: <span class="cur" id="curLabel"></span>
</div>
<div class="kbd"><kbd>1</kbd>-<kbd>0</kbd> select &nbsp;<kbd>Enter</kbd> confirm &nbsp;<kbd>S</kbd> skip &nbsp;<kbd>Space</kbd> play/pause &nbsp;<kbd>&rarr;</kbd> next</div>
</div>
<div class="sidebar">
<h2>What specific trick is this?</h2>
<p>Watch the clip and pick the exact trick. This teaches the model to distinguish between different flips.</p>
<details style="margin-bottom:10px;padding:8px;background:#0d0d14;border:1px solid #222;border-radius:6px">
<summary style="font-size:12px;color:#888;cursor:pointer">+ Add new trick</summary>
<div style="margin-top:8px;display:flex;flex-direction:column;gap:6px">
<input type="text" id="ntName" placeholder="Trick name (e.g. Butterfly Kick)" style="padding:6px;background:#1a1a24;border:1px solid #333;border-radius:4px;color:#fff;font-size:12px">
<input type="text" id="ntId" placeholder="auto-generated ID" readonly style="padding:6px;background:#1a1a24;border:1px solid #333;border-radius:4px;color:#666;font-size:12px">
<button onclick="addNewTrick()" style="padding:6px;background:#4361ee;border:none;border-radius:4px;color:#fff;font-size:12px;cursor:pointer;font-weight:600">Add to list</button>
<div id="ntMsg" style="font-size:11px"></div>
</div>
</details>
<div id="btnList"></div>
<div class="skip">
<button class="btn" id="skipBtn">Not a flip / Skip</button>
</div>
</div>
</div>
<div class="toast" id="toast"></div>
<script>
const TRICKS={tricks_json};
let clips=[],ci=0,sel=null;

async function init(){{
  const r=await fetch('/api/clips').then(r=>r.json());
  clips=r.clips;
  document.getElementById('nTotal').textContent=r.total;
  // Skip already done
  while(ci<clips.length&&clips[ci].relabeled)ci++;
  buildBtns();updStats();
  if(ci<clips.length)loadClip();else showDone();
}}

function updStats(){{
  const done=clips.filter(c=>c.relabeled).length;
  document.getElementById('nDone').textContent=done;
  document.getElementById('nCur').textContent=ci+1;
  document.getElementById('progBar').style.width=(done/clips.length*100)+'%';
}}

function loadClip(){{
  const c=clips[ci];
  document.getElementById('player').src='/video/'+encodeURIComponent(c.filename);
  document.getElementById('fname').textContent=c.filename;
  document.getElementById('curLabel').textContent=c.relabeled||c.current_class;
  sel=null;
  document.querySelectorAll('.btn').forEach(b=>b.classList.remove('sel'));
}}

function buildBtns(){{
  const list=document.getElementById('btnList');
  TRICKS.filter(t=>t.id!=='not_a_flip').forEach(t=>{{
    const b=document.createElement('button');b.className='btn';b.setAttribute('data-t',t.id);
    const sp=document.createElement('span');sp.textContent=t.name;b.appendChild(sp);
    if(t.key){{const k=document.createElement('span');k.className='k';k.textContent=t.key;b.appendChild(k)}}
    b.onclick=function(){{pick(t.id)}};list.appendChild(b);
  }});
  document.getElementById('skipBtn').onclick=function(){{pick('not_a_flip')}};
}}

function pick(id){{
  sel=id;
  document.querySelectorAll('.btn').forEach(b=>{{
    b.classList.toggle('sel',b.getAttribute('data-t')===id);
  }});
  save();
}}

async function save(){{
  if(!sel)return;
  const c=clips[ci];
  await fetch('/api/relabel',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{file:c.filename,trick_id:sel}})}});
  c.relabeled=sel;
  toast(sel==='not_a_flip'?'Skipped':sel.replace(/_/g,' '));
  setTimeout(next,400);
}}

function next(){{
  ci++;
  while(ci<clips.length&&clips[ci].relabeled)ci++;
  updStats();
  if(ci<clips.length)loadClip();else showDone();
}}

function showDone(){{
  const done=clips.filter(c=>c.relabeled&&c.relabeled!=='not_a_flip').length;
  const m=document.getElementById('main');m.textContent='';
  const d=document.createElement('div');d.className='done';
  const h=document.createElement('h2');h.textContent='All flips relabeled!';
  const p=document.createElement('p');p.textContent=done+' clips labeled with specific tricks. Now retrain:';
  const c=document.createElement('code');c.textContent='python scripts/prepare_training.py\\npython scripts/finetune.py --epochs 20';
  d.appendChild(h);d.appendChild(p);d.appendChild(c);m.appendChild(d);
}}

function toast(msg){{const el=document.getElementById('toast');el.textContent=msg;el.classList.add('show');setTimeout(()=>el.classList.remove('show'),800)}}

document.addEventListener('keydown',function(e){{
  if(e.target.tagName==='INPUT')return;
  if(e.key==='Enter'&&sel){{save();return}}
  if(e.key==='s'||e.key==='S'){{pick('not_a_flip');return}}
  if(e.key===' '){{e.preventDefault();const p=document.getElementById('player');p.paused?p.play():p.pause();return}}
  if(e.key==='ArrowRight'){{next();return}}
  const n=parseInt(e.key);
  if(!isNaN(n)){{const t=TRICKS.find(t=>t.key===String(n));if(t)pick(t.id)}}
}});

// Add new trick
var ntNameEl=document.getElementById('ntName');
if(ntNameEl)ntNameEl.oninput=function(){{
  var name=this.value.trim();
  var id=name.toLowerCase().replace(/[^a-z0-9]+/g,'_').replace(/^_|_$/g,'');
  document.getElementById('ntId').value=id;
}};

async function addNewTrick(){{
  var name=document.getElementById('ntName').value.trim();
  var id=document.getElementById('ntId').value.trim();
  var msg=document.getElementById('ntMsg');
  if(!name){{msg.textContent='Enter a name';msg.style.color='#f55';return}}
  var res=await fetch('/api/tricks/add',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{id:id,name:name}})}});
  var data=await res.json();
  if(data.status==='added'){{
    TRICKS.push(data.trick);
    buildBtns();
    msg.textContent='Added! Select it to label.';msg.style.color='#4c4';
    document.getElementById('ntName').value='';document.getElementById('ntId').value='';
  }}else{{
    msg.textContent=data.message||'Error';msg.style.color='#f55';
  }}
}}

init();
</script></body></html>"""


def main():
    clips_dir = Path("data/clips").resolve()
    manifest_path = Path("data/training/manifest.json")
    relabels_path = Path("data/training/relabels.json")

    with open(manifest_path) as f:
        manifest = json.load(f)

    # Get only flip/somersaulting clips
    coarse_flip_classes = {"flip", "trampoline_flip", "tumbling", "back_flip", "cartwheel"}
    flip_clips = [s for s in manifest["samples"] if s["class"] in coarse_flip_classes]

    # Load custom tricks from previous sessions
    custom_tricks_path = Path("data/training/custom_relabel_tricks.json")
    if custom_tricks_path.exists():
        with open(custom_tricks_path) as f:
            custom = json.load(f)
        for t in custom:
            if t["id"] not in [x["id"] for x in FLIP_TRICKS]:
                FLIP_TRICKS.append(t)
        print(f"  Loaded {len(custom)} custom tricks from previous session")

    # Load existing relabels
    existing = {}
    if relabels_path.exists():
        with open(relabels_path) as f:
            existing = json.load(f)

    done = sum(1 for c in flip_clips if c["original"] in existing)

    RelabelHandler.clips_dir = clips_dir
    RelabelHandler.manifest_path = manifest_path
    RelabelHandler.relabels_path = relabels_path
    RelabelHandler.flip_clips = flip_clips

    port = 8502
    server = HTTPServer(("0.0.0.0", port), RelabelHandler)

    print()
    print("  PkVision — Round 2: Relabel Flips")
    print("  " + "=" * 40)
    print(f"  Clips to relabel: {len(flip_clips)} (flip/tumbling/trampoline)")
    print(f"  Already done:     {done}")
    print(f"  Remaining:        {len(flip_clips) - done}")
    print()
    print(f"  Open: http://localhost:{port}")
    print()
    print("  Keys: 1=back_flip 2=front_flip 3=side_flip 4=gainer 5=webster")
    print("        6=aerial 7=wall_flip 8=double_back 9=double_front 0=flash_kick")
    print("        S=skip  Space=play/pause  Enter=confirm")
    print()
    print("  Auto-saves on every selection. Close with Ctrl+C when done.")
    print()

    import webbrowser
    webbrowser.open(f"http://localhost:{port}")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
        if relabels_path.exists():
            with open(relabels_path) as f:
                rl = json.load(f)
            counts = {}
            for v in rl.values():
                counts[v] = counts.get(v, 0) + 1
            print(f"\n{len(rl)} clips relabeled:")
            for t, c in sorted(counts.items(), key=lambda x: -x[1]):
                print(f"  {t}: {c}")
            print(f"\nNext: python scripts/prepare_training.py && retrain on Colab")


if __name__ == "__main__":
    main()
