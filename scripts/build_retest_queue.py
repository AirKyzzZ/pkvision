#!/usr/bin/env python3
"""Build a frozen, unseen clip queue and a blind owner labeling page. No pose or counter output is used.

Strata come from manifest and name parser; order is sha1(slug). Clips already seen by the
probe or the person-selection audit are excluded.

Browsers cannot decode the corpus's MPEG-4 Part 2 clips, so the page plays H.264 copies in a
gitignored folder. Making those copies needs `pip install imageio-ffmpeg`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.recognition.trick_name_parser import parse_trick_name

MANIFEST = ROOT / "data/v5_attribute_training/attribute_manifest.json"
VOTES = ROOT / "data/labeling/claude_probe/gold99_votes.jsonl"
AUDIT = ROOT / "docs/science-superpowers/probes/2026-11-04-person-selection-audit.json"
CLIPS = ROOT / "data/parkourtheory_clips_cropped"
OUT = ROOT / "data/labeling/retest_2026-11"
PLAYABLE = OUT / "clips_h264"
PER_STRATUM = 6


def seen_slugs() -> set[str]:
    audit = json.loads(AUDIT.read_text())
    gold = {json.loads(line)["slug"] for line in VOTES.read_text().splitlines()}
    return gold | set(audit["flagged"]) | set(audit["unflagged"])


def stratum(slug: str, flip: float, twist: float, parsed: dict) -> str | None:
    if "cork" in slug:
        return "cork"
    odd = [t for t in (twist, parsed.get("twist")) if t is not None and float(t) >= 1.5 and float(t) % 1 == 0.5]
    if odd:
        return "odd1.5+"  # taken from either source: none has both agreeing
    if "twist" in parsed and abs(float(parsed["twist"]) - twist) > 0.01:
        return None
    if "flip" in parsed and abs(float(parsed["flip"]) - flip) > 0.01:
        return None
    if twist == 0:
        return "twist0_flip" if flip >= 1 else None
    if twist == 0.5:
        return "twist0.5"
    if twist == 1:
        return "twist1"
    return "twist2+" if twist >= 2 else None


def build_queue() -> list[dict]:
    seen = seen_slugs()
    groups: dict[str, list[dict]] = {}
    for clip in json.loads(MANIFEST.read_text())["clips"]:
        slug = clip["slug"]
        if slug in seen or slug.endswith("_in_back_out") or not (CLIPS / f"{slug}.mp4").exists():
            continue
        flip, twist = float(clip["flip"]), float(clip["twist"])
        key = stratum(slug, flip, twist, parse_trick_name(slug).cues)
        if key:
            groups.setdefault(key, []).append({"slug": slug, "stratum": key, "manifest_flip": flip, "manifest_twist": twist})
    queue = []
    for key in sorted(groups):
        queue += sorted(groups[key], key=lambda c: hashlib.sha1(c["slug"].encode()).hexdigest())[:PER_STRATUM]
    # Label order must not reveal the stratum.
    return sorted(queue, key=lambda c: hashlib.sha1(("order:" + c["slug"]).encode()).hexdigest())


PAGE = """<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Blind Trick Labels</title><style>
:root{--bg:#fff;--fg:#111;--mut:#666;--line:#ccc}@media (prefers-color-scheme:dark){:root{--bg:#141414;--fg:#eee;--mut:#aaa;--line:#444}}
body{background:var(--bg);color:var(--fg);font:16px system-ui;margin:0 auto;max-width:760px;padding:16px}video{width:100%;max-height:60vh;background:#000}
.row{margin:10px 0;display:flex;flex-wrap:wrap;gap:6px;align-items:center}.lab{width:64px;color:var(--mut)}button{font:inherit;padding:6px 11px;border:1px solid var(--line);background:none;color:inherit;border-radius:6px;cursor:pointer}
button.on{background:var(--fg);color:var(--bg)}#s{color:var(--mut)}</style></head><body><h1>Blind trick labels</h1>
<p>Count what you see in the main trick. Names are hidden on purpose; choose "name on screen" if a clip shows one. Answers stay in this browser until you press Download.</p>
<div id="s"></div><video id="v" controls loop muted playsinline></video><p id="err" role="alert"></p>
<div class="row"><span class="lab">Speed</span><button data-r=".1">0.1×</button><button data-r=".25">0.25×</button><button data-r=".5">0.5×</button><button data-r="1">1×</button></div>
<div class="row" id="flip"></div><div class="row" id="twist"></div><div class="row" id="cork"></div><div class="row" id="use"></div>
<div class="row"><button id="p">Previous</button><button id="n">Next</button><button id="d">Download answers</button></div>
<script>const Q=__QUEUE__;let A={};try{A=JSON.parse(localStorage.getItem("blind")||"{}")}catch(e){}
let i=Q.findIndex(s=>!A[s]||!A[s].done);if(i<0)i=0;const F={flip:["flip",[0,.5,1,1.5,2,2.5,3,"unsure"]],twist:["twist",[0,.5,1,1.5,2,2.5,3,3.5,4,"unsure"]],cork:["cork",["no","yes","unsure"]],use:["use",["one trick","several tricks","can't tell","name on screen"]]};
function save(){try{localStorage.setItem("blind",JSON.stringify(A))}catch(e){}}
function draw(){const s=Q[i],a=A[s]||{};document.getElementById("s").textContent="Clip "+(i+1)+" of "+Q.length+" · "+Object.values(A).filter(x=>x.done).length+" done";const v=document.getElementById("v");if(v.dataset.s!==s){v.dataset.s=s;document.getElementById("err").textContent="";v.src="clips_h264/"+s+".mp4";v.playbackRate=.25}
for(const k in F){const b=document.getElementById(k);b.replaceChildren();const l=document.createElement("span");l.className="lab";l.textContent=F[k][0];b.append(l);for(const o of F[k][1]){const e=document.createElement("button");e.textContent=o;if(a[k]===o)e.className="on";e.onclick=()=>{a[k]=o;a.done=["flip","twist","cork","use"].every(x=>a[x]!==undefined);A[s]=a;save();draw()};b.append(e)}}}
document.getElementById("v").onerror=()=>{document.getElementById("err").textContent="This clip could not play. Do not label it; tell Claude which clip number this is."};
document.querySelectorAll("[data-r]").forEach(b=>b.onclick=()=>document.getElementById("v").playbackRate=+b.dataset.r);
document.getElementById("p").onclick=()=>{i=Math.max(0,i-1);draw()};document.getElementById("n").onclick=()=>{i=Math.min(Q.length-1,i+1);draw()};
document.getElementById("d").onclick=()=>{const u=URL.createObjectURL(new Blob([JSON.stringify(A,null,1)],{type:"application/json"}));const a=document.createElement("a");a.href=u;a.download="blind_labels.json";a.click();URL.revokeObjectURL(u)};draw()</script></body></html>"""


def make_playable(queue: list[dict]) -> None:
    import imageio_ffmpeg
    PLAYABLE.mkdir(parents=True, exist_ok=True)
    for clip in queue:
        dest = PLAYABLE / f"{clip['slug']}.mp4"
        if not dest.exists():
            partial = dest.with_suffix(".partial.mp4")
            subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(CLIPS / f"{clip['slug']}.mp4"),
                            "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
                            "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-an", "-map_metadata", "-1", str(partial)], check=True)
            partial.replace(dest)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rebuild", action="store_true", help="replace the frozen queue; invalidates any labels made from it")
    args = parser.parse_args()
    queue = build_queue() if args.rebuild else json.loads((OUT / "queue.json").read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    make_playable(queue)
    if args.rebuild:
        (OUT / "queue.json").write_text(json.dumps(queue, indent=1))
    (OUT / "label.html").write_text(PAGE.replace("__QUEUE__", json.dumps([c["slug"] for c in queue])))
    print(f"{len(queue)} clips ready")  # never print names or counts: the owner runs this before labeling


if __name__ == "__main__":
    main()
