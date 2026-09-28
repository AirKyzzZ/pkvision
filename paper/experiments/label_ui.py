#!/usr/bin/env python3
"""Tiny local labeling UI for paper/experiments/ground_truth.csv.

Runs a FastAPI app at http://127.0.0.1:8765 that shows one clip per page with
a video player, a text box (with autocomplete from the FIG table) for the
trick name, and optional boxes for flip / twist / direction / takeoff. Saving
a row auto-looks-up the D-score from data/fig_tricks_2025.json.

Navigation:
    - N or -> / Enter: save + next
    - P or <-:        previous
    - S:              skip (marks as dropped by leaving fig_name blank)
    - D:              delete current row's label (sets fig_name to empty)

Run:
    ./venv/bin/python paper/experiments/label_ui.py
    open http://127.0.0.1:8766
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, JSONResponse

REPO = Path(__file__).resolve().parents[2]
CSV_PATH = REPO / "paper" / "experiments" / "ground_truth.csv"
FIG_JSON = REPO / "data" / "fig_tricks_2025.json"

FIELDNAMES = [
    "clip_path", "pool", "fig_name", "d_score",
    "flip_count", "twist_count", "direction", "takeoff", "notes",
]


def load_fig_lookup() -> tuple[list[str], dict[str, dict]]:
    """Return (all_names, name_lower -> {d_score, flip, twist, direction, category})."""
    data = json.loads(FIG_JSON.read_text())
    names: list[str] = []
    lookup: dict[str, dict] = {}
    for cat_key, cat in data["categories"].items():
        for t in cat["tricks"]:
            name = t["name"]
            names.append(name)
            entry = {
                "d_score": float(t.get("score", 0) or 0),
                "flip": t.get("flip", 0),
                "twist": t.get("twist", 0),
                "direction": t.get("direction") or "",
                "takeoff": t.get("takeoff") or "",
                "category": cat_key,
            }
            lookup[name.lower()] = entry
            for alias in t.get("aliases", []) or []:
                names.append(alias)
                lookup[alias.lower()] = entry
    return sorted(set(names)), lookup


def read_rows() -> list[dict]:
    with CSV_PATH.open() as f:
        return list(csv.DictReader(f))


def write_rows(rows: list[dict]) -> None:
    with CSV_PATH.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        w.writeheader()
        w.writerows(rows)


FIG_NAMES, FIG_LOOKUP = load_fig_lookup()

app = FastAPI()


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #

@app.get("/", response_class=HTMLResponse)
async def root() -> HTMLResponse:
    rows = read_rows()
    # Jump to the first unfilled row.
    for i, r in enumerate(rows):
        if not r["fig_name"].strip():
            return RedirectResponse(f"/row/{i}")
    return RedirectResponse("/row/0")


@app.get("/clip")
async def clip(path: str) -> FileResponse:
    """Serve a clip file; path is relative to the repo root."""
    target = (REPO / path).resolve()
    if REPO not in target.parents and target != REPO:
        raise HTTPException(400, "bad path")
    if not target.exists():
        raise HTTPException(404, "not found")
    return FileResponse(target)


@app.get("/row/{idx}", response_class=HTMLResponse)
async def row(idx: int) -> HTMLResponse:
    rows = read_rows()
    if not 0 <= idx < len(rows):
        raise HTTPException(404, f"row {idx} out of range (0..{len(rows) - 1})")
    r = rows[idx]
    total = len(rows)
    filled = sum(1 for x in rows if x["fig_name"].strip())
    return HTMLResponse(render_page(idx, total, filled, r))


@app.post("/save/{idx}")
async def save(
    idx: int,
    fig_name: str = Form(""),
    flip_count: str = Form(""),
    twist_count: str = Form(""),
    direction: str = Form(""),
    takeoff: str = Form(""),
    notes: str = Form(""),
    action: str = Form("next"),
) -> RedirectResponse:
    rows = read_rows()
    if not 0 <= idx < len(rows):
        raise HTTPException(404)

    r = rows[idx]
    name = fig_name.strip()
    r["fig_name"] = name
    r["notes"] = notes.strip()

    # Auto-fill d_score from FIG lookup when we know the name.
    entry = FIG_LOOKUP.get(name.lower())
    if entry:
        r["d_score"] = f"{entry['d_score']}"
        if not flip_count.strip():
            r["flip_count"] = str(entry["flip"])
        if not twist_count.strip():
            r["twist_count"] = str(entry["twist"])
        if not direction.strip():
            r["direction"] = entry["direction"] or ""
        if not takeoff.strip():
            r["takeoff"] = entry["takeoff"] or ""

    # User-provided overrides take precedence over auto-fills.
    if flip_count.strip():
        r["flip_count"] = flip_count.strip()
    if twist_count.strip():
        r["twist_count"] = twist_count.strip()
    if direction.strip():
        r["direction"] = direction.strip()
    if takeoff.strip():
        r["takeoff"] = takeoff.strip()

    rows[idx] = r
    write_rows(rows)

    nxt = idx
    if action == "next":
        nxt = min(idx + 1, len(rows) - 1)
    elif action == "prev":
        nxt = max(idx - 1, 0)
    elif action == "stay":
        nxt = idx
    return RedirectResponse(f"/row/{nxt}", status_code=303)


@app.post("/clear/{idx}")
async def clear(idx: int) -> RedirectResponse:
    rows = read_rows()
    rows[idx]["fig_name"] = ""
    rows[idx]["d_score"] = ""
    rows[idx]["flip_count"] = ""
    rows[idx]["twist_count"] = ""
    rows[idx]["direction"] = ""
    rows[idx]["takeoff"] = ""
    write_rows(rows)
    return RedirectResponse(f"/row/{idx}", status_code=303)


@app.get("/api/fig_names")
async def fig_names_api() -> JSONResponse:
    return JSONResponse(FIG_NAMES)


# --------------------------------------------------------------------------- #
# HTML template
# --------------------------------------------------------------------------- #

def render_page(idx: int, total: int, filled: int, r: dict) -> str:
    clip_path = r["clip_path"]
    pool = r["pool"]
    fig_name = r["fig_name"]
    flip = r["flip_count"]
    twist = r["twist_count"]
    direction = r["direction"]
    takeoff = r["takeoff"]
    notes = r["notes"]
    d_score = r["d_score"]

    prev_i = max(idx - 1, 0)
    next_i = min(idx + 1, total - 1)

    # Build autocomplete datalist options.
    options_html = "\n".join(f'<option value="{n}">' for n in FIG_NAMES)

    return f"""<!doctype html>
<html>
<head>
    <meta charset="utf-8">
    <title>PkVision label · {idx+1}/{total}</title>
    <style>
      :root {{ color-scheme: dark light; }}
      body {{
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        max-width: 960px; margin: 1.5rem auto; padding: 0 1rem;
        background: #111; color: #eee;
      }}
      h1 {{ font-size: 1.1rem; font-weight: 600; margin: 0 0 .3rem; color: #8ab4f8; }}
      .meta {{ color: #888; font-size: .85rem; margin-bottom: .5rem; }}
      .bar {{ height: 4px; background: #222; border-radius: 4px; overflow: hidden; margin: .25rem 0 1rem; }}
      .bar > div {{ height: 100%; background: #4caf50; }}
      video {{ width: 100%; max-height: 480px; background: #000; border-radius: 6px; }}
      form {{ margin-top: 1rem; display: grid; gap: .75rem; }}
      label {{ display: block; font-size: .85rem; color: #bbb; margin-bottom: .15rem; }}
      input[type=text], input[type=number], select {{
        width: 100%; padding: .55rem .65rem;
        background: #1c1c1c; color: #eee; border: 1px solid #333; border-radius: 4px;
        font: inherit;
      }}
      input[type=text]:focus, input[type=number]:focus, select:focus {{
        outline: none; border-color: #4caf50;
      }}
      .row {{ display: grid; grid-template-columns: 1fr 1fr 1fr 1fr; gap: .75rem; }}
      .actions {{ display: flex; gap: .5rem; flex-wrap: wrap; margin-top: .5rem; }}
      button {{
        padding: .55rem 1rem; border: 1px solid #444; background: #222; color: #eee;
        border-radius: 4px; cursor: pointer; font: inherit;
      }}
      button.primary {{ background: #4caf50; border-color: #4caf50; color: #000; font-weight: 600; }}
      button.danger {{ background: #2a1212; border-color: #663; color: #f88; }}
      kbd {{
        font-family: monospace; background: #222; border: 1px solid #444;
        border-radius: 3px; padding: 0 .35rem; font-size: .75rem;
      }}
      .hint {{ color: #888; font-size: .8rem; }}
      .skipme {{ color: #aaa; font-size: .85rem; }}
      a {{ color: #8ab4f8; }}
    </style>
</head>
<body>
    <h1>PkVision labeling</h1>
    <div class="meta">
      Row {idx+1} / {total} &nbsp;·&nbsp; {filled} filled &nbsp;·&nbsp; pool <b>{pool}</b>
      &nbsp;·&nbsp; <code>{clip_path}</code>
    </div>
    <div class="bar"><div style="width:{(filled / max(total,1)) * 100:.0f}%"></div></div>

    <video controls autoplay muted playsinline loop preload="auto" src="/clip?path={clip_path}"></video>

    <form method="post" action="/save/{idx}" id="form">
        <div>
            <label for="fig_name">FIG trick name (autocomplete from the 149-row table)</label>
            <input type="text" id="fig_name" name="fig_name" list="fig_names"
                   value="{fig_name}" autofocus autocomplete="off"
                   placeholder="e.g. Gainer Full, Double Cork, Krok">
            <datalist id="fig_names">
                {options_html}
            </datalist>
            <div class="hint">Leave empty + press Next to drop this clip from the benchmark.</div>
        </div>

        <div class="row">
            <div>
                <label for="flip_count">Flips</label>
                <input type="number" step="0.5" id="flip_count" name="flip_count" value="{flip}">
            </div>
            <div>
                <label for="twist_count">Twists</label>
                <input type="number" step="0.5" id="twist_count" name="twist_count" value="{twist}">
            </div>
            <div>
                <label for="direction">Direction</label>
                <select id="direction" name="direction">
                    <option value=""        {_sel('', direction)}>(auto)</option>
                    <option value="backward" {_sel('backward', direction)}>backward</option>
                    <option value="forward"  {_sel('forward', direction)}>forward</option>
                    <option value="side"     {_sel('side', direction)}>side</option>
                </select>
            </div>
            <div>
                <label for="takeoff">Takeoff</label>
                <select id="takeoff" name="takeoff">
                    <option value=""         {_sel('', takeoff)}>(auto)</option>
                    <option value="two_foot" {_sel('two_foot', takeoff)}>two-foot</option>
                    <option value="one_foot" {_sel('one_foot', takeoff)}>one-foot</option>
                    <option value="running"  {_sel('running', takeoff)}>running</option>
                    <option value="wall"     {_sel('wall', takeoff)}>wall</option>
                </select>
            </div>
        </div>

        <div>
            <label for="notes">Notes</label>
            <input type="text" id="notes" name="notes" value="{notes}">
        </div>

        <div class="hint">
          Current D-score (auto-filled from <code>fig_tricks_2025.json</code> on save): <b>{d_score or '—'}</b>
        </div>

        <div class="actions">
            <button type="submit" name="action" value="prev">← Prev</button>
            <button class="primary" type="submit" name="action" value="next">Save &amp; Next →</button>
            <button type="submit" name="action" value="stay">Save</button>
            <button class="danger" type="button" onclick="clearRow()">Clear</button>
            <span class="skipme">
              Shortcuts:
              <kbd>Enter</kbd> save+next · <kbd>←</kbd>/<kbd>→</kbd> prev/next
            </span>
        </div>
    </form>

<script>
const form = document.getElementById('form');
const nameInput = document.getElementById('fig_name');

// Enter submits as "next" (default), Shift+Enter stays on row.
nameInput.addEventListener('keydown', (e) => {{
  if (e.key === 'Enter' && !e.isComposing) {{
    e.preventDefault();
    const action = e.shiftKey ? 'stay' : 'next';
    const hid = document.createElement('input');
    hid.type = 'hidden'; hid.name = 'action'; hid.value = action;
    form.appendChild(hid);
    form.submit();
  }}
}});

document.addEventListener('keydown', (e) => {{
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
  if (e.key === 'ArrowLeft') window.location = '/row/{prev_i}';
  if (e.key === 'ArrowRight') window.location = '/row/{next_i}';
}});

async function clearRow() {{
  if (!confirm('Clear this row?')) return;
  await fetch('/clear/{idx}', {{ method: 'POST' }});
  window.location.reload();
}}
</script>
</body>
</html>"""


def _sel(value: str, current: str) -> str:
    return 'selected' if value == current else ''


def main() -> None:
    if not CSV_PATH.exists():
        raise SystemExit(f"ground_truth.csv not found at {CSV_PATH}. "
                         f"Run make_ground_truth_template.py first.")
    uvicorn.run(app, host="127.0.0.1", port=8766, log_level="warning")


if __name__ == "__main__":
    main()
