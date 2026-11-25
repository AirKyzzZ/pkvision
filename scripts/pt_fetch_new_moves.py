#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

USER_AGENT = "PkVision research crawler"
CRAWL_DELAY_S = 20.0
META = re.compile(r'<meta (?:name|property)="([^"]+)" content="([^"]*)"')
STREAM = re.compile(r"(https://customer-[a-z0-9]+\.cloudflarestream\.com/[0-9a-f]{32})/thumbnails/")
MOVE_ID = re.compile(r"move\?id=([0-9a-f-]{36})")


def parse_move_page(page: str) -> dict:
    meta = dict(META.findall(page))
    stream = STREAM.search(page)
    move_id = MOVE_ID.search(meta.get("og:url", ""))
    return {
        "name": html.unescape(meta.get("og:title", "")).removesuffix(" - Parkour Theory"),
        "description": html.unescape(meta.get("description", "")),
        "video_url": f"{stream.group(1)}/downloads/default.mp4" if stream else "",
        "pt_id": move_id.group(1) if move_id else "",
    }


def fetch(url: str) -> str:
    req = urllib.request.Request(urllib.parse.quote(url, safe=":/?=&"), headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read().decode("utf-8", "replace")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Fetch name, description and demo video URL of the moves a sitemap diff lists as new")
    parser.add_argument("diff", type=Path, help="diff.json written by scrape_parkourtheory_v2.py")
    parser.add_argument("--out", type=Path, help="defaults to new_moves.json next to the diff")
    args = parser.parse_args(argv)

    todo = json.loads(args.diff.read_text(encoding="utf-8"))["new"]
    out = args.out or args.diff.with_name("new_moves.json")
    done = {m["slug"]: m for m in json.loads(out.read_text(encoding="utf-8"))} if out.exists() else {}
    last = 0.0
    for entry in todo:
        if entry["slug"] in done and not done[entry["slug"]].get("error"):
            continue
        time.sleep(max(0.0, CRAWL_DELAY_S - (time.time() - last)))
        try:
            move = {**entry, **parse_move_page(fetch(entry["url"]))}
        except (urllib.error.URLError, TimeoutError) as e:
            move = {**entry, "error": str(e)}
        last = time.time()
        done[entry["slug"]] = move
        out.write_text(json.dumps(list(done.values()), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"[{len(done)}/{len(todo)}] {entry['slug']}{' ERROR' if 'error' in move else ''}", flush=True)


if __name__ == "__main__":
    main()
