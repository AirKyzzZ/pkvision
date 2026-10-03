from __future__ import annotations

import argparse
import difflib
import json
import logging
import re
import time
import urllib.request
from datetime import date
from pathlib import Path
from urllib.parse import unquote

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BASE = "https://www.parkourtheory.com"
USER_AGENT = "PkVision research crawler (contact: airkyzzz.jiji@gmail.com)"
CRAWL_DELAY_S = 20.0
SITEMAPS = ["sitemap-moves-1.xml", "sitemap-categories.xml"]
RENAME_CUTOFF = 0.8

_last_request = 0.0


def polite_get(url: str) -> bytes:
    global _last_request
    wait = CRAWL_DELAY_S - (time.time() - _last_request)
    if wait > 0:
        time.sleep(wait)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return resp.read()
    finally:
        _last_request = time.time()


def parse_sitemap(xml: str) -> list[tuple[str, str]]:
    return [
        (unquote(loc), lastmod)
        for loc, lastmod in re.findall(r"<loc>(.*?)</loc>(?:<lastmod>(.*?)</lastmod>)?", xml)
    ]


def slug_of(name: str) -> str:
    return name.lower().replace(" ", "_").replace("/", "-")


def build_diff(moves: dict[str, str], old: list[dict], since: str) -> dict:
    old_by_slug = {slug_of(t["name"]): t for t in old}
    new = sorted(set(moves) - set(old_by_slug))
    removed = sorted(set(old_by_slug) - set(moves))
    common = set(moves) & set(old_by_slug)

    renames = []
    for slug in removed:
        match = difflib.get_close_matches(slug, new, n=1, cutoff=RENAME_CUTOFF)
        if match:
            renames.append({"old": old_by_slug[slug]["name"], "new_slug": match[0]})

    return {
        "new": [{"slug": s, "url": f"{BASE}/move/{s}", "lastmod": moves[s]} for s in new],
        "removed": [{"name": old_by_slug[s]["name"], "url": old_by_slug[s].get("url", "")} for s in removed],
        "rename_candidates_by_slug": renames,
        "lastmod_after_old_scrape": sorted(
            ({"name": old_by_slug[s]["name"], "lastmod": moves[s]} for s in common if moves[s][:10] > since),
            key=lambda t: t["lastmod"],
        ),
        "failed_in_old_scrape": sorted(old_by_slug[s]["name"] for s in common if old_by_slug[s].get("error")),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="List parkourtheory.com moves and variations from its sitemap and diff against an old scrape")
    parser.add_argument("--old", default="data/parkourtheory_detailed.json")
    parser.add_argument("--old-date", default="2026-03-26")
    parser.add_argument("--out-dir", default=f"data/raw/parkourtheory/{date.today().isoformat()}")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    entries = []
    for name in SITEMAPS:
        xml = polite_get(f"{BASE}/{name}").decode("utf-8")
        (out_dir / name).write_text(xml, encoding="utf-8")
        entries += parse_sitemap(xml)
        logger.info("Fetched %s", name)

    moves = {u.rsplit("/", 1)[1]: m for u, m in entries if "/move/" in u}
    variations = sorted(u.rsplit("/", 1)[1] for u, _ in entries if "/variation/" in u)

    with open(args.old) as f:
        old = json.load(f)

    changes = build_diff(moves, old, args.old_date)
    counts = {"live_moves": len(moves), "live_variations": len(variations), "old_moves": len(old)}
    counts |= {k: len(v) for k, v in changes.items()}
    report = {
        "listed_at": date.today().isoformat(),
        "old_file": args.old,
        "counts": counts,
        "variation_slugs": variations,
        **changes,
    }
    with open(out_dir / "diff.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    logger.info("Counts: %s", counts)
    logger.info("Saved to %s", out_dir)


if __name__ == "__main__":
    main()
