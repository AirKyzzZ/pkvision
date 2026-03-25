"""Parallel scraping of parkourtheory.com using multiple Chrome instances.

Splits 1,837 tricks across N workers, each with its own Chrome browser.
Merges results at the end.

Usage:
    python scripts/scrape_parallel.py --workers 6     # 6 Chrome windows
    python scripts/scrape_parallel.py --workers 8     # 8 Chrome windows (~15 min)
    python scripts/scrape_parallel.py --max 50        # test with 50 tricks
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

OUTPUT_DIR = Path("data/scrape_chunks")
FINAL_OUTPUT = Path("data/parkourtheory_detailed.json")


def extract_trick_data(page) -> dict:
    """Extract all structured fields from a rendered trick page."""
    data = {
        "type": "", "pronunciation": "", "description": "",
        "video_url": "", "video_source": "", "video_time": "",
        "main_image": "", "prerequisites": [], "subsequents": [],
        "explore_related": [],
    }
    try:
        data["type"] = page.evaluate("""() => {
            for (const b of document.querySelectorAll('b')) {
                if (b.innerText.trim() === 'Type:') {
                    const s = b.nextElementSibling;
                    return s ? s.innerText.trim() : '';
                }
            } return '';
        }""")

        data["description"] = page.evaluate("""() => {
            for (const b of document.querySelectorAll('b')) {
                if (b.innerText.trim() === 'Description:') {
                    const s = b.nextElementSibling;
                    return s ? s.innerText.trim() : '';
                }
            } return '';
        }""")

        data["pronunciation"] = page.evaluate("""() => {
            for (const b of document.querySelectorAll('b')) {
                if (b.innerText.trim() === 'Pronunciation:') {
                    let n = b.nextSibling;
                    while (n) {
                        if (n.nodeType === 3 && n.textContent.trim()) return n.textContent.trim();
                        if (n.nodeType === 1 && n.innerText?.trim()) return n.innerText.trim();
                        n = n.nextSibling;
                    }
                }
            } return '';
        }""")

        el = page.query_selector("source[src*='cloudflarestream'], video source, source")
        if el:
            data["video_url"] = el.get_attribute("src") or ""

        data["video_source"] = page.evaluate("""() => {
            for (const b of document.querySelectorAll('b')) {
                if (b.innerText.trim() === 'Source:') {
                    const s = b.nextElementSibling;
                    return s ? s.innerText.trim() : '';
                }
            } return '';
        }""")

        data["video_time"] = page.evaluate("""() => {
            for (const b of document.querySelectorAll('b')) {
                if (b.innerText.trim() === 'Time:') {
                    const s = b.nextElementSibling;
                    return s ? s.innerText.trim() : '';
                }
            } return '';
        }""")

        el = page.query_selector("img[src*='imagedelivery']:not(.search-icon):not([alt='Home'])")
        if el:
            data["main_image"] = el.get_attribute("src") or ""

        data["prerequisites"] = page.evaluate("""() => {
            for (const h of document.querySelectorAll('h3')) {
                if (h.innerText.trim().toLowerCase().startsWith('prerequisite')) {
                    const names = []; let el = h.nextElementSibling;
                    while (el && el.tagName !== 'H3') {
                        el.querySelectorAll('span.move-name').forEach(s => names.push(s.innerText.trim()));
                        el = el.nextElementSibling;
                    } return names;
                }
            } return [];
        }""")

        data["subsequents"] = page.evaluate("""() => {
            for (const h of document.querySelectorAll('h3')) {
                if (h.innerText.trim().toLowerCase().startsWith('subsequent')) {
                    const names = []; let el = h.nextElementSibling;
                    while (el && el.tagName !== 'H3') {
                        el.querySelectorAll('span.move-name').forEach(s => names.push(s.innerText.trim()));
                        el = el.nextElementSibling;
                    } return names;
                }
            } return [];
        }""")

        data["explore_related"] = page.evaluate("""() => {
            const names = [];
            document.querySelectorAll('h3.related-moves-title ~ a span.move-name, h3.related-moves-title ~ div a span.move-name')
                .forEach(s => names.push(s.innerText.trim()));
            return names;
        }""")

    except Exception as e:
        data["extraction_error"] = str(e)
    return data


def scrape_chunk(worker_id: int, trick_names: list[dict], output_path: str) -> dict:
    """Scrape a chunk of tricks in a single Chrome instance."""
    from playwright.sync_api import sync_playwright

    profile = os.path.expanduser(f"~/Library/Application Support/pkvision-scraper-w{worker_id}")
    results = []
    stats = {"total": len(trick_names), "success": 0, "empty": 0, "error": 0}

    with sync_playwright() as p:
        context = p.chromium.launch_persistent_context(
            profile,
            headless=False,
            channel="chrome",
            args=["--disable-blink-features=AutomationControlled"],
            viewport={"width": 1200, "height": 800},
        )
        page = context.pages[0] if context.pages else context.new_page()
        page.add_init_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")

        # Pass Cloudflare
        page.goto("https://parkourtheory.com", timeout=60000)
        time.sleep(8)

        for i, trick in enumerate(trick_names):
            name = trick["name"]
            try:
                page.goto(
                    f"https://parkourtheory.com/search?q={name.replace(' ', '+')}",
                    timeout=12000,
                )
                time.sleep(1.5)

                clicked = False
                for link in page.query_selector_all("a[href*='/move/']"):
                    h4 = link.query_selector("h4.result-header")
                    if h4 and h4.inner_text().strip().lower() == name.lower():
                        link.click()
                        time.sleep(2.5)
                        clicked = True
                        break

                if not clicked:
                    for link in page.query_selector_all("a[href*='/move/']"):
                        h4 = link.query_selector("h4.result-header")
                        if h4:
                            link.click()
                            time.sleep(2.5)
                            clicked = True
                            break

                if clicked:
                    body = page.inner_text("body")
                    if len(body) < 100:
                        time.sleep(2)

                    trick_data = extract_trick_data(page)
                    trick.update({"url": page.url, **trick_data})
                    if trick_data.get("description") or trick_data.get("type"):
                        stats["success"] += 1
                    else:
                        stats["empty"] += 1
                else:
                    trick.update({"error": "no search result"})
                    stats["empty"] += 1

            except Exception as e:
                trick.update({"error": str(e)})
                stats["error"] += 1

            results.append(trick)

            if (i + 1) % 10 == 0 or i == len(trick_names) - 1:
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                print(
                    f"  [W{worker_id}] {i+1}/{len(trick_names)} "
                    f"| ok={stats['success']} empty={stats['empty']} err={stats['error']}"
                )

        context.close()

    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/parkourtheory_tricks.json")
    parser.add_argument("--output", default=str(FINAL_OUTPUT))
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--max", type=int, default=0)
    args = parser.parse_args()

    # Load existing progress
    already_scraped = set()
    if os.path.exists(args.output):
        with open(args.output) as f:
            existing = json.load(f)
        already_scraped = {t["name"] for t in existing if t.get("type") or t.get("description")}
        print(f"Already scraped with content: {len(already_scraped)}")

    with open(args.input) as f:
        all_tricks = json.load(f)

    to_scrape = [t for t in all_tricks if t["name"] not in already_scraped]
    if args.max > 0:
        to_scrape = to_scrape[:args.max]

    print(f"Total: {len(all_tricks)} | Already done: {len(already_scraped)} | To scrape: {len(to_scrape)} | Workers: {args.workers}")

    if not to_scrape:
        print("Nothing to scrape!")
        return

    # Split into chunks
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    chunk_size = len(to_scrape) // args.workers + 1
    chunks = []
    for i in range(args.workers):
        start = i * chunk_size
        end = min(start + chunk_size, len(to_scrape))
        if start < len(to_scrape):
            chunk = to_scrape[start:end]
            chunk_path = str(OUTPUT_DIR / f"chunk_{i}.json")
            chunks.append((i, chunk, chunk_path))

    print(f"Split into {len(chunks)} chunks of ~{chunk_size} tricks each")
    print(f"Starting {len(chunks)} Chrome windows...\n")

    start_time = time.time()

    # Run workers in parallel (each is a separate process with its own Chrome)
    with ProcessPoolExecutor(max_workers=len(chunks)) as executor:
        futures = {
            executor.submit(scrape_chunk, wid, chunk, path): wid
            for wid, chunk, path in chunks
        }

        all_stats = {}
        for future in as_completed(futures):
            wid = futures[future]
            try:
                stats = future.result()
                all_stats[wid] = stats
                print(f"\n  Worker {wid} done: {stats}")
            except Exception as e:
                print(f"\n  Worker {wid} FAILED: {e}")

    # Merge all chunks
    print("\nMerging results...")
    merged = []
    if os.path.exists(args.output):
        with open(args.output) as f:
            merged = [t for t in json.load(f) if t.get("type") or t.get("description")]

    merged_names = {t["name"] for t in merged}

    for _, _, path in chunks:
        if os.path.exists(path):
            with open(path) as f:
                chunk_data = json.load(f)
            for t in chunk_data:
                if t["name"] not in merged_names:
                    merged.append(t)
                    merged_names.add(t["name"])

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)

    elapsed = time.time() - start_time
    has_desc = sum(1 for t in merged if t.get("description"))
    has_video = sum(1 for t in merged if t.get("video_url"))
    has_type = sum(1 for t in merged if t.get("type"))

    print(f"\nDone in {elapsed/60:.1f} min!")
    print(f"Total: {len(merged)} tricks | desc={has_desc} type={has_type} video={has_video}")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
