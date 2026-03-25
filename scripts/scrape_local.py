"""Scrape parkourtheory.com — 100% reliability version.

Strategy: Stay inside the SPA. Never trigger full page reloads.
When content doesn't load (white page), wait and retry patiently.
Only marks a trick as done when we CONFIRM the data is actually there.

Usage:
    python scripts/scrape_local.py              # all 1,837
    python scripts/scrape_local.py --max 20     # test
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def wait_for_content(page, max_wait: int = 20) -> bool:
    """Wait until trick content actually appears on the page.

    Returns True if real content loaded, False if still a white/empty page.
    Checks for the detail-header which only exists on loaded trick pages.
    """
    for sec in range(max_wait):
        time.sleep(1)
        has_content = page.evaluate("""
            () => {
                const header = document.querySelector('h2.detail-header');
                const typeField = document.querySelector('b');
                const body = document.body.innerText || '';
                // Must have the trick header AND more than just nav bar
                return !!(header && body.length > 150);
            }
        """)
        if has_content:
            return True
    return False


def wait_for_search_results(page, max_wait: int = 15) -> bool:
    """Wait until search results actually appear."""
    for sec in range(max_wait):
        time.sleep(1)
        body = page.inner_text("body")
        if "results for" in body.lower() and len(body) > 100:
            return True
    return False


def extract_trick_data(page) -> dict:
    """Extract all fields from a loaded trick page via a single JS evaluation."""
    return page.evaluate("""
        () => {
            const data = {
                type: '', pronunciation: '', description: '',
                video_url: '', video_source: '', video_time: '',
                main_image: '', prerequisites: [], subsequents: [],
                explore_related: [],
            };

            function getField(label) {
                for (const b of document.querySelectorAll('b')) {
                    if (b.innerText.trim() === label) {
                        let text = '';
                        let node = b.nextSibling;
                        while (node) {
                            if (node.nodeType === 1 && node.tagName === 'B') break;
                            if (node.nodeType === 3) text += node.textContent;
                            else if (node.nodeType === 1) text += node.innerText || '';
                            node = node.nextSibling;
                        }
                        return text.trim();
                    }
                }
                return '';
            }

            data.type = getField('Type:');
            data.pronunciation = getField('Pronunciation:');
            data.description = getField('Description:');
            data.video_source = getField('Source:');
            data.video_time = getField('Time:');

            const source = document.querySelector('source[src*="cloudflarestream"], video source, source');
            if (source) data.video_url = source.getAttribute('src') || '';

            const imgs = document.querySelectorAll('img[src*="imagedelivery"]');
            for (const img of imgs) {
                const alt = (img.getAttribute('alt') || '').toLowerCase();
                if (alt !== 'home' && !img.classList.contains('search-icon')) {
                    data.main_image = img.getAttribute('src') || '';
                    break;
                }
            }

            function getSection(heading) {
                const names = [];
                for (const h of document.querySelectorAll('h3')) {
                    if (h.innerText.trim().toLowerCase().startsWith(heading)) {
                        let el = h.nextElementSibling;
                        while (el && el.tagName !== 'H3') {
                            el.querySelectorAll('span.move-name').forEach(s => names.push(s.innerText.trim()));
                            el = el.nextElementSibling;
                        }
                        break;
                    }
                }
                return names;
            }

            data.prerequisites = getSection('prerequisite');
            data.subsequents = getSection('subsequent');

            document.querySelectorAll('h3.related-moves-title ~ a span.move-name, h3.related-moves-title ~ div a span.move-name')
                .forEach(s => data.explore_related.push(s.innerText.trim()));

            return data;
        }
    """)


def go_home(page):
    """Navigate back to homepage via SPA (click logo)."""
    try:
        page.click("a[href='/']")
        time.sleep(1.5)
    except Exception:
        pass


def do_search(page, name: str) -> bool:
    """Type trick name in search bar and wait for results. Returns True if results loaded."""
    search_input = page.query_selector("input")
    if not search_input:
        go_home(page)
        time.sleep(2)
        search_input = page.query_selector("input")
    if not search_input:
        return False

    search_input.click()
    search_input.fill("")
    time.sleep(0.2)
    search_input.fill(name)
    search_input.press("Enter")

    return wait_for_search_results(page)


def click_result(page, name: str) -> bool:
    """Click the best matching search result. Returns True if trick page loaded."""
    result_links = page.query_selector_all("a[href*='/move/']")

    # Exact match first
    for link in result_links:
        h4 = link.query_selector("h4.result-header")
        if h4 and h4.inner_text().strip().lower() == name.lower():
            link.click()
            return wait_for_content(page)

    # Fallback: first result
    for link in result_links:
        h4 = link.query_selector("h4.result-header")
        if h4:
            link.click()
            return wait_for_content(page)

    return False


def scrape_one_trick(page, name: str, trick: dict) -> bool:
    """Scrape one trick entirely via SPA navigation. Returns True on success."""
    # Search
    if not do_search(page, name):
        go_home(page)
        return False

    # Click into trick page
    if not click_result(page, name):
        go_home(page)
        return False

    # Extract — content is confirmed loaded by wait_for_content
    trick_data = extract_trick_data(page)
    trick.update({"url": page.url, **trick_data})

    # Go back home for next trick
    go_home(page)

    # Verify we actually got data
    return bool(trick_data.get("type") or trick_data.get("description"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/parkourtheory_tricks.json")
    parser.add_argument("--output", default="data/parkourtheory_detailed.json")
    parser.add_argument("--max", type=int, default=0)
    args = parser.parse_args()

    from playwright.sync_api import sync_playwright

    # Load existing progress
    detailed = []
    scraped_names = set()
    if os.path.exists(args.output):
        with open(args.output) as f:
            detailed = json.load(f)
        scraped_names = {t["name"] for t in detailed if t.get("type") or t.get("description")}
        print(f"Resuming: {len(scraped_names)} already done")

    with open(args.input) as f:
        tricks = json.load(f)

    to_scrape = [t for t in tricks if t["name"] not in scraped_names]
    if args.max > 0:
        to_scrape = to_scrape[:args.max]

    print(f"Total: {len(tricks)} | Done: {len(scraped_names)} | Remaining: {len(to_scrape)}")
    if not to_scrape:
        print("All done!")
        return

    with sync_playwright() as p:
        session_num = [0]

        def launch_fresh():
            """Launch a fresh Chrome browser and pass Cloudflare."""
            br = p.chromium.launch(
                headless=False,
                channel="chrome",
                args=["--disable-blink-features=AutomationControlled"],
            )
            pg = br.new_page(viewport={"width": 1400, "height": 900})
            pg.add_init_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")

            session_num[0] += 1
            print(f"  [Session #{session_num[0]}] Loading parkourtheory.com...")
            pg.goto("https://parkourtheory.com", timeout=60000)
            time.sleep(10)

            title = pg.title()
            if "attention" in title.lower() or "cloudflare" in title.lower():
                print("  Cloudflare challenge — solve it in Chrome, press Enter here.")
                input("  Press Enter when ready...")
                time.sleep(3)

            # Verify search works
            if not do_search(pg, "Back Flip"):
                print("  Search failed on fresh session, waiting 10s and retrying...")
                time.sleep(10)
                pg.goto("https://parkourtheory.com", timeout=60000)
                time.sleep(10)
                if not do_search(pg, "Back Flip"):
                    print("  Still failing. Will try again on next restart.")
            go_home(pg)
            return br, pg

        browser, page = launch_fresh()
        print("Search works!\n")

        print(f"Scraping {len(to_scrape)} tricks. Ctrl+C saves progress.\n")
        start = time.time()
        failed_tricks = []
        consecutive_fails = 0

        try:
            for i, trick in enumerate(to_scrape):
                name = trick["name"]

                # If too many consecutive failures, restart browser
                if consecutive_fails >= 2:
                    print(f"    Session dead — restarting browser...")
                    try:
                        browser.close()
                    except Exception:
                        pass
                    time.sleep(5)
                    browser, page = launch_fresh()
                    consecutive_fails = 0

                # Try scraping
                success = False
                for attempt in range(3):
                    if scrape_one_trick(page, name, trick):
                        success = True
                        consecutive_fails = 0
                        break
                    else:
                        if attempt == 0:
                            # First fail: maybe just a slow load, retry in same session
                            time.sleep(3)
                        else:
                            # Second+ fail: session is dead, restart browser
                            print(f"    Attempt {attempt+1} failed — restarting browser...")
                            try:
                                browser.close()
                            except Exception:
                                pass
                            time.sleep(5)
                            browser, page = launch_fresh()

                if not success:
                    trick.update({"error": "failed after 3 attempts with restarts"})
                    failed_tricks.append(name)
                    consecutive_fails += 1
                else:
                    consecutive_fails = 0

                detailed.append(trick)

                # Save every 10 tricks
                if (i + 1) % 10 == 0 or i == len(to_scrape) - 1:
                    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
                    with open(args.output, "w", encoding="utf-8") as f:
                        json.dump(detailed, f, indent=2, ensure_ascii=False)

                    elapsed = time.time() - start
                    rate = (i + 1) / elapsed if elapsed > 0 else 1
                    eta = (len(to_scrape) - i - 1) / rate / 60 if rate > 0 else 0
                    has_desc = sum(1 for t in detailed if t.get("description"))
                    errors = len(failed_tricks)
                    pct = has_desc / max(i + 1, 1) * 100

                    print(
                        f"  [{i+1:4d}/{len(to_scrape)}] {name:35s} "
                        f"| {pct:.0f}% ok ({has_desc} done, {errors} failed) "
                        f"| ~{eta:.0f}m left"
                    )

        except KeyboardInterrupt:
            print("\n\nSaving progress...")
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(detailed, f, indent=2, ensure_ascii=False)
            print(f"Saved {len(detailed)} tricks. Run again to resume.")

        browser.close()

    # Summary
    has_desc = sum(1 for t in detailed if t.get("description"))
    has_video = sum(1 for t in detailed if t.get("video_url"))
    errors = sum(1 for t in detailed if t.get("error"))

    print(f"\nDone! {has_desc}/{len(detailed)} success ({has_desc/max(len(detailed),1)*100:.0f}%)")
    print(f"Videos: {has_video} | Errors: {errors}")

    if failed_tricks:
        print(f"\nFailed ({len(failed_tricks)}): {failed_tricks[:20]}")
        print("Run again to retry these.")


if __name__ == "__main__":
    main()
