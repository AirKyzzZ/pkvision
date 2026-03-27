"""Scrape parkourtheory.com trick database using Playwright.

Requires a real browser to bypass Cloudflare protection.
Run on a machine with a desktop browser installed (e.g., windows-dev).

Usage:
    python scripts/scrape_parkourtheory.py --output data/parkourtheory_tricks.json

Output format:
    [
        {
            "name": "Back Flip",
            "url": "https://parkourtheory.com/m/Back%20Flip",
            "description": "...",
            "categories": ["flip"],
            "related_moves": [...],
            "videos": [...]
        },
        ...
    ]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def scrape_all_trick_names(page) -> list[dict]:
    """Get all trick names and URLs from the allMoves category page."""
    logger.info("Loading all moves page...")
    page.goto("https://parkourtheory.com/category/allMoves", wait_until="networkidle", timeout=60000)
    time.sleep(3)

    # Try to find trick links on the page
    # The site structure might use different selectors — try common patterns
    tricks = []

    # Try getting all links that point to /m/ or /move/ paths
    links = page.query_selector_all("a[href*='/m/'], a[href*='/move/']")
    if not links:
        # Fallback: try all links and filter
        links = page.query_selector_all("a")

    seen_urls = set()
    for link in links:
        href = link.get_attribute("href") or ""
        text = (link.inner_text() or "").strip()

        if not text or len(text) > 100:
            continue

        # Filter for trick pages
        if "/m/" in href or "/move/" in href:
            full_url = href if href.startswith("http") else f"https://parkourtheory.com{href}"
            if full_url not in seen_urls:
                seen_urls.add(full_url)
                tricks.append({"name": text, "url": full_url})

    logger.info("Found %d trick links on allMoves page", len(tricks))

    # If the page uses pagination or lazy loading, scroll to load more
    if len(tricks) < 100:
        logger.info("Few tricks found, trying to scroll for more...")
        for _ in range(20):
            page.keyboard.press("End")
            time.sleep(1)

        links = page.query_selector_all("a[href*='/m/'], a[href*='/move/']")
        for link in links:
            href = link.get_attribute("href") or ""
            text = (link.inner_text() or "").strip()
            if text and ("/m/" in href or "/move/" in href):
                full_url = href if href.startswith("http") else f"https://parkourtheory.com{href}"
                if full_url not in seen_urls:
                    seen_urls.add(full_url)
                    tricks.append({"name": text, "url": full_url})

        logger.info("After scrolling: %d trick links", len(tricks))

    # Also try category sub-pages
    category_urls = [
        "https://parkourtheory.com/category/flipMoves",
        "https://parkourtheory.com/category/vaultMoves",
        "https://parkourtheory.com/category/spinMoves",
        "https://parkourtheory.com/category/barMoves",
        "https://parkourtheory.com/category/wallMoves",
        "https://parkourtheory.com/category/groundMoves",
        "https://parkourtheory.com/category/precisionMoves",
    ]

    for cat_url in category_urls:
        try:
            page.goto(cat_url, wait_until="networkidle", timeout=30000)
            time.sleep(2)
            cat_links = page.query_selector_all("a[href*='/m/'], a[href*='/move/']")
            for link in cat_links:
                href = link.get_attribute("href") or ""
                text = (link.inner_text() or "").strip()
                if text and ("/m/" in href or "/move/" in href):
                    full_url = href if href.startswith("http") else f"https://parkourtheory.com{href}"
                    if full_url not in seen_urls:
                        seen_urls.add(full_url)
                        tricks.append({"name": text, "url": full_url})
            logger.info("  %s: found %d new tricks (total: %d)", cat_url.split("/")[-1], len(cat_links), len(tricks))
        except Exception as e:
            logger.warning("  Failed to load %s: %s", cat_url, e)

    return tricks


def scrape_trick_details(page, trick: dict) -> dict:
    """Scrape detailed info from a single trick page."""
    try:
        page.goto(trick["url"], wait_until="networkidle", timeout=30000)
        time.sleep(1)

        # Extract page content
        content = page.inner_text("body")

        # Try to find description
        description = ""
        desc_el = page.query_selector(".move-description, .description, article p, .content p")
        if desc_el:
            description = desc_el.inner_text().strip()

        # Try to find categories/tags
        categories = []
        tag_els = page.query_selector_all(".tag, .category, .badge, [class*='tag'], [class*='category']")
        for el in tag_els:
            text = el.inner_text().strip()
            if text and len(text) < 50:
                categories.append(text)

        # Try to find related moves
        related = []
        related_els = page.query_selector_all("a[href*='/m/'], a[href*='/move/']")
        for el in related_els:
            text = el.inner_text().strip()
            if text and text != trick["name"] and len(text) < 80:
                related.append(text)

        # Try to find video embeds
        videos = []
        video_els = page.query_selector_all("iframe[src*='youtube'], iframe[src*='vimeo'], video source")
        for el in video_els:
            src = el.get_attribute("src") or ""
            if src:
                videos.append(src)

        trick.update({
            "description": description,
            "categories": categories,
            "related_moves": list(set(related))[:10],
            "videos": videos,
            "raw_content": content[:2000],
        })

    except Exception as e:
        logger.warning("Failed to scrape %s: %s", trick["name"], e)
        trick.update({"description": "", "categories": [], "related_moves": [], "videos": [], "error": str(e)})

    return trick


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape parkourtheory.com tricks")
    parser.add_argument("--output", type=str, default="data/parkourtheory_tricks.json")
    parser.add_argument("--names-only", action="store_true", help="Only scrape trick names, skip details")
    parser.add_argument("--max-details", type=int, default=0, help="Max tricks to scrape details for (0=all)")
    parser.add_argument("--resume", action="store_true", help="Resume from existing output file")
    args = parser.parse_args()

    from playwright.sync_api import sync_playwright

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume from existing file
    existing_tricks = []
    scraped_urls = set()
    if args.resume and output_path.exists():
        with open(output_path) as f:
            existing_tricks = json.load(f)
        scraped_urls = {t["url"] for t in existing_tricks if t.get("description") is not None}
        logger.info("Resuming: %d tricks already scraped", len(scraped_urls))

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        )
        page = context.new_page()

        # Phase 1: Get all trick names
        if not existing_tricks:
            tricks = scrape_all_trick_names(page)
            logger.info("Total unique tricks found: %d", len(tricks))

            # Save names immediately
            with open(output_path, "w") as f:
                json.dump(tricks, f, indent=2, ensure_ascii=False)
            logger.info("Saved trick names to %s", output_path)
        else:
            tricks = existing_tricks

        if args.names_only:
            browser.close()
            return

        # Phase 2: Scrape details for each trick
        to_scrape = [t for t in tricks if t["url"] not in scraped_urls]
        if args.max_details > 0:
            to_scrape = to_scrape[:args.max_details]

        logger.info("Scraping details for %d tricks...", len(to_scrape))

        for i, trick in enumerate(to_scrape):
            scrape_trick_details(page, trick)

            if (i + 1) % 25 == 0:
                # Save progress periodically
                all_tricks = existing_tricks + to_scrape[:i + 1]
                with open(output_path, "w") as f:
                    json.dump(all_tricks, f, indent=2, ensure_ascii=False)
                logger.info("  Progress: %d/%d scraped, saved to %s", i + 1, len(to_scrape), output_path)

            # Be respectful - don't hammer the server
            time.sleep(0.5)

        # Final save
        all_tricks = existing_tricks + to_scrape
        with open(output_path, "w") as f:
            json.dump(all_tricks, f, indent=2, ensure_ascii=False)

        browser.close()

    logger.info("Done! Scraped %d tricks total, saved to %s", len(all_tricks), output_path)


if __name__ == "__main__":
    main()
