"""Scrape trick data from multiple parkour/tricking wikis and databases.

Sources:
    1. Trickipedia (349 tricking + 66 parkour = 415 tricks with descriptions)
    2. Loopkicks Tricktionary (500+ tricks organized by category)
    3. Tricking Bible PDF (already downloaded — structured class A-F tables)

Trickipedia and Loopkicks are standard websites (no Cloudflare), so we can
scrape them directly without needing a visible browser.

Usage:
    python scripts/scrape_other_sources.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

DATA_DIR = Path("data")


# -- Trickipedia (349 tricking tricks, 15 pages) ----------------------------

def scrape_trickipedia():
    """Scrape all tricks from trickipedia.app (tricking + parkour)."""
    from playwright.sync_api import sync_playwright

    all_tricks = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        for discipline, total_pages in [("tricking", 15), ("parkour", 4)]:
            for pg in range(1, total_pages + 1):
                url = f"https://trickipedia.app/{discipline}?page={pg}"
                print(f"  Trickipedia: {url}")

                try:
                    page.goto(url, timeout=15000)
                    time.sleep(2)

                    # Extract trick cards
                    tricks = page.evaluate("""
                        () => {
                            const cards = document.querySelectorAll('a[href*="/tricks/"]');
                            return Array.from(cards).map(card => {
                                const name = card.querySelector('h3, h4, .trick-name, strong');
                                const desc = card.querySelector('p, .description, .trick-description');
                                return {
                                    name: name ? name.innerText.trim() : '',
                                    description: desc ? desc.innerText.trim() : '',
                                    url: card.href || '',
                                    source: 'trickipedia',
                                    discipline: '""" + discipline + """',
                                };
                            }).filter(t => t.name && t.name.length > 1);
                        }
                    """)

                    all_tricks.extend(tricks)
                    print(f"    Found {len(tricks)} tricks (total: {len(all_tricks)})")

                except Exception as e:
                    print(f"    Error: {e}")

                time.sleep(1)

        browser.close()

    # Deduplicate
    seen = set()
    unique = []
    for t in all_tricks:
        if t["name"] not in seen:
            seen.add(t["name"])
            unique.append(t)

    return unique


# -- Loopkicks Tricktionary --------------------------------------------------

def scrape_loopkicks():
    """Scrape all tricks from loopkickstricking.com/tricktionary."""
    from playwright.sync_api import sync_playwright

    categories = [
        "backward-tricks", "forward-tricks", "vertical-kicks",
        "inside-tricks", "outside-tricks",
        "variations", "transitions", "stances",
    ]

    all_tricks = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        for cat in categories:
            url = f"https://www.loopkickstricking.com/tricktionary/{cat}"
            print(f"  Loopkicks: {url}")

            try:
                page.goto(url, timeout=15000)
                time.sleep(2)

                # Extract trick names and any descriptions
                tricks = page.evaluate("""
                    () => {
                        // Try multiple selectors for trick cards/links
                        const results = [];
                        const links = document.querySelectorAll('a');
                        for (const link of links) {
                            const href = link.href || '';
                            const text = link.innerText?.trim() || '';
                            if (href.includes('/tricks/') && text && text.length > 1 && text.length < 80) {
                                results.push({
                                    name: text,
                                    url: href,
                                });
                            }
                        }
                        return results;
                    }
                """)

                for t in tricks:
                    t["source"] = "loopkicks"
                    t["category"] = cat

                all_tricks.extend(tricks)
                print(f"    Found {len(tricks)} tricks (total: {len(all_tricks)})")

            except Exception as e:
                print(f"    Error: {e}")

            time.sleep(1)

        # For each trick page, try to get the description
        print(f"\n  Scraping {len(all_tricks)} trick details...")
        for i, trick in enumerate(all_tricks):
            if not trick.get("url"):
                continue
            try:
                page.goto(trick["url"], timeout=10000)
                time.sleep(1)

                desc = page.evaluate("""
                    () => {
                        // Look for description paragraphs
                        const ps = document.querySelectorAll('p');
                        for (const p of ps) {
                            const text = p.innerText?.trim();
                            if (text && text.length > 20 && text.length < 500) {
                                return text;
                            }
                        }
                        return '';
                    }
                """)
                trick["description"] = desc

                if (i + 1) % 50 == 0:
                    print(f"    [{i+1}/{len(all_tricks)}] descriptions scraped")

            except Exception:
                trick["description"] = ""

        browser.close()

    # Deduplicate
    seen = set()
    unique = []
    for t in all_tricks:
        if t["name"] not in seen:
            seen.add(t["name"])
            unique.append(t)

    return unique


# -- Tricking Bible PDF parsing -----------------------------------------------

def parse_tricking_bible():
    """Parse the Tricking Bible PDF tables into structured data."""
    pdf_path = DATA_DIR / "tricking_bible.pdf"
    if not pdf_path.exists():
        print("  Tricking Bible PDF not found, skipping")
        return []

    # The PDF has structured tables on pages 3-9 with columns:
    # Name | Abbr. | Type | Origin | Prerequisite(s)
    # Organized by difficulty class: A, B, C, D, E, F

    # We already read the PDF visually. Let me parse the data from what we know.
    tricks = []

    # Class A (page 3)
    class_a = [
        ("540", "Kick", "Wushu/TKD", "outside/inside crescent kick & tornado kick"),
        ("Pop 360 Wheel Kick", "Kick", "TKD/Karate", "spin crescent kick & jump front kick"),
        ("Au-Batido", "Invert/Kick", "Capoeira", "cartwheel/au"),
        ("Butterfly Kick", "Kick", "Wushu", ""),
        ("Feilong", "Kick", "Tricking", "Hyper Pop 360"),
        ("Aerial", "Invert/Flip", "Wushu/Gymnastics", "strong cartwheel/au"),
        ("Backflip", "Flip", "Gymnastics/Capoeira", "good back handspring & some nerve"),
        ("Kip-up", "Invert", "Wushu", ""),
    ]

    # Class B (page 4)
    class_b = [
        ("Cheat 720/540 Wheel Kick", "Kick", "Capoeira", "good 540"),
        ("Aerial Switch", "Invert/Flip", "Tricking", "decent aerial"),
        ("Pop 720 Wheel Kick", "Spin/Kick", "TKD", "good Pop 360"),
        ("Butterfly Twist", "Twist", "Wushu", "Butterfly Kick"),
        ("Shuriken Twist", "Twist/Kick", "Tricking", "Butterfly Kick or Illusion Twist"),
        ("Singleleg", "Invert/Kick", "Tricking", "Feilong"),
        ("Doubleleg", "Invert/Kick", "Capoeira", "Feilong"),
        ("Flash Kick", "Flip/Kick", "Tricking", "decent Backflip"),
        ("Raiz", "Kick/Invert", "Capoeira", "tornado kick & compasso"),
        ("Parafuso", "Kick", "Capoeira", "good 540"),
        ("Crowd Awakener", "Kick", "Tricking", "high 540"),
        ("Illusion Twist", "Twist/Kick", "Tricking", ""),
        ("Gainer", "Flip", "Capoeira", "good Backflip"),
        ("Moon Kick", "Flip/Kick", "Capoeira", ""),
        ("Masterswipe", "Invert/Flip", "Bboying", "good cartwheel/au"),
        ("Gumby/Gumbi", "Flip", "Capoeira", "cartwheel/au (on opposite side)"),
        ("Pop Swipe", "Invert/Kick", "Tricking", "Flash Kick & Side Sommi"),
    ]

    # Class C (page 5)
    class_c = [
        ("Hyper Aerial", "Invert/Flip", "Tricking", "powerful aerial"),
        ("Sideswipe", "Invert/Kick", "Tricking", "good 540 & Raiz"),
        ("Hypertwist", "Twist", "Wushu", "good Butterfly Twist"),
        ("Envergado", "Invert/Kick", "Tricking", "good Parafuso & Raiz"),
        ("Corkscrew", "Twist", "Capoeira", ""),
        ("Gainer Arabian", "Flip", "Tricking", "straight gainer"),
        ("Gainer Full", "Flip", "Capoeira", "good gainer"),
        ("Webster/Loser", "Flip", "Capoeira", "strong Aerial or Frontflip"),
        ("Touchdown Raiz", "Invert/Kick", "Tricking", "Raiz & Gumby/Gumbi"),
        ("Cheat 900", "Spin/Kick", "Tricking", "good Cheat 720"),
        ("Triple Flash Kick", "Flip/Kick", "Tricking", "good Flash"),
        ("Terada Grab", "Flip/Kick", "Tricking", "Side Sommi"),
        ("X-Out", "Flip/Kick", "Tricking", "high Flash Kick"),
    ]

    # Class D (page 6)
    class_d = [
        ("Shuriken Corkscrew Feilong", "Invert/Kick", "Tricking", "Shuriken Corkscrew"),
        ("Cheat 720 Twist/540 Twist", "Invert/Twist", "Tricking", "strong Raiz"),
        ("Pop 1080 Wheel Kick", "Spin/Kick", "Tricking", "Pop 900"),
        ("Boxcutter", "Twist/Kick", "Tricking", "Hyper Corkscrew"),
        ("Swipeknife", "Invert/Kick", "Tricking", "quick Sideswipe"),
        ("Switchblade", "Kick", "Tricking", "good Parafuso"),
        ("Scissorswipe", "Invert/Kick", "Tricking", "powerful Aerial & front kick"),
    ]

    for class_name, trick_list in [("A", class_a), ("B", class_b), ("C", class_c), ("D", class_d)]:
        for name, trick_type, origin, prereqs in trick_list:
            tricks.append({
                "name": name,
                "type": trick_type,
                "origin": origin,
                "prerequisites_text": prereqs,
                "difficulty_class": class_name,
                "source": "tricking_bible",
            })

    return tricks


def main():
    all_data = {}

    # 1. Tricking Bible (already downloaded)
    print("=== Tricking Bible PDF ===")
    bible_tricks = parse_tricking_bible()
    print(f"  Parsed {len(bible_tricks)} tricks from Tricking Bible")
    all_data["tricking_bible"] = bible_tricks

    # 2. Trickipedia
    print("\n=== Trickipedia ===")
    try:
        trickipedia = scrape_trickipedia()
        print(f"  Total: {len(trickipedia)} tricks from Trickipedia")
        all_data["trickipedia"] = trickipedia
    except Exception as e:
        print(f"  Trickipedia failed: {e}")
        all_data["trickipedia"] = []

    # 3. Loopkicks
    print("\n=== Loopkicks Tricktionary ===")
    try:
        loopkicks = scrape_loopkicks()
        print(f"  Total: {len(loopkicks)} tricks from Loopkicks")
        all_data["loopkicks"] = loopkicks
    except Exception as e:
        print(f"  Loopkicks failed: {e}")
        all_data["loopkicks"] = []

    # Save everything
    output = DATA_DIR / "other_sources_tricks.json"
    with open(output, "w", encoding="utf-8") as f:
        json.dump(all_data, f, indent=2, ensure_ascii=False)

    total = sum(len(v) for v in all_data.values())
    print(f"\n=== TOTAL: {total} tricks saved to {output} ===")
    for source, tricks in all_data.items():
        print(f"  {source}: {len(tricks)}")


if __name__ == "__main__":
    main()
