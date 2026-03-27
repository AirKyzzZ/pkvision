"""Scrape parkourtheory.com — v5: UC + aggressive Chrome restart.

The key insight: Chrome/React degrades after ~20-30 page loads.
Solution: restart Chrome every 10 tricks (proven reliable in tests).

Requirements:
    pip install undetected-chromedriver

Usage:
    DISPLAY=:42 python scripts/scrape_uc.py               # persistent Xvfb
    xvfb-run --auto-servernum python scripts/scrape_uc.py  # auto Xvfb
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("scrape_uc.log"),
    ],
)
logger = logging.getLogger(__name__)

INPUT_FILE = Path("data/parkourtheory_tricks.json")
OUTPUT_FILE = Path("data/parkourtheory_detailed.json")

EXTRACT_JS = """
return (function() {
    const data = {
        name_on_site: '',
        type: '', pronunciation: '', description: '',
        video_url: '', video_source: '', video_time: '',
        main_image: '', prerequisites: [], subsequents: [],
        explore_related: [],
    };
    const header = document.querySelector('h2.detail-header');
    if (header) data.name_on_site = header.innerText.trim();

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

    const source = document.querySelector(
        'source[src*="cloudflarestream"], video source, source'
    );
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
                    el.querySelectorAll('span.move-name').forEach(
                        s => names.push(s.innerText.trim())
                    );
                    el = el.nextElementSibling;
                }
                break;
            }
        }
        return names;
    }

    data.prerequisites = getSection('prerequisite');
    data.subsequents = getSection('subsequent');

    document.querySelectorAll(
        'h3.related-moves-title ~ a span.move-name, '
        + 'h3.related-moves-title ~ div a span.move-name'
    ).forEach(s => data.explore_related.push(s.innerText.trim()));

    return data;
})();
"""


def detect_chrome_version():
    for cmd in ["google-chrome --version", "chromium-browser --version"]:
        try:
            out = subprocess.check_output(cmd, shell=True, text=True).strip()
            for part in out.split():
                if "." in part:
                    return int(part.split(".")[0])
        except Exception:
            continue
    return None


def create_driver(chrome_version):
    """Create Chrome. Retries with cleanup on failure."""
    import undetected_chromedriver as uc

    for attempt in range(3):
        try:
            if attempt > 0:
                subprocess.run("pkill -9 chromedriver 2>/dev/null", shell=True, capture_output=True)
                time.sleep(3)

            options = uc.ChromeOptions()
            options.add_argument("--window-size=1400,900")
            options.add_argument("--no-first-run")
            options.add_argument("--no-default-browser-check")
            options.add_argument("--disable-popup-blocking")
            options.add_argument("--disable-dev-shm-usage")
            options.add_argument("--no-sandbox")

            return uc.Chrome(options=options, headless=False, version_main=chrome_version)
        except Exception as e:
            logger.warning("Chrome attempt %d failed: %s", attempt + 1, e)
            if attempt == 2:
                raise
            time.sleep(5)


def pass_cloudflare(driver, timeout=60):
    driver.get("https://parkourtheory.com")
    time.sleep(5)
    for i in range(timeout):
        title = driver.title.lower()
        if "parkour" in title or "theory" in title:
            logger.info("Cloudflare passed (%ds)", i + 5)
            return True
        time.sleep(1)
    return False


def scrape_trick(driver, name, wait=6.0):
    """Load trick page and extract data."""
    slug = name.lower().replace(" ", "_")
    url = "https://parkourtheory.com/move/" + slug

    try:
        driver.get(url)
    except Exception as e:
        logger.warning("  %s — load failed: %s", name, e)
        return None

    time.sleep(wait)

    # Check Cloudflare
    title = driver.title.lower()
    if "attention" in title or "just a moment" in title:
        logger.warning("  %s — Cloudflare", name)
        return None

    # Extract with retries
    for _ in range(3):
        try:
            data = driver.execute_script(EXTRACT_JS)
            if data and (data.get("type") or data.get("description") or data.get("name_on_site")):
                data["url"] = url
                return data
        except Exception:
            pass
        time.sleep(2)

    # Check if page exists at all
    try:
        has_header = driver.execute_script("return !!document.querySelector('h2.detail-header')")
        if not has_header:
            return {"_status": "not_found"}
    except Exception:
        pass

    logger.warning("  %s — extraction empty", name)
    return None


def save_progress(detailed, output_file):
    tmp = output_file.with_suffix(".tmp")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(detailed, f, indent=2, ensure_ascii=False)
    tmp.rename(output_file)


def load_progress(output_file):
    if not output_file.exists():
        return [], set()
    with open(output_file) as f:
        detailed = json.load(f)
    done = {t["name"] for t in detailed if t.get("type") or t.get("description")}
    return detailed, done


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(INPUT_FILE))
    parser.add_argument("--output", default=str(OUTPUT_FILE))
    parser.add_argument("--max", type=int, default=0)
    parser.add_argument("--delay", type=float, default=3.0)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--restart-every", type=int, default=10,
                        help="Restart Chrome every N tricks (10 is proven reliable)")
    args = parser.parse_args()

    output_file = Path(args.output)
    with open(args.input) as f:
        all_tricks_raw = json.load(f)

    all_tricks = [{"name": t} if isinstance(t, str) else t for t in all_tricks_raw]
    detailed, done_names = load_progress(output_file)
    logger.info("Loaded %d tricks (%d complete)", len(detailed), len(done_names))

    if args.retry_failed:
        to_scrape = [t.copy() for t in detailed if t.get("error")]
        detailed = [t for t in detailed if not t.get("error")]
        done_names = {t["name"] for t in detailed if t.get("type") or t.get("description")}
        logger.info("Retrying %d failed", len(to_scrape))
    else:
        to_scrape = [t for t in all_tricks if t["name"] not in done_names]

    if args.max > 0:
        to_scrape = to_scrape[:args.max]

    logger.info("To scrape: %d | Done: %d", len(to_scrape), len(done_names))
    if not to_scrape:
        return

    chrome_version = detect_chrome_version()
    logger.info("Chrome: %s", chrome_version)

    driver = None
    session_count = 0
    tricks_in_session = 0
    consecutive_fails = 0
    failed_names = []
    start_time = time.time()

    def new_session():
        nonlocal driver, session_count, tricks_in_session
        if driver:
            try:
                driver.quit()
            except Exception:
                pass
            time.sleep(2)
        session_count += 1
        tricks_in_session = 0
        logger.info("[S#%d] Starting Chrome...", session_count)
        driver = create_driver(chrome_version)
        if not pass_cloudflare(driver):
            logger.error("Cloudflare failed!")
            driver.quit()
            sys.exit(1)
        time.sleep(2)

    new_session()

    try:
        for i, trick in enumerate(to_scrape):
            name = trick["name"]

            # Restart Chrome every N tricks — THE key to reliability
            if tricks_in_session >= args.restart_every:
                logger.info("[S#%d] Restart after %d tricks", session_count, tricks_in_session)
                new_session()

            if consecutive_fails >= 3:
                logger.warning("3 consecutive fails. Restarting...")
                time.sleep(10)
                new_session()
                consecutive_fails = 0

            # Scrape (2 attempts: 6s then 10s)
            result = None
            for wait in [6.0, 10.0]:
                result = scrape_trick(driver, name, wait=wait)
                if result is not None:
                    break

            tricks_in_session += 1

            if result is None:
                trick["error"] = "failed"
                failed_names.append(name)
                consecutive_fails += 1
            elif result.get("_status") == "not_found":
                trick["error"] = "not_found_on_site"
                consecutive_fails = 0
            else:
                trick.pop("error", None)
                trick.update(result)
                consecutive_fails = 0

            idx = next((j for j, t in enumerate(detailed) if t["name"] == name), None)
            if idx is not None:
                detailed[idx] = trick
            else:
                detailed.append(trick)
            save_progress(detailed, output_file)

            if (i + 1) % 10 == 0 or i == len(to_scrape) - 1:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed if elapsed > 0 else 1
                eta = (len(to_scrape) - i - 1) / rate / 60 if rate > 0 else 0
                n_ok = sum(1 for t in detailed if t.get("description") or t.get("type"))
                n_err = sum(1 for t in detailed if t.get("error"))
                logger.info(
                    "[%4d/%d] %-30s | %d ok %d err | %.1f/min | ~%.0fm | S#%d",
                    i + 1, len(to_scrape), name[:30], n_ok, n_err, rate * 60, eta, session_count,
                )

            time.sleep(args.delay + random.uniform(0, args.delay))

    except KeyboardInterrupt:
        logger.info("Interrupted!")
    finally:
        if driver:
            try:
                driver.quit()
            except Exception:
                pass
        save_progress(detailed, output_file)

    elapsed = time.time() - start_time
    n_ok = sum(1 for t in detailed if t.get("description") or t.get("type"))
    n_err = sum(1 for t in detailed if t.get("error"))
    logger.info("=" * 50)
    logger.info("Pass 1: %.0f min | %d ok | %d err", elapsed / 60, n_ok, n_err)

    # Auto-retry
    retry = [t for t in detailed if t.get("error") == "failed"]
    if retry and not args.retry_failed:
        logger.info("Auto-retrying %d failed...", len(retry))
        detailed = [t for t in detailed if t.get("error") != "failed"]
        time.sleep(5)
        new_session()
        recovered = 0
        for j, trick in enumerate(retry):
            if tricks_in_session >= args.restart_every:
                new_session()
            result = scrape_trick(driver, trick["name"], wait=8.0)
            tricks_in_session += 1
            if result and not result.get("_status"):
                trick.pop("error", None)
                trick.update(result)
                recovered += 1
            detailed.append(trick)
            save_progress(detailed, output_file)
            time.sleep(args.delay + random.uniform(0, args.delay))
        if driver:
            try:
                driver.quit()
            except Exception:
                pass
        n_ok = sum(1 for t in detailed if t.get("description") or t.get("type"))
        n_err = sum(1 for t in detailed if t.get("error"))
        logger.info("After retry: %d ok | %d err | Recovered: %d", n_ok, n_err, recovered)

    logger.info("Saved: %s", output_file)


if __name__ == "__main__":
    main()
