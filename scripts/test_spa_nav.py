"""Test: true SPA navigation via History API (no page reload, no Cloudflare)."""
import undetected_chromedriver as uc
import time

options = uc.ChromeOptions()
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")
options.add_argument("--window-size=1400,900")
driver = uc.Chrome(options=options, headless=False, version_main=145)

driver.get("https://parkourtheory.com")
time.sleep(6)
print("Title:", driver.title)

# Test 3 approaches to SPA navigation

tricks = [
    "Flyaway Double Layout",  # was failing in v2
    "Double Layout",           # was failing in v2
    "Side Vault",              # was failing in v2
    "Knee Corkscrew",          # was failing before
    "Back Flip",               # known good
]

print("\n=== Method 1: pushState + popstate ===")
for name in tricks:
    slug = name.lower().replace(" ", "_")
    path = "/move/" + slug

    driver.execute_script("""
        window.history.pushState({}, '', arguments[0]);
        window.dispatchEvent(new PopStateEvent('popstate', {state: {}}));
    """, path)
    time.sleep(5)

    info = driver.execute_script("""
        const h = document.querySelector('h2.detail-header');
        const bs = Array.from(document.querySelectorAll('b')).map(b => b.innerText.trim());
        return {
            url: window.location.href,
            header: h ? h.innerText.trim() : 'NONE',
            bolds: bs,
            body_len: document.body.innerText.length,
        };
    """)
    print("  {} -> header={}, bolds={}, body={}".format(
        name, info["header"], len(info["bolds"]), info["body_len"]))

# Go home first before method 2
driver.get("https://parkourtheory.com")
time.sleep(5)

print("\n=== Method 2: location.href (full reload, for comparison) ===")
for name in tricks[:2]:
    slug = name.lower().replace(" ", "_")
    url = "https://parkourtheory.com/move/" + slug

    driver.execute_script("window.location.href = arguments[0]", url)
    time.sleep(6)

    info = driver.execute_script("""
        const h = document.querySelector('h2.detail-header');
        const bs = Array.from(document.querySelectorAll('b')).map(b => b.innerText.trim());
        return {
            url: window.location.href,
            header: h ? h.innerText.trim() : 'NONE',
            bolds: bs,
            body_len: document.body.innerText.length,
        };
    """)
    print("  {} -> header={}, bolds={}, body={}".format(
        name, info["header"], len(info["bolds"]), info["body_len"]))

driver.quit()
print("\nDONE")
