"""Debug: what's actually on the page for failing tricks?"""
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

NAV_JS = """
(function(path) {
    const old = document.getElementById('__pkvision_nav');
    if (old) old.remove();
    const a = document.createElement('a');
    a.href = path;
    a.id = '__pkvision_nav';
    a.style.position = 'fixed';
    a.style.top = '-9999px';
    document.body.appendChild(a);
    a.click();
})(arguments[0]);
"""

# First test a known-good trick
tricks = [
    "Back Flip",                              # known good
    "Running Caster Gainer Layout Touchdown",  # was failing
    "Roll Kip-Up Front",                       # was failing
    "Shoulder Giant",                          # was failing
    "Gargoyle Front",                          # was failing
]

for name in tricks:
    slug = name.lower().replace(" ", "_")
    driver.execute_script(NAV_JS, "/move/" + slug)

    # Wait longer to be sure
    time.sleep(6)

    # Dump everything about the page
    info = driver.execute_script("""
    return {
        url: window.location.href,
        title: document.title,
        h2_header: (document.querySelector('h2.detail-header') || {}).innerText || 'NONE',
        all_bold: Array.from(document.querySelectorAll('b')).map(b => b.innerText.trim()),
        body_length: document.body.innerText.length,
        body_first_500: document.body.innerText.substring(0, 500),
        move_links: document.querySelectorAll('a[href*="/move/"]').length,
        has_video: !!document.querySelector('source[src*="cloudflarestream"], video source'),
        images: document.querySelectorAll('img[src*="imagedelivery"]').length,
    };
    """)

    print("\n" + "=" * 60)
    print("TRICK:", name)
    print("URL:", info["url"])
    print("H2 header:", info["h2_header"])
    print("Bold labels:", info["all_bold"])
    print("Body length:", info["body_length"])
    print("Move links:", info["move_links"])
    print("Has video:", info["has_video"])
    print("Images:", info["images"])
    print("Body preview:", info["body_first_500"][:300])

driver.quit()
print("\nDONE")
