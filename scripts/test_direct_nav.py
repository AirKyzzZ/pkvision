"""Test direct URL navigation for tricks that failed via search."""
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

# Previously failing trick names
test_tricks = [
    "Pimp Double Full",
    "Inward Dive Roll",
    "Kong 360 Dive Roll",
    "Knee Corkscrew",
    "Safety Vault 360",
    "Rail Macaco",
    "Frisbee",
    "Inward Aerial Twist",
    "180 Regrab",
    "Gainer Cat",
]

for name in test_tricks:
    slug = name.lower().replace(" ", "_")

    # SPA navigation via programmatic link click
    driver.execute_script(
        'const a=document.createElement("a");'
        'a.href=arguments[0];a.id="__nav";'
        'document.body.appendChild(a);a.click();',
        "/move/" + slug,
    )
    time.sleep(4)

    has_header = driver.execute_script(
        "return !!document.querySelector('h2.detail-header')"
    )
    body_text = driver.find_element("tag name", "body").text
    body_len = len(body_text)
    cur_url = driver.current_url

    # Check for description field
    desc = driver.execute_script("""
        for (const b of document.querySelectorAll('b')) {
            if (b.innerText.trim() === 'Description:') {
                let t='',n=b.nextSibling;
                while(n){if(n.nodeType===1&&n.tagName==='B')break;t+=n.textContent||'';n=n.nextSibling;}
                return t.trim();
            }
        }
        return '';
    """)

    status = "OK" if (has_header or desc) else "EMPTY"
    print("{:30s} header={} body={:5d} desc={:3d} url={} -> {}".format(
        name, has_header, body_len, len(desc or ""), cur_url, status,
    ))

    driver.execute_script('document.getElementById("__nav")?.remove()')
    time.sleep(1)

driver.quit()
print("DONE")
