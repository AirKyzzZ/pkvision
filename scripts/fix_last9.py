"""Fix the last ~9 tricks using search instead of direct URL."""
import undetected_chromedriver as uc
import json
import time
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys

d = uc.Chrome(headless=False, version_main=145)
d.get("https://parkourtheory.com")
time.sleep(6)
print("Home loaded:", d.title)

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

# Load current data
with open("data/parkourtheory_detailed.json") as f:
    detailed = json.load(f)

# Find tricks with errors
errored = [t for t in detailed if t.get("error")]
print(f"Tricks to fix: {len(errored)}")
for t in errored:
    print(f"  {t['name']} -> {t['error']}")

fixed = 0

for trick in errored:
    name = trick["name"]
    print(f"\nSearching for: {name}")

    # Go home
    d.get("https://parkourtheory.com")
    time.sleep(3)

    # Use search
    try:
        search = d.find_element(By.CSS_SELECTOR, "input")
        search.click()
        search.clear()
        for char in name:
            search.send_keys(char)
            time.sleep(0.03)
        search.send_keys(Keys.RETURN)
        time.sleep(5)

        # Check for results
        body = d.find_element(By.TAG_NAME, "body").text
        if "results for" not in body.lower():
            print(f"  No search results")
            continue

        # Click first result
        links = d.find_elements(By.CSS_SELECTOR, "a[href*='/move/']")
        clicked = False
        for link in links:
            try:
                h4 = link.find_element(By.CSS_SELECTOR, "h4.result-header")
                if h4.text.strip():
                    print(f"  Found: {h4.text.strip()}")
                    h4.click()
                    clicked = True
                    break
            except Exception:
                continue

        if not clicked:
            print(f"  No clickable result")
            continue

        time.sleep(6)

        # Extract
        data = d.execute_script(EXTRACT_JS)
        if data and (data.get("type") or data.get("description") or data.get("name_on_site")):
            trick.pop("error", None)
            trick.update({"url": d.current_url, **data})
            fixed += 1
            print(f"  FIXED! type={data.get('type')}, desc={data.get('description','')[:50]}")
        else:
            print(f"  Extraction empty")

    except Exception as e:
        print(f"  Error: {e}")

d.quit()

# Save
with open("data/parkourtheory_detailed.json", "w", encoding="utf-8") as f:
    json.dump(detailed, f, indent=2, ensure_ascii=False)

ok = sum(1 for t in detailed if t.get("description") or t.get("type"))
err = sum(1 for t in detailed if t.get("error"))
print(f"\nFixed: {fixed} | Total: {ok}/{len(detailed)} | Remaining errors: {err}")
