"""Debug: run the EXACT same flow as scrape_uc.py and see what happens."""
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

# The EXACT EXTRACT_JS from scrape_uc.py
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

tricks = ["Back Flip", "Running Caster Gainer Layout Touchdown", "Gargoyle Front"]

for name in tricks:
    slug = name.lower().replace(" ", "_")
    print("\n" + "=" * 60)
    print("Navigating to:", name, "->", "/move/" + slug)

    driver.execute_script(NAV_JS, "/move/" + slug)
    time.sleep(5)

    # Check page state
    url = driver.current_url
    print("Current URL:", url)

    # Check what bold labels exist
    bolds = driver.execute_script(
        "return Array.from(document.querySelectorAll('b')).map(b => b.innerText.trim())"
    )
    print("Bold labels found:", bolds)

    # Run the EXACT EXTRACT_JS
    result = driver.execute_script(EXTRACT_JS)
    print("EXTRACT_JS returned:", type(result), result)

    # Also try a simpler extraction
    simple = driver.execute_script("""
        const h = document.querySelector('h2.detail-header');
        const bs = document.querySelectorAll('b');
        let type_val = '';
        let desc_val = '';
        for (const b of bs) {
            if (b.innerText.trim() === 'Type:') {
                let n = b.nextSibling;
                while (n && !(n.nodeType === 1 && n.tagName === 'B')) {
                    type_val += n.textContent || '';
                    n = n.nextSibling;
                }
            }
            if (b.innerText.trim() === 'Description:') {
                let n = b.nextSibling;
                while (n && !(n.nodeType === 1 && n.tagName === 'B')) {
                    desc_val += n.textContent || '';
                    n = n.nextSibling;
                }
            }
        }
        return {
            header: h ? h.innerText.trim() : 'NONE',
            type: type_val.trim(),
            desc: desc_val.trim().substring(0, 100),
        };
    """)
    print("Simple extraction:", simple)

driver.quit()
print("\nDONE")
