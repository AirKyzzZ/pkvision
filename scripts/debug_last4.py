"""Debug the 4 tricks that failed."""
import undetected_chromedriver as uc
import time
import urllib.parse

d = uc.Chrome(headless=False, version_main=145)
d.get("https://parkourtheory.com")
time.sleep(6)
print("Home loaded:", d.title)

EXTRACT_JS = """
return (function() {
    const h = document.querySelector('h2.detail-header');
    const data = {header: h ? h.innerText.trim() : '', type: '', desc: ''};
    for (const b of document.querySelectorAll('b')) {
        if (b.innerText.trim() === 'Type:') {
            let n = b.nextSibling;
            while (n && !(n.nodeType === 1 && n.tagName === 'B')) {
                data.type += n.textContent || '';
                n = n.nextSibling;
            }
            data.type = data.type.trim();
        }
        if (b.innerText.trim() === 'Description:') {
            let n = b.nextSibling;
            while (n && !(n.nodeType === 1 && n.tagName === 'B')) {
                data.desc += n.textContent || '';
                n = n.nextSibling;
            }
            data.desc = data.desc.trim();
        }
    }
    return data;
})();
"""

# Standard slugs
tests = [
    ("Giant Release Half Twist Handstand Catch", "giant_release_half_twist_handstand_catch"),
    ("Lache Step One-Hand Palm Walk", "lache_step_one-hand_palm_walk"),
    # Unicode slugs
    ("Au De Frente (no accent)", "au_de_frente"),
    ("Rail Au De Frente (no accent)", "rail_au_de_frente"),
    # With unicode
    ("Au De Frente (unicode)", "a\u00fa_de_frente"),
    ("Rail Au De Frente (unicode)", "rail_a\u00fa_de_frente"),
    # URL-encoded unicode
    ("Au De Frente (encoded)", urllib.parse.quote("a\u00fa_de_frente")),
    # Also try the fraction ones
    ("One-Step Front 1 1/3 Dive Kong Gainer", "one-step_front_1_1/3_dive_kong_gainer"),
    ("Back 1 1/2 Dive Roll", "back_1_1/2_dive_roll"),
]

for label, slug in tests:
    url = "https://parkourtheory.com/move/" + slug
    d.get(url)
    time.sleep(5)
    try:
        result = d.execute_script(EXTRACT_JS)
        ok = bool(result.get("type") or result.get("desc"))
        print("{:45s} -> {} | h={} t={} d={}".format(
            label, "OK" if ok else "EMPTY",
            result.get("header", "")[:30],
            result.get("type", "")[:20],
            result.get("desc", "")[:40],
        ))
    except Exception as e:
        print("{:45s} -> ERROR: {}".format(label, e))

d.quit()
print("DONE")
