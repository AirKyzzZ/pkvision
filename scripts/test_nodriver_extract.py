"""Test nodriver extraction on multiple tricks."""
import asyncio
import json
import nodriver

EXTRACT_JS = """
(function() {
    const data = {
        name_on_site: '',
        type: '', pronunciation: '', description: '',
        video_url: '', video_source: '', video_time: '',
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

    return data;
})()
"""


async def main():
    browser = await nodriver.start()
    page = await browser.get("https://parkourtheory.com")
    await asyncio.sleep(6)
    print("Home page loaded:", await page.evaluate("document.title"))

    tricks = [
        "Back Flip",
        "Gainer Cat",
        "Side Vault",
        "Knee Corkscrew",
        "Gargoyle Front",
    ]

    for name in tricks:
        slug = name.lower().replace(" ", "_")
        url = "https://parkourtheory.com/move/" + slug

        await page.get(url)
        await asyncio.sleep(6)

        try:
            result = await page.evaluate(EXTRACT_JS)
            has_data = bool(result and (result.get("type") or result.get("description")))
            print("{:30s} -> {} desc={}".format(
                name,
                "OK" if has_data else "EMPTY",
                (result.get("description", ""))[:60] if result else "None",
            ))
        except Exception as e:
            print("{:30s} -> ERROR: {}".format(name, e))

    browser.stop()
    print("DONE")


asyncio.run(main())
