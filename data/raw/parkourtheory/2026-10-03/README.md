# parkourtheory.com listing, 2026-10-03

Data courtesy of parkourtheory.com, used with permission.

## Source and method

Listed with `scripts/scrape_parkourtheory_v2.py` on 2026-10-03. The script reads the public sitemaps that `robots.txt` points to and diffs them against `data/parkourtheory_detailed.json` (scraped 2026-03-26). It makes 2 requests, with a 20 s crawl delay and the User-Agent `PkVision research crawler (contact: airkyzzz.jiji@gmail.com)`. No videos were downloaded.

- `sitemap-moves-1.xml`: every `/move/<slug>` URL with its `lastmod`.
- `sitemap-categories.xml`: the 7 category pages and every `/variation/<slug>` URL.
- `diff.json`: counts, the variation slugs, new slugs, removed moves, rename candidates, existing moves whose `lastmod` is after the old scrape, and moves that failed in the old scrape.

Slugs follow `name.lower().replace(" ", "_").replace("/", "-")`. All 1,834 old URLs match this rule.

## Counts

| | |
|---|---|
| live moves | 1,990 |
| live variation pages | 31 |
| old scrape | 1,837 (3 of them failed with no data) |
| new slugs | 166 (most added 2026-03 to 2026-09) |
| removed slugs | 13 |
| rename candidates by slug similarity | 3 (the `parallel_bar_gainer_*_dismount` family) |
| existing moves with `lastmod` after 2026-03-26 | 4 |

## Not done yet

`moves.json` and `variations.json` are not here. Move and variation pages are a client-side app, and their data comes from `https://parkourtheory.com/v3/detail/{move,variation}/<slug>`, which requires a reCAPTCHA token. Getting those fields (description, aliases, prerequisites, variation definitions and examples) needs either a manual browser session or an export from the site owner. New moves in `diff.json` have a slug only, no display name.
