# parkourtheory.com listing, 2026-11-25

Data courtesy of parkourtheory.com, used with permission.

Listed with `scripts/scrape_parkourtheory_v2.py` like the 2026-10-03 listing, with the User-Agent `PkVision research crawler`. Same files, same diff against `data/parkourtheory_detailed.json` (scraped 2026-03-26).

| | 2026-10-03 | 2026-11-25 |
|---|---|---|
| live moves | 1,990 | 2,117 |
| live variation pages | 31 | 31 |
| new since the March scrape | 166 | 296 |
| removed since the March scrape | 13 | 16 |
| rename candidates | 3 | 6 |

131 of the 296 new moves appeared after 2026-10-03, 127 of them with a `lastmod` in October or November 2026. None of the 296 has a clip in the corpus yet.

`new_moves.json` adds each new move's name, description, demo video URL and Parkour Theory id, read from the meta tags of its `/move/<slug>` page by `scripts/pt_fetch_new_moves.py` (one request every 20 s). No videos were downloaded.
