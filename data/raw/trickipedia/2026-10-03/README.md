# Trickipedia (2026-10-03)

**Source:** https://trickipedia.app

**Licence:** CC-BY-SA 4.0 (site footer and /terms). Any file that redistributes the descriptions must keep the credit line and stay CC-BY-SA (share-alike).

**Credit:** Trick data from Trickipedia (https://trickipedia.app) and its contributors, CC-BY-SA 4.0.

## Method

- The server-rendered navigation payload of any page lists every trick (568, the same set as `sitemap.xml`).
- Each public trick page `/<discipline>/<subcategory>/<slug>` was fetched at 1 request/s. The `trick` object was parsed out of the Next.js RSC payload.
- `/api/` was not used, because robots.txt disallows it.
- Page `<title>` tags wrongly say "Trick Not Found". This is a site metadata bug: the body and payload carry the trick.

## Text corruption is upstream, not ours

The "uri" → "slug" damage is in the live site's data, not in our scrape: the live page reads "A Shslugken Corkscrew … prerequisite_ids: Corkscrew, Shslugken Twist". Trick **names are clean**. Descriptions have:
- 21 in-word "slug" for "uri" (Shslugken = Shuriken, dslugng = during);
- 18 "prerequisite_ids:" for "prerequisites:".

The original field is kept as fetched. A repaired copy is added as `<field>_repaired` on the 29 affected records:
- in-word `slug` → `uri`;
- `prerequisite_ids:` → `prerequisites:`.

No standalone word "slug" occurs, so the rule is safe here. Worth reporting to the site owner.

## Counts

| | n |
|---|---|
| Tricks fetched | 568 (0 failed); 533 unique names (some names repeat across disciplines) |
| By discipline | Tricking 350, Parkour & Freerunning 66, Trampolining 65, Trampwall 55, Tumbling 32 |
| With description | 550 |
| With video URLs | 203 |
| With prerequisites | 343 |
| New vs local `data/trickipedia_tricks.json` (527) | 41 records; none removed |
| Empty categories in the nav (no tricks yet) | Surfing, Snowboarding, Skiing, Figure Skating, Partner Stunting, Diving, Wakeboarding, Breakdancing, Døds |

## Files

`tricks.json` holds the run metadata, `categories[]` and `tricks[]`. Each trick has `id`, `name`, `slug`, `url`, `master_category`, `subcategory`, `description` (+ `description_repaired`), `difficulty_level`, guide/tips/mistakes/safety fields, `video_urls`, `image_urls`, `tags`, `inventor_name`, `source_urls`, `prerequisites` (names), `components`, `parent_id`, `is_combo`, `created_at` and `updated_at`.
