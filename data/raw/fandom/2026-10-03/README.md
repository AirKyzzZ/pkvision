# Fandom wikis: parkour, tricking, freerunning (2026-10-03)

**Sources:** https://parkour.fandom.com, https://tricking.fandom.com, https://freerunning.fandom.com

**Licence:** CC-BY-SA (Fandom licensing page, read from `meta=siteinfo&siprop=rightsinfo`). Any file that redistributes this text must keep the credit line below and stay CC-BY-SA.

**Credit:** Text from Parkour Wiki, Tricking Wiki and Freerunning Wiki (fandom.com) and their contributors, CC-BY-SA. See https://www.fandom.com/licensing. Page history is on each article URL.

## Method

MediaWiki API (1.43.9), no HTML scraping, 1 request/s:

1. `generator=allpages&gapnamespace=0&gapfilterredir=nonredirects` with `prop=revisions|categories|info` (latest revision wikitext, categories, URL).
2. The lead is the wikitext before the first `==` heading. `lead_text` strips templates, tables, refs, file links and wiki markup locally. Fandom has no TextExtracts, so this is a best-effort plain text.
3. Redirects: `list=allpages&apfilterredir=redirects`, then resolved with `titles=…&redirects=1`. Redirects are aliases (e.g. "360 Tic-Tac" → "Tic-Tac").

## Counts

| Wiki | Articles (ns0, non-redirect) | Redirects | Non-empty leads | Latest edit |
|---|---|---|---|---|
| parkour | 141 | 161 | 136 | 2026-09-10 |
| tricking | 24 | 6 | 23 | 2024-10-01 |
| freerunning | 16 | 12 | 12 | 2013-07-22 |

Many articles are not moves (teams, people, training, glossary pages). Use `categories` to filter.

## Files

`{parkour,tricking,freerunning}.json` each hold `source`, `fetched_at`, `method`, `license`, `credit`, `siteinfo_statistics`, `articles[]` (`page_id`, `title`, `url`, `last_rev_timestamp`, `last_rev_user`, `categories`, `wikitext_chars`, `lead_wikitext`, `lead_text`) and `redirects[]` (`from`, `to`, `tofragment`).
