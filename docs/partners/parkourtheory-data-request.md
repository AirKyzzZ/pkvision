# What PKVision would need from Parkour Theory

Draft for Maxime to adapt before sending. Numbers are from `data/parkourtheory_detailed.json` (scraped 2026-10-03) and may move with the new update.

PKVision recognizes parkour and freerunning moves from phone video by their physics: how many flips, how many twists, which direction, on or off axis, how the athlete took off and landed. The model is no longer the bottleneck, the labels are. Today each clip's flip and twist counts come from its Parkour Theory name through a parser, and a manual audit found about 17 % of flip counts and 35 % of twist counts wrong. The pose model trained on those labels stays below "always answer the most common count", whatever we fix on the pose side. A few hundred rows verified by the people who built the database would be worth more than another model.

## Tier 1: the move sheet

`data/partner/parkourtheory/move_sheet.csv`, one row per move (1,834 rows), regenerated with `python scripts/pt_partner_data.py template`. Context columns are filled, the rest is blank.

| Priority | Rows | What |
|---|---|---|
| 1 | 580 | twisting moves we have a demo clip for. This is the blocker: even these alone unblock us |
| 2 | 724 | flips without twist, same |
| 3 | 207 | vaults, wall and bar moves without flip, same |
| 4 | 323 | moves we have no clip for |

For priority 1, the first four columns (`flips`, `flip_dir`, `twists`, `axis`) are what matters most: about 10 seconds a row.

| Column | Values | Back Full | Gainer |
|---|---|---|---|
| `flips` | somersaults, quarters allowed | 1 | 1 |
| `flip_dir` | front, back, side, none | back | back |
| `twists` | full twists, halves allowed | 1 | 0 |
| `axis` | on, off (cork, raiz, b-twist, …) | on | on |
| `travel` | forward, backward, sideways, up | backward | forward |
| `shape` | tuck, pike, straight, straddle, open, any | any | any |
| `takeoff` | one_foot, two_feet, sequential, hands, swing | two_feet | one_foot |
| `takeoff_from` | ground, wall, bar, obstacle, ledge, beam | ground | ground |
| `hands` | none, obstacle, wall, bar, ground, several joined with `+` | none | none |
| `landing` | two_feet, one_foot, hands, roll, bar | two_feet | two_feet |
| `demo_is_canonical` | yes, variant, no | yes | yes |
| `move_start_s` / `move_end_s` | seconds in the demo video: last foot leaves, first landing contact | 1.2 / 2.0 | 0.8 / 1.5 |
| `notes` | free text | | |

Rules that keep the sheet useful:

- Blank or `?` means unknown. That is always better than a guess.
- Fill from the move's definition, not from the name alone. If the demo shows a variation, fill the definition and set `demo_is_canonical` to `variant` with a note.
- Directions are relative to the athlete at take-off. A gainer is `flip_dir=back`, `travel=forward`.

If your database already stores some of this, an export in any format (JSON, CSV, SQL dump) is just as good. We convert on our side.

## Tier 2: video, if possible

- Original files rather than the stream re-encodes, and every demo video per move if the update added several, each with a stable video id and the move it shows.
- Per video: the move window (start and end seconds), how many people are visible and which one performs, and the camera (fixed or handheld, side, front or diagonal).

## Tier 3: nice to have

- The changelog of the new update (moves added, renamed, merged), so our copy stays in sync.
- Your naming rules: how a name is built from its parts, and which aliases you accept.

## Use, credit, what Parkour Theory gets back

- Research use. Credit to Parkour Theory in the README, the paper and any screen that uses the data. Parkour Theory already allowed use of its public data with credit (owner note, 2026-10-03).
- Videos are never redistributed. Verified attributes stay out of the public repository unless Parkour Theory agrees.
- Every returned sheet goes through `python scripts/pt_partner_data.py check FILE`. We send back the rows that do not parse, and the list of moves where your attributes and the name disagree, which usually points at a naming ambiguity.
- Once the reader works: attribute pre-tagging for new uploads.

## On our side when a sheet comes back

1. Put it in `data/partner/parkourtheory/returned/` (gitignored).
2. `python scripts/pt_partner_data.py check FILE` validates it, prints errors by line number only and compares flip, twist and direction with the name-derived gold.
3. `--write` saves `labels.json` (gitignored) in the attribute manifest's field names, ready for the CueModel and local-VLM evaluations.

Maxime should finish the blind labels before opening a returned sheet: it contains the answers for the blind-queue clips too.
