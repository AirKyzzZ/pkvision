# 09 · Sources to aggregate: trick vocabularies, video, annotation tools, dataset precedent (2026-09-29)

Desk research only. Nothing was scraped at scale and no dataset was downloaded. $0 spent. I made single-file reads only: the parkourtheory sitemap and JS bundle, one FIG PDF per table, the public VideoNet MCQ JSONL (text, no video), four YouTube search-result pages, and MediaWiki/App Store/GitHub API metadata calls. No repo file was changed except this one.

**Tool limits.** The session's WebSearch budget (200 calls, shared across parallel agents) was already used up before this track started. Instead I used WebFetch, `curl` on known URLs, the GitHub API, the iTunes Search API, the MediaWiki API, and the arXiv/alphaXiv tools. Some items are marked **not found (not searched)** for this reason. Kaggle is one of them.

**Evidence grades:** A = I checked it on the primary source today. B = stated by the primary source but I did not check it independently. C = my estimate or inference. "Not found" means I looked and found nothing. It does not mean the thing doesn't exist.

---

## Bottom line

1. **The cheapest missing vocabulary is already on sources we use.** Two gaps close quickly:
   - parkourtheory has grown to **1,990 moves** (sitemap, 2026-09-29). The local copy has 1,834. There are **169 new moves**, mostly added Mar–Sep 2026, and 13 moves were removed or renamed.
   - The project's `fig_tricks_2025.json` has **149** names, but the official FIG 2025 and 2026 PDFs list **about 189**. The ~40 missing names were already in the 2025 PDF: second names in shared table cells and the multi-name PK Basics cells.

   Together these add about 200 names at near-zero cost.
2. **No new large text source exists.** Trickipedia (569+, CC-BY-SA 4.0) is about 85% covered already, and loopkicks about 100%. The Fandom wikis are small and stale. I found no app or GitHub trick database.
3. **The FIG Trampoline numeric system is a working precedent for the PMC `AIR` segment.** The first digit counts quarter-somersaults, the following digits give half-twists per somersault, and a symbol gives the shape. FIG events already require it.
4. **The only parkour video set with names attached is VideoNet**, and it is gated with auto-approval:
   - 40 basic actions × 5 clips;
   - about 4k–7k web-mined training clips.

   The action list is public in the GitHub JSONL. It has no twist variants beyond "corkscrew".
5. **FIG competition video is on YouTube, but no trick names are attached.** World Gymnastics Channel, FISE and JUMP Freerun post per-run clips and multi-hour streams, titled by athlete, event and run. The trick names would have to come from our own annotation.
6. **Annotation:** a verify-first pass at 20–40 s per clip works only if every field is pre-filled (name parser + skeleton counters) and the UI is keyboard-only. 300 clips come to about 2.5 h of pure verification, plus about 2–3 h for corrections. Label Studio Community (Apache-2.0) covers the needs. Pilot 20 clips first.
7. **Publishability:** 300 clips alone fits a CVsports workshop paper, not a CVPR main-track dataset paper. It needs:
   - an expert-agreement subset;
   - consented own-filmed clips, or YouTube-ID-only release;
   - a weak-label train split;
   - VLM and skeleton baselines;
   - an unseen-trick (open-vocabulary) test split.

---

## 1. Trick vocabularies and wikis

### 1.1 What the project already holds (read from `data/`, grade A)

| Source file | Raw | In `unified_tricks.json` | Notes |
|---|---|---|---|
| `parkourtheory_detailed.json` | 1,837 (1,834 with URL) | 1,750 PT-only + shared | 1,627 have a `video_url` (Cloudflare Stream) |
| `other_sources_tricks.json` → loopkicks | 943 rows, 556 unique names | 424 loopkicks-only | 930 rows have a description |
| → trickipedia | 527, all under `/tricking/` URLs | master_category: Tricking 323, Parkour & Freerunning 63, Trampwall 55, Tumbling 25, Trampolining 19 | 509 with description, 163 with video URLs |
| → tricking_bible | 45 | 10 unique to it | |
| `fig_tricks_2025.json` | 149 (swing 31, wall 29, acrobatics 74, pk_basics 15) | mapped via `fig_to_parkourtheory_map_v2.json` | contains "B-Twist", which is **not** in either official PDF |

### 1.2 Source-by-source

**parkourtheory.com** (grade A unless marked)
- **Size today:** `sitemap-moves-1.xml` lists **1,990** move URLs (fetched 2026-09-29).
- **Update activity (from `lastmod`):**
  - 1,059 entries date from 2014-12 (the legacy import), 180 from 2023, 48 from 2024, 201 from 2025 and **502 from 2026**.
  - Compared with the local copy: **169 slugs are new** (2026-03: 41, 2026-04: 54, 2026-06: 44, 2026-09: 14, the rest older) and **13 are gone** (probably renames, e.g. `wall_macaco`, `pendulum`, `caster_valdez-in_back-out`).
  - Only 4 of the 169 new names already exist in `unified_tricks.json` by name, so **about 165 names are truly new**.
  - Examples: `back_full-in_double_full-out`, `double_full-up_gainer`, `inward_double_side_flip`, `flyaway_stretched_one-and-a-half_full`, `cat_hang_kong_double_side`, `gainer-in_palm_gainer-out`.
- **Also in the sitemap:**
  - **36 `/variation/` pages**: counter, unwind, in, out, up, down, x-out, stall, pistol, pike, layout, straddle, stretched, hook, hyper, handcuffs, lotus, rodeo, feilong, psychocrusher, shuriken, method/japan grab, lazy boy, lonestar, peter pan, spider-man, walkdown, donut boy, double leg, flash kick. These map directly onto the PMC modifier families and open question 3 (Counter/Unwind/Inward).
  - A `/world-first` page (not read).
  - 7 `/category/` pages.
- **Fields per move** (from the project's scrape prompt): type, pronunciation, one-sentence description, one video, performer credit ("Source"), source timestamp, prerequisites, subsequents, related.
- **Access:**
  - React SPA behind Cloudflare. `robots.txt` allows all agents with `Crawl-delay: 20`.
  - The JS bundle calls an **undocumented internal API** at `https://parkourtheory.com/v3` (`/search/init_typeahead`, `/search/move`). There is no public API, export or dump.
  - The GitHub org `parkourtheory` has no public repos and lists `hello@parkourtheory.com` as contact.
- **Licence:** **not found.** There is no terms page in the sitemap and no licence or copyright string in the bundle. The videos are credited excerpts, apparently from third-party footage (timestamps like `25:45:00`), hosted on Cloudflare Stream.
- **Recommendation: YES.** Do a polite delta refresh of the ~169 new moves (plus the 36 variation pages) via the sitemap at the 20 s crawl-delay, about 1 h. Better still, email for an export and permission.

**FIG Parkour Table of Tricks 2025 and 2026** (grade A)
- **Files:**
  - 2025 PDF: "April 2025 – Parkour Technical Committee", 4 pp, created 2025-04-02.
  - 2026 PDF: 6 pp, created **2026-03-30**. It adds a junior table and junior rules (no double rotations, max double twists).
- **Size:** I counted about **189 names** in each: Swing 34, Wall 44, Acrobatics 79, PK Basics 32.
- **2025 → 2026 changes:**
  - One genuinely new name, **Gainer Double Cork (4.1)**.
  - Some abbreviations ("Swing Castaway BF Regrab").
  - Revalued Swing and some Wall moves. Examples: Giant 1.7→1.5, Swing Frontflip 1.8→1.5, Swing Gainer 1080 5.0→4.8, Wall Gainer 360 4.3→3.8, Gaet Pimp Backflip 720 4.5→5.0, Miller 7.7→7.5.
  - Remark 4 changes: unintended tilt now costs −0.3 to −0.5, and off-axis-by-design tricks (cork, b-twist, raiz, butterfly) are exempt.
- **What the project's 149-name file is missing** (all present in both PDFs):
  - Kip, Kip 180 Gainer, Swing Gainer 1080;
  - Reverse Wallspin, Wall Flare, Palm Sideflip, Ginger, Wall Inward Sideflip (+360), Raiden (+180), Gaet Pimp Backflip, Hang Gainer, Trapdoor Wall Flip, Gargoyle Gainer, Handstand Castaway Backflip, Castaway Backflip 360/720;
  - Front Handspring, Butterfly, Looser Sideflip, Double Pistol Frisbee, 360 Kong Gainer;
  - PK Basics Drop, Roll, Precision, Jump, Safety/Speed/Lazy/Thief/Gate/Turn/Dash Vault, Splat, Arm Jump, Underbar, Tap Swing, Rail Flare (Italian Job), Palmspin.
- **Name matching:** only 70/189 match `unified_tricks.json` by exact normalized string. FIG naming ("Swing Gainer 720", "Wall Backflip 360") differs from parkourtheory naming, so the ~40 new names need to go through the existing FIG→PT mapping.
- **Descriptions:** none. **Videos:** none. Values are "guiding values for elements in their most basic form".
- **Older tables** (useful for aliases): 2023 and 2024 tables and the 2017 and 2022–24 CoP are on gymnastics.sport. I found no **2027 table**. The 2026 file suggests FIG publishes yearly.
- **Recommendation: YES.** Re-extract both PDFs into the vocabulary and version the values by year.

**FIG Trampoline/DMT/Tumbling Code of Points 2025–2028** (grade A)
- 65 pp, created 2025-07-09.
- **§J FIG numeric system:** "the first digit describes the number of somersaults, in quarters (¼)", "subsequent digits describe the distribution and quantity of twist in each somersault", and shape is `o` (tuck, or blank), `<` (pike) or `/` (straight). Example: "Half in Rudy out, piked = 8 1 3 <".
- **Difficulty:** "Per 1/4 Somersault 0,1; Per 1/2 Twist 0,1", plus shape and completed-somersault bonuses.
- **§K Tumbling symbols:** 0/1/2 = twist per salto, `.` = front/back, `x` = side.
- **§L Terminology:** Barani, Rudy, Randy, Adolph, Fliffis, Triffis, and the IN/OUT/MIDDLE twist-distribution markers. These are the same as the parkour "Full-In Back-Out" markers.
- **Recommendation: YES, as PMC design precedent, not as vocabulary.** The PMC `AIR(som, twist, shape)` plus `twist_when` is essentially this notation, with parkour contact segments added around it.
- I did not read the MAG CoP (16.7 MB) or WAG CoP element tables. **Not counted.**

**trickipedia.app** (grade A)
- **Size:** "569+ Tricks", "5 Disciplines". Per-discipline pages: **Tricking 350, Parkour & Freerunning 66, Trampolining 65, Trampwall 55, Tumbling 32** (= 568).
- **Content:** descriptions and difficulty 1–3. Videos are YouTube thumbnails and links for many entries, placeholders for others.
- **Licence:** "Content is licensed under CC-BY-SA 4.0" (footer and Terms). Contributors grant the platform a licence. There is no clause on scraping.
- **Overlap:** the project holds 485 in unified (about 85%). The Trampolining gap looks like 65 vs 19, but some may have been merged as duplicates. **Not verified per name.**
- **Staleness:** the project's stored URL format is stale (`/tricking/spindle` returns 404 today).
- **Recommendation: MAYBE.** Pull only the trampoline, tumbling and trampwall delta, with attribution. Any file that redistributes their descriptions must be CC-BY-SA.

**loopkickstricking.com Tricktionary** (grade A)
- **Note:** `loopkicks.com` now redirects to a domain-for-sale page. The live site is `loopkickstricking.com/tricktionary`.
- **Size:** "over 500 tricks", "constantly being updated". Categories: vertical kicks, backward, forward, inside, outside, plus variations/transitions/stances.
- **Videos:** "The majority of clips have been downloaded from YouTube or Instagram. We do not claim ownership over those clips." Trick pages have a short description, prerequisites and a creator credit (e.g. Back Pike → "Yuri Levin").
- **Licence:** footer "© 2026 Loopkicks Tricking"; no terms.
- **Overlap:** about 100% (556 unique already held). **Recommendation: NO**, except an alias refresh.

**Tricking Bible** (grade A)
- 2007 PDF by "Sesshoumaru", 29 pp, 45 tricks. No licence (all rights reserved by default). Stale. Overlap about 100%. **NO.**

**Fandom wikis** (MediaWiki API, grade A)

| Wiki | Articles | Activity | Licence |
|---|---|---|---|
| parkour.fandom.com | 142 (many case duplicates, e.g. "Back Flip"/"Back flip"/"Backflip") | last log 2026-09-10; 2 active users | CC-BY-SA |
| tricking.fandom.com | 24 | last edit 2024-11-14 | CC-BY-SA |
| freerunning.fandom.com | 16 | last edit 2024-11-14 | CC-BY-SA |
| gymnastics.fandom.com (GymnWiki) | 255 | 1 active user | CC-BY-SA |

- parkour.fandom has a few regional aliases (Chan vault, Barrel vault, Catpass, Crane).
- **Recommendation: NO**, except a one-off alias harvest from parkour.fandom.

**VideoNet parkour action list** (grade A)
- The 40 names are recoverable from the **public** `benchmarks/mcq_*.jsonl` on GitHub:
  - aerial, back flip, back handspring, back roll, bar kip, cartwheel, cat balance, cat hang, cat leap;
  - coffee grinder, corkscrew (cork), crane jump / crane landing, dash vault, dive roll, double kong vault;
  - front flip, front handspring, gainer, kash vault, kong vault, lache, lazy vault, muscle up;
  - palm flip, palm spin, pop vault, precision jump, rail precision, reverse vault;
  - safety roll, safety vault, side flip, speed vault, tic tac, turn vault, underbar;
  - wall flip, wall run, wall spin, webster.
- **Overlap:** 28/40 match unified by exact normalized name. The unmatched are naming variants or basics: back roll, bar kip, cat balance, cat hang, coffee grinder, crane jump, double kong vault, muscle up, pop vault, rail precision, safety roll, wall run.
- **Recommendation: YES**, as a basics and alias cross-check, and as a naming bridge to the only public parkour benchmark.

**Kinetics-700 classes** (mmaction2 label map, grade A)
- parkour, backflip (human), somersaulting, gymnastics tumbling, cartwheeling, bouncing on trampoline, capoeira, breakdancing. These are coarse and give no vocabulary value.

**Apps** (iTunes Search API, FR store, 2026-09-29; grade A)
- Queries "parkour tricks", "tricking", "freerunning", "trick tracker parkour" and "trampoline tricks" returned games, running apps, **PKour – Parkour Spots & tricks** (updated 2026-09-25, 0 ratings) and Gymeyes (2019). I found **no app with a public trick database** larger than trickipedia, which is itself a tracker.
- Google Play: not searched.

**GitHub** (API search, grade A)
- No parkour or tricking trick-list datasets found.
- "the Tricktionary" (`the-tricktionary/api`, MIT) is a **jump-rope** dictionary: exclude it.
- `dimitrisv/combogen` is a tricking combo generator (2016, no licence, not inspected).
- `m-a-x-s-e-e-l-i-g/jumpflix.tv` is a parkour **film** catalogue (CC BY-NC-ND 4.0, film-level metadata and spots, no trick names).

**Kaggle:** not found (not searched: search budget exhausted, and the API needs auth).

**Wikibooks Parkour/Movements:** exists (CC BY-SA). Not assessed.

### 1.3 Vocabulary summary

| Source | Moves | Descriptions | Videos | Licence / ToS | Overlap with our 2,677 | Recommend |
|---|---|---|---|---|---|---|
| parkourtheory (delta) | 1,990 total, **+169 new** | yes (1 sentence) | yes, 1/move | none found; robots crawl-delay 20 | ~92% overall; ~165 names new | **YES** |
| parkourtheory `/variation/` | 36 modifiers | not read | ? | none found | not held | **YES** (PMC modifiers) |
| FIG ToT 2025 + 2026 | ~189 each | no (values + scaling rules) | no | © FIG (factual lists) | 149 held; **~40 missing** | **YES** |
| FIG TRA/TUM CoP §J–L | notation, not a list | terminology | no | © FIG | n/a | **YES** as PMC precedent |
| trickipedia | 569+ (5 disciplines) | yes | partial (YouTube) | **CC-BY-SA 4.0** | ~85% | MAYBE (trampoline delta) |
| loopkicks Tricktionary | "500+" | short | third-party clips | © Loopkicks; clips not owned | ~100% | NO |
| Tricking Bible | 45 | prereqs | no | none (2007) | ~100% | NO |
| parkour.fandom | 142 articles | uneven | some | CC-BY-SA | high | alias harvest only |
| tricking/freerunning.fandom | 24 / 16 | uneven | few | CC-BY-SA | ~100% | NO |
| VideoNet parkour list | 40 basics | definitions (gated) | 5 each | gated, research only | 28/40 exact | YES (alias check) |
| Apps / GitHub / Kaggle | none found | | | | | not found |

---

## 2. Video sources

**parkourtheory per-move videos** (grade A counts; licence not found)
- Local scrape: 1,627 of 1,837 moves have a video. The project holds 1,618 clips. The 169 new moves may add up to about 169 more; how many have video is **not measured**.
- **How names attach:** one move page = one named clip, curated by the site owner. Prior project work found twist-level label noise and about 35% wrong labels on the crux attribute (survey 2026-09-28). The name is reliable at family level and weaker at modifier level.
- Hosting: Cloudflare Stream MP4, credited to the performer.
- **Feasibility:** high (already in hand). Redistribution: no.

**VideoNet – Parkour domain** (CVPR 2026 Highlight, arXiv 2605.02834v2; grade A)
- **Benchmark:** 40 actions × 5 clips = **200 clips, mean 4.4 s, median 3.0 s**. MCQ uses 160 questions for Parkour.
- **How the benchmark clips were made:**
  - Prolific non-experts, given name and definition, found 7 clips per action from distinct videos.
  - Clip verification: 3 annotators, majority vote.
  - Trimming: another stage.
  - "Five distinct annotators review each clip". "One of the authors manually inspected and adjusted the labels and trimmings of all 5,000 clips."
  - Expert verification was run on 7 domains (97.6% correct overall). **Parkour was not one of them.**
- **Training data**, mined from YouTube titles and WhisperX transcripts, localized by Gemini 2.5 Flash:
  - Parkour **6,424** (TranscriptLocalized), **4,034** (TitleMatch), **7,109** (SingleAction) clips.
  - Molmo2-4B Parkour MCQ rose 46.88 → 56.88 with SingleAction.
- **Access:**
  - HF `raivn/VideoNet` is **`gated: auto`** (HF API).
  - The form asks for institution, purpose (Research/Education), and checkboxes: "the original creators of the videos hold all rights", "use the clips ONLY as is allowed under 'fair use' doctrine".
  - No licence field. Benchmark videos are hosted with onscreen text blurred. Metadata includes YouTube IDs and timestamps.
  - The GitHub repo has no licence (last push 2026-05-06).
- **Recommendation: YES.** Request access. It gives 200 human-checked basic-move clips as an external sanity eval, and about 7k weakly named clips to mine for the basics. No twist or cork variants beyond "corkscrew".

**FIG and competition footage on YouTube** (one search page per query, 2026-09-29; grade A for existence)
- **World Gymnastics Channel** (FIG official) posts per-athlete runs, e.g. "TORHALL Elis (SWE) – 2024 Parkour Worlds, Kitakyushu (JPN) Qualification Parkour Freestyle Run 1" (0:50).
- **FISE** posts per-run clips from the **FIG Parkour Freestyle World Cup, Montpellier 2026** (0:35–0:53, ~4 months old) and a 1:21:35 livestream of the 2025 final.
- **JUMP Freerun** streamed a FIG World Cup finals day (3:45:41).
- Worlds editions (Wikipedia): FIG 2022 Tokyo, 2024 Kitakyushu. There are also Sport Parkour League Worlds (Vancouver 2022–2025) and Parkour Earth Worlds 2026 (Brno).
- **How names attach:** titles give athlete, event and run, **never trick names**. Livestream commentary sometimes names tricks (not measured). I found no official per-trick judging sheets (not found). FIG publishes D and E scores, not trick lists (not verified).
- **Feasibility:** good for a FIG-aligned eval. Freestyle runs are 20–45 s, contain the high-value aerials that drive the D-score, and come from a fixed broadcast-style camera. Every trick needs our own annotation.

**Red Bull Art of Motion** (grade A for existence)
- Official Red Bull replays exist: Matera 2019 (2:27:21), Piraeus 2021 (2:11:21), Santorini 2017 (2:30:25), and an 8:17:20 "best moments" stream.
- Judging (Wikipedia): creativity, flow, execution, difficulty.
- No trick names in titles. **MAYBE**: very high-difficulty freerunning, but multi-angle edits and slow-motion replays complicate clip extraction.

**Tutorial channels** (grade A for existence)
- Titles name the trick. For "how to cork": Kojos Trick Lab, Bob Reese, Plan Zero, MrJumptrix, Alex Destreza (Shorts). For "castaway backflip tutorial": Bob Reese, Joey Adrian, urbanamadei, FliplikeZ, 3runTube, Ethan Guzman.
- **Names attach through title and transcript.** Videos contain drills, progressions, failed attempts and slow motion, so each clip needs localization. VideoNet's SingleAction filter (title names one action, localizer finds exactly one clip) was the best of their three filters.
- **Recommendation: YES** as a weak-label train source for named tricks. **NO** as an eval source.

**Instagram / TikTok trick pages** (grade A for TikTok ToS, not verified for Instagram)
- **TikTok EEA ToS** (last updated July 2026) forbids you to "extract any data or content from the Platform using any automated system or software that is not provided by TikTok or approved in writing by TikTok". The user-to-user licence covers use "using the Platform for entertainment purposes".
- The **TikTok Research API** requires affiliation with an "academic institution in the U.S., EEA, UK, Canada, or Switzerland", an ethics review, and non-commercial purpose. It serves metadata, not video files (as described).
- **Instagram/Meta terms and Meta Content Library:** the pages did not render (JS). **Not verified this session.**
- **Recommendation: NO** for collection. Manual link-only references at most.

**Academic datasets that include parkour or tricking**

| Dataset | Content | Names | Licence / access | Use |
|---|---|---|---|---|
| VideoNet Parkour (2026) | 200 eval + ~4–7k train clips | 40 basics | gated (auto), research only | yes |
| LAAS Parkour (Maldonado/Watier; used in Li et al. CVPR 2019, arXiv 1904.02683) | "two-hand jump, moving-up, pull-up and a single-hand hop"; Vicon 400 Hz, force plates 2200 Hz, RGB 25 fps | 4 actions | GitHub `zongmianli/Parkour-dataset`, **no licence file**, last push 2020-10-21 | no (biomechanics only) |
| HIL (Wang et al., arXiv 2505.12619) | 19 YouTube clips, 15 parkour skills, 30 s total, for physics-based character control | per skill | release not found | no |
| Kinetics-700 | classes parkour, backflip (human), somersaulting, gymnastics tumbling, cartwheeling, bouncing on trampoline | coarse | YouTube IDs | pretraining only |
| SportSkills (arXiv 2603.25163) | 55 sports, 638k clips | none of parkour / gymnastics / tricking / trampoline in its sport list | — | no |
| ActionAtlas (NeurIPS D&B 2024, arXiv 2410.05774) | 934 videos, 580 actions, 56 sports | whether parkour is included: **not verified** | YouTube IDs | check later |
| Trampoline pose synthetic set (arXiv 2604.01322) | pose, not moves | — | see survey 07-03 | pose only |

Any 2024–2026 parkour or tricking recognition dataset with twist or modifier-level labels: **not found** (consistent with survey 2026-09-28).

### 2.1 Video summary

| Source | Scale | How names attach | Licence / legality (research) | Feasibility | Recommend |
|---|---|---|---|---|---|
| parkourtheory per-move clips | ~1,627 held; ≤ ~1,990 | 1 curated name per clip (modifier noise) | none found; no redistribution | in hand | **YES** (pool for eval candidates) |
| VideoNet Parkour | 200 eval / ~4–7k train | human-verified (eval); title/transcript (train) | gated auto; research/education; "fair use" checkbox | 1 form | **YES** |
| FIG runs (World Gymnastics Channel, FISE, JUMP Freerun) | dozens of runs per event; multi-hour streams | athlete/run in title only | YouTube ToS; EU TDM exception may apply (see §5) | clip by run, annotate | **YES** (FIG-aligned eval) |
| Red Bull Art of Motion | ~2 h replays × several editions | none | YouTube ToS | edits and replays complicate | MAYBE |
| Tutorial channels | many per trick | title + transcript | YouTube ToS | needs localization | **YES** (weak train) |
| Instagram / TikTok | large, not measured | captions / hashtags | ToS forbid automated extraction; research APIs gated | poor | **NO** |
| LAAS, HIL, Kinetics, SportSkills | small or coarse | — | mixed | — | NO |

---

## 3. Labelling tools, annotation schemes, and a verify-first workflow

### 3.1 Precedent schemes

- **FineGym** (CVPR 2020, arXiv 2004.06704):
  - Categories come from "official documentation" (the FIG CoP).
  - Element labels are chosen by **"a decision-tree consisting of attribute-based queries"**. The annotator walks from the set to the leaf by answering attribute questions (e.g. "3 turn or more?").
  - Annotation was by a "team trained specifically for this job", not MTurk. Quality control: "training annotators with domain-specific knowledge, pretesting the annotators rigorously … referential slides as well as demos, and cross-validating across annotators".
  - The decision-tree path also produces the attribute vector for free. This is the closest precedent to annotating a PMC field by field.
- **FineDiving** (CVPR 2022, arXiv 2204.03646):
  - A lexicon built with "three professional athletes of the diving association".
  - A two-stage coarse-to-fine pass: action type and boundaries, then the step sub-action types and each step's start frame. Steps follow the FINA dive number (take-off / somersault / twist / entry).
  - "Six workers who have prior knowledge in the diving domain", each part checked by another worker, using a public frame-labelling toolbox. **"The total time of the whole annotation process is about 120 hours"** for 3,000 videos, about **2.4 min per video** including step boundaries.
- **VideoNet:** non-experts become reliable when the task is reduced from k-way choice to a **binary verification** (is this clip action X, yes/no?) with a written definition. The definitions were written with an LLM plus web search and cross-checked. 97.6% of clips were correct on expert audit. This is the verify-first model.
- **SportSkills CoachGT:** experts spent about 4 min per rating task (15 tasks per hour). That is a useful ceiling for expert time on richer judgments.

### 3.2 Tools

| Tool | Licence | Activity | Video features relevant to PMC | Verdict |
|---|---|---|---|---|
| **Label Studio Community** | Apache-2.0 | v1.23.1, 2026-09-25 | `<Video>`, **`<TimelineLabels>`** ("a single frame or a span of frames"), **`<Choices>`** with per-choice `hotkey`, `perRegion`, `visibleWhen`/`whenTagName`/`whenChoiceValue`, `<Taxonomy>` (works with video, remote `apiUrl` beta); **predictions import** "for review and correction" with `model_version` and `score` | **Recommended first** |
| CVAT | MIT | v2.77.0, 2026-09-28 | video tracks, tags with attributes; Docker self-host | heavier than needed |
| VGG VIA 3 | BSD-2 | pushed 2026-03-05 | single offline HTML page, temporal segments + attributes | fallback, simplest to host |
| Custom page (as VideoNet built) | own | — | keyboard-only verify of a pre-filled PMC string | if the Label Studio pilot is slower than 40 s/clip |

Not verified: Label Studio docs show prediction examples for images, text and OCR only, not for video `TimelineLabels`. Task-level `Choices` predictions are documented, and video-region predictions should be tested in the pilot.

### 3.3 Recommended workflow for ~300 clips at 20–40 s each

1. **Pick clips** from parkourtheory (named), FIG runs (segmented per trick) and own filming. Balance by PMC cells (som × twist × axis × context), not by name.
2. **Pre-fill every field before a human sees the clip:**
   - Name → PMC via the G1 parser, for named clips.
   - Skeleton rule counters for `som`, `twist` and `som_dir`, from the RTMPose keypoints already extracted.
   - Optional VLM second opinion on `CONTACT.element`, `axis` and `EXIT`.
   - Store the disagreements and a confidence score per field.
3. **Verify in Label Studio.** Use 0.25×–0.5× looped playback. Show the pre-filled compact PMC string; highlight only disagreeing fields; use one-key accept (hotkeys on `Choices`). `TimelineLabels` holds the SETUP/CONTACT/AIR/EXIT spans only where boundaries are needed. Add a **flag** key for "ambiguous / unwatchable".
4. **Time budget** (C, estimate):
   - Clips are about 3–5 s (VideoNet parkour median 3.0 s). Two loops at 0.5× take about 12–20 s, plus 5–15 s for decisions. So **20–40 s is plausible for clips where pre-fill is right**.
   - 300 × 30 s ≈ **2.5 h**.
   - If about 30% need edits at about 2 min each (FineDiving averaged 2.4 min for full annotation), add about **3 h**.
   - Total **about 5–6 h**, in 30-min sessions.
5. **Agreement:** a second rater (e.g. a FIG judge or coach via the stakeholder contact) double-labels **10–15% (30–45 clips)**, blind to the pre-fill. Report per-field Cohen's κ, or exact match for the ordinal `som`/`twist`.
6. **Pilot gate:** time the first 20 clips. If the median is over 40 s, or the pre-fill is wrong on more than 40% of clips, fix the pre-fill or UI before continuing.

> Tension with the project memory (`project_no_manual_labels`): this is verification of machine proposals, which is how the flywheel was defined. But 300 clips at 5–6 h is more human time than the "~99 gold clips only" constraint implies. **Owner decision.**

---

## 4. Dataset-paper precedent and what a parkour dataset needs

### 4.1 What made the precedents valuable

| Dataset | Venue | Size | Label depth | What made it matter |
|---|---|---|---|---|
| **FineGym** | CVPR 2020 (arXiv 2004.06704) | 303 competitions (~708 h); 4,883 event instances; **32,697 sub-actions**; 530 element classes (354 with ≥1 instance) | 3 semantic levels (event / set / element) × 2 temporal levels | Official taxonomy, decision-tree labels (attributes for free), benchmarks Gym99/Gym288 (Gym530 defined), and a headline failure (ST-GCN Gym99 36.4% top-1) |
| **FineDiving** | CVPR 2022 (arXiv 2204.03646) | **3,000** videos; 52 action types, 29 sub-action types, 23 difficulty degrees | action + step boundaries + official scores | First procedure-level AQA set; expert lexicon; FINA dive-number decomposition; code MIT |
| **Diving48** | ECCV 2018 (RESOUND) | **~18k** clips, 48 classes, 1 official split | label = 4 attributes (take-off, somersault, twist, flight position) | Built to have **low scene bias** (bias 1.26 per Choi et al. NeurIPS 2019), so models must read motion |
| **MultiSports** | ICCV 2021 (arXiv 2105.07404) | 3,200 clips, **37,701 instances**, 902k boxes, 66 classes, 4 sports | spatio-temporal tubes | Dense multi-person detection benchmark |
| Small AQA sets (from the FineDiving table) | various | MIT Dive 159, UNLV Dive 370, AQA-7-Dive 549, MTL-AQA 1,412 | score (± action) | Show that a few hundred clips can publish if the task is new |
| **ActionAtlas** | NeurIPS 2024 D&B | 934 videos, 580 actions, 56 sports | MCQ; test-only | Shows a test-only benchmark is acceptable at a D&B track |
| **VideoNet** | CVPR 2026 Highlight | 5,000 eval clips (Parkour 200) + ~160k train | name MCQ / binary | Breadth + training data + open-vs-closed model gap |

Common ingredients:
- an authoritative label system (FIG, FINA);
- compositional labels (Diving48 attributes, FineGym decision trees, FineDiving steps);
- low scene bias;
- fixed splits and baselines showing current models fail;
- release as annotations + YouTube IDs (FineGym, ActionAtlas);
- a documented annotation protocol with cross-checking.

### 4.2 What a parkour dataset paper would need

- **Novelty claim:** the first parkour/freerunning dataset with a **compositional physical code** (the PMC, the "dive number" for parkour), plus open-vocabulary naming. Include a held-out **unseen-trick split**, reusing the locked design/test name split. FIG-aligned D-score-relevant subsets strengthen it.
- **Scale by venue** (C, judgement):
  - 300 verified clips is **workshop scale**: CVsports @ CVPR. The 2026 edition was the 12th, datasets are in scope, and papers were due 2026-03-05, so expect about March 2027 for the next one.
  - For **NeurIPS D&B** or a WACV applications paper, pair the 300-clip gold test set with a larger **weakly labelled train split**: the parkourtheory ~1.6–2k named clips with parser PMCs, plus VideoNet-style mined clips.
  - A CVPR main-track dataset paper would likely need thousands of clips (FineDiving 3k, FineGym 33k).
  - CVPR/WACV formal dataset tracks: **not verified**.
- **Must-haves:**
  - (a) Per-field inter-rater agreement on a subset.
  - (b) Athlete- and source-disjoint splits.
  - (c) Baselines: frontier VLMs at dense fps, open VLMs including VideoNet-fine-tuned Molmo2, skeleton rule counters (NS-AQA style), and a supervised skeleton model.
  - (d) A datasheet and **Croissant metadata** (NeurIPS 2025 D&B requires Croissant, hosting on HF/Kaggle/Dataverse/OpenML, and reviewer access without contacting authors).
  - (e) Clean licensing (§5).
  - (f) A clear statement of what is out of scope (execution quality).

---

## 5. Legal and licence risks

(Not legal advice. Sources as cited.)

1. **parkourtheory:** no licence or terms found.
   - Descriptions are copyrighted text by default.
   - Clips are credited third-party excerpts.
   - Trick names are facts and are fine to use.
   - Do not redistribute descriptions or videos.
   - Crawling is allowed by robots.txt at 20 s delay. Asking for permission (`hello@parkourtheory.com`) removes the ambiguity.
2. **FIG PDFs:** © FIG. Using trick names and values as facts with citation is low-risk. Do not republish the PDFs.
3. **Trickipedia:** CC-BY-SA 4.0 → attribution plus **share-alike** on any derived file that includes their text.
4. **Fandom:** CC-BY-SA.
5. **loopkicks:** © plus clips it doesn't own. Names only.
6. **YouTube ToS** (effective 2026-01-09): no download or automated access "except … (c) as permitted by applicable law".
   - The EU DSM Directive (2019/790) **Art. 3** allows text and data mining for scientific research by **research organisations** with lawful access, and rightholders cannot opt out.
   - **Art. 4** allows it for anyone, but rightholders can opt out in machine-readable form.
   - Whether a student project counts as a "research organisation" depends on university affiliation. **Not verified.**
   - French transposition, CPI L122-5-3: page not fetched.
   - Safe release pattern: publish **YouTube IDs + timestamps + labels**, not clips. Accept link rot (Kinetics lost about 5% within a few years, per Choi et al. 2019's footnote).
7. **VideoNet:** research/education only, "fair use" self-certification (a US doctrine; the EU has no general fair use). No licence, so derived data can't be relicensed.
8. **TikTok:** ToS forbid automated extraction; the Research API needs an institution and an ethics review. **Instagram:** terms not verified this session.
9. **GDPR:** athletes are identifiable (faces, names in titles). An EU-hosted dataset of identifiable people needs:
   - a lawful basis (research or legitimate interest);
   - a data-protection notice;
   - takedown on request;
   - **written consent forms for own-filmed clips.**

   Own-filmed consented clips are the cleanest core for a publishable test set.
10. **Non-commercial licences** (FineGym annotations CC BY-NC 4.0, per survey 05): fine for research. Decide before any commercial FIG judge-assist.

---

## 6. Gaps (not found or not verified)

- Searched and not found:
  - a FIG **2027** table (only 2025 and 2026 exist);
  - any public parkour/tricking trick database on GitHub;
  - an app with a public trick database;
  - any parkour/tricking dataset with twist or modifier labels;
  - official per-trick judging data for FIG competitions;
  - release of the HIL parkour clips.
- Not searched (WebSearch budget exhausted): Kaggle, Google Play apps, other tricking wikis beyond Fandom, parkour video channels beyond two YouTube queries, WACV/CVPR dataset-track rules, the NeurIPS 2026 D&B call (the 2025 call was used).
- Not verified:
  - Instagram/Meta terms and Meta Content Library eligibility (pages did not render);
  - the Diving48 official page (connection refused; figures from secondary citations);
  - whether ActionAtlas includes parkour;
  - Label Studio prediction import for video `TimelineLabels`;
  - the MAG/WAG CoP element counts;
  - how many of the 169 new parkourtheory moves have videos;
  - whether FIG streams carry trick-naming commentary;
  - the French L122-5-3 text.

---

## Sources (all accessed 2026-09-29 unless noted)

**Vocabularies**
- parkourtheory: https://www.parkourtheory.com/robots.txt ; https://www.parkourtheory.com/sitemap.xml → `sitemap-moves-1.xml`, `sitemap-pages.xml`, `sitemap-categories.xml` (lastmod 2026-09-29) ; JS bundle `/static/js/main.8d32c91a.js` (API base `https://parkourtheory.com/v3`) ; GitHub org https://github.com/parkourtheory (no public repos). No licence found.
- FIG PK Table of Tricks 2026: https://www.gymnastics.sport/publicdir/rules/files/en_1.1.1%20-%20PK%20Code%20of%20Points%202025-2028%20-%20Table%20of%20tricks%202026.pdf (PDF created 2026-03-30). © FIG.
- FIG PK Table of Tricks 2025: https://www.gymnastics.sport/publicdir/rules/files/en_1.1.1%20-%20PK%20Code%20of%20Points%202025-2028%20-%20Table%20of%20tricks%202025.pdf (April 2025, created 2025-04-02).
- FIG PK 2023/2024 tables and CoP 2022–2024/2025–2028: gymnastics.sport `publicdir/rules/files/` (URLs seen in search results before the budget ran out; not re-read).
- FIG Trampoline Gymnastics CoP 2025–2028: https://www.gymnastics.sport/publicdir/rules/files/en_1.1%20-%20TRA%20Code%20of%20Points%202025-2028.pdf (created 2025-07-09), §B, §J, §K, §L.
- Trickipedia: https://trickipedia.app ; /parkour ; /tricking ; /trampoline ; /trampwall ; /tumbling ; /terms. CC-BY-SA 4.0.
- Loopkicks Tricktionary: https://www.loopkickstricking.com/tricktionary ; example https://www.loopkickstricking.com/tricks/back-pike ; `loopkicks.com/tricks` → 302 to strongdomains.com (domain for sale). © 2026 Loopkicks Tricking.
- Fandom (MediaWiki API `api.php?action=query&meta=siteinfo`): parkour.fandom.com, tricking.fandom.com, freerunning.fandom.com, gymnastics.fandom.com. CC-BY-SA.
- Tricking Bible: local `data/tricking_bible.pdf` (2007, author "Sesshoumaru").
- App Store: iTunes Search API, `country=fr`, 5 queries.
- GitHub: `the-tricktionary/api` (jump rope), `dimitrisv/combogen`, `m-a-x-s-e-e-l-i-g/jumpflix.tv` (CC BY-NC-ND 4.0).
- Kinetics-700 labels: https://raw.githubusercontent.com/open-mmlab/mmaction2/main/tools/data/kinetics/label_map_k700.txt

**Video**
- VideoNet: arXiv 2605.02834v2 (Tables 9, 20, 21; §3.2–3.3; App. B); https://github.com/RAIVNLab/VideoNet (`benchmarks/mcq_val.jsonl`, `mcq_test.jsonl`; no licence; pushed 2026-05-06); https://huggingface.co/api/datasets/raivn/VideoNet (`gated: auto`, gating form fields).
- YouTube search pages (2026-09-29): "FIG parkour world cup freestyle final", "red bull art of motion full event", "how to cork tutorial", "castaway backflip tutorial parkour". Channels: World Gymnastics Channel, FISE, JUMP Freerun, Red Bull, Kojos Trick Lab, Bob Reese, Plan Zero, MrJumptrix, Ethan Guzman, 3runTube, FliplikeZ, Alex Destreza, Joey Adrian, urbanamadei.
- Wikipedia: https://en.wikipedia.org/wiki/Parkour_World_Championships ; https://en.wikipedia.org/wiki/Red_Bull_Art_of_Motion
- LAAS Parkour: Li et al., "Estimating 3D Motion and Forces of Person-Object Interactions from Monocular Video", CVPR 2019, arXiv 1904.02683 §6.1 ; https://github.com/zongmianli/Parkour-dataset (no licence; pushed 2020-10-21) ; Maldonado et al. 2017, CMBBE 20(sup1):123–124.
- HIL: Wang et al., arXiv 2505.12619v2 §6.1 "Dataset".
- SportSkills: Ashutosh et al., arXiv 2603.25163 (sport list, App. Fig. 4).
- ActionAtlas: Salehi et al., NeurIPS 2024 D&B, arXiv 2410.05774.
- TikTok EEA ToS (last updated July 2026): https://www.tiktok.com/legal/page/eea/terms-of-service/en ; Research API: https://developers.tiktok.com/products/research-api/
- YouTube ToS (effective 2026-01-09): https://www.youtube.com/static?template=terms&gl=FR&hl=en
- EU DSM Directive 2019/790 Arts. 3–4: summary via https://en.wikipedia.org/wiki/Directive_on_Copyright_in_the_Digital_Single_Market (EUR-Lex did not render).

**Annotation and precedent**
- FineGym: Shao et al., CVPR 2020, arXiv 2004.06704 (§3.2–3.3, Table 1, Table 3).
- FineDiving: Xu et al., CVPR 2022, arXiv 2204.03646 (§3.1–3.2, App. D.2); https://github.com/xujinglin/FineDiving (MIT).
- MultiSports: Li et al., ICCV 2021, arXiv 2105.07404; https://github.com/MCG-NJU/MultiSports.
- Diving48: Li, Li, Vasconcelos, "RESOUND", ECCV 2018 (http://www.svcl.ucsd.edu/projects/resound/dataset.html, unreachable today); figures via Choi et al., NeurIPS 2019, arXiv 1912.05534 §4.1.
- Label Studio: https://github.com/HumanSignal/label-studio (Apache-2.0, v1.23.1 2026-09-25); https://labelstud.io/tags/timelinelabels ; /tags/choices ; /tags/taxonomy ; /guide/predictions ; /templates/video_timeline_segmentation.
- CVAT: https://github.com/cvat-ai/cvat (MIT, v2.77.0 2026-09-28). VIA: https://github.com/ox-vgg/via (BSD-2).
- CVsports 2026: https://vap.aau.dk/cvsports/ ; /call-for-papers/.
- NeurIPS 2025 D&B call: https://neurips.cc/Conferences/2025/CallForDatasetsBenchmarks

**Project files read (not modified):** `data/unified_tricks.json`, `data/other_sources_tricks.json`, `data/parkourtheory_detailed.json`, `data/fig_tricks_2025.json`, `data/openclaw_scraping_prompt.md`, `docs/superpowers/specs/2026-09-28-parkour-motion-code.md`, `docs/science-superpowers/survey/2026-09-28/*` (grep only).
