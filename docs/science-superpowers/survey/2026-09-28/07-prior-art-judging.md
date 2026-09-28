# 07: Prior art, deployed judging systems, and the full-run judge-assist

Survey date: 2026-09-28. Desk research only: no project data touched, no code run on project data, $0 spent.
Track: (1) automatic trick recognition / judging in parkour and action-sport analogs, (2) the Fujitsu Judging
Support System (JSS) as of 2025-2026, (3) the FIG Parkour Code of Points and what a judge-assist must output,
(4) full-run temporal action segmentation / localisation in sports, (5) competitive landscape and novelty.

Grading: **A** = measured in-domain (parkour). **A\*** = primary FIG rule text (authoritative, in-domain, normative
rather than measured). **B** = measured on an analog sport. **C** = claim, vendor statement, press or app-store copy.
"Snippet only" means I saw the search-engine excerpt but could not open the page (403 or JS-rendered); treat
those as C regardless of the outlet.

**No grade-A evidence of automatic parkour trick recognition or judging exists in anything I could find.** The
only A-grade items are FIG rule documents, one competition result, and one human-rater reliability study on a
beginner parkour skills test.

---

## 0. Bottom line

1. **Nobody is doing camera-based parkour judging.** No paper, dataset, product, startup or FIG pilot does
   automatic parkour or freerunning trick recognition or judging (searched arXiv, alphaXiv, web, app stores,
   FIG, Parkour Earth, Red Bull Art of Motion). The nearest parkour items are a 4-action biomechanics dataset
   (LAAS), a 0-100 "technique score" app for 5 beginner moves that has not launched (Better Form), and a
   flip-detection training app (TrickCam) that does not identify tricks.
2. **The run-length premise in the brief is wrong for FIG.** FIG freestyle runs last **20 to 45 s**, not
   60-70 s. Timing starts at the first step, jump or swing. There is a warning at 40 s. **No trick after 45 s
   earns D**, but E is still judged. Source: PK CoP 2025-2028 §3.2/§4.1 and TR 2026 and 2027 §7 Art. 1.2 (A\*).
3. **FIG D-score = the three highest scaled trick values**, with at most two from the same category (the category
   cap is new in the 2026 Table of Tricks). Values are "guiding values" that judges scale up for placement,
   form, entry and exit. Failed tricks and repeated tricks do not count. The 2026 table added a −0.3 to −0.5
   penalty for unintentional off-axis tilt (A\*). So the assist does not need to parse every movement. It needs
   high recall on the few high-value aerial tricks, plus the context needed for scaling.
4. **Fujitsu JSS is the only FIG-approved judging AI.** It covers artistic gymnastics only (10 apparatus) and
   uses 4-8 fixed HD cameras per apparatus (it used 3D lidar until 2023). It recognises about 2,000 elements at
   a claimed ~90% agreement with judges (C). Until 2025 it was used only for inquiries and blocked scores. From
   the October 2026 Rotterdam Worlds, **Apparatus Supervisors will review JSS output before submitting their
   D-score** (FIG MAG Newsletter, July 2026; C). The path took 2017 (collaboration), 2018 (tests), 2019 (first
   official use on 4 apparatus after test events and FIG Executive Committee approval), and 2023 (all 10
   apparatus). The 2018 roadmap promised "automation of judges' scores from 2020". That still has not happened
   in 2026.
5. **FIG has made no statement on AI for parkour.** None found. FIG rules keep video subordinate to judges: IRCOS
   "is NOT intended to replace the existing judging system by a video judging system" (A\*). For parkour, FIG
   requires only **one** video recording setup per event (TR 2026/2027 Art. 4.10.4.1; A\*), so the
   infrastructure a 1-2 iPhone system would replace or augment is minimal.
6. **Segmentation of a full routine works on single moving-camera broadcast in analog sports (B), without being
   solved.** Figure-skating TAS (VIFSS 2025) gets **92.6% F1@50 element-level** (jump type plus 1-4 rotations)
   on 371 broadcast programs, with 2D-pose baselines at 78.8%. It collapses to ~50% at F1@90 (±1-2 frame
   boundaries). Precise take-off/landing spotting (T-DEED 2024) reaches 85-87 mAP at 1-frame tolerance on
   skating broadcast. An unsupervised skeleton method (UTAL-GNN 2025) gets 82.7 mAP on diving boundaries with
   no labels. On in-the-wild phone/social video, a calisthenics TAS baseline gets only **mASF1 0.674**, and that
   is with one skill per clip. No acrobatic TAS result on handheld multi-trick footage exists.
7. **Camera-only is no longer novel. Parkour, consumer phones, and CoP alignment are.** Fujitsu, Swiss
   Timing/Omega and Owl AI are all camera-based. The open niche is a validated, 1-2-phone,
   parkour-CoP-aligned proposer that outputs trick identity, attributes, scaling cues and a top-3 D proposal
   for a human to confirm. The closest threat is **Owl AI**: $11M seed, X Games / action sports, claims
   "camera quality available on most iPhones". It has not published any validation (C) and has not mentioned
   parkour.

---

## 1. Parkour, freerunning and action-sport analogs: recognition and judging (2020-2026)

### 1.1 Parkour / freerunning

| # | Item | What it does | Sensors | Accuracy | Deployed? | Grade |
|---|---|---|---|---|---|---|
| P1 | FIG Parkour CoP 2025-2028 + Tables of Tricks 2025/2026 | Human judging only: 3 D-judges and 3 E-judges | Eyes, plus one required video setup (TR) | n/a | Yes (rules) | A\* |
| P2 | Automatic parkour trick recognition / judging research | **Not found.** An arXiv search for parkour/freerunning returns only legged-robot and humanoid "parkour" locomotion papers | n/a | n/a | n/a | n/a |
| P3 | LAAS Parkour dataset (Li et al.; GitHub `zongmianli/Parkour-dataset`, linked to arXiv 2111.01591, 2021-11-02) | Videos of 4 actions (safety vault, kong vault, pull-up, muscle-up) with ground-truth 3D pose and contact forces. Biomechanics, not recognition | RGB + mocap/force | n/a | Research dataset | B (not recognition) |
| P4 | Dvořák, Baláš, Martin, "The Reliability of Parkour Skills Assessment", *Sports* 2018-01-24 | 3 raters score a 10-technique course from **video** (0-45 pts) | Video | Inter-rater Krippendorff α **0.910-0.916**, between trials 0.828-0.874, between days 0.839-0.924. 20 young males (beginner and advanced) | Research test | A (human raters, not competition, not AI) |
| P5 | Parkour AI (Better Form) | 0-100 "technique score" for 5 basic moves (precision, safety vault, wall run, roll, cat leap). No trick ID, no FIG D | Single phone video | None published | **Not launched** (waitlist; page updated 2026-04-30) | C |
| P6 | TrickCam (Android) | Detects standing backflip/frontflip take-off, auto slow-mo, jump height, air time, rotation speed estimate. No trick naming | Single phone | None published | App store | C (snippet only) |
| P7 | Red Bull Art of Motion / Parkour Earth Unified Guidelines 2026 / WFPF | Human judging. No AI/technology mention found | n/a | n/a | n/a | C (absence) |
| P8 | FIG pilot of AI for parkour | **Not found** | | | | |
| P9 | FIG statement on AI judging for parkour | **Not found.** The FIG AIPS talk by Watanabe ("AI in gymnastics", date not retrievable) shows only its meta description | | | | |
| P10 | Parkour at The World Games 2025, men's freestyle final (Chengdu, 2025-08-12) | Winner Shiohata (JPN) **D 16.0**, E 12.8, total 28.8. Qualification D 14.8 | n/a | n/a | Record | A (result record) |

P10 matters for design: D 16.0 is three tricks averaging ~5.3 each, i.e. doubles (Double Backflip 5.3, Double
Gainer 5.5) plus scaling. At elite level, PK Basics (0.1-1.8) never enter the top 3.

### 1.2 Analogs

| # | Sport / system | What it does | Sensors | Accuracy | Deployed? | Grade |
|---|---|---|---|---|---|---|
| S1 | **Owl AI** (X Games spin-out; Jeremy Bloom, with Sergey Brin's help; CEO Josh Gwyther; $11M seed led by S32, June 2025) | AI judge for action sports. X Games Aspen 2025 SuperPipe trial ran alongside humans. In 2026 it "re-judged" the Milano Cortina women's snowboard slopestyle and flipped gold/silver (86.67 vs 89.13) | Video. Built on Google Cloud / Vertex AI, "thousands of hours" of footage. Bloom: needs only "camera quality available on most iPhones" | X Games: "overscored compared to the humans, but accurately predicted podium outcomes". **No published comparison** | Unofficial. Also partners with Super Motocross and Major League Pickleball. Bloom mentions gymnastics and figure skating as future targets | C |
| S2 | **Figure skating, ISU + Swiss Timing/Omega** (Milano Cortina 2026) | 3D tracking: jump height, air time, landing speed, blade angle. Rotation-completeness judging support planned. ISU calls it "a data-support tool rather than integrating it into the judging framework" | **14 × 8K cameras** around the rink, markerless | None published | Broadcast graphics at the Games. Judges "could have access later". Singles first, then pairs and dance | C |
| S3 | China "Figure Skating AI-Assisted Scoring System 1.0" (pre-Beijing 2022) | Tracks 8 key points | CV | n/a | Unclear | C (via CGTN) |
| S4 | **Diving, Swiss Timing** (Paris 2024) | 3D reconstruction of the dive: air time, entry speed, "safe gap" to the board, as judge support | Multi-camera | n/a | Judge-review aid | C (snippet only) |
| S5 | Diving, Baidu training system (China team, Paris 2024) | Training feedback via an Ernie LLM | Video | n/a | Training only | C (snippet only) |
| S6 | **Breaking** (WDSF Trivium, Paris 2024) | Digital slider system (5 criteria, head-to-head). **No AI** | Judge tablets | Human judge ICC (single, absolute) **0.206-0.452** across criteria (Sato, *Front. Psychol.* 2025-12-03). Low-moderate reliability plus national bias (Braeunig, arXiv 2511.00553, 2025-11-01) | Digital scoring deployed. AI not | B (human judges) |
| S7 | Breaking, research | Classifies BRACE (Red Bull BC One) segments into toprock/footwork/powermove. Video encoders beat VLMs (arXiv 2510.20287, 2025-10-23) | Broadcast video | Coarse 3-class | Research | B |
| S8 | **Skateboarding** (World Skate, Olympics) | **No AI judging trial found.** Research: SkateboardAI dataset of in-the-wild trick clips (arXiv 2311.11467, 2023) | Video | Not in abstract | Research | B/C |
| S9 | **BMX freestyle** (UCI) | **No AI judging found.** Aug 2025 UCI judges' course only | n/a | n/a | n/a | C (absence) |
| S10 | Freestyle ski/snowboard (FIS) | No FIS AI judging found beyond Owl AI's unofficial work and IOC "considering" AI for big air/halfpipe measurement | n/a | n/a | n/a | C |
| S11 | Tricking | **Nothing found** (research, apps or competitions) | | | | |
| S12 | Consumer gymnastics apps: **Stickd** (USAG optional code, single iPhone 60 fps, D/E breakdown, $9.99/mo, early access with a Feb 2026 sample) and **CheckForm** ("Google Gemini" cloud scoring; claims FIG D/E) | Skill ID + score estimate | Single phone | None published. A CheckForm user review: "it keeps saying the wrong skill" (Tsukahara called a front handspring) | App stores | C |
| S13 | Trampoline, single camera (Connolly et al., arXiv 1709.03399, 2017-09-11) | Identifies 20 trampoline skills from pose-angle trajectories with nearest neighbour | 1 video camera | **80.7%** on 714 skills | Research | B |
| S14 | Calisthenics (Finocchiaro et al., arXiv 2507.12245, 2025-07-16) | Temporal segmentation of 9 static skills for judging hold time | Social-media and ad-hoc video (839 clips), OpenPose | mASF1 **0.674** | Research | B |

---

## 2. Fujitsu Judging Support System (JSS), status 2025-2026

### 2.1 Timeline and approval path

| Date | Event | Grade | Source |
|---|---|---|---|
| Oct 2017 | FIG and Fujitsu announce collaboration on a JSS for artistic gymnastics | C | Fujitsu PR 2017 |
| 2018-11-20 | "Practical implementation" decision. **3D laser sensors.** Roadmap: 2019 Worlds; leading countries by 2024; FIG's 146 members by 2028; **"automation of judges' scores" from 2020**; training, e-learning and broadcast uses | C | Fujitsu PR 2018 |
| 2018-2019 | Technical verification at 2018 Worlds (Doha), then tests at the Tokyo All-Around World Cup (Apr 2019) and Junior Worlds (Győr, Jun 2019). **FIG Executive Committee approved** the official debut | C | Fujitsu PR 2019-10-02 |
| Oct 2019 | First official use at the Stuttgart Worlds on **4 apparatus** (pommel, rings, men's and women's vault). The Superior Jury uses it "in cases of inquiry or blocked scores" | C | Fujitsu PR 2019-10-02 |
| 2021 | Used at the Tokyo Olympics for some events | C | Asia News Network (snippet only) |
| 2023-10-05 | All **10 apparatus** at the Antwerp Worlds. **Replaced sensors with camera-based image analysis** ("Human Motion Analytics", 4D capture, >100 basic motions, jitter-correction algorithm) | C | Fujitsu PR 2023 |
| 2024-01-16 | MIT Tech Review: 4-8 HD cameras per apparatus, trained on 8,000 routines, **~2,000 elements at "about 90% accuracy" vs human judges**. Used only for inquiries and blocked scores. Cannot judge artistry, beam connections or dance elements. Coaches blocking cameras impair it. Score sheets record nothing about JSS use | C | MIT TR |
| 2024-08-05 | Same figures repeated (4-8 cameras; ~2,000 elements; 90%) | C | R&D World |
| Oct 2025 | Jakarta Worlds: Watanabe announces "AI Judging", with JSS output displayed in the venue | C | Gymnastics Now |
| 2025-11-20 | Fujitsu Research paper on multi-camera gymnast tracking "applied to recent Gymnastics World Championships". Few cameras fit in the arena. Often only **two opposing views** give valid detections. Needs a ray-plane prior (gymnast in a vertical plane) | B | arXiv 2511.16532 |
| Jul 2026 | MAG Newsletter N°4: after a "successful test in Jakarta", JSS will be used "throughout all phases" of the **Rotterdam 2026 Worlds**. **"Before submitting their D-score evaluation, Apparatus Supervisors will have the opportunity to review the Judging Support System output."** FIG is also rolling out "Real-Time Judging": E-jury deductions entered live on PC, phone, tablet or Stream Deck, with training on the STS video library | C (federation statement) | FIG MAG NL 4/2026 |

### 2.2 Answers to the brief

- **Sensors now:** camera-only (4-8 HD cameras per apparatus), since 2023. It was lidar ("3D laser sensors")
  from 2018 to about 2022. (C)
- **Apparatus:** all 10 artistic apparatus. **No other FIG discipline** (rhythmic, trampoline, acro, aerobic,
  parkour) was found using JSS. The FIG TR 2025/2026/2027 do not mention JSS at all.
- **Accuracy claims:** ~90% agreement on ~2,000 elements (journalism relaying Fujitsu/FIG; C). No
  per-apparatus, per-element or peer-reviewed figures were found.
- **Approval process:** test events, then FIG Executive Committee approval, then use restricted to the Superior
  Jury for inquiries and blocked scores, then (2026) Supervisor pre-review of D. IRCOS rules (Appendix to the
  CoP, July 2024; A\*) restrict video to inquiries and blocked scores. They give access to the Superior Jury
  President, to Supervisors *after* they score, and to D-judges only in special cases (MAG/WAG "0-vault"). They
  state IRCOS "is NOT intended to replace the existing judging system by a video judging system".
- **Camera-only / AI roadmap:** no published FIG roadmap for fully automatic or AI judging found. IOC Olympic AI
  Agenda (April 2024) lists "improved refereeing/judging" as a goal (C). The first Olympic AI Forum was in
  November 2025 (C, via The Conversation).
- **FIG statements on AI for parkour:** not found.

---

## 3. FIG Parkour Code of Points and what a judge-assist must output

### 3.1 Rule facts (A\*)

Sources: PK CoP 2025-2028 (published 2024-04-22, in force 2025-01-01), Table of Tricks 2025 (April 2025),
Table of Tricks 2026, TR 2026 Section 7 v1.0 (May 2025), TR 2027 Section 7 v1.0 (May 2026), Appendix to the CoP
(July 2024).

- **Panel:** 6 judges, 3 E and 3 D. E is out of 15. D has "no theoretical maximum (15+)".
- **Run:** 20-45 s. Timing starts at the "first step, jump or swing". A run under 20 s costs −2.00. Signal at
  40 s, maximum at 45 s. "The judges will not consider any tricks or movements for Difficulty after the maximum
  time", but E still applies. The athlete ends with arms crossed. Qualifications allow a second run, and only the
  second counts.
- **Field of play:** at least 40 × 10 m, with a designated freestyle zone. Leaving it means disqualification.
- **D construction:** "A run as difficult as possible is desired. The three (3) best D scores add up to the final
  score for this criterion." The Table of Tricks gives "guiding values for elements in their most basic form. The
  job of the judges is to identify the element and adjust its value according to Scaling."
- **Categories:** Swing Moves, Wall Moves, Acrobatics Moves, PK Basics. The 2026 table lists ~150 senior entries,
  from 0.1 (stride, drop, roll, precision) to 7.5 (Quad Cork; Castaway Double Backflip; Swing Double Gainer 1080
  "Miller").
- **Remarks (2026):**
  1. A failed trick gets no D. That means an uncontrolled landing lying down (−4.00 E) or a major crash (−6.00 E).
  2. **Repeated tricks are not considered**, "even if they differ in form, entry, placement, or exit".
  3. One table for men and women.
  4. **New in 2026:** an on-axis trick "performed with visible unintentional tilt" gets negative scaling of **−0.3
     to −0.5**. This does not apply to designed off-axis tricks (cork, b-twist, raiz, butterfly). In 2025 the rule
     was a flat −0.5 for "slanted axis".
  5. **New in 2026:** "Of an athlete's three (3) highest D-scores, no more than two (2) may come from the same
     category."
- **Scaling up** (judges' discretion, no numeric increments published):
  - *Placement:* travel distance or height difference, narrow take-off area, landing on a narrow or elevated
    surface (rail, bar).
  - *Form:* body shape (layout, pike, pistol, spider, stall), twist timing (full-up, full-down), reversed twist
    (unfull), touchdowns and kicks.
  - *Entry:* a challenging move directly before. Round-off, scoot, cartwheel and kip do not count.
  - *Exit:* a challenging move directly after.
  - "More than one situation can be applied to one element."
- **2025 to 2026 value changes (examples):**
  - Swing moves mostly lowered: Giant 1.7 → 1.5, Swing Frontflip 1.8 → 1.5, Swing Sideflip 1.9 → 1.6, Swing
    Gainer 1080 5.0 → 4.8, Swing Triple Gainer 7.5 → 7.3.
  - Wall moves mixed: Wall Gainer 3.0 → 2.7, Wall Gainer 360 4.3 → 3.8, Gaet Pimp Backflip 720 4.5 → 5.0,
    Castaway Double Backflip 7.3 → 7.5, Wall Gainer 1080 7.5 → 7.3.
  - Added: Gainer Double Cork 4.1.
  - Added junior rules: no double rotations, at most double twists, obstacle height limits.
- **Update cadence:** "The Reference List / Table of Tricks will be published and updated three months before
  the start of each competition season." A 2027 table would be due about Oct 2026. **Not found online as of
  2026-09-28.**
- **E-score (2026):** Safety 6, Landing Quality 3 (0-3 reward), Flow 5, Flow Quality 1, total 15. The PDF text
  layer also contains 9 and 6 for Safety and Flow, which look like struck-through older values. I inferred the
  current split from the 15-point total. Full stop: −1.00 for a brief 1 s stop, up to −5.00.
- **Tie-break (freestyle):** higher E, then the mean of all E-judges, then the mean of all D-judges.
- **2027 TR changes:** new Speed 100 m (FIG world records) and Team Speed Mixed. **Freestyle unchanged** (still
  20-45 s). The TR cites CoP §3.1.2 and §3.1.3 for these, which implies a revised CoP text. **Not found online.**
- **Video:** one video recording setup for parkour, "immediately available in case an exercise needs to be
  reviewed" (TR 2026/2027 Art. 4.10.4.1).
- **Not found:** how the three D-judges' scores are combined (mean, median, or panel consensus), and numeric
  scaling increments.

### 3.2 What a FIG PK judge-assist must output to be useful (derived from the rules above)

Required, in priority order:

1. **Run clock:** t0 = first step, jump or swing, with flags at 20 s, 40 s and 45 s. Every trick is tagged as
   before or after 45 s. Tricks straddling 45 s are flagged, because the rule does not say whether take-off or
   landing time governs.
2. **Trick timeline:** start/take-off/landing timestamps for every candidate trick. **Recall on high-value
   aerial and rotational tricks matters most**, because D uses only the top three. Full coverage is still needed
   to apply the repetition rule and the category cap. PK Basics can be low priority for senior runs (P10), but
   not for juniors or lower levels.
3. **Trick identity as a Table-of-Tricks entry:** name, base value, category, plus top-k alternatives with
   confidence, and compositional attributes (flips, twists, direction, axis/cork, body shape, and contact type:
   ground, wall, palm, bar, kong). Attributes are what let the assist survive the yearly table updates.
4. **Scaling cues**, each with evidence:
   - distance or height difference
   - narrow take-off
   - landing on a rail, bar or elevated surface
   - body shape
   - twist timing and direction
   - touchdowns and kicks
   - challenging entry or exit connection
   - **unintentional tilt** (−0.3 to −0.5)
   Several of these depend on scene geometry (obstacles), not only on pose.
5. **Validity flags:** failed trick (lying landing or crash, so excluded from D), and **repeated trick**
   (identity match regardless of form, entry, placement or exit).
6. **Proposed D:** the three highest scaled values under the ≤2-per-category constraint, the runner-up
   combination, and sensitivity (which uncertain call changes D the most).
7. **Evidence package:** per-trick clip with slow motion (IRCOS needs 50 fps; iPhone can record 120/240 fps)
   and a pose overlay, so a judge can confirm in seconds. This matches the only deployment slot FIG currently
   accepts: the Supervisor / Superior Jury review of a proposal, as JSS does in 2026.
8. **Uncertainty routing:** low-confidence tricks and near-ties in the top three are pushed to the human.
   Anything else defeats the "human confirms" contract.

Optional (E-panel aids): full-stop durations (−1 per second, up to −5), hand, knee or seated landings,
stutter steps.

---

## 4. Full-run temporal action segmentation / localisation (2024-2026)

### 4.1 Evidence table

| # | Work | Task / sport | Footage | Result | Labels needed | Grade |
|---|---|---|---|---|---|---|
| T1 | **VIFSS** (Tanaka, Suzuki, Fujii; arXiv 2508.10281, 2025-08-14) | Element-level TAS of figure-skating jumps: type + rotation level (single to quad), with entry and landing phases | **371 broadcast** short programs (Olympics, Worlds), single moving camera, 25 fps | Element-level: **Acc 85.8, F1@50 92.6, F1@75 90.7, F1@90 49.6**. 2D-pose baseline: F1@50 78.8, F1@90 35.4. 3D-lifted pose is *worse* than 2D at element level, because MotionAGFormer fails on triples and quads it never saw. With 1% of labels plus contrastive pretraining on mocap-derived virtual views: >60% element-level F1@50 | Supervised segments, plus a 3D mocap set (FS-Jump3D, 12 cameras) for pretraining | B |
| T2 | FS-Jump3D (Tanaka et al., arXiv 2408.16638, 2024-08-29) | Same group. 3D pose features help TAS | Mocap + broadcast | See T1 | Supervised | B |
| T3 | **T-DEED** (Xarles et al., arXiv 2404.05392, 2024-04-08) | Precise event spotting of take-off and landing (4 classes) | FigureSkating broadcast (371 performances, events 0.23% of frames); FineDiving (3,000 clips) | FS-Comp **85.2 mAP@1 frame / 91.7@2**; FS-Perf 86.8 / 96.1; FineDiving 71.5 / 87.6 | Supervised, end-to-end RGB | B |
| T4 | E2E-Spot (Hong et al., arXiv 2207.10213, 2022-07-20) | Defines precise spotting. Annotations for FineGym, FineDiving, FigureSkating, Tennis | Broadcast | Baseline for T3 | Supervised | B |
| T5 | **UTAL-GNN** (Badatya et al., arXiv 2508.19647, 2025-08-27) | Unsupervised skeleton boundary detection: denoising-pretrained ST-GCN plus curvature of an "action dynamics metric" | DSV Diving, plus in-the-wild diving | **82.66 mAP**, 29 ms latency, "matching SOTA supervised". Generalises to unseen footage without retraining (authors' claim) | **None** | B |
| T6 | **NS-AQA** (Okamoto & Parmar, arXiv 2403.13798, 2024-03-20) | Neuro-symbolic diving: segmentation into phases, rule-based element symbols, per-element report with visual evidence. Experts preferred it to neural AQA | Broadcast diving | "SOTA action recognition and temporal segmentation" (see file 01 for twist and somersault numbers) | Rules + small detectors | B |
| T7 | Calisthenics TAS (arXiv 2507.12245, 2025-07-16) | Per-frame MLP on OpenPose, then Viterbi / heuristic smoothing | **In-the-wild social and ad-hoc video**, 24 fps, one skill per clip | mASF1 **0.674** (Viterbi), 0.631 (heuristic), 0.114 raw. Worst on inverted skill (one-arm handstand F1 0.64) | Supervised | B |
| T8 | Weakly-supervised fine-grained TAD (Li et al., arXiv 2207.11805, 2022) | FineGym / FineAction detection from video-level labels via atomic-action clustering | Broadcast gymnastics | SOTA weakly-supervised at the time (numbers not extracted) | Weak | B |
| T9 | FineGym (Shao et al., arXiv 2004.06704, 2020) | Event → set → element hierarchy over full gymnastics routines | Broadcast | Reference hierarchy | Supervised | B |
| T10 | Trampoline pose (Drolet-Roy et al., arXiv 2604.01322, 2026-04-01) | Fine-tuning ViTPose on synthetic extreme poses from trampoline mocap | Multi-view | 3D MPJPE **−46.1 mm (−42.7%)** vs pretrained. Off-the-shelf pose under-performs on inverted and extreme poses | Synthetic | B |
| T11 | Fujitsu tracking (arXiv 2511.16532, 2025-11-20) | Multi-camera gymnast tracking inside the deployed JSS | Fixed arena cameras | Needs domain priors, because often only 2 opposing views detect the gymnast | n/a | B |
| T12 | VLMs for AQA (Monte e Freitas et al., arXiv 2604.08294, 2026-04-09) | Gemini 3.1 Pro, Qwen3-VL and InternVL3.5 on fitness, figure skating and diving AQA | Video | "Only marginally above random chance". Biased toward predicting correct execution | Zero/few-shot | B |
| T13 | VLM diving judge (arXiv 2609.19354, 2026-09-16) | Open VLMs zero-shot on AQA-7 diving | Broadcast | Standalone Spearman < 0.32. Ensemble regression over VLM text: 0.67 | Zero-shot + regression | B |
| T14 | Figure-skating full programs: MCFS (Liu et al., AAAI 2021), YourSkatingCoach (arXiv 2410.20427, 2024: 454 jump videos, air-time detection), FSBench (arXiv 2504.19514, 2025: existing models show "significant limitations") | Various | Broadcast | See papers | Supervised | B |

### 4.2 Feasibility for a 20-45 s parkour run filmed on 1-2 iPhones

**Evidence that helps (B):**
- Broadcast figure skating is the closest analog: one athlete, a single camera panning to follow them, short
  airborne rotational elements (a skating jump averages 16 frames ≈ 0.65 s at 25 fps), and labels combining type
  and rotation count. There, pose-based TAS reaches 92.6% F1@50 element-level (T1). Precise take-off and landing
  spotting reaches 85 mAP at ±1 frame (T3).
- Airborne phases give a strong, label-free segmentation cue: unsupervised boundary detection reaches 82.7 mAP on
  diving (T5). TrickCam's take-off detection shows it runs on-device (C).
- A judge-assist does not need frame-exact boundaries. It needs to find each trick and hand a clip to a human.
  F1@50-level overlap is the relevant operating point, not F1@90.

**Evidence that hurts (B):**
- **Occlusion and scale.** A freestyle run crosses a ≥40 × 10 m field with walls and bars. Even Fujitsu's fixed
  multi-camera rigs lose detections in most views (T11). One static wide phone makes the athlete small, and one
  panning phone adds motion blur. The FIG requirement of a single video setup suggests event footage will be a
  single operator-panned view.
- **Pose on inverted and extreme poses** is the weak link: off-the-shelf pose fails on trampoline poses (T10), and
  inverted skills score worst in calisthenics TAS (T7). 3D lifting got *worse* than 2D on unseen high-rotation
  jumps (T1).
- **In-the-wild phone footage** is much harder than broadcast: calisthenics TAS gets mASF1 0.674 with only one
  skill per clip (T7).
- **Labels.** Every strong TAS result above is supervised on segment-annotated full routines (371 programs in T1).
  PkVision's no-manual-labels constraint rules that out. The label-free options are unsupervised boundaries (T5),
  rule-based airborne detection, or synthetic full runs stitched from the trimmed parkourtheory clips. None of
  these is validated on acrobatic multi-element footage.
- **Context.** The Swing / Wall / Acro category and most scaling cues depend on obstacle contact and geometry
  (bar, wall, rail, height difference). Skating and diving never need that.

**Verdict.**
- **Segmenting aerial and rotational tricks in a 20-45 s run from 1-2 iPhones is plausible, but unproven.** By
  analogy, expect somewhere between the calisthenics floor (~0.67 mASF1) and the skating ceiling (~0.93 F1@50).
  It will be lower than skating broadcast because of occlusion, no labelled full runs, and phone framing. No
  A-grade number exists.
- Segmentation is **not the binding constraint** for a judge-assist. Trick identity (see file 01 for twist
  counting) and scaling context are. Segmentation errors cost little when a human confirms every trick.
- **Cheapest decisive check** (no money, no manual labels): run an airborne-phase detector (COM or hip
  trajectory, feet-off-ground) on a few public full-run videos, and count missed or extra tricks against a video
  of the same run. This is one human viewing pass per run, as the "~99 gold clips" rule allows.

---

## 5. Competitive landscape, novelty and value

### 5.1 Closest systems and how they differ

| System | Domain | Capture | Output | Validation | Status | Difference from PkVision |
|---|---|---|---|---|---|---|
| Fujitsu JSS | Artistic gymnastics, 10 apparatus | 4-8 fixed HD cameras per apparatus, calibrated, multi-view 3D | Element ID for ~2,000 closed-list elements, D and E support | ~90% vs judges (C) | FIG-official. Supervisors review before D (2026) | Closed element list, fixed venue rigs, costly, 8,000 labelled routines. No parkour, no obstacle context |
| Swiss Timing / Omega | Figure skating, diving | 14 × 8K (skating), multi-cam (diving) | Kinematics (height, air time, rotation), not element D | None published | Broadcast and judge-review aid | Metrics, not trick identity or D. Olympic-venue hardware |
| Owl AI | Snowboard, freestyle, motocross, pickleball | Video; claims iPhone-grade suffices | Trick difficulty, landing quality, a full score | None published (X Games "overscored" but got the podium right) | Unofficial, VC-funded ($11M) | Closest in spirit (action sports, camera-only, trick-based). Unvalidated, holistic scoring, no FIG PK alignment, no parkour |
| ISU AI tool | Figure skating | Omega cameras | Rotation and edge support | None | Data support, not scoring | Analog precedent for "human confirms" |
| Consumer apps (Stickd, CheckForm) | Gymnastics (USAG/FIG) | Single phone | Skill ID + D/E estimate | None. User-reported misidentification | Retail | Unvalidated, VLM-based (CheckForm uses Gemini). T12 suggests VLM AQA is near chance |
| TrickCam, Better Form Parkour AI | Flips, parkour basics | Single phone | Flip detection or a 0-100 technique score | None | Retail / not launched | No trick identity, no CoP |
| JudgeMate, WDSF Trivium | Many | Human input | Digital score entry | n/a | Deployed | Scoring software, not recognition. JudgeMate states it "is not an AI judging system" |

### 5.2 Is anyone close to a parkour judge-assist?

**No one found.** No research group, federation pilot, or company has shown automatic parkour trick recognition
or FIG PK D-score proposal. The most likely entrant is Owl AI: funded, action-sports DNA, iPhone-grade claim,
stated interest in gymnastics. Nothing public connects it to parkour.

### 5.3 Novelty and value of a camera-only, iPhone-based PK judge-assist

- **Novel:**
  1. The first parkour trick recognizer and judge-assist.
  2. Open-vocabulary compositional output (flips, twists, direction, axis, context) mapped onto a yearly-changing
     Table of Tricks, instead of a closed element list (JSS ~2,000 elements).
  3. 1-2 consumer phones instead of fixed multi-camera rigs.
  4. Explicit handling of obstacle-dependent scaling and categories, which no analog system handles.
- **Not novel:** "camera-only" by itself (Fujitsu since 2023, Owl AI, Swiss Timing), and the "human confirms the
  AI proposal" workflow (JSS 2026 Supervisor pre-review; ISU data support).
- **Value:**
  - **Fit with FIG reality.** Parkour events need only one video setup (A\*), so a phone-based proposer does not
    fight existing infrastructure.
  - **Fit with FIG policy.** IRCOS and JSS set the accepted pattern: an AI proposal reviewed by
    Supervisors/Superior Jury, used for inquiries and education (A\*/C).
  - **Consistency.** The tie-break uses judge means and D is top-3 with discretionary scaling, so judge
    consistency matters. Nobody has measured it for PK freestyle (not found). An assist doubles as a consistency
    instrument, like FIG and Longines' judge-evaluation work in artistic gymnastics (Mercier & Heiniger, arXiv
    1807.10021, 2018).
  - **Training and self-assessment**: athletes, coaches and judge education. JSS's first commercial path was
    also training.
- **Risks:**
  - The FIG adoption clock is years long: JSS took ~2 years from collaboration to limited official use on 4
    apparatus, and its 2018 promise of automated scores by 2020 is still unmet in 2026.
  - The rules forbid replacing judges with video.
  - Validation will have to be judge-level and item-level (per-trick agreement), not a podium-match anecdote.
  - Rule volatility: the table is updated yearly and 2026 changed values and added constraints.

---

## 6. Gaps (not found or unresolved)

1. Any paper, dataset or product doing automatic parkour, freerunning or tricking trick recognition or judging.
2. Any FIG statement, pilot or roadmap for AI or JSS in parkour, or in any discipline other than artistic
   gymnastics.
3. Fujitsu JSS per-apparatus or per-element accuracy, independent validation, current camera count, and cost.
   Only the ~90% / ~2,000 elements figure (journalism) was found.
4. Owl AI method, validation data, and whether its re-judging is trick-level or holistic.
5. How the 3 PK D-judges' scores are combined, and numeric scaling increments.
6. Whether a trick straddling 45 s counts (take-off vs landing time).
7. The Table of Tricks 2027 (due about Oct 2026) and the revised PK CoP text that TR 2027 references (Speed 100 m,
   Team Speed Mixed).
8. Tricks per FIG freestyle run (distribution), and the size of the publicly available full-run video corpus
   (FIG World Cups, Worlds, World Games).
9. Human inter-rater reliability of FIG PK freestyle D and E judging (only a beginner skills test exists, P4).
10. Any TAS or TAL result on handheld or phone footage of an acrobatic sport with **multiple** elements per
    video. Calisthenics has one skill per clip, and all strong results are on broadcast.
11. A validated label-free TAS method reaching >85% F1@50 on acrobatic element segmentation (UTAL-GNN is
    boundary-only, on diving).
12. Diving48 current SOTA top-1 (the benchmark page returned HTTP 500; not verified).
13. Full text of FIG news items (JS-rendered pages; titles only): "A hard skill: Difficulty Judging in Parkour",
    "Unravelling the secrets of the Parkour Code", "FIG President showcases use of AI in gymnastics at AIPS
    Congress", "Parkour: What's new for 2025-2028?".

---

## 7. Sources

FIG primary documents (A\*)
- FIG, *Parkour Code of Points 2025-2028*, approved Feb 2024, published 2024-04-22.
  https://www.gymnastics.sport/publicdir/rules/files/en_1.1%20-%20PK%20Code%20of%20Points%202025-2028.pdf
- FIG PK Technical Committee, *Table of Tricks 2025*, April 2025.
  https://www.gymnastics.sport/publicdir/rules/files/en_1.1.1%20-%20PK%20Code%20of%20Points%202025-2028%20-%20Table%20of%20tricks%202025.pdf
- FIG, *Table of Tricks 2026* (PK CoP 2025-2028), date not printed; retrieved 2026-09-28.
  https://www.gymnastics.sport/publicdir/rules/files/en_1.1.1%20-%20PK%20Code%20of%20Points%202025-2028%20-%20Table%20of%20tricks%202026.pdf
- FIG, *Technical Regulations 2026* (v4.0 May 2026; Section 7 Parkour v1.0 May 2025).
  https://www.gymnastics.sport/publicdir/rules/files/en_1.1%20-%20Technical%20Regulations%202026.pdf
- FIG, *Technical Regulations 2027* (v1.1 June 2026; Section 7 Parkour v1.0 May 2026).
  https://www.gymnastics.sport/publicdir/rules/files/en_1.1%20-%20Technical%20Regulations%202027.pdf
- FIG, *Technical Regulations 2025*.
  https://www.gymnastics.sport/publicdir/rules/files/en_1.1%20-%20Technical%20Regulations%202025.pdf
- FIG, *Appendix to the Codes of Points 2025-2028* (IRCOS rules; PK IRM), Lausanne, July 2024.
  https://www.gymnastics.sport/publicdir/rules/files/en_1.2%20-%20Appendix%20to%20the%20CoP%202025-2028.pdf
- FIG, *General Judges' Rules 2025-2028*.
  https://www.gymnastics.sport/publicdir/rules/files/en_1.2%20-%20General%20Judges'%20Rules%202025-2028.pdf
- FIG MTC, *MAG Newsletter N°4*, July 2026. http://www.fig-docs.com/website/newsletters/MAG/2026/MAG_NL_4_en.pdf (C)

Fujitsu JSS
- Fujitsu PR, "The International Gymnastics Federation and Fujitsu to Collaborate on Building a Judging Support
  System", Oct 2017.
  https://info.archives.global.fujitsu/global/about/resources/news/press-releases/2017/1007-01.html (seen in search
  results; not opened)
- Fujitsu PR, "The International Gymnastics Federation to Implement Fujitsu's Judging Support System",
  2018-11-20. https://info.archives.global.fujitsu/global/about/resources/news/press-releases/2018/1120-01.html
- Fujitsu PR, "'A step towards the future' with the first official use of Fujitsu technology to support judging at
  the 2019 Artistic Gymnastics World Championships", 2019-10-02.
  https://info.archives.global.fujitsu/global/about/resources/news/press-releases/2019/1002-01.html
- Fujitsu PR, "Fujitsu and the International Gymnastics Federation launch AI-powered Fujitsu Judging Support
  System for use in competition for all 10 apparatuses", 2023-10-05.
  https://info.archives.global.fujitsu/global/about/resources/news/press-releases/2023/1005-02.html
- MIT Technology Review, "How AI is changing gymnastics judging", 2024-01-16.
  https://www.technologyreview.com/2024/01/16/1086498/ai-gymnastics-judging-jss-world-championships-antwerp-paris-olympics/
- R&D World, "Fujitsu JSS applies AI to gymnastics judging", 2024-08-05.
  https://www.rdworldonline.com/ai-assisted-gymnastics-judging-system/
- Asia News Network, "New AI scoring system ... eyed for 2024 Paris Olympics", date not verified (snippet only).
  https://asianews.network/new-ai-scoring-system-available-for-all-10-gymnastic-apparatuses-eyed-for-2024-paris-olympics/
- Gymnastics Now, "Recap: Podium training at the 2025 World Gymnastics Championships in Jakarta", Oct 2025.
  https://gymnastics-now.com/worlds-2025-podium-training-live-updates/
- Yang et al. (Fujitsu Research), "Enhancing Multi-Camera Gymnast Tracking Through Domain Knowledge Integration",
  arXiv 2511.16532, 2025-11-20. https://arxiv.org/abs/2511.16532
- Forbes (C. Price), "Beyond The Naked Eye: Experts Discuss The Future Of AI Gymnastics Judging", 2025-11-11
  (403; snippet only).
  https://www.forbes.com/sites/carolineprice/2025/11/11/beyond-the-naked-eye-experts-discuss-the-future-of-ai-gymnastics-judging/
- FIG News, "FIG President showcases use of Artificial Intelligence in gymnastics at AIPS Congress" (meta
  description only; date not retrievable).
  https://www.gymnastics.sport/site/news/displaynews.php?urlNews=4155540

Other deployed or claimed systems
- The Conversation (W. Standaert), "AI is coming to Olympic judging: what makes it a game changer?", 2026-02-03.
  https://theconversation.com/ai-is-coming-to-olympic-judging-what-makes-it-a-game-changer-274313
- CGTN, "International Skating Union weighs AI's role in judging", 2026-02-11.
  https://news.cgtn.com/news/2026-02-11/International-Skating-Union-weighs-AI-s-role-in-judging--1KFKl1sSgLK/p.html
- IEEE Spectrum, "How Will AI Transform Figure Skating Judging at the Olympics?", 2026-02-04.
  https://spectrum.ieee.org/winter-olympics-2026-tech
- CPR News, "AI experiment in halfpipe judging at X Games will give snowboarders a glimpse into the future",
  2025-01-22. https://www.cpr.org/2025/01/22/ai-experiment-snowboard-halfpipe-judging-x-games/
- Deadline, "Emerging Sports Firm The Owl AI Raises $11M Seed Round, Taps Google Vet Josh Gwyther As CEO",
  June 2025 (snippet only).
  https://deadline.com/2025/06/emerging-sports-firm-the-owl-ai-raises-11m-seed-round-google-josh-gwythe-ceo-1236442279/
- Colorado Sun, "The X Games debuted AI judging in Aspen. Now they are building AI referees for all sports.",
  2025-06-26 (snippet only). https://coloradosun.com/2025/06/26/x-games-ai-judging/
- Snowboarder, "The X Games' Owl AI Just Re-Judged The Women's Olympic Slopestyle Contest", Feb 2026 (snippet
  only). https://www.snowboarder.com/news/owl-ai-womens-olympic-slopestyle
- Denver7, "Former Olympic skier, CU Buffs star using AI in sports to prevent missed calls", 2026 (date not
  shown). https://www.denver7.com/decodedc/technology/former-olympic-skier-cu-buffs-star-using-ai-in-sports-to-prevent-missed-calls
- Swatch Group, "Omega: New Technology for Paris 2024" (snippet only).
  https://www.swatchgroup.com/en/swatch-group/innovation-powerhouse/internet-things/omega-new-technology-paris-2024
- China Daily, "Chinese AI making all the smart moves at Paris Olympics", 2024-08-07 (snippet only).
  https://www.chinadaily.com.cn/a/202408/07/WS66b2afe5a3104e74fddb8c2e.html
- IOC, *Olympic AI Agenda*, April 2024.
  https://stillmed.olympics.com/media/Documents/International-Olympic-Committee/AI/Olympic-AI-Agenda.pdf
- WDSF, "Breaking Down the Rules: Storm and Renegade on the Trivium Judging System" (undated).
  https://www.worlddancesport.org/News/Breaking_Down_the_Rules_Storm_and_Renegade_on_the_Trivium_Judging_System-3183
- UCI, "BMX Freestyle Park: UCI World Cycling Centre hosts international judges and athletes", Aug 2025 (snippet
  only).
  https://www.uci.org/article/bmx-freestyle-park-uci-world-cycling-centre-hosts-international-judges-and/7rWBcPQhdRDG12r8BbLGUB
- JudgeMate, "AI in Sports Judging", 2026-04-07. https://www.judgemate.com/en/guides/ai-sports-judging
- Stickd, product page, retrieved 2026-09-28. https://stickd.app/
- CheckForm Gymnastics (App Store), retrieved 2026-09-28. https://apps.apple.com/us/app/gymnastics-ai-coach-judge/id6757922269
- TrickCam (Google Play; snippet only). https://play.google.com/store/apps/details?id=com.trickcam.app&hl=en_US
- Better Form, "Parkour AI", updated 2026-04-30. https://www.buildbetterform.com/parkour/
- Parkour UK, "Parkour Earth's Unified Guidelines for Parkour Competitions" (2026).
  https://parkour.uk/news/parkour-earth-competitions-guidelines
- Red Bull, "The Red Bull Art of Motion Judging Criteria".
  https://www.redbull.com/int-en/videos/the-red-bull-art-of-motion-judging-criteria
- Wikipedia, "Parkour at the 2025 World Games – Men's freestyle", event 2025-08-12.
  https://en.wikipedia.org/wiki/Parkour_at_the_2025_World_Games_%E2%80%93_Men%27s_freestyle
- Apple, `VNDetectHumanBodyPose3DRequest` documentation (iOS 17+, single-person 3D pose on-device).
  https://developer.apple.com/documentation/vision/vndetecthumanbodypose3drequest

Research
- Tanaka, Suzuki, Fujii, "VIFSS: View-Invariant and Figure Skating-Specific Pose Representation Learning for
  Temporal Action Segmentation", arXiv 2508.10281, 2025-08-14. https://arxiv.org/abs/2508.10281
- Tanaka, Suzuki, Fujii, "3D Pose-Based Temporal Action Segmentation for Figure Skating", arXiv 2408.16638,
  2024-08-29. https://arxiv.org/abs/2408.16638
- Xarles et al., "T-DEED: Temporal-Discriminability Enhancer Encoder-Decoder for Precise Event Spotting in Sports
  Videos", arXiv 2404.05392, 2024-04-08. https://arxiv.org/abs/2404.05392
- Hong et al., "Spotting Temporally Precise, Fine-Grained Events in Video", arXiv 2207.10213, 2022-07-20.
  https://arxiv.org/abs/2207.10213
- Badatya, Baghel, Hegde, "UTAL-GNN: Unsupervised Temporal Action Localization using Graph Neural Networks",
  arXiv 2508.19647, 2025-08-27. https://arxiv.org/abs/2508.19647
- Okamoto & Parmar, "Hierarchical NeuroSymbolic Approach for Comprehensive and Explainable Action Quality
  Assessment", arXiv 2403.13798, 2024-03-20. https://arxiv.org/abs/2403.13798
- Finocchiaro, Farinella, Furnari, "Calisthenics Skills Temporal Video Segmentation", arXiv 2507.12245,
  2025-07-16. https://arxiv.org/abs/2507.12245
- Connolly, Silvestre, Bleakley, "Automated Identification of Trampoline Skills Using Computer Vision Extracted
  Pose Estimation", arXiv 1709.03399, 2017-09-11. https://arxiv.org/abs/1709.03399
- Drolet-Roy et al., "Human Pose Estimation in Trampoline Gymnastics: How to Improve Performance on Extreme Poses",
  arXiv 2604.01322, 2026-04-01. https://arxiv.org/abs/2604.01322
- Li, He, Xu, "Weakly-Supervised Temporal Action Detection for Fine-Grained Videos with Hierarchical Atomic
  Actions", arXiv 2207.11805, 2022-07-24. https://arxiv.org/abs/2207.11805
- Shao et al., "FineGym: A Hierarchical Video Dataset for Fine-grained Action Understanding", arXiv 2004.06704,
  2020-04-14. https://arxiv.org/abs/2004.06704
- Li, Li, Vasconcelos, "RESOUND: Towards Action Recognition without Representation Bias" (Diving48), ECCV 2018.
  https://openaccess.thecvf.com/content_ECCV_2018/papers/Yingwei_Li_RESOUND_Towards_Action_ECCV_2018_paper.pdf
- Chen et al., "YourSkatingCoach: A Figure Skating Video Benchmark for Fine-Grained Element Analysis",
  arXiv 2410.20427, 2024-10-27. https://arxiv.org/abs/2410.20427
- Gao et al., "FSBench: A Figure Skating Benchmark for Advancing Artistic Sports Understanding", arXiv 2504.19514,
  2025-04-28. https://arxiv.org/abs/2504.19514
- Monte e Freitas et al., "Can Vision Language Models Judge Action Quality? An Empirical Evaluation",
  arXiv 2604.08294, 2026-04-09. https://arxiv.org/abs/2604.08294
- Velesaca et al., "Can Vision-Language Models Judge Olympic Diving?", arXiv 2609.19354, 2026-09-16.
  https://arxiv.org/abs/2609.19354
- Dhar, Ramakrishnan, Munson, "Breakdance Video classification in the age of Generative AI", arXiv 2510.20287,
  2025-10-23. https://arxiv.org/abs/2510.20287
- Braeunig, "Breaking Down the Scoring: Interrater Reliability and National Bias in Olympic Breaking",
  arXiv 2511.00553, 2025-11-01. https://arxiv.org/abs/2511.00553
- Sato, "Reliability of judging in Olympic breaking at the 2024 Paris games", *Frontiers in Psychology*,
  2025-12-03. https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2025.1593158/full
- Chen, "SkateboardAI: The Coolest Video Action Recognition for Skateboarding", arXiv 2311.11467 (v1 2023-08-02).
  https://arxiv.org/abs/2311.11467
- Dvořák, Baláš, Martin, "The Reliability of Parkour Skills Assessment", *Sports* 6(1):6, 2018-01-24.
  https://pmc.ncbi.nlm.nih.gov/articles/PMC5969187/
- LAAS Parkour dataset (Z. Li et al.). https://github.com/zongmianli/Parkour-dataset ; related: "Estimating 3D
  Motion and Forces of Human-Object Interactions from Internet Videos", arXiv 2111.01591, 2021-11-02.
  https://arxiv.org/abs/2111.01591
- Mercier & Heiniger, "Judging the Judges: Evaluating the Performance of International Gymnastics Judges",
  arXiv 1807.10021, 2018-07-26. https://arxiv.org/abs/1807.10021
