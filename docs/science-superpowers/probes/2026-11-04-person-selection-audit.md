# Person selection audit and selector test (2026-11-04)

Exploratory, with no decision weight. It runs on clips already seen, so it cannot reverse the pre-registered kill in `2026-11-04-rule-counter-results.md`. All visual codes come from one AI agent, not blinded and not owner-checked. Codes: `2026-11-04-person-selection-audit.json`. Scores: `2026-11-04-person-selection-results.json`.

**Corrected after code review.** The first version of arm S used `body(frame)`, and rtmlib poses a full-frame box when nobody is detected, which invents a person. All arm S numbers below come from a clean re-extraction with the detector run first. That removed the apparent gain to 4/12. The selector itself was not changed.

## Answer

1. **Wrong-person skeletons are common in the corpus.** A rough estimate is 12–37% of clips with at least one wrong sampled frame, from a small sample.
2. **Fixing person selection would not rescue the counter.** The selector did not raise Decision-12 at all. With the skeleton on the athlete, the counter was exact on 3 of 8 clips and missed both double-fulls.
3. **The earlier "mostly upstream" reading was too generous**, and the always-zero baseline below makes the counter's aggregate score look worse.

## Baselines missing from the first report

| Predictor | Decision-12 twist | All-47 twist | All-47 flip |
|---|---|---|---|
| Always 0 twists / always 1 flip | 3/12 | **36/47** | 28/47 |
| Counter, arm A | 3/12 | 22/47 | 30/47 |
| Counter, arm S (selector) | 3/12 | 20/47 | 27/47 |

36 of the 47 reference clips have zero twist, so all-47 twist accuracy says little. On those 36, the counter is exact on only 19 (A) and 17 (S). On the 11 clips with twists it is exact on 3 on both arms. Twist 2+ is 0/4 on both. Coverage falls from 42 (A) to 34 (S).

## Wrong-person rate in the corpus

`scripts/extract_pose.py` output, all 1,618 clips, computed by `scripts/audit_person_continuity.py`. The 32 coded clips were sampled from the first 1,100 alphabetically, the shards done at that point, and the overlays were not committed. A frame counts with at least 6 joints above confidence 0.3. A jump is a centre move above 1 median body diagonal across at most 2 frames.

| Jumps | Share of clips | Coded wrong | Unclear |
|---|---|---|---|
| ≥ 2 | 12.8% | 9/16 | 1/16 |
| 0 | 78.2% | 1/16 | 3/16 |
| 1 | 9.0% | not sampled | |

Bounds: low 12% counts only wrong codes and treats the one-jump group as clean. High 37% counts unclear as wrong and the whole one-jump group as wrong. These bounds ignore sampling error, and with 16 clips per group that error is large. Six sampled frames per clip also miss brief errors, and a wrong frame does not always fall during the trick. The effect on CueModel's macro-F1 is **not measured**.

**The corpus has the invented-person bug too.** `scripts/extract_pose.py`, `scripts/extract_pose_modal.py` and `scripts/pose_audit.py` still call `body(frame)`, so frames with no detection hold a skeleton fitted to the whole frame. On the 47 reference clips, 316 of 2,775 frames (11.4%) have no detection. Using the jump definition above, 3 of 24 jumps on those clips touch such a frame, as do 17 of 69 on the 16 flagged audit clips. All 7 jumps in `1080_dive_roll` do, so that clip would not be flagged without invented frames, though it still counts in the 9/16. So some jumps above are invented, and the shards are contaminated by more than wrong people. The corpus-wide rate is not measured, and fixing it means re-extracting the shards.

Jump count flags wrong clips well (9/16 vs 1/16) and could triage the corpus, but a skeleton that stays on one bystander produces no jump.

**Also seen:** some clips contain camera cuts or replays (`540_dive_roll_bomb`, `one_elbow_palm_double_gainer`, `backward_roll_bomb`). Any counter that assumes one continuous shot is exposed to this. The rate is not measured.

## Selector test

`core/pose/athlete_select.py` tracks people, then keeps the track with the most rotation and travel. It was committed in `242be08` before any real-clip score and has not been tuned since.

On the 10 audit clips previously coded wrong, by overlay:
- **Fixed by the selector, 4:** `back_double_full_in_back_out`, `mukhametshin_gargoyle_tunnel`, `one_elbow_palm_double_gainer`, `castaway_stall_back`.
- **Improved by clean detection, 1:** `1080_dive_roll`. Its wrong frame is gone, but the track stops at about frame 50 while the athlete stays visible.
- **Still wrong, 5:** `one_hand_raiden`, `ew_stall_gainer`, `baby_knee_giant_gainer`, `one_step_wall_flare_back`, `dive_half_back_unwind_double_full_down` (no sampled frame is drawn; the track sits on a seated bystander).
- **New errors:** `palm_spin_gainer` and `lache_scissor_180_double_leg_in_double_leg_out`, both previously unclear, now sit on bystanders in sampled frames.

Likely causes, not verified: travel is measured in body diagonals, which rewards jitter on small distant people, and a fast athlete splits into several short tracks. **This selector is not ready for production use.**

## Counter with the skeleton on the athlete

Decision-12 under arm S, reading overlays for sampled frames only:

| Clip | Truth twist | Counted | |
|---|---|---|---|
| `flyaway_half_down` | 0.5 | 0.5 | exact |
| `lache_dark_arabian` | 0.5 | 0.5 | exact, flip wrong |
| `360_precision` | 1 | 1 | exact |
| `raiz_dismount_push_half_twist` | 0.5 | 1 | |
| `castaway_back` | 0 | 0.5 | |
| `stomach_webster` | 0 | 0.5 | |
| `side_double_full` | 2 | 1 | |
| `wall_double_full` | 2 | 1.5 | |

`swing_castaway_double_back`, `swing_castaway_double_full`, `kong_gainer_full` and `palm_spin` abstain. If all four were right, the score would be 7/12, still below the 10/12 "promising" line. In the earlier smoke test the owner's own `backflip`, with known truth, was also wrong. At frames 82–84 the skeleton jumps to a bystander, which breaks the flip count. With those frames removed, flip becomes 1 and direction backward on all arms, but twist stays wrong at 1 / 1.5 / 0.5 against a truth of 0.

Limit: n = 8, thresholds untuned, truth provisional, and no human checked these overlays. This supports not investing further in the counter as the primary reader. It does not prove that 2D twist counting is impossible.
