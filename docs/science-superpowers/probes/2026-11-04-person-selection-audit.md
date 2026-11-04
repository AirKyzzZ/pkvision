# Person selection audit and selector test (2026-11-04)

Exploratory, with no decision weight. It runs on clips already seen, so it cannot reverse the pre-registered kill in `2026-11-04-rule-counter-results.md`. All visual codes come from one AI agent, not blinded and not owner-checked. Codes: `2026-11-04-person-selection-audit.json`. Scores: `2026-11-04-person-selection-results.json`.

## Answer

1. **Wrong-person skeletons are common in the corpus.** A rough estimate is 12–37% of clips with at least one wrong sampled frame, from a small sample.
2. **Fixing person selection would not rescue the counter.** With the skeleton on the athlete, it was exact on 4 of 9 Decision-12 clips and missed both double-fulls.
3. **The earlier "mostly upstream" reading was too generous**, and the always-zero baseline below makes the counter's aggregate score look worse.

## Baselines missing from the first report

| Predictor | Decision-12 twist | All-47 twist | All-47 flip |
|---|---|---|---|
| Always 0 twists / always 1 flip | 3/12 | **36/47** | 28/47 |
| Counter, arm A | 3/12 | 22/47 | 30/47 |
| Counter, arm S (selector) | 4/12 | 22/47 | 27/47 |

36 of the 47 reference clips have zero twist, so all-47 twist accuracy says little. On those 36, the counter is exact on only 19 (A) and 18 (S). On the 11 clips with twists it is exact on 3 (A) and 4 (S). Twist 2+ is 0/4 on both.

## Wrong-person rate in the corpus

`scripts/extract_pose.py` output, all 1,618 clips. The 32 coded clips were sampled from the first 1,100 alphabetically, the shards done at that point. A jump is a centre move above 1 median body diagonal across at most 2 frames.

| Jumps | Share of clips | Coded wrong | Unclear |
|---|---|---|---|
| ≥ 2 | 12.8% | 9/16 | 1/16 |
| 0 | 78.2% | 1/16 | 3/16 |
| 1 | 9.0% | not sampled | |

Bounds: low 12% counts only wrong codes and treats the one-jump group as clean. High 37% counts unclear as wrong and the whole one-jump group as wrong. These bounds ignore sampling error, and with 16 clips per group that error is large. Six sampled frames per clip also miss brief errors, and a wrong frame does not always fall during the trick. The effect on CueModel's macro-F1 is **not measured**.

Jump count flags wrong clips well (9/16 vs 1/16) and could triage the corpus, but a skeleton that stays on one bystander produces no jump.

**Also seen:** some clips contain camera cuts or replays (`540_dive_roll_bomb`, `one_elbow_palm_double_gainer`, `backward_roll_bomb`). Any counter that assumes one continuous shot is exposed to this. The rate is not measured.

## Selector test

`core/pose/athlete_select.py` tracks people, then keeps the track with the most rotation and travel. It was committed in `242be08` before any real-clip score and has not been tuned since.

On the 10 audit clips previously coded wrong, by overlay:
- **Fixed, 4:** `back_double_full_in_back_out`, `mukhametshin_gargoyle_tunnel`, `one_elbow_palm_double_gainer`, `castaway_stall_back`.
- **Still wrong or nearly empty, 6:** `one_hand_raiden`, `ew_stall_gainer`, `baby_knee_giant_gainer`, `1080_dive_roll`, `one_step_wall_flare_back`, `dive_half_back_unwind_double_full_down` (nothing drawn in any sampled frame).
- **New error:** `palm_spin_gainer`, previously unclear, now sits on a spectator in most sampled frames.

Likely causes, not verified: travel is measured in body diagonals, which rewards jitter on small distant people, and a fast athlete splits into several short tracks. **This selector is not ready for production use.**

## Counter with the skeleton on the athlete

Decision-12 under arm S, reading overlays for sampled frames only:

| Clip | Truth twist | Counted | |
|---|---|---|---|
| `flyaway_half_down` | 0.5 | 0.5 | exact |
| `lache_dark_arabian` | 0.5 | 0.5 | exact, flip wrong |
| `360_precision` | 1 | 1 | exact |
| `palm_spin` | 1 | 1 | exact, flip wrong, an end frame on another person |
| `raiz_dismount_push_half_twist` | 0.5 | 1 | |
| `castaway_back` | 0 | 0.5 | |
| `stomach_webster` | 0 | 0.5 | |
| `side_double_full` | 2 | 1 | |
| `wall_double_full` | 2 | 0.5 | arm A gave 1.5 on the same clip |

`swing_castaway_double_back`, `swing_castaway_double_full` and `kong_gainer_full` abstain. If all three were right, the score would be 7/12, still below the 10/12 "promising" line. The owner's own `backflip`, clean and with known truth, was also wrong in the earlier smoke test.

Limit: n = 9, thresholds untuned, truth provisional, and no human checked these overlays. This supports not investing further in the counter as the primary reader. It does not prove that 2D twist counting is impossible.
