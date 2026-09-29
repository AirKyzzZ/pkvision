# Parkour Motion Code (PMC) v1: draft for owner review

**Status:** DRAFT v1, 2026-09-29. It replaces v0 after the owner's review ("decompose as much as possible"; tricking in scope) and the naming glossary (`docs/science-superpowers/survey/2026-09-29/08-naming-glossary.md`).

**Purpose:** a precise, machine-readable description of any parkour, freerunning or tricking move, sitting between video and names:

```
video ──(vision readers)──▶ PMC ──(naming grammar + lexicon)──▶ name(s) + aliases
                                ◀──(name parser)──────────────── name + description
```

## Principles

1. **Everything is physical.** There are no named building blocks. `kong`, `gainer`, `webster` and `dash` are **lexicon entries whose definition is a PMC**. A new trick is any PMC; its name is generated from the lexicon and the grammar.
2. **Airborne rotation follows the FIG Trampoline Code.** The quarter-somersault count plus half-twists per somersault is a published standard, extended with the fields parkour naming needs that trampoline doesn't encode: travel direction, axis tilt and twist direction.
3. **Directions are relative to the body at take-off.** Relative naming terms (inward, gainer, counter, unwind, hyper) are **derived** from physical fields by the naming grammar, never stored. Example: *inward* = somersault direction opposite to travel direction.
4. **Hierarchical and abstainable.** Every field has a level:
   - L0: segment structure;
   - L1: counts and directions;
   - L2: fine detail.

   A vision reader may write `?` at any level and still produce a coarse name ("vault(?) > back flip, full twist").
5. **Names never enter the PMC.** Numbers in names (kick "720", vault-internal turns) are resolved through the lexicon, because tricking kick numbers are not air rotation (glossary §2.8).

## Structure

A PMC is an ordered list of segments. JSON is canonical.

| Segment | Meaning |
|---|---|
| `SETUP` | approach and take-off preparation |
| `CONTACT` | any support phase: hands, feet or other body parts on ground, obstacle, wall or bar |
| `AIR` | a flight phase, with or without rotation |
| `TRANSITION` | tricking link between moves (combos) |
| `EXIT` | landing |

A contact made **during** a rotation (Back Dash pushing the vault mid-flip; the -In/-Out "opened early" markers) is written as a `CONTACT` segment between two `AIR` segments that carry `continues: true`. The rotation is then counted across them.

## `AIR` (flight)

| Field | Level | Values | Notes / source |
|---|---|---|---|
| `som_q` | L1 | int ≥ 0, **quarter** somersaults | FIG-TRA first digit. 4 = one full somersault |
| `som_dir` | L1 | front, back, side_l, side_r | somersault direction relative to facing at take-off. It stays constant through twists, so an Arabian is `back` with a half twist |
| `twist_h` | L1 | list of ints, **half**-twists per somersault (FIG-TRA subsequent digits) | `[1,3]` on `som_q=8` = half-in rudy-out. With no somersault: one entry, e.g. Dash 360 = `som_q=0, twist_h=[2]` |
| `travel` | L1 | forward, backward, left, right, up (on the spot) | centre-of-mass travel relative to facing at take-off. Back + forward = gainer family; front + backward = inward front; side + forward = tunnel; lateral = caster |
| `twist_dir` | L2 | per twist component: L, R (counter-clockwise / clockwise seen from above, at take-off) | counter and unwind are derived by comparing with the entry turn and the previous twist |
| `twist_when` | L2 | per somersault: up, early, mid, late, down, whole | -Up/-Down, full-in/full-out. Trampoline "in/out" maps to the position in the list |
| `axis` | L2 | on, off | off = cork, raiz, slant, corkscrew (designed off-axis, FIG ToT26 remark 4) |
| `shape` | L2 | tuck, pike, straight, straddle, open, arch | per somersault if it changes. FIG-TRA symbols o < / |
| `legs` | L2 | list: split (x-out), double_leg, kick_<type>, grab_<type>, switch, missleg | cosmetic or identity-bearing per the lexicon |
| `continues` | L0 | bool | the rotation continues after an intervening CONTACT |

## `CONTACT` (support phase)

| Field | Level | Values | Notes |
|---|---|---|---|
| `surface` | L1 | ground, obstacle_top, ledge_top, wall_face, bar, rail, beam | |
| `seq` | L2 | ordered list of `{limb, action}`. Limb: LH, RH, H?, BH (both hands), LF, RF, F?, BF, knee, elbow, shoulder, back, toes. Action: plant, push, step, grip, release, pivot, slide, catch | This is where vaults become physical. Lazy vault = `[near_H push, near_F step, far_H push]` (glossary 3.4) |
| `legs_path` | L2 | between_arms, side, feet_first, around, none | kong/monkey = between_arms; dash = feet_first; speed/safety = side |
| `feet_off_first` | L2 | bool | feet leave before the hands land: kong (true) vs monkey (false). Glossary 3.4, TP |
| `facing` | L2 | toward, sideways, away | body orientation to the surface during contact |
| `yaw` | L1 | degrees turned during contact (multiple of 90), direction L/R | turn vault 180, reverse vault 360, palm spin 180 |
| `swing` | L1 (bars) | `{dir: front, back; circle_deg; release: frontswing, backswing}` | giant = circle 360; flyaway = release on the forward swing into a back somersault |
| `count` | L1 | int | "Double Kong" = 2 consecutive push-offs |
| `steps` | L1 (walls) | int | one-step / two-step wall; tic tac |

## `SETUP`, `TRANSITION`, `EXIT`

| Segment | Field | Level | Values |
|---|---|---|---|
| SETUP | `approach` | L1 | stand, run, walk_back, from_hang, from_ledge, from_previous |
| SETUP | `takeoff` | L1 | one_foot, two_feet, sequential (cheat/vanish), hands, swing_release |
| SETUP | `turn` | L2 | degrees turned on the ground before leaving (cheat 180) |
| SETUP | `from` | L1 | ground, wall, bar, ledge_top, obstacle_top |
| SETUP | `position` | L2 | cat_hang, gargoyle, kneel, sitting, … (open list) |
| TRANSITION | `kind` | L1 | LK vocabulary: punch, vanish, skip, swing_through, missleg, reversal, pop, rapid, … |
| EXIT | `limbs` | L1 | one_foot, two_feet, hands, roll, back |
| EXIT | `surface` | L1 | ground, ledge_top (precision), bar (catch/regrab), lower_level (down, bomb) |
| EXIT | `stance` | L2 | complete, hyper, mega, semi, frontside, backside, turbo (relative to spin direction, LK) |
| EXIT | `precision` | L2 | bool: lands on a narrow target |
| EXIT | `same_bar` | L2 | bool: regrab (true) vs catch (false) |

## Compact display string

Segments are joined with ` > `. `AIR` is written trampoline-style:

`<som_dir><som_q>:<twist_h joined by .><shape> [travel] [~off]`

- `som_dir`: f / b / l / r
- `shape`: o tuck, < pike, / straight, x straddle, - open
- `[travel]`: a travel arrow, `→f` or `→b`
- `~off`: off-axis

Examples below were built from glossary definitions. The owner should check them.

| Name | PMC display |
|---|---|
| Back Flip | `SETUP(stand, 2F) > AIR(b4:0o →b) > EXIT(2F)` |
| Gainer Layout | `SETUP(run, 1F) > AIR(b4:0/ →f) > EXIT(2F)` |
| Inward Front | `SETUP(2F) > AIR(f4:0o →b) > EXIT(2F)` |
| Arabian | `SETUP(2F) > AIR(b4:1o, twist early) > EXIT(2F)` |
| Half-in Rudy-out (trampoline) | `AIR(f8:1.3)` |
| Kong Vault | `SETUP(run) > AIR(-0:0 →f) > CONTACT(obstacle_top, [BH push], between_arms, feet_off_first) > EXIT(2F)` |
| Dash 360 | `CONTACT(obstacle_top, [BH push], feet_first) > AIR(-0:2 →f) > EXIT(2F)` |
| Wall Full | `SETUP(run) > CONTACT(wall_face, [F? step], steps=1) > AIR(b4:2- →b) > EXIT(2F)` |
| Cork | `SETUP(run, 1F) > AIR(b4:2- →f ~off) > EXIT(2F)` |
| Flyaway (Swing Gainer) | `CONTACT(bar, [BH grip], swing{front, release frontswing}) > AIR(b4:0o →f) > EXIT(2F)` |

## Observability (which reader must produce each field)

| Fields | Reader | Survey evidence |
|---|---|---|
| `som_q`, `som_dir`, `twist_h`, `travel` | skeleton rule counters (root trajectory + torso/hip vectors) | B: diving somersault 97.3%, twist 93.3% (NS-AQA) |
| `twist_dir`, `twist_when`, `shape` | skeleton, with a VLM second opinion | partial. Twist direction needs L/R keypoints and face points |
| `axis` | skeleton + VLM | **no evidence**: the biggest unknown |
| `CONTACT.seq`, `legs_path`, `feet_off_first`, `facing` | VLM or learned reader + skeleton hand/foot proximity | weakest. The old pk_basics wall |
| `surface`, `EXIT.surface` | scene / VLM | partial |
| `stance`, `legs`, `TRANSITION` | skeleton + VLM | partial |

## Known naming conflicts the grammar must carry (not resolve silently)

From glossary §4:
- Gainer take-off: one or two feet.
- "Cheat": tricking set-up vs parkour slant gainer.
- "Double cork": tricking/FIG twist count vs snowsports inversion count.
- "Hyper": four definitions.
- "Reverse": five meanings.
- Aerial vs Sideflip.
- Aerial twist: 180 or 360.
- Tic tac: one step or two.
- Wall run: horizontal vs vertical.
- Which hand touches in a touchdown.
- Tricking kick degree tables (TP vs LK disagree).

The lexicon stores community-tagged variants. The generator outputs every matching name with its community.

## Held-out split for G1 (locked 2026-09-29, before any parsing)

See `data/splits/g1_name_split.json`:
- **Design:** `sha1(canonical_name) % 10 < 3`, plus the 229 test names that research notes 08, 09 or this spec had quoted. 1,049 names in total.
- **Test:** the other 1,628.
- **FIG:** the FIG Tables of Tricks form a separate test set. `data/fig_tricks_2025.json` misreads the table (same-row tricks stored as aliases, wrong cork/double-cork/B-360 values; verified 2026-09-29). It must be rebuilt from the official 2025/2026 PDFs before use.

## Open questions for the owner

1. Is **take-off facing** the right reference for all directions? With it, an Arabian is "back + half twist" and a Barani is "front + half twist".
2. **Tic tac:** one wall step or two? **Wall run:** horizontal or vertical? (Sources disagree.)
3. **Aerial vs Sideflip:** what separates them for you? Shape, footwork, or something else?
4. **Hyper / stances:** keep LK's modern stance system as the default?
5. Are there fields you'd add that name a trick differently in practice: hand placement, height, which leg kicks?
