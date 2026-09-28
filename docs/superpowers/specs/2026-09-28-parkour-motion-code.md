# Parkour Motion Code (PMC) v0: draft for owner review

**Status:** DRAFT v0, 2026-09-28. Written from the 30% design split of trick names only (see below). Every field is open for owner review.

**Purpose:** give every trick a precise, machine-readable description of the motion, playing the role the dive number (`5253B`) plays in diving. It sits between video and names:

```
video ──(vision readers)──▶ PMC ──(naming grammar)──▶ name(s) + aliases
                                ◀──(name parser)────── name + description
```

The vision side is judged on how accurately it produces the PMC. The naming side is judged on round-trip fidelity (name → PMC → name). An unseen trick is a PMC that no known name matches; it gets a generated name and always keeps its PMC.

## What the naming convention looks like (from the design split)

1. **Recursive over base moves.** "Pimp Double Full" is Pimp Flip plus a double full twist. Descriptions define tricks through other tricks ("Circle into Gaet Pimp").
2. **Sequential.** "Kong Front Dash" is kong, then a front flip, then a dash vault. "X into Y" is the usual chaining.
3. **Modifier families:**
   - setup: One-Step, Two-Step, Kneel, Pop, Turn, Toe-On, Punch, Palm;
   - limb: One-Hand, Elbow, Shoulder;
   - rotation: Full, Double Full, Half, Quarter, 180/360/540, One-And-A-Half, Double, Triple, Inward, Counter, Unwind;
   - shape and style: Layout, X-Out, Single Ankle, Method Grab;
   - exit: Dive Roll, Down (off a ledge), Precision, Regrab, Catch, Dismount, Bomb.
4. **Phase markers.** In "A-In B-Out", one rotation starts during element A and finishes during element B.
5. **Degrees are the native unit** of longitudinal rotation (180 = half twist). Quarters exist.

## Design choice (the main thing to review)

A **hybrid, two-level code**:

- **Contact and setup phases** (vaults, wall contacts, bar elements, landings) use a **closed vocabulary of named elements** (`kong`, `dash`, `tic_tac`, `cast`, `lache`, …). Their inner limb sequence is not broken down in v0.
- **Airborne phases** are fully **physical**: somersault degrees, twist degrees, direction, axis, shape, twist timing. Most new tricks are new rotation combinations on known elements, and rotation is what skeleton rule counters can measure.
- **Named aerial base moves** (gainer, webster, raiz, pimp, arabian, macaco, cork, …) are **not** primitives. Each gets a lexicon entry defining it *as a PMC*, to be induced from the descriptions in G1 and verified by the owner. So "Gainer Full" = the gainer entry + a twist of 360.

Alternatives considered:
- Fully physical, with no named elements: the most general, but vault limb sequences are hard both to parse and to see.
- Flat attribute tuple (the 2026 cue contract): can't represent sequences or the -In/-Out markers. Its pk_basics degeneracy came from this.

## Schema

A PMC is an ordered list of **segments** (start-time order). JSON is canonical; the compact string is for display.

### Segment types

| Type | Meaning | Fields |
|---|---|---|
| `SETUP` | approach / set before the main element | `from`: ground, wall, bar, obstacle_top, ledge. `action`: stand, run, step_n (steps up a wall), kneel, pop, turn. `feet`: 1, 2 |
| `CONTACT` | named contact element | `element`: closed vocabulary (below). `surface`: ground, obstacle, wall, bar, rail_beam. `limbs`: 1H, 2H, feet, 1F, knee, elbow, shoulder, toe |
| `AIR` | airborne rotation phase | see below |
| `EXIT` | how it ends | `kind`: feet, precision, dive_roll, roll, catch, regrab, down_ledge, handstand. `surface` as above |

### `AIR` fields

| Field | Unit / values | Notes |
|---|---|---|
| `som` | degrees, multiple of 90, ≥ 0 | somersault (inversion) rotation |
| `som_dir` | forward, backward, side | relative to the body at take-off |
| `travel` | with, against, lateral | travel relative to the rotation. `against` + backward = gainer family |
| `twist` | degrees, multiple of 90, ≥ 0 | longitudinal rotation |
| `twist_rel` | same, counter, unwind | relative to the preceding rotation or twist (Counter, Unwind) |
| `twist_when` | early, mid, late, whole | where the twist happens in the flight |
| `axis` | on, off | off = cork/corkscrew/raiz-style tilted axis |
| `shape` | tuck, pike, layout, open, straddle | "open" = no defined shape (e.g. twist-only) |
| `style` | list: x_out, single_ankle, double_leg, grab_<name>, kick_<name> | cosmetic. Never used for identity in v0 |
| `during` | list of `CONTACT` | contact made mid-rotation (Back Dash pushes the vault while rotating; the -In/-Out markers) |

### Closed element vocabulary (v0 seed, to be extended in G1)

- **Vaults:** kong, dive_kong, dash, speed, lazy, reverse, safety, turn, split, butterfly, aerial_twist_vault, double_kong
- **Wall:** step, tic_tac, wall_run, punch, palm, wall_spin, climb_up
- **Bar / pole:** swing, cast, caster, castaway, lache, giant, toe_on, flyaway, kip
- **Ground:** roll, handspring, handstand, cartwheel

### Compact display string

Segments are joined with ` > `. `AIR` is written as `AIR(<dir><som> t<twist>[,counter][,off] <shape>)`:

- `som` in degrees, with direction f / b / s;
- `t` twist in degrees, followed by `,counter` or `,off` when set;
- shape is `T` / `P` / `L` / `O` (tuck, pike, layout, open).

Illustrative examples, taken from the parkourtheory descriptions. The owner should check these:

| Name | Description (parkourtheory) | PMC string |
|---|---|---|
| Wall Full | A full twisting Wall Flip | `SETUP(run) > CONTACT(step, wall, 1F) > AIR(b360 t360 O) > EXIT(feet)` |
| Dash 360 | Dash Vault into a 360 twist off the obstacle | `CONTACT(dash, obstacle, 2H) > AIR(t360 O) > EXIT(feet)` |
| Kong Front Dash | Kong Front off the first wall to a Dash Vault off the second | `CONTACT(kong, obstacle, 2H) > AIR(f360 T) > CONTACT(dash, obstacle, 2H) > EXIT(feet)` |
| Back Dash | A 180 jump into a back tuck while pushing off the vault | `SETUP(run) > AIR(b360 t180 T during[CONTACT(push, obstacle, 2H)]) > EXIT(feet)` |
| Gainer Layout | A laid out Gainer | `SETUP(run, 1F) > AIR(b360 against L) > EXIT(feet)` |

## Observability tags (which reader must produce each field)

| Field | Reader | Evidence (survey 2026-09-28) |
|---|---|---|
| `AIR.som`, `som_dir` | skeleton rule counter | B: diving somersault 97.3% |
| `AIR.twist` | skeleton rule counter, VLM as second opinion | B: diving twist 93.3%; corks unknown |
| `AIR.axis` | skeleton + VLM | no evidence: biggest unknown |
| `AIR.travel`, `twist_when`, `shape` | skeleton (root trajectory, joint angles) | B, partial |
| `CONTACT.element`, `limbs`, `surface` | VLM / learned + scene | weakest: this is the old pk_basics problem |
| `SETUP`, `EXIT` | VLM + skeleton | partial |

Anything a reader can't decide is written `?` (abstain). A PMC containing `?` still names coarsely, e.g. "vault(?) > back flip with full twist".

## Out of scope for v0

- Execution quality.
- FIG scaling cues (height/distance/placement).
- Obstacle geometry beyond `surface` and `down_ledge`.
- Tricking ground kicks as full elements: kept as `style`/`kick_*` unless G1 shows they carry identity.

## Held-out split (locked 2026-09-28, before any parsing)

- **Design split:** `int(sha1(canonical_name).hexdigest(), 16) % 10 < 3` over `data/unified_tricks.json`. That is 820 of 2,677 names, and the only ones used to write or refine this spec and its lexicon.
- **Test split:** the other 1,857 names, plus all 149 FIG tricks (the FIG table was not consulted for this draft). Used only for G1 metrics.

## Open questions for the owner

1. Is hybrid (named contact elements + physical rotation) the right depth?
2. Is `travel: against` the right definition of the gainer family? It also has to cover cast/caster gainers from bars.
3. What do Inward, Counter and Unwind mean exactly? Proposed: inward = back rotation traveling toward the wall/obstacle; counter = twist opposite to the natural direction; unwind = twist back in the opposite direction. Needs your definitions.
4. Are the element vocabulary groups complete for FIG? What's missing?
5. Scope: include tricking kicks (loopkicks/trickipedia) in G1 metrics, or report them only as secondary?
