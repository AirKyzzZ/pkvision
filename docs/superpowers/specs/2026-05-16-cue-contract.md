# PkVision Cue Contract — what P1–P3 must predict (2026-05-16)

**Purpose:** defines the exact structured cues the future P3 pose/cue model must output to feed the **P0-validated, non-circular** frozen FIG decoder, with honest single-camera extractability flags and the now-known ontology data-quality risks. This is the authoritative target for P1 (pose extraction) → P3 (cue model).

## 1. Decoder cue vocabulary & weights

From `core/recognition/fig_decoder.py` `CUE_WEIGHTS` (the validated, frozen decoder — P0 confirmed it is genuine structured reasoning, not POOL-B memorization):

| Cue | Weight | What it is |
|---|---|---|
| `context` | 3.0 | ground(acrobatics) / wall / swing / pk_basics |
| `flip` | 3.0 | flip count (0, 0.5, 1, 2, 3, …) |
| `direction` | 2.0 | backward / forward / side |
| `twist` | 2.0 | twist count (0, 0.5, 1, 1.5, …) |
| `axis` | 1.5 | lateral / longitudinal / sagittal / off_axis |
| `entry` | 1.5 | standing / running / kong / caster / roundoff / … |
| `takeoff` | 1.5 | running_forward / standing / cheat / gainer / swing_release / … |
| `hand_contact` | 1.5 | bool — key for Backflip vs Backhandspring |
| `kick` | 1.0 | bool — only ~2 tricks |
| `body_shape` | 0.5 | tuck / pike / layout — cosmetic |

There is **no `movement` cue** — the proposed `movement`/vault-type dimension (would help the pk_basics family) is **DEFERRED** (see §4). The decoder skips absent cues with no penalty, so emitting `None` (abstain) for an unreliable cue is graceful by design.

## 2. Single-camera extractability (the honest part)

What a P3 model can realistically predict from **one** 2D camera (per P0 results + `research-2026-05-feasibility` memory):

| Cue | Single-camera extractable? | Ceiling / notes |
|---|---|---|
| `context` | Mostly (scene/apparatus visible) | Reliable enough; key disambiguator |
| `flip` | Yes for integer counts (sagittal rotation) | Reliable; the decoder's strongest real signal |
| `direction` | Yes (gross) | Reliable |
| `twist` | **NO — geometric axis ambiguity** | `twist ≥ 1.5` essentially unrecoverable mono. **Model must ABSTAIN** (emit `None`). Accepted hard ceiling; this is what the deferred synced multi-iPhone path exists to fix. |
| `axis` | Partial | off_axis vs lateral is hard mono |
| `entry` / `takeoff` | Only gross cases (running vs standing) | Ontology defines these on only 15–21/149 tricks; Layer-3 distinguishers are sparse in the ontology AND hard from one view |
| `hand_contact` | Sometimes (wall/bar contact visible) | Useful (Backflip vs Backhandspring) |
| `kick` | Rarely needed | 2 tricks |
| `body_shape` | Partial | Low weight, cosmetic |

## 3. P0-validated decoder performance (the UPPER BOUND)

Oracle cues (perfect cue extraction), N=67 blind, decontaminated, hard-dominated set:

- Trick **top-1 73.1%**, **top-3 85.1%**, **D-score MAE 0.225**, **`dscore_correct` 74.6%**.
- `dscore_correct` is only +1.5pp over top-1 → **confusions are real D-score errors, NOT zero-cost** (the "many errors are scoring-equivalent" hypothesis was tested and **falsified**).
- **Non-circular:** removing the POOL-B-tuned canonical bonus costs only 1.5pp; pure ontology-only scoring = 71.6%.

**P1–P3 (real single-camera cue extraction) will perform BELOW this oracle bound**, capped especially by the twist ceiling. Treat 73% oracle top-1 as the ceiling, not the target.

## 4. Known gaps / risks that MUST inform P1–P3

1. **Cue-degenerate families (~25% real scoring error):** `pk_basics` (13 vault tricks — Stride/Plyo/Tic Tac/Kong/Side/Pop/Reverse/Kash/Dong/Double Kong Vault/Wallrun/Climb Up/Dyno — identical physics signature, 11% top-1), plus some swing/cork groups (0%). These have **no distinguishing cue in the ontology**. A `movement`/vault-type cue would help pk_basics but Task 4 (adding it + mutating the ontology) is **deferred** pending the data-quality resolution and the P1-vs-ontology priority reassessment.
2. **The FIG ontology itself is data-quality-unreliable** (`scripts/fig_ontology_audit.py`, Task 3a): 15 alias-duplicate entries, several biomechanically wrong (`Double Frontflip`→`Triple Cork`, `Kong Gainer`→`Backflip 900`, `Double Sideflip`→`Gainer 720`). Implications: `data/fig_tricks_2025.json` cannot be blindly trusted as the scoring source-of-truth for the degenerate families; **automated alias-dedup is unsafe**; per-group P0 numbers for affected families carry data-quality uncertainty. Any ontology mutation must be human/FIG-Code-of-Points-verified.
3. **Twist ceiling (geometric, single camera):** unrecoverable for `twist ≥ 1.5`; the model must abstain; this caps single-camera accuracy regardless of model quality. The deferred synced multi-iPhone capture path is the only thing that addresses it.

## 5. Strategic note (open decision)

The decoder is a **validated, non-circular foundation (~73% oracle top-1)**. The remaining ceiling splits into:
(a) **ontology data quality** — fixable, but needs careful human/FIG-CoP work, not automation; and
(b) **single-camera cue extractability** — especially twist, which is a hardware/multi-view problem, not a model problem.

Before investing further in ontology work (deferred Task 4), the open question for the next session is whether **P1 (real pose/cue extraction — the actual product bottleneck)** is higher leverage than polishing a quality-questionable ontology. This contract is the fixed target either way: P1–P3 must predict the §1 cues, accept the §2 extractability limits, and not assume the ontology is clean (§4.2).
