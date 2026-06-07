# Parkour Trick-Notation Parser — Compositional Name → Cue Labels (v1)

**Date:** 2026-06-07
**Status:** Design (awaiting review → implementation plan)

## 1. Context & motivation

Diagnostics established that the pkvision recognition stack is **viable but blocked by label quality, not by the model** (per-cue val macro-F1 ~0.4; auto-derived labels ~1.36 MAE off gold D-scores). The quantified wall is **noisy automatic labels** — specifically *slug→trick match errors* — under the hard no-manual-labels-at-scale constraint.

Current label derivation (`scripts/build_attribute_dataset.py`) is a **pure dictionary lookup**: `unified.get(slug)` / a hand-built FIG map. It throws away the single richest free signal we have — the **trick name itself**. Parkour/tricking names are a **compositional notation**: numeric rotations (`360`/`720`/`1080`), count modifiers (`half`/`full`/`double`/`triple`/`one_and_a_half`), flip-phase structure (`in … out`, `unwind`, `counter`), and a move lexicon (`gainer`, `cork`, `arabian`, `layout`, `cat`, `kong`, …). When a slug matches the wrong/variant dictionary entry, or the dictionary physics are coarse, the label is silently wrong.

Data evidence (1,622-clip manifest):
- Names are highly compositional: avg 3.8 tokens/slug, 286 distinct non-numeric tokens.
- High-precision anchors are near-perfect: `gainer`→backward 220/254, `cork`→off_axis 33/33 (non-null), `layout`→body_shape 60/61, `full`→1 twist 221/282.
- Name-parsing **confirms ~78%** of existing labels and **flags a real minority** that diverge; on divergences *either side* can be right (FIG gold `back_double_full`=3.5 beats a naive parse; a wrong `unified` entry or an `unmatched` clip is the parser's to fix).

This spec covers **v1: a compositional trick-name parser that cleans the cue labels** (and cross-checks existing ones), validated against known answer keys, with corrections routed through the existing human-verify flywheel before they are trusted for training.

## 2. Goal & non-goals

**Goal:** Parse the compositional trick notation into cue labels (`flip, twist, direction, axis, context, body_shape, entry`) with calibrated per-cue confidence, use it under a layered authority to **fill** missing cues and **correct** noisy ones, **flag** every disagreement, route the high-value changes through **human verification**, and demonstrate a measurable **core-mF1 lift over the 0.390 baseline** after retraining.

**Non-goals (v1):**
- No LLM parsing (held as an optional later escape hatch only if measured coverage demands it).
- No auxiliary training signal / CLIP-style name↔skeleton alignment yet (Phase 2, noted below).
- No new video scraping / data scaling.
- No replacement of FIG gold — FIG remains the 149-trick ground truth.

## 3. Key decisions (brainstorm outcomes)

1. **Both, parser first.** Build a reusable name-grammar module that first cleans labels (and powers a verify-UI cross-check); the auxiliary training signal is a sequenced Phase 2.
2. **Layered authority, fill + flag.** Priority **FIG gold → confident parser → unified physics → frame-category.** Parser fills missing cues and overrides `unified` only when confident *and* disagreeing; FIG always wins but disagreements are still logged. **High-precision-or-abstain** — never emit a guess.
3. **Hybrid grammar build.** Auto-mine candidate `token→cue` associations from FIG + `unified_tricks` to draft the lexicon; hand-curate the compositional core (phases, modifiers, numeric flip-vs-twist family routing). Domain expert (user) disposes on ambiguous calls.
4. **Staged success.** Gate 1 = intrinsic precision on FIG (149) + gold (99). Gate 2 = retrain CueModel on cleaned+verified labels, report core-mF1 lift over 0.390 + #labels filled/corrected.
5. **Human-verify the corrections.** Because these labels train the model, the parser's corrections/fills route through the existing `verify_server` + `VerifiedStore` (verify-not-label). The bulk where parser agrees needs no human touch — verification effort stays bounded.

## 4. Architecture — notation parse → diff → verify → clean labels

```
                                  ┌──── FIG 149 + gold 99 (answer keys) ──── Gate 1 (precision) ───┐
                                  ▼                                                                 │
trick name ─► PARSER ─► ParsedCues{cues, confidence, trace, unparsed} ─► LAYERED MERGE ─► manifest_v2
(notation)   (5 stages)                                            (FIG>parser>unified>cat + provenance)
                                                                          │
                                                  changes.json (fills + corrections + FIG-disagreements)
                                                                          ▼
                                              VERIFY UI (human confirms/corrects) ─► VerifiedStore ─► trusted labels
                                                                          ▼
                                              retrain CueModel ─► Gate 2 (core-mF1 lift vs 0.390)
```

## 5. Components (units, interfaces, reuse vs build)

**`trick_name_parser` (build — `core/recognition/trick_name_parser.py`)**
Pure, side-effect-free. One entry point:
```python
@dataclass
class ParsedCues:
    cues: dict[str, float | str]      # only cues the grammar is confident about
    confidence: dict[str, float]      # per-cue 0..1
    trace: list[str]                  # rules that fired (full provenance)
    unparsed_tokens: list[str]        # tokens no rule consumed (coverage signal)

def parse_trick_name(name: str) -> ParsedCues: ...
```
Abstention is first-class: an uncertain cue is *absent* from `cues`. Five ordered stages:
1. **Normalize + tokenize** — lowercase; split on `_`/`-`/space; fold multiword atoms (`one_and_a_half`→`1.5x`, `double`→`2x`).
2. **Phase segmentation** — split on `in`/`out`/`unwind`/`down`; flip count from phase count + explicit flip tokens; twist sums across phases. (Kills the flat-lexicon failure mode.)
3. **Modifier resolution** — `half`/`full`/`double`/`triple`/`one_and_a_half` resolve by **adjacency** to a twist- vs flip-keyword (`double full`=2 twists; `double back`=2 flips; `dive_half_back` half-flip ≠ `inward_half` half-twist).
4. **Numeric rotations** — `180/360/540/720/900/1080…` → degrees, mapped to **flip or twist by governing move family** (somersault→flip like `1080_dive_roll`=3 flips; turning/standing→twist like `180_cat`=½ turn).
5. **Aggregate** — per-cue confidence = min of contributing rules; conflicting confident rules → **abstain + flag** (never silently pick).

**Move lexicon (data — `data/name_grammar/lexicon.json`)**
Curated `token → {cue contributions, confidence, family}` table feeding stages 3–5. Drafted by the miner, finalized by curation.

**Miner (build — `scripts/mine_name_grammar.py`)**
For each token, aggregate FIG + `unified_tricks` physics across all names containing it → `token→cue` associations with **support count + purity**. Writes `data/name_grammar/lexicon_draft.json` for curation.

**Validator / Gate 1 (build — `scripts/validate_name_parser.py`)**
Runs `parse_trick_name` on the **149 FIG tricks** (vs known flip/twist/direction/score) and the **~99 gold clips** (vs verified cues). Reports per-cue **precision on fired predictions**, **abstention rate**, confusion breakdown. Bar: per-cue precision ≥ 0.90 on non-abstained outputs. Doubles as a regression test.

**Label integration (adapt — `scripts/build_attribute_dataset.py`)**
Resolution becomes layered (§3.2). Writes versioned `data/v5_attribute_training/attribute_manifest_v2.json` with **`label_provenance` per cue** + `data/name_grammar/changes.json` (every fill / correction / FIG-disagreement).

**Verify hook (adapt — `scripts/make_proposals.py` + `scripts/verify_server.py`)**
A parser-backed proposal source surfaces each changed clip showing **parse cues vs old label**, prioritized by disagreement. Human confirms/corrects in the existing UI; verdicts append to `VerifiedStore` and become trusted training labels. Bulk agreements skip verification.

**Retrain + Gate 2 (reuse — `scripts/train_cue_model.py`)**
Retrain on the cleaned+verified manifest; report core-mF1 vs 0.390 baseline + counts filled/corrected/verified.

**Reused as-is:** `core/labeling/{cue_model,store,clip_ref,proposer}.py`, `core/recognition/{fig_decoder,oracle_cues}.py`, the FIG table, `unified_tricks.json`, the gold set.

## 6. Data flow & stores

- **Lexicon draft / final** — `data/name_grammar/lexicon_draft.json` (mined), `data/name_grammar/lexicon.json` (curated, the source of truth for stages 3–5).
- **Cleaned manifest** — `data/v5_attribute_training/attribute_manifest_v2.json` with per-cue `label_provenance ∈ {fig, parser, unified, frame_cat}` (auditable, reversible).
- **Changes report** — `data/name_grammar/changes.json`: `{slug, cue, old_value, old_source, parser_value, parser_conf, kind ∈ {fill, correct, fig_disagree}}`.
- **Verified store** — the existing append-only `data/labeling/verified.jsonl` (parser-sourced verifications carry `proposer_source="parser"`).

## 7. Success metrics & testing

- **Gate 1 (intrinsic):** per-cue precision ≥ 0.90 on fired predictions on FIG 149 + gold 99; abstention reported honestly. No downstream work until it passes.
- **Gate 2 (payoff):** core-mF1 vs the 0.390 baseline after retraining on the cleaned+verified manifest; plus #labels filled / corrected / human-verified.
- **Human throughput:** clips/min on the verification of changed clips (the human-cost metric).
- **Unit tests** per construct: numeric (flip *and* twist families), phases (`back_full_in_full_out`), modifiers (`double_full`, `one_and_a_half`, `dive_half_back` vs `inward_half`), lexicon anchors, abstention on gibberish, conflict→abstain. A golden snapshot of ~30 representative slug parses catches lexicon-curation regressions; the FIG-149 check is a regression gate.

## 8. Error handling

- Parser never throws on unknown tokens (→ `unparsed_tokens` + abstain). Fail-fast only on non-string / malformed input.
- Conflicting confident rules → abstain + flag (no silent winner).
- Coverage (unparsed-token rate) is reported, not hidden — it sizes the residual the optional LLM escape hatch would target later.
- Layered merge is non-destructive: the original manifest is untouched; `_v2` + provenance make every change reversible.

## 9. Build sequence (phases — detailed in the plan)

- **P-A** `trick_name_parser.py` skeleton: dataclass, tokenize/normalize, stage scaffolding, abstention contract + unit tests for the contract.
- **P-B** Miner → `lexicon_draft.json`; curation pass → `lexicon.json` (compositional core + anchors).
- **P-C** Implement stages 2–5 against the lexicon; per-construct unit tests + golden snapshot.
- **P-D** Validator / **Gate 1** on FIG 149 + gold 99; tune lexicon until the bar is cleared.
- **P-E** Layered integration in `build_attribute_dataset.py` → `manifest_v2` + provenance + `changes.json`.
- **P-F** Verify hook (parser proposal source + disagreement prioritization) → human verification batch.
- **P-G** Retrain → **Gate 2** (core-mF1 lift) + report.

## 10. Risks & open questions

- **Flip-vs-twist numeric routing is the trickiest rule** → calibrated against FIG 149 in Gate 1; abstains when family is ambiguous.
- **Curation is where precision is won or lost** → miner provides purity stats; user is domain authority; Gate 1 is the objective check before any label is changed.
- **Long-tail coverage unknown** → unparsed-token rate is measured (§8); optional LLM escape hatch deferred until the residual is quantified, not assumed.
- **Verification throughput on changed clips** is unproven until P-F; disagreement-first ordering of the verify queue is the lever.
- **Phase 2 (deferred):** name-derived attributes as auxiliary supervision (the train wrapper already supports aux heads) or CLIP-style skeleton↔name alignment — straightforward once labels are clean + provenance exists.
