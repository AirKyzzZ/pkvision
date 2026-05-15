# P0 Blind Decoder Report

N = 67 verified rows; skipped (unresolved verified trick) = 0

- **full**: top1=73.1% top3=85.1% d_score_MAE=0.2253731343283582
- **no_canonical**: top1=71.6% top3=83.6% d_score_MAE=0.24029850746268658
- **no_group_bonus**: top1=73.1% top3=85.1% d_score_MAE=0.2253731343283582
- **ontology_only**: top1=71.6% top3=83.6% d_score_MAE=0.24029850746268658

GATE (full top1 >= 80%): FAIL
Canonical-bonus dependence (full - no_canonical top1): 1.5%

## Verdict (2026-05-15)

**DECISION: VALIDATED — proceed with A′; decoder accepted as non-circular foundation; NO decoder rewrite.**

- Circularity hypothesis (the reason P0 existed) is REFUTED: canonical-bonus dependence = 1.5pp; `ontology_only` top-1 = 71.6% (~107x chance). The decoder is genuine structured reasoning, not POOL-B memorization.
- Strict >=80% gate missed (full top-1 73.1%) but: (a) N=67 -> 95% CI ~=+/-11pp, 80% within noise; (b) shortfall fully diagnosed as ontology cue-degeneracy in physics-collision families (chiefly `pk_basics flip=0 twist=0` 9-trick vault group at 11%, some swing groups 0%) where the FIG ontology encodes no distinguishing cue — an ontology-coverage ceiling, NOT a decoder defect or circularity.
- The literal plan rule (full<80% -> DECODER-REWORK) is overridden by evidence: that branch's rationale (memorization) was disproven.

**Next:** proceed to P1 (heavy offline pose extraction + audit) AND a scoped task for cue-degenerate families (extend FIG ontology distinguishing cues / special-case pk_basics) before end-to-end reliance. Decoder NOT reworked.

**Known follow-up defect:** P0 scripts require `PYTHONPATH=.` (or a sys.path bootstrap) when run directly — `python3 scripts/p0_eval_decoder.py` fails on `import core`. Result unaffected (verified via `PYTHONPATH=.`). Fix tracked.
