# P0 Blind Decoder Report

N = 67 verified rows; skipped (unresolved verified trick) = 0

- **full**: top1=73.1% top3=85.1% d_score_MAE=0.2253731343283582
- **no_canonical**: top1=71.6% top3=83.6% d_score_MAE=0.24029850746268658
- **no_group_bonus**: top1=73.1% top3=85.1% d_score_MAE=0.2253731343283582
- **ontology_only**: top1=71.6% top3=83.6% d_score_MAE=0.24029850746268658

GATE (full top1 >= 80%): FAIL
Canonical-bonus dependence (full - no_canonical top1): 1.5%