# PkVision: what is next

The plan and its reasons are in `docs/research/2026-11-26-direction-review.md`. The older roadmap (Docker, live webcam, GVHMR) is dropped until one attribute is read reliably.

## Now

- [ ] Report the pre-registered label fix (`scripts/probe_pt_label_fix.py report`)
- [ ] Merge PR #4, #5, #6; update PR #7 with the 296 new Parkour Theory moves
- [ ] Owner: label the 24 remaining blind clips (`data/labeling/retest_2026-11/label.html`)

## Next

- [ ] Twist counts from the name parser and the Parkour Theory progression graph, conflicts to a small owner check queue
- [ ] Second evaluation set: the 36 owner labels, and a held-out half scored on Parkour Theory categories
- [ ] Athlete selection, measured with `scripts/probe_flip_yardstick.py`
- [ ] One small pose swap on 100 clips for upside-down frames

## Later, each behind its own pre-registration

- [ ] Skeleton-free features (V-JEPA 2) on the repaired labels
- [ ] VLM blind probe, once the labels and a budget of $5 exist
- [ ] Rewrite `README.md`
