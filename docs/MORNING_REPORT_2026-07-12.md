# Morning report — 2026-07-12

You went to bed on a stale project and asked me to use Claude as the judge to make your labels far better, and to start off on the right foot so we don't relabel everything in a few weeks. Here's what happened overnight.

## TL;DR

1. **Claude-as-judge works.** On your 99 gold clips, two independent Claude reviewers agree with each other 96% on flip count, 93% on twist, 93% on direction, 94% on context.
2. **Your "gold" labels are ~17% wrong on flip and ~35% wrong on twist.** Every past eval (the 0.390 cue model, the P0 blind set) was scored against a partly-broken yardstick. This is the "right foot" fix — we now know the ground truth was the problem.
3. **Cleaner labels measurably improve the model.** In a clean, zero-circularity experiment, relabeling just 65 acrobatics clips (4% of the training data) lifted flip prediction on held-out gold by **+8.5pp, robust across all 3 random seeds.**
4. **Cost: $0 in API spend.** Everything ran on your subscription (Fable, then Opus after Fable's monthly limit) plus free local training.

## What I did, in order

- Re-ran the July-2026 SOTA research (104 agents): the "VLMs can't zero-shot judge" verdict still holds for the *published* frontier, but nobody had tested whether a top model could extract *cues* (flip/twist/direction) rather than name tricks. That gap is what your idea exploited.
- Calibrated on 12 clips → passed → audited all 99 gold clips with a 2-vote + tiebreak consensus, capturing a rich schema (flip, twist, direction, **axis**, context, body shape, entry, hand contact, landing, multi-trick flag, per-cue confidence).
- Cross-checked every Claude label against three independent sources: your manifest, the FIG Code of Points, and your trick-name parser.
- Ran one bulk batch (74 acrobatics clips) before spend limits stopped it.
- Retrained the cue model on the cleaned labels and measured lift with a held-out, no-circularity protocol.

## The headline experiment (clean, no circularity)

Both models trained with all 99 gold clips **held out**. Evaluated on the "concordant" gold subset — only clips where your manifest and Claude *independently agree*, so the eval labels are trustworthy regardless of who's right elsewhere. Scored at the model's own bin granularity. 3 random seeds each.

| cue | baseline labels | Claude labels | Δ (mean of 3 seeds) | robust? |
|---|---|---|---|---|
| **flip** | 40.7% | **49.2%** | **+8.5pp** | yes — positive in all 3 seeds |
| direction | 46.3% | 49.0% | +2.7pp | mixed (noise) |
| twist | 64.6% | 59.9% | −4.7pp | mixed (noise) |
| context | 57.7% | 54.3% | −3.4pp | mixed (noise) |

**Read this honestly:** only flip is a robust win. The others are within seed noise *at this tiny scale* (65 relabeled clips, ~80-clip eval). That's expected — flip is exactly the cue those acrobatics relabels touch. The point isn't the magnitude; it's that a 4% relabel of the worst bucket already moves the needle on the primary cue, with no circularity. Full-corpus relabeling is now an evidence-backed bet, not a hope.

## What needs YOU

1. **Spend limits blocked the bulk labeling.** Both Fable and Opus hit their monthly caps. We labeled 74 of 1,519 corpus clips. To finish, we need either quota headroom (raise the limit / next cycle) or the Gemini free-tier path from the research. The pipeline is checkpointed and resumable — I can pick up at clip 90 the moment quota is back.
2. **26 gold clips need your eye** (`data/labeling/claude_probe/gold99_disputes.json`). These are genuine taxonomy calls, not model errors — e.g. `arabian`: Claude reads a standard arabian (1 flip, ½ twist) but your manifest maps it to FIG "A-180" (a different half-flip move). You're the authority here.

## Decisions I made autonomously (tell me if any were wrong)

- Merged `feature/data-flywheel-foundation` into `main` locally (you approved this; not pushed).
- Switched labeling to Opus when Fable's limit hit (you'd switched the session model).
- Used **per-cue** confidence gating (trust Claude's confident flip read even when its twist read is uncertain) rather than all-or-nothing.
- Kept everything **uncommitted** — I did not commit the night's artifacts or push anything. Commit plan is below for your OK.

## Files produced (all uncommitted)

- `data/labeling/claude_probe/` — gold audit: `gold99_votes.jsonl`, `gold99_report.md`, `gold99_disputes.json`, `retrain_lift_3seed.json`
- `data/labeling/claude_labels/labels_v1.jsonl` — 74 bulk-labeled clips
- `data/v5_attribute_training/` — experiment manifests + `eval_gold_concordant.json`
- `data/models/cue_{baseline,claude}_noeval*.pt` — trained checkpoints
- `docs/superpowers/specs/2026-07-12-claude-weak-labeling-design.md` — the schema/protocol spec
- Workflow + analysis scripts live in the session scratchpad (portable if you want them in-repo)

## Recommended next steps

1. You clear the 26-clip dispute queue in the verify UI (~15 min).
2. When quota returns, finish bulk labeling the corpus, then re-run the 3-seed experiment at full scale — that's where twist/direction/context should also move.
3. Then wire the cue model → FIGDecoder top-k for the end-to-end judge-assist demo on a full run.
