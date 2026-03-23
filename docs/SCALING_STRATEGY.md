# PkVision — Data Scaling Strategy

## The Problem

Parkour has thousands of tricks. Manual labeling doesn't scale. We need a system where:
- Adding a new trick requires describing it, not labeling 100 clips
- The model improves itself with minimal human oversight
- The community contributes labels as a side effect of using the system

## The Solution: Four-Tier Data Pipeline

### Tier 1: Foundation (Zero Human Effort)

Pre-trained models already understand human movement. We leverage:
- **VideoMAE** (Kinetics-400) — knows somersaulting, tumbling, parkour, cartwheeling
- **CLIP** — matches video to text descriptions
- **Kinetics-400/700** — 4,000+ labeled flip/parkour clips available

These give us coarse categories (flip vs vault vs parkour movement) for free.

### Tier 2: Trick Descriptions as Labels

Instead of labeling 100 clips per trick, we DESCRIBE each trick in natural language. CLIP matches videos to descriptions automatically.

Example trick description file (`data/tricks/descriptions/back_flip.txt`):
```
A person jumping upward and rotating backward in the air.
The body is tucked with knees pulled toward the chest.
A single complete backward rotation, landing on both feet.
A standing or running backflip, somersault backward.
```

Example trick description file (`data/tricks/descriptions/gainer.txt`):
```
A person running forward then flipping backward while still moving forward.
Forward momentum with backward rotation in the air.
A backflip done while moving forward, often from a ledge or platform.
```

The CLIP model computes similarity between each video and each trick description. The best match becomes the label. **No human watching required.**

For 3,000 tricks, you write 3,000 text descriptions (5 sentences each). That's a few hours of work by someone who knows parkour, not months of video labeling.

### Tier 3: Community Consensus

The system becomes self-labeling through community participation:

1. **Upload and self-label**: Athletes upload a clip and tag it with the trick name (they know what they just did)
2. **Peer verification**: Other athletes confirm or correct the label (like Waze reports)
3. **Confidence threshold**: After 3 confirmations, the label is accepted into training data
4. **Gamification**: Leaderboard for contributions, badges for verified labels

This turns every parkour athlete into a data contributor. The more popular PkVision gets, the faster the data grows.

### Tier 4: Active Learning Loop

The model drives its own improvement:

```
New video uploaded
    → Model classifies with confidence score
    → If confidence > 90%: auto-label (no human needed)
    → If confidence 50-90%: show to community for verification
    → If confidence < 50%: flag for expert review
    → Verified labels added to training set
    → Model retrained periodically
    → Repeat
```

Over time, the model gets more confident → fewer human reviews needed → scales infinitely.

## Implementation Phases

### Phase 1 (Now): Foundation + Your Labels
- Pre-trained VideoMAE + Kinetics-400 data
- Your 89 manually labeled parkour tricks
- 24 trick classes, 313 training clips

### Phase 2: CLIP-Based Auto-Labeling
- Write text descriptions for 50-100 tricks
- Use CLIP to auto-label YouTube/Instagram parkour clips
- Human review only for uncertain matches
- Target: 1,000+ labeled clips with minimal manual work

### Phase 3: Community Portal
- Web app where athletes upload and self-label clips
- Peer verification system
- Public leaderboard
- Target: 5,000+ community-labeled clips

### Phase 4: Self-Improving Pipeline
- Active learning: model asks humans only when uncertain
- Automatic retraining on new verified data
- Continuous accuracy improvement
- Target: 10,000+ clips, 100+ tricks, 90%+ accuracy

## Why This Works

- **Tier 1** eliminates the cold-start problem (model already knows movement)
- **Tier 2** replaces "watch 100 videos" with "write 5 sentences" per trick
- **Tier 3** makes every user a labeler (supply grows with demand)
- **Tier 4** reduces human work over time (model handles easy cases)

The result: a system where adding trick #3,000 is as easy as adding trick #10.
