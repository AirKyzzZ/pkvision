# ActionAtlas acrobatic subset (2026-10-03)

**Source:** https://huggingface.co/datasets/mrsalehi/ActionAtlas-v1.0 (`data/test-00000-of-00001.parquet`, 3.9 MB, not gated). Paper: Salehi et al., ActionAtlas, NeurIPS 2024 D&B, arXiv 2410.05774.

**Licence:** the code repo https://github.com/mrsalehi/action-atlas is Apache-2.0. The HF dataset card declares no licence. Rows are YouTube references (ID + start/end seconds), not clips.

**Credit:** ActionAtlas, Salehi et al. 2024.

## Method

Downloaded the parquet into a scratch directory (not kept). Kept only the rows whose `domain` is parkour, cheerleading, gymnastics, diving or skating. Fields kept: `id`, `domain`, `action`, `youtube_id`, `start_s`, `end_s`, `youtube_title`, `choices` (the MCQ distractors). No video downloaded.

## Counts

114 of 934 rows: Cheerleading 63, Figure skating 41 (two spellings), Rhythmic Gymnastics 3, Gymnastics 3, Diving 2, Parkour 2, Ice Skating 1.

Cheerleading rows name tumbling skills: back tuck, standing back tuck, layout, full up, back handspring, double back handspring, cartwheel full twist, front tuck, punch front, aerial. Parkour has only 2 rows ("Parkour roll").

## Files

`acrobatic_subset.json`
