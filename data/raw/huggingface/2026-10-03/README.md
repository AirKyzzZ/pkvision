# Hugging Face Hub sweep (2026-10-03)

**Source:** Hugging Face Hub public API (`/api/datasets?search=`, `/api/models?search=`, `/api/datasets/<id>`, `/api/datasets/<id>/treesize/main`, `/api/datasets/<id>/tree/main/<dir>`).

**Licence:** this file is our own metadata summary. Each dataset keeps its own licence, listed per row (`license`, `gated`).

## Method

- Searched about 70 terms: parkour, freerun(ning), tricking, acrobatic(s), gymnastics, gymnast, trampoline, tumbling, diving, figure skating, skating jump, somersault, flip, backflip, FineGym, FineDiving, Diving48, FS-Jump3D, FineFS, SkatingVerse, AthletePose3D, MMFS, AQA, sport(s), mocap and others.
- Read metadata, card text where public, and the storage size.
- Only one file was downloaded: the 3.9 MB ActionAtlas parquet (see `../../actionatlas/2026-10-03`). No video was downloaded.
- HF search matches substrings of repo ids only, so datasets whose ids don't contain the term can be missed.

## Counts

27 datasets and 4 models rated. Terms with no relevant hit:
- freerun, freerunning, tricking, acrobatic(s), somersault;
- FineDiving, FSD-10, FineFS, SkatingVerse, FS-Jump3D, AthletePose3D;
- tumbling (no dataset).

## Files

`hf_sweep.json`: `datasets[]` (`id`, `modality`, `labels`, `license`, `gated`, `last_modified`, `size_gb`, `usefulness`) and `models[]`.
