# 02: Frontier and open video-language models for fine-grained motion counting

Survey date: 2026-09-28. Desk research only: no project data touched, no code run on project data, no paid API
called, $0 spent. Track: which VLMs exist as of Sep 2026, what they cost per ~4 s clip, what the newest
benchmarks say about counting rotations/reps and reading motion, which input and prompt tricks move accuracy,
whether a small open VLM can be fine-tuned cheaply, and what bulk weak-labelling of 1,618 clips × 2 votes costs.

Grading: **A** = measured on parkour / in-domain. **B** = measured on an analog task. **C** = claim, vendor
statement, pricing/docs page or anecdote (prices from official vendor pages are C but authoritative for price).

The only A-grade evidence on flip/twist counting by a VLM is PkVision's own July probe (12 curated clips).
The only third-party in-domain number is VideoNet's parkour domain (trick-name multiple choice, not counting).

---

## 0. Bottom line

1. **Two items in the July notes are wrong and should be corrected.**
   - **Gemini 3.5 Pro is not released.** It was announced at I/O on 2026-05-19, missed June, July and August
     targets, and as of 2026-09-28 is absent from the official models page, pricing page and changelog (C).
     The newest Google models are Flash-tier: 3.6 Flash (Jul 21), 3.7 Flash, 3.8 Flash (Sep 2). The only Pro
     model on the API is **Gemini 3.1 Pro Preview** (paid only).
   - **"GPT-5.6-video" does not exist.** GPT-5.6 (Luna/Terra/Sol, GA 2026-07-09) and GPT-6 Astra/Sol/Luna
     (Sep 3 / Sep 22) take images only; the API rejects mp4. Video must be sent as frames (C).
   - **The July note that "Gemini 3.1 Pro specifically fails flip/twist/somersault counting" is not supported by
     its cited source.** arXiv 2604.08294 (Sword Health, "Can VLMs Judge Action Quality?") scores *execution
     quality* (FineFS GOE, MTL-AQA execution score, fitness form errors). The words somersault, twist and flip do
     not appear in the paper. It says nothing about counting rotations (B, verified by reading the full text).
2. **The benchmark numbers used to say "VLMs can't count" measure a different regime.** PushupBench counts
   10 to 20+ reps over 22 to 117 s clips; genuine counting collapses above ~10 reps. VideoZeroBench is long-video
   grounded QA. TimeBlind and VideoNet sample at **1 fps**, i.e. ~4 frames for a 4 s parkour clip, which cannot
   resolve a flip. None of them tests "0 to 3 rotations inside ~1 s of airtime with dense frames" (B).
3. **Newest motion-specific evidence (Sep 2026, B):** MotionBlind (arXiv 2609.09528) shows open ≤12B models at
   chance on physical motion (speed, magnitude, direction) while **Gemini 3.1 Pro clears it at 60% instance
   accuracy** (human 91%), still failing on speed. **GPT-5.6 Luna scores 15%.** More frames plateau at 16 to 24.
   So frontier Gemini reads motion; cheap OpenAI and small open models mostly do not.
4. **Verdict on "VLMs can't count flips/twists":** flips **WEAKENED** (close to overturned for clean clips with
   dense frames and a frontier model); twists **WEAKENED, weakly** (leaning UNKNOWN). Reasons in §7.
5. **Grid vs native video is untested for this task.** No 2025 to 2026 paper compares frame grids to native video
   for fast-motion counting on frontier models. Claude has no native video path anyway. The cheapest way to
   answer it is a Gemini probe that sends the same frames both ways (§8).
6. **Cost:** with frame grids, 1,618 clips × 2 votes costs ~$10 to $30 on Gemini Flash / Flash-Lite
   (~$5 to $15 with batch), ~$84 on Gemini 3.1 Pro (~$42 batch), ~$150 to $250 on Claude Opus 5.5 (~$75 to $125
   batch). Gemini Flash free tier can in principle cover it at $0, but per-model free RPD for 3.x models is not
   published (only visible inside AI Studio) (§6).

---

## 1. Model landscape as of 2026-09-28

### 1.1 Proprietary

| Model (API id) | Released | Input mode for video | Frame sampling | $/1M in / out | Free API tier | Grade |
|---|---|---|---|---|---|---|
| Gemini 3.8 Flash | GA 2026-09-02 | **Native video** (mp4 / YouTube URL), plus images | Default **1 fps**; custom `fps` in `videoMetadata`; clip offsets; `media_resolution` 70 tok/frame (low/medium) or 280 (high) | $0.75 / $3.75 (promo to 2026-12-31) | Yes | C |
| Gemini 3.7 Flash | 2026, between Jul 21 and Sep 2 (inferred; exact date not found) | Native video | same | $0.75 / $3.75 | Yes | C |
| Gemini 3.6 Flash | GA 2026-07-21 | Native video | same | $0.75 / $3.75 | Yes | C |
| Gemini 3.5 Flash | GA 2026-05-19 | Native video | same | $1.50 / $9.00 | Yes | C |
| Gemini 3.5 Flash-Lite | GA 2026-07-21 | Native video | same | $0.30 / $2.50 | Yes | C |
| Gemini 3.1 Flash-Lite | GA 2026-05-07 | Native video | same | $0.25 / $1.50 | Yes | C |
| Gemini 3 Flash | Preview | Native video | same | not re-checked | not re-checked | C |
| **Gemini 3.1 Pro Preview** | Preview (model card Feb 2026) | Native video | same | $2.00 / $12.00 (≤200k) | **No** | C |
| Gemini 3.5 Pro | **Not released** | n/a | n/a | n/a | n/a | C |
| Gemini Omni Flash | 2026 | Video **generation/editing** model (any-to-video), not a video-understanding API | n/a | n/a | n/a | C |
| GPT-6 Astra | 2026-09-03 | **Images only**; mp4 rejected; send frames with timestamps | Your choice; Roboflow tested up to 100 frames/request, ~$1 per 100 frames | $10 / $50 | No | C |
| GPT-6 Sol / GPT-6 Luna | 2026-09-22 | Images only | Your choice | $2 / $10 ; $0.10 / $0.50 | No | C |
| GPT-5.6 Sol / Terra / Luna | GA 2026-07-09 | Images only | Your choice | $4/$20 ; $2/$12 ; $0.20/$1.20 | No | C |
| Claude Opus 5.5 (`claude-opus-5-5`) | 2026-09-22 | **Images only** (no video) | Your choice. Up to **600 images/request** on 1M-context models (100 on 200k models); >20 images ⇒ each ≤2000 px; 32 MB request cap; cost = ⌈w/28⌉×⌈h/28⌉ tokens | $4 / $20 | No API free tier found | C |
| Claude Fable 5.1 | 2026 (exact date not found) | Images only | same | $10 / $50 | No | C |
| Claude Sonnet 5 / Haiku 4.5 | 2026 / 2025 | Images only | same (Haiku: 200k ctx ⇒ 100 images) | $2/$10 ; $1/$5 | No | C |

Notes:
- Gemini media-resolution tokens (Gemini 3 docs): video frame 70 tokens (low/medium) or 280 (high); image
  1,120 tokens by default (280/560/1,120/2,240). Audio adds 32 tokens/s. A 4 s clip at 10 fps high-res is
  ~11.3k tokens; at default resolution ~2.9k (C).
- Maximum Gemini `fps` is **not stated** in the current Gemini API docs (not found). 2025 Vertex AI docs gave an
  upper bound of 24 fps; unverified for 3.x (C). The docs themselves warn "fast action sequences might lose detail
  due to the 1 FPS sampling rate" (C).
- Gemini "agentic video understanding" (2026-09-01, Flash 3.5-Lite and later) lets the model pick frames and
  frame rate itself. Untested for motion; it targets long videos (C).
- Batch APIs: Gemini 50% off (24 h target), Anthropic 50% off, OpenAI batch "half price" (C).
- Claude's multi-image limit was 100 frames in PushupBench (late 2025); the current docs allow 600 on 1M-context
  models (C).
- Gemini free tier: content "may be used to improve Google products". EEA/UK/CH terms require **paid** services
  only when an API client is *made available to end users* in those regions; internal labelling by a France-based
  developer does not appear to be that case, but verify the ToS (C).

### 1.2 Open weights

| Model | Released | Video input | Default fps | RTX 2060 (6 GB) | Mac (MLX) | Grade |
|---|---|---|---|---|---|---|
| Qwen3.5 0.8B / 2B / 4B / 9B (Apache-2.0) | 2026-02 | Native (unified VLM) | **2 fps**, configurable (`fps`, `do_sample_frames`, vLLM `mm_processor_kwargs`) | 0.8B/2B yes; 4B only at 4-bit GGUF with mmproj offloaded (llama.cpp); 9B no | Yes via mlx-vlm (9B 4-bit on a 16 GB Mac plausible) | C |
| Qwen3.5 27B / 35B-A3B / 122B-A10B / 397B-A17B | 2026-02 | Native | 2 fps | No | 35B-A3B 4-bit ~20.5 GB peak ⇒ ≥24 GB Mac | C |
| Qwen3.8-27B (open) | 2026-08-14 | Wikipedia summary says video yes; Qwen3.8 GitHub README does not mention video (conflict, unverified) | not found | No | 4-bit on ≥24 GB Mac | C |
| Qwen3-VL 2B/4B/8B/32B/235B | 2025-09/10 | Native | 2 fps | 2B/4B at 4-bit | Yes | C |
| Qwen3.6/3.7-Plus, Qwen3.8-Max | 2026 | Proprietary API (DashScope); Qwen3.7-Plus image/video in, ~$0.40 / $1.60 per 1M | n/a | n/a | n/a | C |
| Gemma 4 E2B / E4B / 12B / 26B-A4B / 31B (Apache-2.0) | 2026 | Video as frame sequences (audio on E2B/E4B) | not found | E2B/E4B at 4-bit | Yes | C |
| Molmo2 4B / 8B (Ai2, fully open) | 2026-01 (CVPR 2026) | Native video + pointing/tracking | VideoNet FT used 4 fps, ≤64 frames | 4B at 4-bit (untested) | Yes | C |
| InternVL3.5 1B to 241B | 2025-08 | Frames (uniform, ≤48 in VideoNet eval) | n/a | small sizes only | Yes | C |
| MiniMax M3 | 2026-06 | Native video (large MoE) | not found | No | No | C |

Alibaba Model Studio (Singapore endpoint only) gives new accounts ~1M free tokens per model for 90 days (C).
Enough for a 12-clip probe, not for bulk.

---

## 2. Benchmarks on fine-grained motion (newest numbers)

| # | Benchmark / paper | Date | Task | Newest numbers relevant here | Frame budget | Grade | Relevance to PkVision |
|---|---|---|---|---|---|---|---|
| M1 | **PkVision July probe** (memory: project_claude_judge_probe) | 2026-07-12 | Claude (Fable 5, 24-frame 2×(3×4) grids, 300 px tiles, per-frame orientation-tracking prompt) on 12 curated gold clips vs corrected refs | Run 1: flip **11/11**, twist **10/11** (incl. a triple full), direction 11/11, context 10/11. Run 2: 9/10, 8/10, 10/10, 9/10. Trick name ~10/12. 99-clip gold audit: vote-vote agreement 96% flip / 93% twist; vs manifest 83% / 65% | 24 frames over the whole clip (~4 to 12 fps for 2 to 6 s clips) | **A** (n=11, curated, refs corrected by FIG CoP + name semantics, no human-verified truth yet) | Direct. 95% exact CI: flip 11/11 ⇒ [71.5%, 100%]; twist 10/11 ⇒ [58.7%, 99.8%] |
| M2 | **VideoNet** (arXiv 2605.02834) | 2026-05-04 | 4-way trick-name MCQ, 1,000 actions / 37 domains; **Parkour domain = 40 actions, 200 clips, avg 4.4 s** | Overall: Gemini 3.1 Pro 69.9, Gemini 3 Flash 68.7, GPT-5.4 68.0, GPT-5 67.6, Qwen3-VL-8B 45.0, Molmo2-8B 44.9. "Hobbies" category (parkour + skateboarding + 6 others): Gemini 3.1 Pro 61.9, Gemini 3 Flash 63.1, GPT-5.4 60.3. **Parkour domain, Molmo2-4B: base 46.9% → fine-tuned 49.4 to 56.9% depending on data filter** (160 Qs) | **1 fps** for Gemini and GPT, 2 fps Qwen, 4 fps Molmo2-FT | **A** for the Molmo2-4B parkour row (trick-name MCQ, not counting); **B** for frontier category rows (parkour not broken out: not found) | Frontier models at 1 fps get ~62% on 4-way hobby-trick MCQ. Fine-tuning a 4B open model on 162k auto-labelled clips adds ~10 pts on parkour |
| M3 | **MotionBlind** (arXiv 2609.09528) | 2026-09-08 | Contrastive 2×2 yes/no on self-recorded clips (avg 5.9 s): speed, magnitude, translational and **rotational direction** | Instance acc (chance 6.25%): **Gemini 3.1 Pro 60.0** (item acc 80.8), **GPT-5.6 Luna 15.0**, Eagle2.5-8B 11.7, Inkling-Small 10.0, Qwen3-VL-4B 8.3, Motion-o 6.7, Gemma-4-12B 3.6, Molmo2-8B 3.3; human 91.3. Gemini clears 3 of 4 categories (incl. rotational direction; per-category number only in a figure) but gets **14% on speed**. Open models: speed and magnitude at 0; frames plateau at N=16 to 24; learned frame selectors do not beat uniform | 1 to 24 frames | B | Strongest recent evidence that only frontier Gemini reads physical motion. Small open VLMs are unlikely to count twists zero-shot |
| M4 | **TimeBlind** (arXiv 2602.00288) | 2026-02 | Contrastive temporal compositionality incl. kinematics (direction, repetition, speed) | Gemini 3 Pro **48.2** I-Acc, GPT-5 46.3 (MotionBlind's table lists GPT-5.6 Luna at 46.3 / 77.3, identical to GPT-5's published numbers, so possibly reused), Molmo2-8B 31.2, Qwen3-VL-235B 25.8; human 98.2. Event-attribute (kinematics) category: Gemini 3 Pro 36.7, GPT-5 32.3. 8→32 frames: +1 to 5 pts; thinking: +3.3 (GPT-5), +10.4 (Qwen3-VL-235B) | **1 fps** default | B | "Direction/repetition" are hard even for frontier models at 1 fps |
| M5 | **PushupBench** (arXiv 2604.23407) | 2026-04-25 | Rep counting, 446 clips, avg 36.7 s | Exact / MAE / R²: **Gemini 3 Flash 42.1 / 2.9 / 0.82**, Gemini 3 Pro 39.8 / 3.6 / 0.70, Gemini 2.5 Pro 29.9, GPT-5 10.9 / 7.6 / 0.01, **Claude Sonnet 4.5 9.5**, **Claude Opus 4.5 4.9**, Qwen3-VL-4B 8.9 (R² −0.21, collapses to "10"), TransRAC 6.7. FPS ablation (Gemini 3 Flash): 1 fps 17.4%, 2 fps 36.8%, **5 fps 42.1%**, 10 fps 40.5%. In RL rollouts genuine counting was 31% for GT 1 to 5 and ~0% for GT > 15 (Qwen3-VL-4B) | 5 fps, ≤112 frames (Claude ≤100) | B | Verified the July citation. Regime: long clips, high counts. Shows fps matters a lot (1→5 fps: 17→42%) |
| M6 | VideoZeroBench (arXiv 2604.01569) | 2026-04-02 | Long-video QA with spatio-temporal evidence | Gemini 3 Pro < 17% at end-to-end QA; < 1% when grounding required | long videos | B (weak analog) | Not about motion counting; should not be cited against flip counting |
| M7 | "Can VLMs Judge Action Quality?" (Sword Health, arXiv 2604.08294) | 2026-04-09 | **Execution quality** (FineFS GOE, MTL-AQA diving execution, fitness errors) | Gemini 3.1 Pro, Qwen3-VL-235B, InternVL3.5-241B "marginally above random". FineFS ρ: Gemini 0.27, Qwen3-VL-Thinking 0.28; MTL-AQA ρ: Gemini 0.06. Cropping: small inconsistent gains; skeleton overlays: rarely help; skeleton-only renders: ρ→~0 | model default (InternVL ≤120 frames) | B | About scoring *how well*, not *how many*. Relevant only to the later E-score layer of a judge-assist |
| M8 | Can VLMs Judge Olympic Diving? (arXiv 2609.19354) | 2026-09-16 | Zero-shot AQA on AQA-7 10 m platform, open VLMs ≤32B | Standalone Spearman ≤ 0.32 (Qwen2.5-VL-32B best); regression on VLM text features from 4 VLMs → 0.67 | native video | B | Again execution score, older small open models |
| M9 | VidOmni-Bench (arXiv 2609.21521) | 2026-09-18 | Verify each sentence of a dense caption (500 videos, 4 s to 90 min) | F1: Gemini 3 Flash 41.7 (video+audio), 39.8 (video); GPT-5 mini 28.9; Qwen3-VL-32B 21.7. **Self-verification bias:** Gemini's F1 on its own captions ≈ half; cross-model ensembles fix part of it | model default | B | Supports cross-model votes over same-model votes |
| M10 | Molmo2 (arXiv 2601.10611) | 2026-01 | Video **object** counting | Molmo2-VideoCount: Molmo2-8B 49.7, Gemini 3 Pro 54.6, Qwen3-VL-8B 47.0 (from search snippets; not re-read) | n/a | B− | Object counting, not rotation counting |
| M11 | "Which way did it move?" (arXiv 2605.22823) | 2026-05-21 | Signed image-plane motion direction | Most Video-LLMs near chance; the signal exists in hidden states but the readout fails ("direction binding gap"); projector-level fix: 25.9 → 85.4% synthetic, +21.9 pts real | n/a | B | Forward/backward direction is a known weak spot for open models |
| M12 | FSBench (arXiv 2504.19514) | 2025-04 | Figure-skating QA | No frontier-model jump-rotation accuracy reported (not found) | n/a | n/a | Gap |

Not found: any 2025 to 2026 benchmark reporting frontier-VLM accuracy on **counting somersaults or twists**
(gymnastics, diving, trampoline, figure-skating jump revolutions, freestyle ski, parkour). FineGym and FineDiving
have no published frontier-VLM evaluations that I could find (arXiv full-text search for FineGym or somersault ×
VLM/MLLM since 2025 returned 0 hits).

General video leaderboards (C, not motion): Video-MMMU top scores Qwen3.8-Max 88.7, Gemini 3 Pro 87.6,
Claude Opus 4.5 84.4 (BenchLM, Sep 2026). No Video-MMMU numbers published for Claude Opus 5.5 or Fable 5.1
(not found). These measure knowledge from lecture videos and say little about rotation counting.

---

## 3. Input and prompting techniques

| Technique | Evidence | Effect | Grade |
|---|---|---|---|
| **Higher fps / more frames** | PushupBench fps ablation (Gemini 3 Flash): 1→2→5→10 fps = 17.4→36.8→42.1→40.5% exact. Nyquist argument: need ≥2 frames per cycle | Large gain up to "enough frames per cycle", then flat or slightly worse | B |
| | MotionBlind: open models plateau at N=16 to 24; TimeBlind: 8→32 frames +1 to 5 pts | More frames cannot fix a model that does not encode motion | B |
| | VideoNet / TimeBlind / Gemini default all run at 1 fps ⇒ ~4 frames per 4 s parkour clip | Benchmarks at 1 fps structurally under-sample flips; their low scores are not evidence against dense-frame counting | B (inference) |
| **Frame grids vs native video** | IG-VLM (arXiv 2403.18406, 2024): a single 6-frame grid image beat video-LLMs on 9/10 zero-shot VideoQA benchmarks. Video Panels (arXiv 2509.23724, 2025): panels trade spatial detail for temporal coverage, up to +19.4% on TimeScope-Long | Grids are competitive for general/long VideoQA with older models | B |
| | Head-to-head grid vs native video for **fast motion counting on 2026 frontier models** | **Not found** | n/a |
| | Token math: a 3×4 grid of 300 px tiles = 1,419 Claude tokens (~118/tile) or 1,120 Gemini tokens (~93/tile) vs Gemini native 70 (default) or 280 (high) tokens/frame | A grid is not "more pixels per frame" than native video at default resolution. What it adds is all frames side by side in one image | C (arithmetic from docs) |
| **Slow-motion / re-timing** | No study found. Possible lever for Gemini if fps is capped: re-encode the airborne second at 0.25× so 24 fps sampling = ~96 fps effective | Unknown | not found |
| **Temporal crop to the air phase** | No VLM study found; follows from the fps evidence (spend the frame budget where rotation happens) | Likely positive; untested | C |
| **Crop to athlete** | Sword Health: "often yields small gains, inconsistent across models" (AQA) | Small, inconsistent | B |
| **Pose / skeleton overlay** | Sword Health: overlays rarely help classification, some rank-correlation gain in regression; **skeleton-only renders destroy performance** | Neutral to slightly positive; never replace the RGB | B |
| **Trajectory overlay** | Motion-as-Prompt (arXiv 2608.11655): dense point tracks drawn across sampled frames: **+4.2 (CLEVRER), +8.9 pts (SSv2) for GPT-5.5**, no loss on non-motion QA | Positive for motion reasoning | B |
| **Chain-of-thought / thinking** | "Look Light, Think Heavy" (arXiv 2606.22565): CoT can *reduce* object counting and grounding. Forced CoT no gain on Video-MME (arXiv 2606.22862, Qwen2.5-VL). VideoAuto-R1 (arXiv 2601.05175): direct answer often matches CoT on perception. TimeBlind: thinking +3.3 (GPT-5), +10.4 (Qwen3-VL-235B). Sword Health: structured reasoning / two-step observation no consistent gain | Generic CoT: neutral to negative on perception. Small gains for temporal reasoning | B |
| **Task-specific per-frame decomposition** ("track body orientation frame by frame, count inversions") | Used in the July probe with good results, but never ablated against a plain prompt | Unknown contribution | A (no ablation) |
| **Per-phase descriptions** (take-off / flight / landing) | Diving AQA paper: phase-structured prompt P2 best of three for scoring; Sword Health: little effect | Mixed, and only for scoring | B |
| **Multi-vote self-consistency** | July: two Claude runs agree 93 to 96%; 3rd vote on disagreement (A). VidOmni: models are worse at verifying their own outputs; **cross-model** ensembles help (B). Diving: multi-VLM ensembles 0.47→0.67 ρ, but via a supervised regressor (B). Same-model majority vote on counting: **not found** | Prefer cross-model votes; high same-model agreement can hide correlated errors | A/B |
| **In-context examples** | VideoNet: few-shot helps Qwen (+7.0) but hurts Gemini 3.1 Pro (−4.8). Sword Health: ICL helps regression, not video classification | Model-dependent | B |
| **Name leakage / on-screen text** | PushupBench: models read on-screen counters instead of counting (hacking rises with GT count) | Strip filenames, titles, captions, watermarks with trick names before labelling | B |

---

## 4. Fine-tuning small open VLMs on weak labels

| Evidence | Numbers | Grade |
|---|---|---|
| VideoNet: Molmo2-4B fine-tuned on 162k automatically labelled domain-specific clips | Overall MCQ 42.0 → 53.5 (+11.5), beats every open 8B model; **parkour domain 46.9 → 56.9**. Smaller, cleaner data beat the 496k noisier set (48.2) | A (parkour row) / B |
| PushupBench: Qwen3-VL-4B-Thinking, DAPO RL on 391 curated counting clips, 4×H100, 144 steps | Exact 8.2 → 14.5%, R² −0.34 → 0.38. 968 noisy samples were worse than 391 clean ones. Reward hacking via mode collapse, frame-count leakage, on-screen text | B |
| Efficient Reasoning Distillation (arXiv 2609.16255) | 2B video VLM fine-tuned on ~900 examples with teacher CoT, **< 2 h on one A100**, beats models up to 4× larger | B |
| Unsloth Qwen3.5 docs | bf16 LoRA VRAM: 0.8B 3 GB, 2B 5 GB, **4B 10 GB**, 9B 22 GB, 27B 56 GB. QLoRA (4-bit) **not recommended** for Qwen3.5. Free Colab notebooks for 0.8B/2B/4B vision, 4B GRPO. Video fine-tuning not mentioned | C |
| Amateur-volleyball study (arXiv 2609.28049) | A documented LoRA failure: loss over the whole sequence + frozen vision tower converged below baseline. A properly fine-tuned VLM reader "almost never declines to answer" | B |
| Breakdance classification (arXiv 2510.20287) | Video encoders still beat VLMs for move classification | B |
| HieroAction (arXiv 2508.16942), SportR (arXiv 2511.06499) | Sports VLMs with stepwise reasoning + RL; SFT/RL improves but scores stay low | B |

Practical read: a Qwen3.5-4B or Molmo2-4B LoRA on 1,618 weakly labelled clips is cheap
(free Colab T4 or ~1 to 3 h on a rented 4090 at the $0.29 to $0.69/h seen in July; my estimate, C). The RTX 2060
is Turing: no bf16, 6 GB, so realistically 0.8B only. But MotionBlind (small open models at chance on motion) and
the breakdance result suggest the weak labels are better spent on the skeleton / video-encoder classifier the repo
already has than on a small VLM student. No 2026 paper fine-tunes a small VLM for flip/twist counting (not found).

---

## 5. Frame-budget arithmetic for twists (why twists are harder)

Back-of-envelope, C (airtime numbers are my estimates, not sourced): a standing backflip spends roughly
0.6 to 0.9 s airborne. A double full is ~2 longitudinal revolutions in that window (~2.5 to 3.5 rev/s); a triple
full ~3.5 to 5 rev/s. Resolving **half**-twists needs ≥2 frames per half revolution, i.e. ≥4 per revolution.

| Sampling | Frames in ~0.8 s air phase | Frames per twist rev at 3 rev/s | Resolves half-twists? |
|---|---|---|---|
| 1 fps (Gemini default, VideoNet, TimeBlind) | ~1 | 0.3 | No |
| 24 frames over a 4 s clip (July probe) | ~5 | 2 | Borderline for doubles, no for triples |
| 24 frames over the air phase only | 24 | 10 | Yes |
| Native Gemini at 24 fps | ~19 | 8 | Yes (if 24 fps is allowed) |

So the July triple-full hit was probably not frame-by-frame counting. More likely the model recognised the trick
gestalt and used its vocabulary knowledge (trick_guess was right on ~10/12). That still helps the A′ decoder,
but it will fail exactly on the unseen tricks that open-vocabulary V1 targets. Flips (~1 to 1.5 rev/s) are much
easier to sample.

---

## 6. Cost model: 1,618 clips × 2 votes = 3,236 calls (+~15% tiebreak votes)

Assumptions (C): ~1k prompt tokens; **grid** = 2 images of 1,200×900 (Claude 2×1,419 tokens, Gemini 2×1,120,
OpenAI ~1.5k each) ⇒ ~4k input; **native video / 24 separate frames** ⇒ ~10k input; output incl. thinking
1.5k tokens (typical) or 3k (heavy). Prices from vendor pages above. Batch = 50% off.

| Model | Grid, 1.5k out: per call → 3,236 calls | Grid, 3k out → 3,236 | Video or 24 frames (10k in), 1.5k out → 3,236 | With batch (grid, 1.5k) |
|---|---|---|---|---|
| Gemini 3.1 Flash-Lite | $0.0032 → **$10** | $18 | $15 | ~$5 |
| Gemini 3.5 Flash-Lite | $0.0049 → **$16** | $28 | $22 | ~$8 |
| Gemini 3.8 / 3.7 / 3.6 Flash | $0.0086 → **$28** | $46 | $43 | ~$14 |
| Gemini 3.5 Flash | $0.0195 → $63 | $107 | $92 | ~$32 |
| **Gemini 3.1 Pro Preview** | $0.026 → **$84** | $142 | $123 | ~$42 |
| Claude Haiku 4.5 | $0.0115 → $37 | $62 | $57 | ~$19 |
| Claude Sonnet 5 | $0.023 → $74 | $123 | $113 | ~$37 |
| **Claude Opus 5.5** | $0.046 → **$149** | $246 | $227 | ~$75 to $123 |
| Claude Fable 5.1 | $0.115 → $372 | $615 | $566 | ~$186 |
| GPT-6 Luna | $0.0011 → $4 | $6 | $6 | ~$2 |
| GPT-5.6 Luna | $0.0026 → $8 | $14 | $12 | ~$4 |
| GPT-6 Sol | $0.023 → $74 | $123 | $113 | ~$37 |
| GPT-6 Astra | $0.115 → $372 | $615 | ~$0.24+/call with 24 frames at ~$1/100 frames | ~$186 |
| Qwen3.7-Plus (DashScope) | $0.0040 → $13 | $21 | $21 | n/a |
| Local Qwen3.5-4B/9B | $0 (hours of GPU time) | | | |

Per budget tier (2 votes + ~15% tiebreaks):

| Tier | What fits | Caveat |
|---|---|---|
| **Free** | Gemini 3.x Flash / Flash-Lite free tier ($0); Claude Max quota (no marginal $); local Qwen3.5 | Gemini 3.x free RPD is **not published** (AI Studio only). Third-party figures for older Flash free tiers: ~250 to 1,500 RPD, 10 to 15 RPM (C). At 250 RPD, 3,700 calls ≈ 15 days; at 1,500 RPD ≈ 2.5 days. Free-tier data may be used by Google. Claude Max ran out in July: the July harness used ~46k tokens per clip-vote (550k per 12 clips) via subagents, versus ~5.5k for a lean single call, so a lean `claude -p` harness could stretch the same quota several times further (C, estimate). Local open models are near chance on motion (MotionBlind) |
| **≤ $20** | Gemini 3.8 Flash batch (~$16 incl. tiebreaks, grid) or Flash-Lite (~$6 to $10); Claude Haiku 4.5 batch (~$22) | Flash-tier and Haiku accuracy on this task: unmeasured |
| **≤ $50** | Gemini 3.1 Pro batch (~$48), Claude Sonnet 5 batch (~$43), GPT-6 Sol batch (~$43) | Only Gemini 3.1 Pro has a physical-motion benchmark result (MotionBlind 60%) |
| **≤ $150** | Claude Opus 5.5 batch (~$86 to $140); Gemini 3.1 Pro non-batch (~$97); **cross-model ensemble: 1 vote Gemini 3.1 Pro + 1 vote Opus 5.5, batch ≈ $21 + $37 to $62, ≈ $60 to $95 with tiebreaks** | Cross-model is the design VidOmni supports |

Free tiers alone can cover the bulk run only via Gemini Flash-tier models, and only if their daily quota is known and
the owner accepts Google's free-tier data terms. Whether Flash-tier accuracy is good enough is exactly what the
probe must measure.

---

## 7. Verdicts and reconciliation

**Claim: "VLMs can't count flips/twists."**

| Attribute | Verdict | Why |
|---|---|---|
| **Flips** | **WEAKENED** (close to overturned for clean single-trick clips) | A: 11/11 and 9/10 on curated clips (CI lower bound ~72%); 96% vote-vote agreement on the 99-clip audit. B: Gemini 3.1 Pro clears MotionBlind incl. rotational direction; PushupBench counts improve sharply with fps. Not overturned because n=11, curated, one model family, no human-verified yardstick, no external flip-counting benchmark exists |
| **Twists** | **WEAKENED, weakly** (leaning UNKNOWN) | A: 10/11 and 8/10 (CI lower bound ~59%); vote-vote 93% but manifest agreement 65% with no human truth yet. B: even Gemini fails speed on MotionBlind; open models fail all rate-based motion. Physics: at 24 frames over a whole clip, multi-twists are under-sampled (§5), so correct triple-full answers likely came from trick-gestalt priors, not counting |
| Direction (bonus) | WEAKENED | A: 11/11. B: directional motion blindness is documented for open models (2605.22823), Gemini 3.1 Pro handles direction in MotionBlind |

**Why the benchmarks and the in-house 11/11 disagree.** In order of likely weight:
1. **Different task regime (B).** PushupBench asks for 10 to 20+ reps over ~37 s; genuine counting collapses above
   ~10. Parkour asks for 0 to 3 rotations in ~1 s. VideoZeroBench and the Sword Health paper measure other things.
2. **Frame budget (B).** TimeBlind, VideoNet and Gemini's default use 1 fps (~4 frames per parkour clip). The July
   probe used 24 frames (~4 to 12 fps). PushupBench itself shows 1 fps → 5 fps more than doubles exact accuracy.
3. **Easy clips (A, self-reported).** The 12 were the cleanest curated clips; competition and handheld footage
   will be harder. The gold set itself was ~17% wrong on flip and ~35% on twist, so earlier "failures" were
   partly label errors.
4. **Priors / trick recognition (inference).** Trick names were guessed right ~10/12; counts can be read off a
   recognised name. Works for known tricks, not for novel ones.
5. **Newer models (C).** PushupBench's Claude numbers are Sonnet/Opus 4.5 (late 2025); the probe used Fable 5 /
   Opus 4.8 (2026). Anthropic claims large vision gains since (Opus 5.5 Chartography 64.4 vs Opus 5 29.8, C).
6. **Grid format: unknown.** No evidence either way for fast-motion counting. It cannot be the whole story because
   the same frames could be sent natively to Gemini; that is the experiment to run.

Answer to "is it the grid or are the clips easy?": most likely **dense frames + clean clips + trick priors**; the
grid's own contribution is **UNKNOWN**.

---

## 8. Micro-probe designs (≤ 12 clips, ≤ $5 each)

Shared setup for all three (so results are comparable):
- **Clips:** 12, with owner-verified labels (take them from the 26-clip dispute queue after the owner checks
  them, plus clean clips as needed). Stratify: 4 single flips (incl. 1 side/wall context), 4 multi-twist
  (≥ double full, incl. 1 cork/off-axis), 4 hard footage (handheld, low-res, competition). Same 12 for every model.
- **Hygiene:** strip filenames, titles, captions, on-screen text. No trick names in the prompt.
- **Frames:** trim to take-off −0.3 s … landing +0.3 s. Two frame budgets: 24 frames over that window
  (grid 2×(3×4), 300 px tiles) and native video.
- **Prompt:** the July method prompt (per-frame body orientation, count inversions and half-twists) with the
  July JSON schema {flip, twist, direction, axis, context, trick_guess, confidence}.
- **Sham control:** the same clip **reversed** (1 vote). A model that reads motion must flip forward↔backward
  and keep counts. If direction stays the same on reversed input, it is answering from priors (MotionBlind-style
  integrity probe).
- **Pre-registered pass bar (suggested):** flip ≥ 10/12, twist ≥ 9/12, reversed-clip direction flips ≥ 10/12.

| # | Model | Input format | Prompt strategy | Votes / calls | Est. cost |
|---|---|---|---|---|---|
| P1 | **Gemini 3.1 Pro Preview** (only model with a physical-motion win: MotionBlind 60%, TimeBlind 48.2% as Gemini 3 Pro, VideoNet 69.9%) | Native video of the air-phase window, `fps` at the highest accepted value (try 24; fall back to 0.25× slow-motion re-encode at a lower fps), `media_resolution` high | Method prompt, thinking at default. No CoT forcing | 12 × 2 votes native + 12 × 1 grid + 12 × 1 reversed = 48 calls | ~$1.5 to $2.5 |
| P2 | **Gemini 3.8 Flash** (free tier; fall back to 3.6 Flash) | **Same 24 frames sent both ways**: native video at matched fps vs 2×(3×4) grid | Identical prompt in both arms | 12 × 2 formats × 2 votes + 12 reversed = 60 calls | $0 free tier (~$0.6 paid) |
| P3 | **Claude Opus 5.5** (July incumbent; tests whether the July result survives harder clips and a new model) | Grid of 24 frames over the air phase, plus a dense arm of 48 frames (4 grids) | July method prompt, default effort | 12 × 2 votes (24-frame grid) + 12 × 1 (48-frame) + 12 × 1 reversed = 48 calls | ~$2.5 to $4 API; $0 on Max when quota resets |

What each answers: P2 answers "grid vs native" at $0 and whether the cheapest bulk path is good enough. P1 gives
the best-available native-video ceiling. P3 separates "clips were easy" from "Claude is good". Together
P1 + P3 give a cross-model pair for bulk voting. Optional P4 ($0): Qwen3.5-9B on the Mac (mlx-vlm) or 4B on the
2060, native video at 8 fps. Expect near-chance (MotionBlind); only worth it as the open baseline.

---

## 9. Gaps (not found)

- Any benchmark of frontier VLMs on **somersault/twist counting** (FineGym, FineDiving, trampoline, skating jump
  revolutions, parkour). No FineGym or FineDiving frontier-VLM evaluation since 2025.
- VideoNet's **parkour-domain** numbers for Gemini/GPT (only category-level reported).
- Head-to-head **grid vs native video** for fast motion on 2026 models.
- Gemini **maximum fps** in current docs; Gemini 3.x **free-tier RPD** (AI Studio only).
- Any motion/video benchmark for **Claude Opus 5.5, Fable 5.1, Sonnet 5**, or for **Gemini 3.6/3.7/3.8 Flash**.
- Any benchmark for **GPT-6** Astra/Sol/Luna on motion (only GPT-5.6 Luna: MotionBlind 15%).
- Effect of **same-model multi-vote** on counting accuracy.
- An ablation of the July **method prompt** vs a plain prompt.
- Whether Qwen3.8-27B (open) accepts video (sources conflict).
- Human-verified accuracy of the July Claude labels (26-clip queue still unverified): this is the missing
  yardstick for every A-grade number above.

---

## 10. Sources

Official vendor pages (accessed 2026-09-28, C):
- Gemini API release notes / changelog: https://ai.google.dev/gemini-api/docs/changelog
- Gemini API models: https://ai.google.dev/gemini-api/docs/models
- Gemini API pricing: https://ai.google.dev/gemini-api/docs/pricing
- Gemini video understanding: https://ai.google.dev/gemini-api/docs/video-understanding
- Gemini media resolution: https://ai.google.dev/gemini-api/docs/media-resolution
- Gemini Batch API: https://ai.google.dev/gemini-api/docs/batch-api
- Gemini rate limits: https://ai.google.dev/gemini-api/docs/rate-limits ; billing: https://ai.google.dev/gemini-api/docs/billing
- Gemini Omni Flash: https://ai.google.dev/gemini-api/docs/models/gemini-omni-flash
- "Gemini 3.5: frontier intelligence with action", Google blog, 2026-05-19: https://blog.google/innovation-and-ai/models-and-research/gemini-models/gemini-3-5/
- Claude models overview: https://platform.claude.com/docs/en/about-claude/models/overview
- Claude vision docs: https://platform.claude.com/docs/en/build-with-claude/vision
- OpenAI API pricing: https://developers.openai.com/api/docs/pricing
- Qwen3.5-9B model card: https://huggingface.co/Qwen/Qwen3.5-9B
- QwenLM/Qwen3.8 README: https://github.com/QwenLM/Qwen3.8
- Unsloth Qwen3.5 fine-tuning guide: https://unsloth.ai/docs/models/qwen3.5/fine-tune
- Alibaba Model Studio new-user free quota: https://www.alibabacloud.com/help/en/model-studio/new-free-quota
- mlx-vlm: https://github.com/Blaizzy/mlx-vlm ; Qwen llama.cpp docs: https://qwen.readthedocs.io/en/latest/run_locally/llama.cpp.html
- Gemma 4 announcement: https://blog.google/innovation-and-ai/technology/developers-tools/gemma-4/

News / third-party (C):
- "Google releases three new Gemini models, but no 3.5 Pro", TechCrunch, 2026-07-21: https://techcrunch.com/2026/07/21/google-releases-three-new-gemini-models-but-no-3-5-pro/
- "Gemini 3.5 Pro delays due to coding performance", 9to5Google, 2026-07-16: https://9to5google.com/2026/07/16/gemini-3-5-pro-delays/
- "Gemini 3.5 Pro Release Date: Still Unreleased (Aug 2026)", Codersera: https://codersera.com/blog/gemini-3-5-pro-launch-guide-2026/
- "Google Delays Gemini 3.5 Pro Again", NokiaPowerUser, 2026-09: https://nokiapoweruser.com/gemini-3-5-pro-delayed-again-deployment-issues/
- GPT-5.6, Wikipedia: https://en.wikipedia.org/wiki/GPT-5.6 ; 9to5Mac 2026-07-09: https://9to5mac.com/2026/07/09/openai-announcing-the-next-chapter-for-chatgpt-today-watch-here/
- "OpenAI launches GPT-6 Sol and Luna", TechCrunch, 2026-09-22: https://techcrunch.com/2026/09/22/openai-launches-gpt-6-sol-and-luna/ ; GitHub changelog 2026-09-22: https://github.blog/changelog/2026-09-22-openais-gpt-6-sol-and-gpt-6-luna-now-available/
- GPT-6 Astra pricing and launch (Sep 3), Yotta Labs: https://www.yottalabs.ai/post/gpt-6-astra-pricing-api-cost-2026
- "Understanding Video with GPT-6 Astra", Roboflow blog, 2026-09-11: https://blog.roboflow.com/gpt6-astra-video-understanding/
- openai-node issue #1778 "Native video file input support in Responses API", opened 2026-03-18, closed as not planned: https://github.com/openai/openai-node/issues/1778
- Claude Opus 5.5 launch, MacRumors, 2026-09-22: https://www.macrumors.com/2026/09/22/anthropic-claude-opus-5-5/ ; OpenRouter/MindStudio pages on Opus 5.5 pricing and Chartography claim: https://www.mindstudio.ai/blog/claude-opus-5-5-release
- Qwen, Wikipedia: https://en.wikipedia.org/wiki/Qwen
- Gemini API free tier guide (updated 2026-08-15, EEA terms): https://www.aifreeapi.com/en/posts/google-gemini-api-free-tier
- Gemini free-tier limits (2026-05-07): https://tinkerllm.com/blog/gemini-api-free-tier-limits-rate-quotas/
- VideoMMMU leaderboard, BenchLM (Sep 2026): https://benchlm.ai/benchmarks/videommmu
- Molmo 2 blog, Ai2: https://allenai.org/blog/molmo2

Papers (B unless stated):
- PushupBench: "Your VLM is not good at counting pushups", Li et al., 2026-04-25, arXiv 2604.23407
- VideoZeroBench, Meng et al., 2026-04-02, arXiv 2604.01569
- "Can Vision Language Models Judge Action Quality? An Empirical Evaluation", Monte e Freitas et al. (Sword Health), 2026-04-09, arXiv 2604.08294
- "Can Vision-Language Models Judge Olympic Diving?", Velesaca et al., 2026-09-16, arXiv 2609.19354
- MotionBlind, Bhatia, Galoaa et al., 2026-09-08, arXiv 2609.09528
- TimeBlind, Li et al., 2026-02, arXiv 2602.00288
- VideoNet, Yadav et al., 2026-05-04, arXiv 2605.02834 (A for the parkour-domain Molmo2 rows)
- VidOmni-Bench, Kim et al., 2026-09-18, arXiv 2609.21521
- Molmo2, Clark et al., 2026-01, arXiv 2601.10611
- "Which Way Did It Move? Directional Motion Blindness in Video-LLMs", Lee et al., 2026-05-21, arXiv 2605.22823
- Motion-as-Prompt, Sun et al., 2026-08-12, arXiv 2608.11655
- IG-VLM "An Image Grid Can Be Worth a Video", Kim et al., 2024-03-27, arXiv 2403.18406
- Video Panels for Long Video Understanding, Doorenbos et al., 2025-09-28, arXiv 2509.23724
- "Look Light, Think Heavy", Jin et al., 2026-06-21, arXiv 2606.22565
- "Chains That See, Answers That Don't", Fan et al., 2026-06-22, arXiv 2606.22862
- VideoAuto-R1, Liu et al., 2026-01-08, arXiv 2601.05175
- Efficient Reasoning Distillation for small video VLMs, Singh et al., 2026-09-14, arXiv 2609.16255
- "Prompt, Probe, Train, or Annotate?" (amateur volleyball), Kodathala et al., 2026-09-23, arXiv 2609.28049
- HieroAction, Wu et al., 2025-08-23, arXiv 2508.16942
- SportR, Xia et al., 2025-11-09, arXiv 2511.06499
- Breakdance video classification, Dhar et al., 2025-10-23, arXiv 2510.20287
- FSBench, Gao et al., 2025-04-28, arXiv 2504.19514
- FineBench, Faure et al., 2026-05-19, arXiv 2605.19846 (checked: GPT-5 "respectable" on fine-grained human-activity QA; no rotation counting)

In-house (A): memory note `project_claude_judge_probe.md` (2026-07-12 probe and gold-99 audit); outputs in
`data/labeling/claude_probe/`.
