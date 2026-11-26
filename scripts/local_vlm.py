#!/usr/bin/env python3
"""A free local VLM arm for the blind probe: Qwen3-VL-4B-Instruct in 4-bit on the RTX 2060.

Exploratory only. MotionBlind puts models of this size near chance on physical motion, so its answers
carry no decision weight in the pre-registered probe; they test the pipeline with a real model and give a
$0 floor on the full dataset. Same frozen prompt and grids as the paid grid arms; greedy decoding.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
QUANT = "nf4"  # 4-bit NormalFloat, fp16 compute: a 6 GB card holds the 4B weights with room for two grids
MAX_NEW_TOKENS = 160
CACHE_DIR = Path(os.environ.get("PKVISION_HF_CACHE", Path.home() / "hf-cache"))


class LocalQwenProvider:
    """Same interface as the API providers in run_vlm_blind.py: ask(spec, paths) -> (text, usage)."""

    def __init__(self, model_id: str = MODEL_ID, cache_dir: Path = CACHE_DIR):
        import torch
        from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration

        self.torch = torch
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type=QUANT, bnb_4bit_use_double_quant=True,
                                   bnb_4bit_compute_dtype=torch.float16)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id, quantization_config=quant, dtype=torch.float16, device_map="cuda:0", cache_dir=str(cache_dir))
        self.processor = AutoProcessor.from_pretrained(model_id, cache_dir=str(cache_dir))
        self.model.eval()
        self.model_id = model_id

    def ask(self, spec: dict, paths: list[Path]) -> tuple[str, dict]:
        from PIL import Image

        if spec.get("input") != "grid":
            raise ValueError("the local arm reads grids only")
        images = [Image.open(p).convert("RGB") for p in paths]
        return self.ask_images(images)

    def ask_images(self, images: list, prompt: str | None = None) -> tuple[str, dict]:
        from scripts.run_vlm_blind import PROMPT

        content = [{"type": "image"} for _ in images] + [{"type": "text", "text": prompt or PROMPT}]
        text = self.processor.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
                                                  add_generation_prompt=True)
        inputs = self.processor(text=[text], images=images, return_tensors="pt").to(self.model.device)
        started = time.time()
        with self.torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
        n_in = int(inputs["input_ids"].shape[1])
        generated = out[:, n_in:]
        reply = self.processor.batch_decode(generated, skip_special_tokens=True)[0]
        usage = {"input_tokens": n_in, "output_tokens": int(generated.shape[1]), "seconds": round(time.time() - started, 2)}
        return reply, usage


def main() -> None:
    """Smoke test on one grid pair: python scripts/local_vlm.py grid1.jpg grid2.jpg"""
    from PIL import Image

    provider = LocalQwenProvider()
    reply, usage = provider.ask_images([Image.open(p).convert("RGB") for p in sys.argv[1:]])
    print(reply)
    print(usage)


if __name__ == "__main__":
    main()
