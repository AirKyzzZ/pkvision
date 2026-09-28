#!/usr/bin/env python3
"""Benchmark VLM models on parkour trick identification.

Tests multiple models on ground truth clips and produces an accuracy report.
Uses OpenRouter to access all models through one API.

Usage:
    # Benchmark default models on single-trick test clips
    python scripts/vlm_benchmark.py

    # Benchmark specific models
    python scripts/vlm_benchmark.py --models "qwen/qwen3.5-omni,google/gemini-2.5-pro"

    # Benchmark on custom clips with ground truth
    python scripts/vlm_benchmark.py --clips data/final_clips/backflip.mp4:Backflip data/final_clips/gainer.MOV:Gainer

    # Test consensus mode
    python scripts/vlm_benchmark.py --consensus
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from core.vlm.fig_matcher import FIGMatcher
from core.vlm.openrouter_provider import OpenRouterProvider, DEFAULT_BENCHMARK_MODELS
from core.vlm.consensus import ConsensusJudge

# Ground truth test set
DEFAULT_TEST_SET = [
    ("data/final_clips/backflip.mp4", "Backflip"),
    ("data/final_clips/gainer.MOV", "Gainer"),
    ("data/final_clips/double_cork.mp4", "Double Cork"),
    ("data/final_clips/frontflip.mp4", "Frontflip"),
    ("data/final_clips/back_double_full.mp4", "Backflip 720"),
]


def benchmark_models(
    test_set: list[tuple[str, str]],
    models: list[str],
    api_key: str | None = None,
):
    """Benchmark each model on the test set."""
    matcher = FIGMatcher()
    results = {}

    for model_name in models:
        print(f"\n{'='*60}")
        print(f"MODEL: {model_name}")
        print(f"{'='*60}")

        correct = 0
        family_correct = 0
        total = 0
        model_results = []

        for clip_path_str, ground_truth in test_set:
            clip_path = ROOT / clip_path_str
            if not clip_path.exists():
                print(f"  SKIP: {clip_path_str} (not found)")
                continue

            total += 1
            print(f"\n  Clip: {clip_path.name} (GT: {ground_truth})")

            try:
                provider = OpenRouterProvider(model=model_name, api_key=api_key)
                response = provider.analyze_trick(clip_path)

                if response.raw_text.startswith("ERROR:"):
                    print(f"    ERROR: {response.raw_text[:100]}")
                    model_results.append({
                        "clip": clip_path.name,
                        "ground_truth": ground_truth,
                        "prediction": "ERROR",
                        "correct": False,
                    })
                    continue

                # Take the first trick identified
                if response.tricks:
                    trick = response.tricks[0]
                    fig_match = matcher.match(
                        trick.trick_name,
                        flip_count=trick.flip_count,
                        twist_count=trick.twist_count,
                        direction=trick.direction,
                    )
                    pred_name = fig_match.fig_name if fig_match else trick.trick_name
                    d_score = fig_match.d_score if fig_match else 0.0

                    # Check exact match
                    gt_match = matcher.match(ground_truth)
                    gt_fig = gt_match.fig_name if gt_match else ground_truth

                    is_correct = pred_name.lower() == gt_fig.lower()
                    # Family match: same flip count and direction
                    gt_entry = gt_match
                    is_family = (
                        trick.flip_count == (gt_entry.d_score if gt_entry else 0)  # rough
                        or pred_name.split()[0].lower() == gt_fig.split()[0].lower()
                    )

                    if is_correct:
                        correct += 1
                        family_correct += 1
                        status = "EXACT"
                    elif is_family:
                        family_correct += 1
                        status = "FAMILY"
                    else:
                        status = "WRONG"

                    print(f"    Predicted: {pred_name} (D={d_score}) [{status}]")
                    print(f"    Reasoning: {trick.reasoning[:120]}")
                    print(f"    Tokens: {response.input_tokens}in + {response.output_tokens}out")

                    model_results.append({
                        "clip": clip_path.name,
                        "ground_truth": gt_fig,
                        "prediction": pred_name,
                        "vlm_raw": trick.trick_name,
                        "d_score": d_score,
                        "correct": is_correct,
                        "status": status,
                        "reasoning": trick.reasoning,
                        "tokens_in": response.input_tokens,
                        "tokens_out": response.output_tokens,
                    })
                else:
                    print(f"    No tricks identified!")
                    model_results.append({
                        "clip": clip_path.name,
                        "ground_truth": ground_truth,
                        "prediction": "NONE",
                        "correct": False,
                    })

            except Exception as e:
                print(f"    EXCEPTION: {e}")
                model_results.append({
                    "clip": clip_path.name,
                    "ground_truth": ground_truth,
                    "prediction": "EXCEPTION",
                    "correct": False,
                    "error": str(e),
                })

            # Rate limit courtesy
            time.sleep(1)

        accuracy = correct / total if total > 0 else 0
        results[model_name] = {
            "exact_accuracy": accuracy,
            "exact_correct": correct,
            "total": total,
            "details": model_results,
        }

        print(f"\n  RESULT: {correct}/{total} exact ({accuracy:.0%})")

    return results


def benchmark_consensus(
    test_set: list[tuple[str, str]],
    models: list[str],
    api_key: str | None = None,
):
    """Benchmark consensus mode."""
    judge = ConsensusJudge(models=models, api_key=api_key)
    matcher = FIGMatcher()
    correct = 0
    total = 0

    print(f"\n{'='*60}")
    print(f"CONSENSUS MODE: {len(models)} models")
    print(f"{'='*60}")

    for clip_path_str, ground_truth in test_set:
        clip_path = ROOT / clip_path_str
        if not clip_path.exists():
            continue

        total += 1
        print(f"\n  Clip: {clip_path.name} (GT: {ground_truth})")

        result = judge.judge(clip_path)

        gt_match = matcher.match(ground_truth)
        gt_fig = gt_match.fig_name if gt_match else ground_truth

        is_correct = result.consensus_name.lower() == gt_fig.lower()
        if is_correct:
            correct += 1

        status = "CORRECT" if is_correct else "WRONG"
        print(f"  → Consensus: {result.consensus_name} (D={result.d_score}) "
              f"[{result.confidence}, {result.agreement:.0%} agreement] {status}")
        print(f"    Votes: {[v.fig_name for v in result.votes]}")

    print(f"\n{'='*60}")
    print(f"CONSENSUS ACCURACY: {correct}/{total} ({correct/total:.0%})" if total else "No clips tested")


def print_summary(results: dict):
    """Print comparison table."""
    print(f"\n{'='*60}")
    print(f"BENCHMARK SUMMARY")
    print(f"{'='*60}")
    print(f"{'Model':<40} {'Accuracy':<12} {'Correct':<10}")
    print(f"{'-'*40} {'-'*12} {'-'*10}")

    ranked = sorted(results.items(), key=lambda x: x[1]["exact_accuracy"], reverse=True)
    for model, data in ranked:
        short = model.split("/")[-1] if "/" in model else model
        acc = f"{data['exact_accuracy']:.0%}"
        cnt = f"{data['exact_correct']}/{data['total']}"
        print(f"{short:<40} {acc:<12} {cnt:<10}")

    # Save detailed results
    out_path = ROOT / "data" / "vlm_benchmark_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed results saved to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark VLM models on parkour tricks")
    parser.add_argument("--models", default=None,
                        help="Comma-separated model list (default: top 5)")
    parser.add_argument("--clips", nargs="*",
                        help="Clips as path:label pairs (default: built-in test set)")
    parser.add_argument("--consensus", action="store_true",
                        help="Test consensus mode instead of individual models")
    parser.add_argument("--api-key", default=None,
                        help="OpenRouter API key (or set OPENROUTER_API_KEY env)")
    args = parser.parse_args()

    models = args.models.split(",") if args.models else DEFAULT_BENCHMARK_MODELS

    if args.clips:
        test_set = []
        for item in args.clips:
            parts = item.rsplit(":", 1)
            if len(parts) == 2:
                test_set.append((parts[0], parts[1]))
            else:
                print(f"Invalid clip format: {item} (expected path:label)")
                return
    else:
        test_set = DEFAULT_TEST_SET

    print(f"PkVision VLM Benchmark")
    print(f"Models: {len(models)}")
    print(f"Test clips: {len(test_set)}")

    if args.consensus:
        benchmark_consensus(test_set, models, args.api_key)
    else:
        results = benchmark_models(test_set, models, args.api_key)
        print_summary(results)


if __name__ == "__main__":
    main()
