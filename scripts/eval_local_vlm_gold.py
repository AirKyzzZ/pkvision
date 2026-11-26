#!/usr/bin/env python3
"""First trick-recognition numbers on the whole dataset, at $0: the local VLM against the gold attributes.

Runs Qwen3-VL-4B (4-bit, RTX 2060) with the probe's frozen prompt and frame grids over every cropped clip
that is NOT in the blind queue, then scores flip count, twist count and direction against the gold in
`data/v5_attribute_training/attribute_manifest_v2.json`. The gold is name-derived and noisy (about 17 % of
flip counts and 35 % of twist counts are wrong by the September audit), so agreement is a floor-and-ceiling
reading, not a verdict. Blind-queue clips are excluded here and go only through run_vlm_blind.py.

    python scripts/eval_local_vlm_gold.py run [--limit N] [--aerial-only]
    python scripts/eval_local_vlm_gold.py ground-truth      # the 5 owner-verified single-trick clips
    python scripts/eval_local_vlm_gold.py report            # accuracies, baselines, confusions
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.prepare_vlm_blind import CLIPS, QUEUE, grids, read_frames  # noqa: E402
from scripts.score_vlm_blind import parse_reply  # noqa: E402

GOLD = ROOT / "data/v5_attribute_training/attribute_manifest_v2.json"
SPLIT = ROOT / "data/splits/g1_name_split.json"
GROUND_TRUTH = ROOT / "paper/experiments/ground_truth.csv"
OUT_DIR = ROOT / "data/labeling/local_vlm"
PREDS = OUT_DIR / "gold_preds.jsonl"
GT_PREDS = OUT_DIR / "ground_truth_preds.jsonl"
ATTRIBUTES = ("flip", "twist", "direction")
BINS = ("0", "0.5", "1", "1.5", "2", "3+")


def bin_count(value) -> str | None:
    """Same bins as build_attribute_dataset.py; None for a missing or non-numeric answer."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if v <= 0:
        return "0"
    if v <= 0.75:
        return "0.5"
    if v <= 1.25:
        return "1"
    if v <= 1.75:
        return "1.5"
    if v <= 2.25:
        return "2"
    return "3+"


def norm_direction(value) -> str:
    return "none" if value in (None, "none", "null", "") else str(value).lower()


def signature(flip, twist, direction) -> str:
    """The trick as its three counted attributes, e.g. 'backward 1 flip 1 twist'. Unknown parts read '?'."""
    return f"{norm_direction(direction)} {bin_count(flip) or '?'} flip {bin_count(twist) or '?'} twist"


def load_gold() -> dict[str, dict]:
    data = json.loads(GOLD.read_text())
    entries = data["clips"] if isinstance(data, dict) and "clips" in data else data
    if isinstance(entries, dict):
        entries = list(entries.values())
    split = json.loads(SPLIT.read_text()) if SPLIT.exists() else {}
    design = {n.replace(" ", "_") for n in split.get("design", [])}
    gold = {}
    for e in entries:
        slug = e["slug"]
        if not (CLIPS / f"{slug}.mp4").is_file():
            continue
        gold[slug] = {"flip": e.get("flip"), "twist": e.get("twist"), "direction": norm_direction(e.get("direction")),
                      "context": e.get("context"), "family": e.get("family"), "source": e.get("source"),
                      "split": "design" if slug in design else "test"}
    return gold


def queue_slugs() -> set[str]:
    return {c["slug"] for c in json.loads(QUEUE.read_text())}


def is_aerial(g: dict) -> bool:
    return g.get("family") == "flip" or g.get("context") == "acrobatics"


def order(gold: dict[str, dict], excluded: set[str], aerial_only: bool = False) -> list[str]:
    """Aerial flips first, each group in a fixed pseudo-random order, blind-queue clips never."""
    keep = [s for s in gold if s not in excluded and (is_aerial(gold[s]) or not aerial_only)]
    key = lambda s: (not is_aerial(gold[s]), hashlib.sha1(s.encode()).hexdigest())  # noqa: E731
    return sorted(keep, key=key)


def read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def ask_clip(provider, path: Path) -> dict:
    from PIL import Image

    frames, _ = read_frames(path)
    images = [Image.fromarray(g[:, :, ::-1]) for g in grids(frames)]  # BGR -> RGB
    reply, usage = provider.ask_images(images)
    row = {"usage": usage}
    try:
        row.update(parse_reply(reply))
    except ValueError:
        row.update({"flip": None, "twist": None, "direction": None, "cork": None, "unsure": None, "raw": reply[:500]})
    return row


def run(slugs: list[str], gold: dict[str, dict], out: Path, limit: int | None = None, provider=None) -> dict:
    done = {r["slug"] for r in read_rows(out)}
    todo = [s for s in slugs if s not in done][:limit]
    summary = {"already": len(done), "made": 0, "answered": 0, "malformed": 0, "failed": 0}
    if not todo:
        return summary
    if provider is None:
        from scripts.local_vlm import LocalQwenProvider

        provider = LocalQwenProvider()
    out.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with out.open("a") as fh:
        for i, slug in enumerate(todo, 1):
            row = {"slug": slug, "gold": gold[slug], "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}
            try:
                row.update(ask_clip(provider, CLIPS / f"{slug}.mp4"))
                summary["malformed" if row.get("twist") is None and "raw" in row else "answered"] += 1
            except Exception as error:  # noqa: BLE001 - one bad clip must not stop the night run
                row["error"] = type(error).__name__ + ": " + str(error)[:200]
                summary["failed"] += 1
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            summary["made"] += 1
            if i % 25 == 0:
                rate = (time.time() - started) / i
                print(f"{i}/{len(todo)} clips, {rate:.1f} s/clip, {rate * (len(todo) - i) / 60:.0f} min left", file=sys.stderr)
    return summary


def ground_truth_clips(excluded: set[str]) -> list[dict]:
    """The owner-verified single-trick clips of ground_truth.csv that exist on disk and are not blind-queue clips."""
    rows = []
    for r in csv.DictReader(GROUND_TRUTH.open()):
        path = ROOT / r["clip_path"]
        if r["pool"] != "POOL-B" or not path.is_file() or path.stem in excluded or r["flip_count"] in ("", None):
            continue
        rows.append({"slug": path.stem, "path": str(path), "fig_name": r["fig_name"],
                     "gold": {"flip": float(r["flip_count"]), "twist": float(r["twist_count"]),
                              "direction": norm_direction(r["direction"]), "source": "owner"}})
    return rows


def run_ground_truth(out: Path, provider=None) -> list[dict]:
    clips = ground_truth_clips(queue_slugs())
    done = {r["slug"] for r in read_rows(out)}
    todo = [c for c in clips if c["slug"] not in done]
    if todo and provider is None:
        from scripts.local_vlm import LocalQwenProvider

        provider = LocalQwenProvider()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a") as fh:
        for c in todo:
            row = {"slug": c["slug"], "fig_name": c["fig_name"], "gold": c["gold"]}
            try:
                row.update(ask_clip(provider, Path(c["path"])))
            except Exception as error:  # noqa: BLE001
                row["error"] = type(error).__name__ + ": " + str(error)[:200]
            fh.write(json.dumps(row) + "\n")
    return read_rows(out)


def _pred(row: dict, attr: str):
    if attr == "direction":
        return norm_direction(row.get("direction")) if "twist" in row else None
    return bin_count(row.get(attr))


def _gold(row: dict, attr: str):
    return row["gold"]["direction"] if attr == "direction" else bin_count(row["gold"][attr])


def report(rows: list[dict]) -> dict:
    """Accuracy per attribute against the gold bins, with the best constant as the bar to beat."""
    scored = [r for r in rows if "error" not in r]
    out = {"clips": len(rows), "scored": len(scored), "failed": len(rows) - len(scored),
           "malformed": sum("raw" in r for r in scored), "unsure": sum(bool(r.get("unsure")) for r in scored),
           "mean_seconds": round(sum(r.get("usage", {}).get("seconds", 0) for r in scored) / max(1, len(scored)), 2),
           "attributes": {}, "confusion": {}, "by_group": {}, "trick": {}}
    if not scored:
        return out
    for attr in ATTRIBUTES:
        gold = [_gold(r, attr) for r in scored]
        pred = [_pred(r, attr) for r in scored]
        hits = sum(g == p for g, p in zip(gold, pred))
        best_value, best_n = Counter(gold).most_common(1)[0]
        pred_counts = Counter(p if p is not None else "malformed" for p in pred)
        out["attributes"][attr] = {"accuracy": round(hits / len(scored), 3), "hits": hits,
                                   "best_constant": {"value": best_value, "accuracy": round(best_n / len(scored), 3)},
                                   "beats_constant": hits > best_n, "pred_distribution": dict(pred_counts)}
        conf = defaultdict(Counter)
        for g, p in zip(gold, pred):
            conf[str(g)][str(p)] += 1
        out["confusion"][attr] = {g: dict(c) for g, c in sorted(conf.items())}
    exact = [all(_gold(r, a) == _pred(r, a) for a in ATTRIBUTES) for r in scored]
    gold_sigs = [signature(r["gold"]["flip"], r["gold"]["twist"], r["gold"]["direction"]) for r in scored]
    pred_sigs = [signature(r.get("flip"), r.get("twist"), r.get("direction")) if "raw" not in r else "malformed" for r in scored]
    best_sig, best_sig_n = Counter(gold_sigs).most_common(1)[0]
    pairs = Counter((g, p) for g, p in zip(gold_sigs, pred_sigs) if g != p)
    per_trick = defaultdict(lambda: [0, 0])
    for g, ok in zip(gold_sigs, exact):
        per_trick[g][0] += ok
        per_trick[g][1] += 1
    out["trick"] = {"exact_accuracy": round(sum(exact) / len(scored), 3), "exact_hits": sum(exact),
                    "best_constant": {"signature": best_sig, "accuracy": round(best_sig_n / len(scored), 3)},
                    "per_trick": {s: {"hits": h, "n": n, "accuracy": round(h / n, 2)}
                                  for s, (h, n) in sorted(per_trick.items(), key=lambda kv: -kv[1][1]) if n >= 5},
                    "top_confusions": [{"gold": g, "pred": p, "n": n} for (g, p), n in pairs.most_common(12)]}
    for key in ("context", "split", "source"):
        groups = defaultdict(list)
        for r, ok in zip(scored, exact):
            groups[str(r["gold"].get(key))].append((r, ok))
        out["by_group"][key] = {
            g: {"n": len(v), "twist_acc": round(sum(_gold(r, "twist") == _pred(r, "twist") for r, _ in v) / len(v), 3),
                "flip_acc": round(sum(_gold(r, "flip") == _pred(r, "flip") for r, _ in v) / len(v), 3),
                "exact_acc": round(sum(ok for _, ok in v) / len(v), 3)} for g, v in sorted(groups.items())}
    aerial = [(r, ok) for r, ok in zip(scored, exact) if is_aerial(r["gold"])]
    if aerial:
        out["by_group"]["aerial_flips"] = {
            "n": len(aerial),
            "twist_acc": round(sum(_gold(r, "twist") == _pred(r, "twist") for r, _ in aerial) / len(aerial), 3),
            "twist_best_constant": round(Counter(_gold(r, "twist") for r, _ in aerial).most_common(1)[0][1] / len(aerial), 3),
            "flip_acc": round(sum(_gold(r, "flip") == _pred(r, "flip") for r, _ in aerial) / len(aerial), 3),
            "flip_best_constant": round(Counter(_gold(r, "flip") for r, _ in aerial).most_common(1)[0][1] / len(aerial), 3),
            "direction_acc": round(sum(_gold(r, "direction") == _pred(r, "direction") for r, _ in aerial) / len(aerial), 3),
            "exact_acc": round(sum(ok for _, ok in aerial) / len(aerial), 3)}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cmd", choices=("plan", "run", "ground-truth", "report"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--aerial-only", action="store_true", help="only clips whose gold family is flip / acrobatics")
    parser.add_argument("--out", type=Path, default=PREDS)
    args = parser.parse_args()
    gold = load_gold()
    excluded = queue_slugs()
    slugs = order(gold, excluded, args.aerial_only)
    if args.cmd == "plan":
        print(json.dumps({"gold_clips": len(gold), "blind_queue_excluded": len(excluded), "to_run": len(slugs),
                          "aerial": sum(is_aerial(gold[s]) for s in slugs),
                          "already": len(read_rows(args.out)), "ground_truth_clips": len(ground_truth_clips(excluded))}, indent=1))
    elif args.cmd == "run":
        print(json.dumps(run(slugs, gold, args.out, args.limit), indent=1))
    elif args.cmd == "ground-truth":
        rows = run_ground_truth(GT_PREDS)
        for r in rows:
            print(json.dumps({k: r.get(k) for k in ("slug", "fig_name", "gold", "flip", "twist", "direction", "cork", "unsure", "error")}))
        print(json.dumps(report(rows)["attributes"], indent=1))
    else:
        print(json.dumps(report(read_rows(args.out)), indent=1))


if __name__ == "__main__":
    main()
