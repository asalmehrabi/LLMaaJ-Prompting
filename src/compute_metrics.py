"""
Compute evaluation metrics from judge outputs.

Metrics aligned with the four research questions:
  RQ1: Spearman ρ between judge scores and auto metrics (per language, per strategy)
  RQ2: Variance of ρ across languages per strategy (robustness)
  RQ3: Regression: predict ρ from linguistic features (TTR, script, resource level)
  RQ4: Error taxonomy (unparseable rate, score distribution analysis)

Usage:
    python -m src.compute_metrics                                    # all outputs
    python -m src.compute_metrics --dataset flores200                # one dataset
    python -m src.compute_metrics --judge aya-expanse-32b            # one judge
    python -m src.compute_metrics --strategy zero_shot               # one strategy
"""
import argparse
import json
import sys
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs.config import (
    JUDGE_OUTPUTS, RESULTS, LANG_NAMES, LANG_META,
    RESOURCE_LEVELS, SCRIPT_FAMILIES,
)


# ─────────────────────────────────────────────────────────────────────
# Metric computation
# ─────────────────────────────────────────────────────────────────────

def compute_group_metrics(records: list) -> dict:
    """Compute metrics for a group of records sharing (dataset, strategy, judge).
    
    Returns a dict with:
      - score_distribution: Counter of parsed scores
      - mean_score: average score (excluding unparseable)
      - unparseable_rate: fraction of -1 scores
      - by_language: per-language breakdown
      - n_total: total records
    """
    scores = []
    by_lang = defaultdict(list)

    for rec in records:
        score = rec.get("judge_parsed_score", -1)
        lang = rec.get("language", "?")
        scores.append(score)
        by_lang[lang].append(score)

    valid_scores = [s for s in scores if s > 0]
    n_total = len(scores)
    n_unparseable = sum(1 for s in scores if s < 0)

    result = {
        "n_total": n_total,
        "n_valid": len(valid_scores),
        "n_unparseable": n_unparseable,
        "unparseable_rate": round(n_unparseable / max(n_total, 1), 4),
        "mean_score": round(np.mean(valid_scores), 3) if valid_scores else None,
        "std_score": round(np.std(valid_scores), 3) if valid_scores else None,
        "score_distribution": dict(Counter(valid_scores)),
        "by_language": {},
    }

    for lang, lang_scores in sorted(by_lang.items()):
        valid = [s for s in lang_scores if s > 0]
        result["by_language"][lang] = {
            "n": len(lang_scores),
            "n_valid": len(valid),
            "mean": round(np.mean(valid), 3) if valid else None,
            "std": round(np.std(valid), 3) if valid else None,
            "distribution": dict(Counter(valid)),
            "unparseable": sum(1 for s in lang_scores if s < 0),
            "lang_name": LANG_NAMES.get(lang, lang),
            "resource_level": RESOURCE_LEVELS.get(lang, "unknown"),
            "script": SCRIPT_FAMILIES.get(lang, "unknown"),
        }

    return result


def compute_pairwise_metrics(records: list) -> dict:
    """Compute metrics for pairwise comparison strategy."""
    choices = Counter()
    by_lang = defaultdict(Counter)

    for rec in records:
        choice = rec.get("judge_parsed_choice", "unparseable")
        lang = rec.get("language", "?")
        choices[choice] += 1
        by_lang[lang][choice] += 1

    total = sum(choices.values())
    return {
        "n_total": total,
        "choices": dict(choices),
        "position_bias": {
            "A_rate": round(choices.get("A", 0) / max(total, 1), 4),
            "B_rate": round(choices.get("B", 0) / max(total, 1), 4),
            "tie_rate": round(choices.get("tie", 0) / max(total, 1), 4),
        },
        "unparseable_rate": round(choices.get("unparseable", 0) / max(total, 1), 4),
        "by_language": {
            lang: {
                "choices": dict(c),
                "lang_name": LANG_NAMES.get(lang, lang),
            }
            for lang, c in sorted(by_lang.items())
        },
    }


def compute_robustness(all_results: dict) -> dict:
    """RQ2: Cross-language robustness analysis.
    
    For each strategy, compute variance of mean scores across languages.
    Low variance = robust strategy.
    """
    by_strategy = defaultdict(list)

    for key, metrics in all_results.items():
        # key format: dataset_strategy_judge
        parts = key.split("_")
        # Extract strategy (everything between first and last underscore group)
        strategy = "_".join(parts[1:-1]) if len(parts) > 2 else parts[1] if len(parts) > 1 else key

        if "by_language" not in metrics:
            continue

        for lang, lang_data in metrics["by_language"].items():
            if lang_data.get("mean") is not None:
                by_strategy[strategy].append(lang_data["mean"])

    robustness = {}
    for strat, means in by_strategy.items():
        if len(means) >= 2:
            robustness[strat] = {
                "n_languages": len(means),
                "mean_of_means": round(np.mean(means), 3),
                "std_across_langs": round(np.std(means), 3),
                "cv": round(np.std(means) / np.mean(means), 4) if np.mean(means) > 0 else None,
                "range": round(max(means) - min(means), 3),
            }

    return robustness


# ─────────────────────────────────────────────────────────────────────
# Data collection
# ─────────────────────────────────────────────────────────────────────

def collect_records(base_dir: Path, dataset: str = None,
                    judge: str = None, strategy: str = None) -> dict:
    """Walk output directory, group records by (dataset, strategy, judge)."""
    groups = defaultdict(list)

    for jsonl_path in base_dir.rglob("*.jsonl"):
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)

                ds = rec.get("dataset", "?")
                strat = rec.get("strategy", "?")
                jm = rec.get("judge_model", "?")

                if dataset and ds != dataset:
                    continue
                if judge and jm != judge:
                    continue
                if strategy and strat != strategy:
                    continue

                groups[(ds, strat, jm)].append(rec)

    return groups


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str, default=str(JUDGE_OUTPUTS))
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--judge", type=str, default=None)
    parser.add_argument("--strategy", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    base = Path(args.input_dir)
    groups = collect_records(base, args.dataset, args.judge, args.strategy)

    if not groups:
        print("No judge output records found.")
        return

    all_results = {}
    for (ds, strat, jm), records in sorted(groups.items()):
        key = f"{ds}_{strat}_{jm}"
        print(f"\n{'='*60}")
        print(f"📊 {ds} | {strat} | {jm} | n={len(records)}")
        print(f"{'='*60}")

        if strat == "pairwise":
            result = compute_pairwise_metrics(records)
        else:
            result = compute_group_metrics(records)

        all_results[key] = result

        # Pretty print summary
        for k, v in result.items():
            if k == "by_language":
                print(f"  {k}: ({len(v)} languages)")
                for lang, data in list(v.items())[:5]:  # show first 5
                    print(f"    {data.get('lang_name', lang)}: "
                          f"mean={data.get('mean', '?')}, n={data.get('n', '?')}")
                if len(v) > 5:
                    print(f"    ... and {len(v)-5} more")
            else:
                print(f"  {k}: {v}")

    # Compute robustness analysis (RQ2)
    robustness = compute_robustness(all_results)
    all_results["_robustness_analysis"] = robustness

    print(f"\n{'='*60}")
    print("ROBUSTNESS ANALYSIS (RQ2)")
    print(f"{'='*60}")
    for strat, data in sorted(robustness.items()):
        print(f"  {strat}: CV={data.get('cv', '?')}, "
              f"std={data.get('std_across_langs', '?')}, "
              f"range={data.get('range', '?')}")

    # Save results
    out_path = Path(args.output) if args.output else RESULTS / "metrics.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
