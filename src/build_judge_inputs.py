"""
Build judge input JSONL files from downloaded datasets.

For each (dataset, language, strategy) combination:
  1. Load the parquet file from raw_data/
  2. Extract the relevant columns (input, reference, output)
  3. Build the judge prompt using the selected strategy
  4. Write to judge_inputs/{dataset}/{lang}_{strategy}.jsonl

The "output" field (model-generated text to evaluate) can come from:
  - A separate model output file (if evaluating another model's generations)
  - The reference itself (for calibration / ceiling experiments)
  - A synthetic perturbation of the reference (for floor experiments)

Usage:
    python -m src.build_judge_inputs                              # all combos
    python -m src.build_judge_inputs --dataset flores200          # one dataset
    python -m src.build_judge_inputs --strategy zero_shot         # one strategy
    python -m src.build_judge_inputs --dataset xlsum --lang swa   # one slice
    python -m src.build_judge_inputs --dry-run                    # count only
"""
import argparse
import json
import random
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs.config import (
    DATASETS, STRATEGIES, RAW_DATA, JUDGE_INPUTS,
    LANG_NAMES, output_filename,
)
from prompts.templates import build_judge_prompt


random.seed(42)


def load_raw_data(dataset: str, lang: str) -> pd.DataFrame:
    """Load a parquet file from raw_data/{dataset}/{lang}.parquet."""
    path = RAW_DATA / dataset / f"{lang}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"No data file: {path}")
    return pd.read_parquet(path)


def extract_fields(row: pd.Series, columns: dict) -> dict:
    """Extract standardized fields from a dataset row.
    
    Handles the different column naming conventions across datasets.
    """
    fields = {}
    for field_name, col_name in columns.items():
        val = row.get(col_name, "")
        # Handle XQuAD-style answers (dict with 'text' list)
        if isinstance(val, dict) and "text" in val:
            val = val["text"][0] if val["text"] else ""
        elif isinstance(val, list):
            val = val[0] if val else ""
        fields[field_name] = str(val) if val is not None else ""
    return fields


def generate_model_output(fields: dict, task: str, mode: str = "reference") -> str:
    """Generate or retrieve the model output to be evaluated.
    
    Modes:
      - 'reference': Use the gold reference as the output (ceiling test)
      - 'perturbed': Randomly perturb the reference (synthetic errors)
      - 'external': Load from a separate model output file (production)
    
    For the thesis, we start with 'reference' mode for calibration,
    then switch to 'external' when evaluating actual model generations.
    """
    if mode == "reference":
        return fields.get("reference", "")
    elif mode == "perturbed":
        ref = fields.get("reference", "")
        # Simple perturbation: shuffle words in the middle
        words = ref.split()
        if len(words) > 4:
            mid = words[1:-1]
            random.shuffle(mid)
            return " ".join([words[0]] + mid + [words[-1]])
        return ref
    else:
        return fields.get("output", "")


def build_inputs_for_combo(
    dataset: str, lang: str, strategy: str,
    output_mode: str = "reference",
) -> int:
    """Build judge input JSONL for one (dataset, lang, strategy) combo.
    
    Returns number of records written.
    """
    ds_cfg = DATASETS[dataset]
    strat_cfg = STRATEGIES[strategy]
    task = ds_cfg["task"]

    # Check compatibility
    if task not in strat_cfg["tasks"]:
        return 0
    if "restrict_to_langs" in strat_cfg and lang not in strat_cfg["restrict_to_langs"]:
        return 0

    # Load data
    try:
        df = load_raw_data(dataset, lang)
    except FileNotFoundError:
        print(f"    ⚠ No data for {dataset}/{lang}")
        return 0

    # Prepare output directory and file
    out_dir = JUDGE_INPUTS / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{lang}_{strategy}.jsonl"

    # Skip if exists (skip-if-exists logic)
    if out_file.exists():
        existing = sum(1 for _ in open(out_file))
        print(f"    📂 {out_file.name}: exists ({existing} records)")
        return existing

    lang_name = LANG_NAMES.get(lang, lang)
    columns = ds_cfg["columns"]
    count = 0

    with open(out_file, "w", encoding="utf-8") as f:
        for idx, row in df.iterrows():
            fields = extract_fields(row, columns)
            model_output = generate_model_output(fields, task, output_mode)

            # Build the judge prompt
            prompt = build_judge_prompt(
                strategy=strategy,
                task=task,
                input_text=fields.get("input", ""),
                reference=fields.get("reference", ""),
                output_text=model_output,
                lang=lang,
                lang_name=lang_name,
                context=fields.get("context", ""),
            )

            record = {
                "id": f"{dataset}_{lang}_{strategy}_{idx}",
                "dataset": dataset,
                "language": lang,
                "task": task,
                "strategy": strategy,
                "prompt_system": prompt["system"],
                "prompt_user": prompt["user"],
                "meta": {
                    "reference": fields.get("reference", ""),
                    "model_output": model_output,
                    "output_mode": output_mode,
                    "original_index": int(idx),
                },
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

    print(f"    ✅ {out_file.name}: {count} records")
    return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--lang", type=str, default=None)
    parser.add_argument("--strategy", type=str, default=None)
    parser.add_argument("--output-mode", type=str, default="reference",
                        choices=["reference", "perturbed", "external"])
    parser.add_argument("--required-only", action="store_true",
                        help="Only build inputs for required strategies")
    parser.add_argument("--dry-run", action="store_true",
                        help="Count combinations without writing")
    args = parser.parse_args()

    datasets = [args.dataset] if args.dataset else list(DATASETS.keys())
    strategies = [args.strategy] if args.strategy else list(STRATEGIES.keys())

    if args.required_only:
        strategies = [s for s in strategies if STRATEGIES[s]["required"]]

    print("=" * 70)
    print("BUILDING JUDGE INPUTS")
    print(f"  Datasets:   {', '.join(datasets)}")
    print(f"  Strategies: {', '.join(strategies)}")
    print(f"  Output:     {JUDGE_INPUTS}")
    print(f"  Mode:       {args.output_mode}")
    print("=" * 70)

    grand = 0
    for ds_name in datasets:
        ds_cfg = DATASETS[ds_name]
        langs = [args.lang] if args.lang else list(ds_cfg["configs"].keys())

        print(f"\n{'─'*60}")
        print(f"📊 {ds_name} ({ds_cfg['task']})")
        print(f"{'─'*60}")

        for lang in langs:
            for strat in strategies:
                if args.dry_run:
                    strat_cfg = STRATEGIES[strat]
                    if ds_cfg["task"] in strat_cfg["tasks"]:
                        grand += 1
                else:
                    grand += build_inputs_for_combo(
                        ds_name, lang, strat, args.output_mode
                    )

    print(f"\n{'='*70}")
    action = "combinations found" if args.dry_run else "total records"
    print(f"Done: {grand} {action}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
