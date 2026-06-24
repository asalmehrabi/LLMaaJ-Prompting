"""
Run judge inference using the OpenAI API (for GPT-4o).

Same interface as run_judge_vllm.py but calls the OpenAI API instead.
Handles rate limiting with exponential backoff.

Usage:
    python -m src.run_judge_api --model gpt-4o
    python -m src.run_judge_api --model gpt-4o --dataset flores200 --strategy zero_shot
"""
import argparse
import json
import os
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs.config import (
    JUDGE_MODELS, JUDGE_INPUTS, JUDGE_OUTPUTS, INFERENCE_DEFAULTS, STRATEGIES,
)
from src.run_judge_vllm import parse_response


def call_openai(client, model_name: str, system: str, user: str,
                temperature: float, max_tokens: int,
                max_retries: int = 5) -> dict:
    """Call OpenAI API with retry logic."""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return {
                "text": response.choices[0].message.content.strip(),
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
            }
        except Exception as e:
            wait = 2 ** attempt
            print(f"    ⚠ API error (attempt {attempt+1}): {str(e)[:80]}. "
                  f"Retrying in {wait}s...")
            time.sleep(wait)

    return {"text": "ERROR: max retries exceeded", "prompt_tokens": 0,
            "completion_tokens": 0}


def run_file_api(client, jsonl_path: Path, output_path: Path,
                 model_name: str, api_model: str) -> int:
    """Process a single JSONL file through the OpenAI API."""
    records = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if not records:
        return 0

    strategy = records[0].get("strategy", "zero_shot")
    strat_cfg = STRATEGIES.get(strategy, {})
    temperature = strat_cfg.get("temperature", INFERENCE_DEFAULTS["temperature"])
    max_tokens = INFERENCE_DEFAULTS["max_tokens"]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"    {jsonl_path.name}: {len(records)} prompts")

    with open(output_path, "w", encoding="utf-8") as f:
        for i, rec in enumerate(records):
            t0 = time.time()
            resp = call_openai(
                client, api_model,
                rec.get("prompt_system", ""),
                rec.get("prompt_user", ""),
                temperature, max_tokens,
            )
            elapsed = time.time() - t0

            parsed = parse_response(resp["text"], strategy)

            result = {
                "id": rec["id"],
                "dataset": rec["dataset"],
                "language": rec["language"],
                "task": rec["task"],
                "strategy": rec["strategy"],
                "judge_model": model_name,
                "judge_raw_output": resp["text"],
                "judge_parsed_score": parsed["parsed_score"],
                "judge_parsed_choice": parsed["parsed_choice"],
                "judge_prompt_tokens": resp["prompt_tokens"],
                "judge_completion_tokens": resp["completion_tokens"],
                "judge_time_s": round(elapsed, 4),
                "meta": rec.get("meta", {}),
            }
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

            if (i + 1) % 50 == 0:
                print(f"      {i+1}/{len(records)} done")

    return len(records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--strategy", type=str, default=None)
    parser.add_argument("--lang", type=str, default=None)
    args = parser.parse_args()

    if args.model not in JUDGE_MODELS:
        print(f"Unknown model: {args.model}")
        sys.exit(1)

    model_cfg = JUDGE_MODELS[args.model]
    if model_cfg["backend"] != "openai":
        print(f"Model {args.model} uses backend '{model_cfg['backend']}', not openai.")
        sys.exit(1)

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not set")
        sys.exit(1)

    from openai import OpenAI
    client = OpenAI(api_key=api_key)

    print("=" * 70)
    print(f"JUDGE INFERENCE (API): {args.model}")
    print(f"  API model: {model_cfg['model_name']}")
    print("=" * 70)

    total = 0
    for jsonl_path in sorted(JUDGE_INPUTS.rglob("*.jsonl")):
        file_dataset = jsonl_path.parent.name
        if args.dataset and file_dataset != args.dataset:
            continue

        stem = jsonl_path.stem
        if args.strategy and not stem.endswith(f"_{args.strategy}"):
            continue
        if args.lang and not stem.startswith(args.lang):
            continue

        out_path = JUDGE_OUTPUTS / file_dataset / args.model / jsonl_path.name

        if out_path.exists():
            existing = sum(1 for _ in open(out_path))
            print(f"    📂 {file_dataset}/{jsonl_path.name}: exists ({existing})")
            total += existing
            continue

        n = run_file_api(client, jsonl_path, out_path, args.model,
                         model_cfg["model_name"])
        total += n

    print(f"\nDone: {total} records processed")


if __name__ == "__main__":
    main()
