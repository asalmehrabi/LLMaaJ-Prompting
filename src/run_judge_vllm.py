"""
Run judge inference using vLLM on Snellius GPUs.

Reads JSONL files from judge_inputs/, runs inference with the specified
judge model, and writes results to judge_outputs/.

Features:
  - Skip-if-exists: won't re-run completed files
  - Batched inference: all prompts in a file processed together
  - Structured output: raw response + parsed score + metadata
  - Per-file timing and token counts

Usage:
    python -m src.run_judge_vllm --model aya-expanse-32b
    python -m src.run_judge_vllm --model aya-expanse-32b --dataset flores200
    python -m src.run_judge_vllm --model aya-expanse-32b --strategy zero_shot
    python -m src.run_judge_vllm --model aya-expanse-32b --dataset xlsum --lang swa
"""
import argparse
import json
import re
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs.config import (
    JUDGE_MODELS, JUDGE_INPUTS, JUDGE_OUTPUTS,
    INFERENCE_DEFAULTS, STRATEGIES,
)


# ─────────────────────────────────────────────────────────────────────
# Response parsers — extract structured output from raw judge response
# ─────────────────────────────────────────────────────────────────────

def parse_rating(raw: str) -> int:
    """Extract a 1-5 rating from the judge response.
    
    Tries multiple patterns:
      - Explicit "Rating: 4" or "Score: 4"
      - First standalone digit 1-5
      - Returns -1 if unparseable
    """
    text = raw.strip()

    # Pattern 1: "Rating: N" or "Score: N"
    match = re.search(r'(?:rating|score)\s*[:=]\s*([1-5])', text, re.IGNORECASE)
    if match:
        return int(match.group(1))

    # Pattern 2: First standalone 1-5 digit
    nums = re.findall(r'\b([1-5])\b', text)
    if nums:
        return int(nums[0])

    return -1  # unparseable


def parse_pairwise(raw: str) -> str:
    """Extract A/B/Tie from pairwise comparison response."""
    text = raw.strip().upper()
    if "TIE" in text:
        return "tie"
    # Look for isolated A or B (not part of a word)
    match = re.search(r'\b([AB])\b', text)
    if match:
        return match.group(1)
    return "unparseable"


def parse_response(raw: str, strategy: str) -> dict:
    """Parse judge response based on strategy type."""
    if strategy == "pairwise":
        choice = parse_pairwise(raw)
        return {"parsed_choice": choice, "parsed_score": -1}
    else:
        score = parse_rating(raw)
        return {"parsed_score": score, "parsed_choice": ""}


# ─────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────

def load_vllm_model(model_name: str):
    """Load a vLLM model based on config."""
    from vllm import LLM, SamplingParams

    cfg = JUDGE_MODELS[model_name]
    print(f"  Loading {cfg['hf_id']}...")
    print(f"  Tensor parallel: {cfg.get('tp_size', 1)} GPUs")

    llm = LLM(
        model=cfg["hf_id"],
        tensor_parallel_size=cfg.get("tp_size", 1),
        max_model_len=cfg.get("max_model_len", 8192),
        trust_remote_code=True,
        dtype="auto",
    )
    return llm


def get_sampling_params(strategy: str):
    """Get vLLM SamplingParams, respecting strategy-specific overrides."""
    from vllm import SamplingParams

    strat_cfg = STRATEGIES.get(strategy, {})
    temp = strat_cfg.get("temperature", INFERENCE_DEFAULTS["temperature"])
    max_tok = INFERENCE_DEFAULTS["max_tokens"]
    top_p = INFERENCE_DEFAULTS["top_p"]

    return SamplingParams(temperature=temp, max_tokens=max_tok, top_p=top_p)


# ─────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────

def format_chat_prompt(record: dict, model_name: str) -> str:
    """Format system + user messages into a single prompt string.
    
    For chat models, we use the model's chat template if available.
    For base models, we concatenate system and user.
    """
    system = record.get("prompt_system", "")
    user = record.get("prompt_user", "")

    # Simple concatenation — vLLM handles chat templates internally
    # if the model has one. For Aya Expanse, this works well.
    return f"{system}\n\n{user}"


def run_file(llm, jsonl_path: Path, output_path: Path, model_name: str) -> int:
    """Run inference on a single JSONL file.
    
    Returns number of records processed.
    """
    # Load records
    records = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if not records:
        return 0

    # Determine strategy from first record (all records in a file share it)
    strategy = records[0].get("strategy", "zero_shot")
    sampling = get_sampling_params(strategy)

    # Build prompts
    prompts = [format_chat_prompt(r, model_name) for r in records]

    print(f"    {jsonl_path.name}: {len(prompts)} prompts → ", end="", flush=True)
    t0 = time.time()
    outputs = llm.generate(prompts, sampling)
    elapsed = time.time() - t0
    print(f"{elapsed:.1f}s ({elapsed/len(prompts):.2f}s/prompt)")

    # Write results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for rec, output in zip(records, outputs):
            raw_text = output.outputs[0].text.strip()
            parsed = parse_response(raw_text, strategy)

            result = {
                "id": rec["id"],
                "dataset": rec["dataset"],
                "language": rec["language"],
                "task": rec["task"],
                "strategy": rec["strategy"],
                "judge_model": model_name,
                "judge_raw_output": raw_text,
                "judge_parsed_score": parsed["parsed_score"],
                "judge_parsed_choice": parsed["parsed_choice"],
                "judge_prompt_tokens": len(output.prompt_token_ids),
                "judge_completion_tokens": len(output.outputs[0].token_ids),
                "judge_time_s": round(elapsed / len(records), 4),
                "meta": rec.get("meta", {}),
            }
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    return len(records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                        choices=list(JUDGE_MODELS.keys()))
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--strategy", type=str, default=None)
    parser.add_argument("--lang", type=str, default=None)
    args = parser.parse_args()

    model_cfg = JUDGE_MODELS[args.model]
    if model_cfg["backend"] != "vllm":
        print(f"Model {args.model} uses backend '{model_cfg['backend']}', not vllm.")
        print("Use src.run_judge_api instead.")
        sys.exit(1)

    print("=" * 70)
    print(f"JUDGE INFERENCE: {args.model}")
    print(f"  Model: {model_cfg['hf_id']}")
    print(f"  Input: {JUDGE_INPUTS}")
    print(f"  Output: {JUDGE_OUTPUTS}")
    print("=" * 70)

    # Load model once
    llm = load_vllm_model(args.model)

    # Find all input files matching filters
    total = 0
    for jsonl_path in sorted(JUDGE_INPUTS.rglob("*.jsonl")):
        # Parse filename: {lang}_{strategy}.jsonl
        stem = jsonl_path.stem
        parts = stem.rsplit("_", 1)
        if len(parts) != 2:
            continue
        file_lang, file_strategy = parts[0], parts[1]

        # The lang part might contain underscores (e.g., for composite names)
        # but our lang codes don't, so this split works.

        # Get dataset name from parent directory
        file_dataset = jsonl_path.parent.name

        # Apply filters
        if args.dataset and file_dataset != args.dataset:
            continue
        if args.strategy and file_strategy != args.strategy:
            continue
        if args.lang and not stem.startswith(args.lang):
            continue

        # Output path mirrors input structure + model name
        out_path = JUDGE_OUTPUTS / file_dataset / args.model / jsonl_path.name

        # Skip if exists
        if out_path.exists():
            existing = sum(1 for _ in open(out_path))
            print(f"    📂 {file_dataset}/{jsonl_path.name}: exists ({existing} records)")
            total += existing
            continue

        print(f"  📊 {file_dataset}/{jsonl_path.name}")
        n = run_file(llm, jsonl_path, out_path, args.model)
        total += n

    print(f"\n{'='*70}")
    print(f"Done: {total} records processed")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
