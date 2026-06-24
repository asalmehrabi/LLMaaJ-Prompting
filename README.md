# LLM-as-a-Judge Multilingual Prompting Pipeline

**Thesis:** Evaluating the Multilingual Transferability of LLM-as-a-Judge Prompting Protocols  
**Author:** Asal Mehrabi (UvA/CWI) — Supervisor: Clemencia Siro  

---

## Pipeline Overview

```
                    ┌───────────────────────────────┐
                    │  1. DOWNLOAD DATASETS         │
                    │  src/prepare_datasets.py       │
                    │  HuggingFace → parquet files   │
                    └──────────┬────────────────────┘
                               │
                    ┌──────────▼────────────────────┐
                    │  2. BUILD JUDGE INPUTS         │
                    │  src/build_judge_inputs.py     │
                    │  parquet + strategy → JSONL    │
                    └──────────┬────────────────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
    ┌─────────▼──────┐  ┌─────▼──────────┐     │
    │ 3a. VLLM JUDGE │  │ 3b. API JUDGE  │     │
    │ Aya Expanse    │  │ GPT-4o         │     │
    │ (Snellius GPU) │  │ (OpenAI API)   │     │
    └─────────┬──────┘  └─────┬──────────┘     │
              │                │                │
              └────────────────┼────────────────┘
                               │
                    ┌──────────▼────────────────────┐
                    │  4. COMPUTE METRICS            │
                    │  src/compute_metrics.py        │
                    │  Score distributions, ρ, bias  │
                    └───────────────────────────────┘
```

## Project Structure

```
thesis_pipeline/
│
├── configs/
│   ├── config.py               # Central config: datasets, models, strategies, paths
│   └── monolingual_prompts.py  # Target-language prompt translations
│
├── prompts/
│   └── templates.py            # All prompt templates + strategy wrappers
│
├── src/
│   ├── prepare_datasets.py     # Step 1: Download HF datasets → parquet
│   ├── build_judge_inputs.py   # Step 2: Build JSONL inputs per (dataset, lang, strategy)
│   ├── run_judge_vllm.py       # Step 3a: vLLM inference (Aya Expanse on GPU)
│   ├── run_judge_api.py        # Step 3b: API inference (GPT-4o)
│   ├── compute_metrics.py      # Step 4: Compute evaluation metrics
│   └── logger.py               # Structured experiment logging
│
├── scripts/
│   ├── setup_env.sh            # One-time Snellius environment setup
│   ├── download_data.sbatch    # SLURM: download datasets (CPU)
│   ├── build_inputs.sbatch     # SLURM: build judge inputs (CPU)
│   ├── smoke_test.sbatch       # SLURM: end-to-end test (1 dataset, 1 strategy)
│   ├── run_experiment.sbatch   # SLURM: single experiment (1 dataset, all strategies)
│   ├── run_array.sbatch        # SLURM: array job (all datasets in parallel)
│   └── run_gpt4o.sbatch        # SLURM: GPT-4o experiments (CPU, API)
│
├── notebooks/                  # Analysis notebooks (not committed)
├── requirements.txt
└── README.md
```

## Data Flow

### On disk (scratch storage)

```
/scratch-shared/<user>/
├── raw_data/                       # Step 1 output
│   ├── flores200/
│   │   ├── eng.parquet
│   │   ├── swa.parquet
│   │   └── ...
│   ├── xlsum/
│   │   └── ...
│   └── ...
│
├── judge_inputs/                   # Step 2 output
│   ├── flores200/
│   │   ├── eng_zero_shot.jsonl
│   │   ├── eng_cot.jsonl
│   │   ├── swa_zero_shot.jsonl
│   │   ├── swa_monolingual.jsonl
│   │   └── ...
│   └── ...
│
├── judge_outputs/                  # Step 3 output
│   ├── flores200/
│   │   ├── aya-expanse-32b/
│   │   │   ├── eng_zero_shot.jsonl
│   │   │   └── ...
│   │   └── gpt-4o/
│   │       └── ...
│   └── ...
│
├── results/                        # Step 4 output
│   ├── metrics_aya-expanse-32b_all_20260415.json
│   └── metrics_gpt-4o_all_20260415.json
│
├── logs/                           # Experiment logs
│   └── judge_inference.jsonl
│
└── hf_cache/                       # HuggingFace cache
```

### Output record format (judge_outputs)

```json
{
  "id": "flores200_swa_zero_shot_42",
  "dataset": "flores200",
  "language": "swa",
  "task": "mt",
  "strategy": "zero_shot",
  "judge_model": "aya-expanse-32b",
  "judge_raw_output": "Rating: 4\nThe translation correctly conveys...",
  "judge_parsed_score": 4,
  "judge_parsed_choice": "",
  "judge_prompt_tokens": 312,
  "judge_completion_tokens": 87,
  "judge_time_s": 0.45,
  "meta": {
    "reference": "Sentensi ya Kiswahili...",
    "model_output": "Tafsiri ya mfano...",
    "output_mode": "reference",
    "original_index": 42
  }
}
```

## Quick Start

```bash
# 1. Clone and setup on Snellius
scp -r thesis_pipeline/ <user>@snellius.surf.nl:~/thesis_pipeline/
ssh <user>@snellius.surf.nl
bash ~/thesis_pipeline/scripts/setup_env.sh

# 2. Download all datasets (CPU job, ~30 min)
sbatch scripts/download_data.sbatch

# 3. Smoke test (verify everything works)
sbatch scripts/smoke_test.sbatch
# Check: cat logs/smoke_<jobid>.out

# 4. Run full experiments
# Option A: One dataset at a time
sbatch scripts/run_experiment.sbatch flores200
sbatch scripts/run_experiment.sbatch xlsum

# Option B: All datasets in parallel (array job)
sbatch scripts/run_array.sbatch

# 5. Run GPT-4o experiments
sbatch scripts/run_gpt4o.sbatch

# 6. Analyze results
python -m src.compute_metrics --output results/final_metrics.json
```

## Experimental Design

### Datasets (13 datasets, 4 tasks)

| Task | Datasets | Languages |
|------|----------|-----------|
| MT | FLORES-200, MAFAND-MT | 12 + 10 langs |
| Summarization | XL-Sum, MLSUM, CaSum | 13 + 2 + 1 langs |
| QA | AfriQA, XQuAD, PersianQA, CatalanQA | 9 + 3 + 1 + 1 langs |
| Math | AfriMGSM, MGSM | 16 + 4 langs |

### Prompting Strategies (6 required + 3 optional)

| # | Strategy | Tasks | Description |
|---|----------|-------|-------------|
| 1 | Zero-shot | All | Baseline — direct evaluation |
| 2 | Few-shot anchored | All | Examples at scale extremes |
| 3 | Prometheus-style | MT, Sum | Rubric + reference (Siro et al. 2026) |
| 4 | Chain-of-Thought | Sum, QA, Math | Step-by-step reasoning |
| 5 | Cross-lingual | All | English prompt, non-English content |
| 6 | Monolingual | All | Prompt in target language |
| 7* | Pairwise | MT, Sum | A/B comparison, position randomized |
| 8* | Self-consistency | QA, Math | 3x runs, majority vote |
| 9* | Tree-of-Thought | Math, QA | Multi-path reasoning (anchor langs) |

\* = optional

### Judge Models

| Model | Backend | Setup |
|-------|---------|-------|
| Aya Expanse 32B | vLLM (2×A100) | Snellius GPU partition |
| GPT-4o | OpenAI API | API key, CPU partition |

## Key Design Decisions

### Why this structure differs from Esteban's pipeline

Esteban's pipeline was designed around **AfroBench CSV outputs** with a scenario-based
evaluation framework (Scenario 1: wrong-case verification, Scenario 2: quality rating,
Scenario 3: pairwise comparison). His pipeline reads pre-generated model outputs from
CSV files and sends them to a judge.

This pipeline is designed differently for several reasons:

1. **Strategy-centric, not scenario-centric.** The thesis research questions are about
   comparing *prompting strategies*, so the primary axis is strategy (zero-shot, CoT,
   monolingual, etc.), not scenario type. Each JSONL file corresponds to one
   (dataset, language, strategy) triple.

2. **Direct HuggingFace integration.** Rather than depending on pre-existing AfroBench
   CSVs, datasets are downloaded directly from HuggingFace and stored as parquet. This
   makes the pipeline self-contained and reproducible.

3. **Modular prompt system.** Prompt templates are separated from the inference code.
   Adding a new strategy means adding a wrapper function in `prompts/templates.py` and
   an entry in `configs/config.py` — no changes to the runner.

4. **Dual-backend inference.** The same JSONL inputs work for both vLLM (Aya Expanse)
   and the OpenAI API (GPT-4o), enabling direct comparison.

### What was reused from Esteban's work

- **SLURM structure:** Module loading sequence, partition selection, environment
  variable setup, and log file organization follow the same patterns.
- **vLLM configuration:** Tensor parallel size (2), max model length, sampling
  parameters were validated by his successful runs.
- **Skip-if-exists logic:** The pattern of checking for existing output files before
  re-running inference.
- **Response parsing:** The regex-based score extraction approach.

### What was reused from the notebook

- **HuggingFace dataset IDs:** All verified working IDs (MotaOcean/xlsum,
  lighteval/mgsm, etc.) and the fixes for broken alternatives.
- **Language-config mappings:** The exact config strings for each dataset-language pair.
- **Parquet storage pattern:** Save raw data as parquet for fast local loading.

## Adding a New Strategy

1. Add the strategy metadata to `configs/config.py` → `STRATEGIES` dict
2. Write a wrapper function in `prompts/templates.py` (receives system + user, returns modified system + user)
3. Register the wrapper in `STRATEGY_WRAPPERS` dict
4. Rebuild inputs: `python -m src.build_judge_inputs --strategy new_strategy`
5. Run inference: `python -m src.run_judge_vllm --model aya-expanse-32b --strategy new_strategy`

## Adding a New Dataset

1. Add the dataset to `configs/config.py` → `DATASETS` dict with HF ID, configs, columns
2. Download: `python -m src.prepare_datasets --dataset new_dataset`
3. Build inputs: `python -m src.build_judge_inputs --dataset new_dataset`

## Estimated Compute Budget

- **Aya Expanse 32B:** ~1.5 sec/prompt on 2×A100. With ~84,000 core judge calls across
  all (dataset, language, strategy) combos, expect ~35 GPU-hours.
- **GPT-4o:** ~0.5 sec/prompt via API. Same call count, ~12 CPU-hours.
- **Storage:** ~2 GB for datasets, ~5 GB for judge I/O, ~50 GB for model cache.
