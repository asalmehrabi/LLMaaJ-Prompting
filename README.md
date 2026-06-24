# Multilingual LLM-as-a-Judge Prompting Benchmark

**Evaluating the Transferability and Robustness of LLM-as-a-Judge Prompting Protocols in Multilingual Scenarios**

> MSc Thesis — Data Science & Artificial Intelligence  
> University of Amsterdam / Centrum Wiskunde & Informatica (CWI)  
> Author: Asal Mehrabi · Supervisor: Clemencia Siro

---

## Abstract

This repository accompanies the thesis *Evaluating the Transferability and Robustness of LLM-as-a-Judge Prompting Protocols in Multilingual Scenarios*. It provides the complete experimental pipeline for a large-scale systematic benchmark of nine prompting strategies for LLM-as-a-Judge evaluation, covering four open-source judge models, 27 languages spanning high- to low-resource tiers, four NLP tasks, and 13 datasets — producing 1,068,528 individual judge scores. Strategy rankings are validated against human quality annotations from three external datasets (SSA-MTE, SummEval, WMT MQM).

Inference runs on the [Snellius supercomputer](https://www.surf.nl/en/snellius-the-national-supercomputer) (SURF, Netherlands) via [vLLM](https://github.com/vllm-project/vllm). All experiments use deterministic decoding (temperature = 0, seed = 42) except Self-Consistency (temperature = 0.7, 3 runs).

---

## Repository Structure

```
.
├── configs/
│   ├── config.py                      # Central configuration: datasets, models,
│   │                                  # strategies, paths, language metadata
│   └── monolingual_prompts.py         # Native-language prompt templates
│                                      # (8 languages verified by native speakers)
├── prompts/
│   └── templates.py                   # All 9 prompting strategy templates per task
├── src/
│   ├── prepare_datasets.py            # Stage 1: download and preprocess datasets
│   ├── prepare_math_datasets.py       # Stage 1: math-specific preprocessing
│   ├── build_judge_inputs.py          # Stage 2: assemble JSONL inference inputs
│   ├── run_judge_vllm.py              # Stage 3: batched vLLM inference
│   ├── run_judge_validation.py        # Stage 3: validation dataset inference
│   ├── extract_results.py             # Stage 4: parse outputs → per-cell CSV
│   ├── compute_metrics.py             # Stage 4: η², CV, Spearman ρ, range
│   ├── compute_ssa_correlation.py     # Stage 4: SSA-MTE human alignment
│   ├── compute_validation_correlation.py  # Stage 4: SummEval + WMT MQM alignment
│   └── logger.py                      # Logging utilities
├── scripts/
│   ├── setup_env.sh                   # Environment setup
│   ├── download_data.sh               # Dataset acquisition
│   ├── build_inputs.sh                # Input construction
│   ├── run_all_datasets_h100.sh       # Full inference (H100 partition)
│   ├── run_mt_sum_v4_h100.sh          # MT + Summarization inference
│   ├── run_qa_math_v4_h100.sh         # QA + Math inference
│   ├── run_validation_h100.sh         # External validation inference
│   ├── run_missing_v4_h100.sh         # Re-run missing cells
│   ├── check_results.sh               # Completeness verification
│   └── smoke_test.sh                  # Sanity check before full run
├── requirements.txt
└── README.md
```

---

## Experimental Setup

### Judge Models

| Model | HuggingFace ID | Parameters | Access |
|-------|---------------|-----------|--------|
| Aya Expanse | `CohereForAI/aya-expanse-32b` | 32B | Public |
| Gemma 3 | `google/gemma-3-27b-it` | 27B | Gated† |
| Qwen 2.5 | `Qwen/Qwen2.5-32B-Instruct` | 32B | Public |
| Qwen 2.5 AWQ | `Qwen/Qwen2.5-72B-Instruct-AWQ` | 72B | Public |

† Gemma 3 requires accepting the model license at [huggingface.co/google/gemma-3-27b-it](https://huggingface.co/google/gemma-3-27b-it) before downloading.

### Prompting Strategies

| # | Strategy | MT | Sum | QA | Math |
|---|----------|----|-----|----|------|
| 1 | Zero-Shot | ✓ | ✓ | ✓ | ✓ |
| 2 | Few-Shot Anchored | ✓ | ✓ | ✓ | ✓ |
| 3 | Prometheus | ✓ | ✓ | ✓ | ✓ |
| 4 | Chain-of-Thought | — | ✓ | ✓ | ✓ |
| 5 | Cross-Lingual | ✓ | ✓ | ✓ | ✓ |
| 6 | Monolingual | ✓ | ✓ | ✓ | ✓ |
| 7 | Translate-then-Evaluate | ✓ | ✓ | — | — |
| 8 | Role Prompting | ✓ | ✓ | ✓ | — |
| 9 | Self-Consistency | — | — | ✓ | ✓ |

Strategies 7–9 share the Zero-Shot scoring structure and differ only in prompt language and target-text handling. All prompt templates are in `prompts/templates.py`. Monolingual prompts for French, Spanish, Catalan, Persian, Swahili, Hausa, Yoruba, and Amharic were verified by native speakers (`configs/monolingual_prompts.py`); remaining languages use machine-translated templates.

### Datasets

| Task | Dataset | HuggingFace ID | Languages |
|------|---------|----------------|-----------|
| MT | FLORES-200 | `Muennighoff/flores200` | 12 |
| MT | MAFAND-MT | `masakhane/mafand` | 10 |
| MT | SSA-MTE | `McGill-NLP/SSA-MTE` | 11 |
| Summarization | XL-Sum | `csebuetnlp/xlsum` | 13 |
| Summarization | MLSUM | `mlsum` | 2 |
| Summarization | CaSum | `projecte-aina/casum` | 1 |
| QA | AfriQA | `masakhane/afriqa` | 9 |
| QA | XQuAD | `google/xquad` | 3 |
| QA | PersianQA | `SajjadAyoubi/persian_qa` | 1 |
| QA | CatalanQA | `projecte-aina/catalanqa` | 1 |
| QA | Belebele | `facebook/belebele` | 15 |
| Math | AfriMGSM | `masakhane/afrimgsm` | 16 |
| Math | MGSM | `juletxara/mgsm` | 4 |
| Validation | SummEval | `mteb/summeval` | 1 |
| Validation | WMT MQM | `RicardoRei/wmt-mqm-human-evaluation` | 1 |

SSA-MTE provides human direct-assessment scores and is the primary external validation dataset. SummEval and WMT MQM are used for cross-dataset human alignment analysis.

### Language Resource Tiers

| Tier | Languages |
|------|-----------|
| High | English, French, German, Spanish |
| Mid | Arabic, Catalan, Persian |
| Low-mid | Hausa, Swahili |
| Low | Amharic, Bemba, Ewe, Fon, Igbo, Kinyarwanda, Lingala, Luganda, Oromo, Nigerian Pidgin, Somali, Southern Sotho, Tigrinya, Twi, Wolof, isiXhosa, Yoruba, isiZulu |

---

## Reproducing the Experiments

### Prerequisites

- Python 3.12
- CUDA 12.6
- vLLM ≥ 0.8
- HPC cluster with A100 or H100 GPUs (~300 GPU-hours for the full experiment)
- HuggingFace account and access token

### HuggingFace Authentication

A token is required to download gated models (Gemma 3) and some datasets.

```bash
# Set as environment variable (recommended for HPC)
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxx

# Or authenticate via CLI
huggingface-cli login
```

### Step 1 — Environment

```bash
git clone https://github.com/asalmehrabi/LLMaaJ-Prompting.git
cd LLMaaJ-Prompting
bash scripts/setup_env.sh

# Or manually
python3.12 -m venv ~/venvs/thesis_pipeline
source ~/venvs/thesis_pipeline/bin/activate
pip install -r requirements.txt
```

### Step 2 — Configure scratch directory

All data, inputs, outputs, and results are written to a configurable scratch path. Set before running any script:

```bash
# On Snellius
export THESIS_SCRATCH=/scratch-shared/$USER

# Local testing
export THESIS_SCRATCH=~/thesis_scratch
```

Subdirectories (`raw_data/`, `judge_inputs_v4/`, `judge_outputs_v4/`, `results/`) are created automatically on first run.

### Step 3 — Data preparation

```bash
bash scripts/download_data.sh
```

Downloads and preprocesses all 13 benchmark datasets from HuggingFace into parquet files under `$THESIS_SCRATCH/raw_data/`.

### Step 4 — Build inference inputs

```bash
bash scripts/build_inputs.sh
```

Assembles all strategy × dataset × language combinations into JSONL files under `$THESIS_SCRATCH/judge_inputs_v4/`. Skip-if-exists logic makes this safe to rerun after partial failures.

### Step 5 — Run inference

Submit SLURM jobs per model and task group. On the Snellius H100 partition:

```bash
# MT + Summarization
bash scripts/run_mt_sum_v4_h100.sh aya-expanse-32b
bash scripts/run_mt_sum_v4_h100.sh gemma-3-27b-it
bash scripts/run_mt_sum_v4_h100.sh qwen-2.5-32b
bash scripts/run_mt_sum_v4_h100.sh qwen-2.5-72b

# QA + Math
bash scripts/run_qa_math_v4_h100.sh aya-expanse-32b
bash scripts/run_qa_math_v4_h100.sh gemma-3-27b-it
bash scripts/run_qa_math_v4_h100.sh qwen-2.5-32b
bash scripts/run_qa_math_v4_h100.sh qwen-2.5-72b

# Or submit all at once
bash scripts/run_all_datasets_h100.sh
```

Raw judge responses are saved as JSONL under `$THESIS_SCRATCH/judge_outputs_v4/`.

### Step 6 — External validation

```bash
bash scripts/run_validation_h100.sh aya-expanse-32b
bash scripts/run_validation_h100.sh gemma-3-27b-it
bash scripts/run_validation_h100.sh qwen-2.5-32b
bash scripts/run_validation_h100.sh qwen-2.5-72b
```

Runs judges on SSA-MTE, SummEval, and WMT MQM for Spearman ρ computation against human annotations.

### Step 7 — Results extraction

```bash
python src/extract_results.py
```

Parses all judge outputs and produces `all_results_v4.csv` — the primary results file used for all analyses in the thesis.

### Step 8 — Metrics computation

```bash
python src/compute_metrics.py                    # η², CV, cross-language range
python src/compute_ssa_correlation.py            # Spearman ρ vs SSA-MTE human scores
python src/compute_validation_correlation.py     # Spearman ρ vs SummEval + WMT MQM
```

### Step 9 — Verify completeness

```bash
bash scripts/check_results.sh
```

Reports any missing model × strategy × dataset × language combinations and estimates remaining compute time.

---

## Output Format

The primary results file `all_results_v4.csv` contains 2,876 per-cell means aggregated over 1,068,528 individual judge scores.

| Column | Type | Description |
|--------|------|-------------|
| `model` | str | Judge model identifier |
| `strategy` | str | Prompting strategy |
| `task` | str | MT / Sum / QA / Math |
| `dataset` | str | Source dataset name |
| `lang` | str | ISO 639-3 language code |
| `lang_name` | str | Full language name |
| `resource` | str | high / mid / low-mid / low |
| `mean` | float | Mean holistic score (1–5) |
| `std` | float | Standard deviation across examples |
| `n` | int | Number of evaluated examples |
| `human_score_mean` | float | Human quality annotation mean (SSA-MTE only) |

---


---

## References

- Zheng et al. (2023). Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena. *NeurIPS 2023*.
- Ojo et al. (2025). SSA-MTE: A Benchmark for Multilingual Machine Translation Evaluation. *arXiv:2506.04557*.
- Üstün et al. (2024). Aya Expanse: Combining Research Breakthroughs for a New Multilingual Frontier. *arXiv:2412.04261*.
- Gemma Team (2025). Gemma 3 Technical Report. *arXiv:2503.19786*.
- Yang et al. (2024). Qwen2.5 Technical Report. *arXiv:2412.15115*.
- Fabbri et al. (2021). SummEval: Re-evaluating Summarization Evaluation. *TACL*.
- Freitag et al. (2021). Experts, Errors, and Context: A Large-Scale Study of Human Evaluation for Machine Translation. *TACL*.
- Kwon et al. (2023). Efficient Memory Management for Large Language Model Serving with PagedAttention. *SOSP 2023*.

---

## License

MIT License. Dataset and model licenses vary — consult individual HuggingFace repository pages before use.
