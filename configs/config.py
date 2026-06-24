"""
Central configuration for the LLM-as-a-Judge multilingual pipeline.

All datasets, models, prompting strategies, language metadata, paths,
and experimental parameters are defined here. Change configs — not code.
"""
from pathlib import Path
import os

# ─────────────────────────────────────────────────────────────────────
# 1. PATHS — everything flows from SCRATCH
# ─────────────────────────────────────────────────────────────────────
# On Snellius, SCRATCH should be /scratch-shared/<username>
# Locally, falls back to ~/thesis_scratch
SCRATCH = Path(os.environ.get("THESIS_SCRATCH", os.path.expanduser("~/thesis_scratch")))
PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DATA       = SCRATCH / "raw_data"         # downloaded parquet files
JUDGE_INPUTS   = SCRATCH / "judge_inputs"     # prepared JSONL for judge inference
JUDGE_OUTPUTS  = SCRATCH / "judge_outputs"    # raw judge responses
RESULTS        = SCRATCH / "results"          # aggregated metrics / analysis
LOGS           = SCRATCH / "logs"             # experiment-level logs
HF_CACHE       = SCRATCH / "hf_cache"         # Hugging Face model/dataset cache

for d in [RAW_DATA, JUDGE_INPUTS, JUDGE_OUTPUTS, RESULTS, LOGS, HF_CACHE]:
    d.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────
# 2. LANGUAGE METADATA
# ─────────────────────────────────────────────────────────────────────
# Canonical 3-letter codes used everywhere in the pipeline.
# fmt: off
LANG_META = {
    # code: (full_name, script_family, resource_level, iso_639_3)
    "eng": ("English",           "Latin",    "high",    "eng"),
    "fra": ("French",            "Latin",    "high",    "fra"),
    "spa": ("Spanish",           "Latin",    "high",    "spa"),
    "deu": ("German",            "Latin",    "high",    "deu"),
    "cat": ("Catalan",           "Latin",    "mid",     "cat"),
    "fas": ("Persian",           "Arabic",   "mid",     "fas"),
    "ara": ("Arabic",            "Arabic",   "mid",     "ara"),
    "hau": ("Hausa",             "Latin",    "low-mid", "hau"),
    "swa": ("Swahili",           "Latin",    "low-mid", "swh"),
    "yor": ("Yoruba",            "Latin",    "low",     "yor"),
    "amh": ("Amharic",           "Ethiopic", "low",     "amh"),
    "ibo": ("Igbo",              "Latin",    "low",     "ibo"),
    "kin": ("Kinyarwanda",       "Latin",    "low",     "kin"),
    "xho": ("isiXhosa",          "Latin",    "low",     "xho"),
    "zul": ("isiZulu",           "Latin",    "low",     "zul"),
    "twi": ("Twi",               "Latin",    "low",     "twi"),
    "lug": ("Luganda",           "Latin",    "low",     "lug"),
    "som": ("Somali",            "Latin",    "low",     "som"),
    "orm": ("Oromo",             "Latin",    "low",     "orm"),
    "pcm": ("Nigerian Pidgin",   "Latin",    "low",     "pcm"),
    "sna": ("Shona",             "Latin",    "low",     "sna"),
    "tir": ("Tigrinya",          "Ethiopic", "low",     "tir"),
    "lin": ("Lingala",           "Latin",    "low",     "lin"),
    "wol": ("Wolof",             "Latin",    "low",     "wol"),
    "ewe": ("Ewe",               "Latin",    "low",     "ewe"),
    "sot": ("Southern Sotho",    "Latin",    "low",     "sot"),
    "bem": ("Bemba",             "Latin",    "low",     "bem"),
    "fon": ("Fon",               "Latin",    "low",     "fon"),
}
# fmt: on

LANG_NAMES = {code: meta[0] for code, meta in LANG_META.items()}
SCRIPT_FAMILIES = {code: meta[1] for code, meta in LANG_META.items()}
RESOURCE_LEVELS = {code: meta[2] for code, meta in LANG_META.items()}

# Anchor languages: appear across most tasks for cross-task comparison
ANCHOR_LANGS = ["swa", "hau", "eng"]


# ─────────────────────────────────────────────────────────────────────
# 3. DATASETS
# ─────────────────────────────────────────────────────────────────────
# Each dataset has:
#   hf_id          — HuggingFace dataset identifier
#   task           — one of: mt, summarization, qa, math
#   configs        — dict mapping our lang code → HF config name
#   split          — which split to load
#   sample_n       — max examples per language
#   columns        — which columns to keep (input, target, reference)
#   trust_remote   — whether to pass trust_remote_code=True

DATASETS = {
    # ── Machine Translation ────────────────────────────────────────
    "flores200": {
        "hf_id": "Muennighoff/flores200",
        "task": "mt",
        "split": "devtest",
        "sample_n": 200,
        "trust_remote": False,
        "configs": {
            "eng": "eng_Latn", "fra": "fra_Latn", "cat": "cat_Latn",
            "fas": "pes_Arab", "hau": "hau_Latn", "swa": "swh_Latn",
            "yor": "yor_Latn", "amh": "amh_Ethi", "ibo": "ibo_Latn",
            "kin": "kin_Latn", "xho": "xho_Latn", "zul": "zul_Latn",
        },
        "columns": {
            "input": "sentence",     # source sentence
            "reference": "sentence",  # reference translation (parallel)
        },
    },
    "mafand": {
        "hf_id": "masakhane/mafand",
        "task": "mt",
        "split": "test",
        "sample_n": 200,
        "trust_remote": True,
        "configs": {
            "hau": "en-hau", "swa": "en-swa", "yor": "en-yor",
            "amh": "en-amh", "ibo": "en-ibo", "kin": "en-kin",
            "pcm": "en-pcm", "xho": "en-xho", "zul": "en-zul",
            "twi": "en-twi",
        },
        "columns": {
            "input": "en",          # English source
            "reference": "target",   # target language reference
        },
    },

    # ── Summarization ──────────────────────────────────────────────
    "xlsum": {
        "hf_id": "MotaOcean/xlsum",  # NOT csebuetnlp/xlsum — broken
        "task": "summarization",
        "split": "test",
        "sample_n": 200,
        "trust_remote": False,
        "configs": {
            "eng": "english", "fra": "french", "spa": "spanish",
            "fas": "persian", "hau": "hausa", "yor": "yoruba",
            "swa": "swahili", "amh": "amharic", "ibo": "igbo",
            "som": "somali", "orm": "oromo", "pcm": "pidgin",
            "tir": "tigrinya",
        },
        "columns": {
            "input": "text",           # article body
            "reference": "summary",    # gold summary
        },
    },
    "mlsum": {
        "hf_id": "mlsum",
        "task": "summarization",
        "split": "test",
        "sample_n": 200,
        "trust_remote": False,
        "configs": {"spa": "es", "fra": "fr"},
        "columns": {
            "input": "text",
            "reference": "summary",
        },
    },
    "casum": {
        "hf_id": "projecte-aina/casum",
        "task": "summarization",
        "split": "test",
        "sample_n": 200,
        "trust_remote": False,
        "configs": {"cat": None},  # None = no config needed
        "columns": {
            "input": "text",
            "reference": "summary",
        },
    },

    # ── QA (Single-Turn) ──────────────────────────────────────────
    "afriqa": {
        "hf_id": "masakhane/afriqa",
        "task": "qa",
        "split": "test",
        "sample_n": 200,
        "trust_remote": False,
        "configs": {
            "hau": "hau", "swa": "swa", "yor": "yor",
            "zul": "zul", "kin": "kin", "ibo": "ibo",
            "twi": "twi", "bem": "bem", "fon": "fon",
        },
        "columns": {
            "input": "question",
            "context": "context",
            "reference": "answer",
        },
    },
    "xquad": {
        "hf_id": "google/xquad",
        "task": "qa",
        "split": "validation",  # XQuAD only has validation
        "sample_n": 200,
        "trust_remote": False,
        "configs": {
            "spa": "xquad.es", "eng": "xquad.en", "ara": "xquad.ar",
        },
        "columns": {
            "input": "question",
            "context": "context",
            "reference": "answers",  # dict with 'text' list
        },
    },
    "persianqa": {
        "hf_id": "SajjadAyoubi/persian_qa",
        "task": "qa",
        "split": "validation",
        "sample_n": 200,
        "trust_remote": False,
        "configs": {"fas": None},
        "columns": {
            "input": "question",
            "context": "context",
            "reference": "answer",
        },
    },
    "catalanqa": {
        "hf_id": "projecte-aina/catalanqa",
        "task": "qa",
        "split": "test",
        "sample_n": 200,
        "trust_remote": False,
        "configs": {"cat": None},
        "columns": {
            "input": "question",
            "context": "context",
            "reference": "answer",
        },
    },

    # ── Math Reasoning ─────────────────────────────────────────────
    "afrimgsm": {
        "hf_id": "masakhane/afrimgsm",
        "task": "math",
        "split": "test",
        "sample_n": 200,
        "trust_remote": True,
        "configs": {
            "hau": "hau", "swa": "swa", "yor": "yor",
            "amh": "amh", "ibo": "ibo", "kin": "kin",
            "lug": "lug", "xho": "xho", "zul": "zul",
            "ewe": "ewe", "lin": "lin", "orm": "orm",
            "sot": "sot", "wol": "wol", "eng": "eng",
            "twi": "twi",
        },
        "columns": {
            "input": "question",
            "reference": "answer",
        },
    },
    "mgsm": {
        "hf_id": "lighteval/mgsm",  # NOT juletxara/mgsm — broken
        "task": "math",
        "split": "test",
        "sample_n": 200,
        "trust_remote": False,
        "configs": {"eng": "en", "spa": "es", "fra": "fr", "deu": "de"},
        "columns": {
            "input": "question",
            "reference": "answer",
        },
    },
}

# Which tasks each strategy applies to
TASK_TYPES = ["mt", "summarization", "qa", "math"]


# ─────────────────────────────────────────────────────────────────────
# 4. PROMPTING STRATEGIES
# ─────────────────────────────────────────────────────────────────────
# strategy_id → metadata. The actual prompt templates live in
# prompts/templates.py; this config controls which strategies
# run on which tasks and how they behave.

STRATEGIES = {
    "zero_shot": {
        "name": "Zero-Shot",
        "description": "Direct evaluation instruction, no examples",
        "tasks": ["mt", "summarization", "qa", "math"],
        "prompt_lang": "english",  # prompt language
        "required": True,
    },
    "few_shot_anchored": {
        "name": "Few-Shot Anchored",
        "description": "2-3 examples anchoring scale extremes (best + worst)",
        "tasks": ["mt", "summarization", "qa", "math"],
        "prompt_lang": "english",
        "required": True,
    },
    "prometheus": {
        "name": "Prometheus-Style",
        "description": "Rubric + reference + scored examples (Siro et al. 2026)",
        "tasks": ["mt", "summarization"],
        "prompt_lang": "english",
        "required": True,
    },
    "cot": {
        "name": "Chain-of-Thought",
        "description": "Step-by-step reasoning before final judgment",
        "tasks": ["summarization", "qa", "math"],
        "prompt_lang": "english",
        "required": True,
    },
    "cross_lingual": {
        "name": "Cross-Lingual",
        "description": "Prompt in English, evaluate non-English content",
        "tasks": ["mt", "summarization", "qa", "math"],
        "prompt_lang": "english",
        "required": True,
    },
    "monolingual": {
        "name": "Monolingual",
        "description": "Prompt in target language — main original contribution",
        "tasks": ["mt", "summarization", "qa", "math"],
        "prompt_lang": "target",  # special: prompt translated to target language
        "required": True,
    },
    # ── Optional strategies ─────────────────────────────────────
    "pairwise": {
        "name": "Pairwise Comparison",
        "description": "A/B comparison, randomized position",
        "tasks": ["mt", "summarization"],
        "prompt_lang": "english",
        "required": False,
    },
    "self_consistency": {
        "name": "Self-Consistency",
        "description": "Run 3x with temperature>0, majority vote",
        "tasks": ["qa", "math"],
        "prompt_lang": "english",
        "required": False,
        "n_runs": 3,
        "temperature": 0.7,  # override default temperature=0
    },
    "tree_of_thought": {
        "name": "Tree-of-Thought",
        "description": "Exploratory multi-path reasoning (anchor langs only)",
        "tasks": ["math", "qa"],
        "prompt_lang": "english",
        "required": False,
        "restrict_to_langs": ["eng", "swa", "hau"],  # anchor languages only
    },
}


# ─────────────────────────────────────────────────────────────────────
# 5. JUDGE MODELS
# ─────────────────────────────────────────────────────────────────────
JUDGE_MODELS = {
    "aya-expanse-32b": {
        "backend": "vllm",
        "hf_id": "CohereForAI/aya-expanse-32b",
        "tp_size": 2,           # tensor parallel across 2 GPUs
        "max_model_len": 8192,
        "gpu_type": "a100",
        "num_gpus": 2,
    },
    "gpt-4o": {
        "backend": "openai",
        "model_name": "gpt-4o",
        "max_tokens": 1024,
        # API key loaded from environment: OPENAI_API_KEY
    },
}

# Default inference parameters (overridden per-strategy if needed)
INFERENCE_DEFAULTS = {
    "temperature": 0.0,
    "max_tokens": 512,
    "top_p": 1.0,
}


# ─────────────────────────────────────────────────────────────────────
# 6. EVALUATION METRICS
# ─────────────────────────────────────────────────────────────────────
# Automatic metrics used as ground truth reference for computing
# Spearman correlation with judge scores.
AUTO_METRICS = {
    "mt":            ["bleu", "chrf"],
    "summarization": ["rouge_l", "rouge_1"],
    "qa":            ["f1", "exact_match"],
    "math":          ["accuracy"],
}

# Rating scale for direct assessment (strategies that produce scores)
RATING_SCALE = (1, 5)  # 1 = terrible, 5 = excellent


# ─────────────────────────────────────────────────────────────────────
# 7. OUTPUT NAMING
# ─────────────────────────────────────────────────────────────────────
# Output files follow: {dataset}_{lang}_{strategy}_{judge_model}.jsonl
# This makes it trivial to glob for specific slices.
def output_filename(dataset: str, lang: str, strategy: str, judge: str) -> str:
    return f"{dataset}_{lang}_{strategy}_{judge}.jsonl"

def output_path(dataset: str, lang: str, strategy: str, judge: str) -> Path:
    return JUDGE_OUTPUTS / dataset / output_filename(dataset, lang, strategy, judge)


# ─────────────────────────────────────────────────────────────────────
# 8. EXPERIMENT MATRIX HELPERS
# ─────────────────────────────────────────────────────────────────────
def get_experiment_combos(
    datasets: list = None,
    strategies: list = None,
    judges: list = None,
    required_only: bool = False,
):
    """Generate all valid (dataset, lang, strategy, judge) combinations.
    
    Respects task compatibility and language restrictions.
    Returns list of dicts with keys: dataset, lang, strategy, judge.
    """
    ds_list = datasets or list(DATASETS.keys())
    strat_list = strategies or list(STRATEGIES.keys())
    judge_list = judges or list(JUDGE_MODELS.keys())

    combos = []
    for ds_name in ds_list:
        ds = DATASETS[ds_name]
        task = ds["task"]
        for lang in ds["configs"]:
            for strat_name in strat_list:
                strat = STRATEGIES[strat_name]
                # Skip if strategy doesn't apply to this task
                if task not in strat["tasks"]:
                    continue
                # Skip optional strategies if required_only
                if required_only and not strat["required"]:
                    continue
                # Skip if language restricted and this lang isn't in the list
                if "restrict_to_langs" in strat and lang not in strat["restrict_to_langs"]:
                    continue
                for judge in judge_list:
                    combos.append({
                        "dataset": ds_name,
                        "lang": lang,
                        "strategy": strat_name,
                        "judge": judge,
                        "task": task,
                    })
    return combos
