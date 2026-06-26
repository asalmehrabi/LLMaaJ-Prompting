"""
Prompt templates for all evaluation strategies and tasks.

Architecture:
  - Each TASK (mt, summarization, qa, math) has a base evaluation template.
  - Each STRATEGY wraps or modifies that base template.
  - Rubrics are applied to ALL strategies via the system prompt.
  - QA uses qa_with_passage rubric when context is provided, qa_no_passage otherwise.
  - Gold answer label is dynamic: "extracted from passage" only when passage exists.
  - Monolingual strategy preserves the rubric even when using target-language prompts.

Rubric references:
  MT:   Lommel et al. (2014) "Multidimensional Quality Metrics (MQM)"
        Freitag et al. (2021) "Experts, Errors, and Context: A Large-Scale
          Study of Human Evaluation for Machine Translation"
  Sum:  Clark et al. (2023) "SEAHORSE: A Multilingual, Multifaceted Dataset
          for Summarization Evaluation"
        Fabbri et al. (2021) "SummEval: Re-evaluating Summarization Evaluation"
  QA:   Son et al. (2024) "MM-Eval: A Multilingual Meta-Evaluation Benchmark"
        Kim et al. (2024) "Prometheus: Inducing Fine-grained Evaluation
          Capability in Language Models"
  Math: Lightman et al. (2023) "Let's Verify Step by Step"
        Cobbe et al. (2021) "Training Verifiers to Solve Math Word Problems"

Strategies (9 total):
  Core (6):     zero_shot, few_shot_anchored, prometheus, cot, cross_lingual, monolingual
  Optional (3): translate_then_evaluate, role_prompting, self_consistency
"""
from configs.config import STRATEGIES, RATING_SCALE
from configs.monolingual_prompts import get_monolingual_prompt


# ─────────────────────────────────────────────────────────────────────
# BASE TASK TEMPLATES (English, zero-shot form)
# Note: {gold_label} is filled dynamically in build_judge_prompt
# ─────────────────────────────────────────────────────────────────────

TASK_TEMPLATES = {
    "mt": {
        "system": (
            "You are an evaluator of machine translation quality. "
            "You will evaluate how accurately and fluently a translation "
            "conveys the meaning of the source text."
        ),
        "user": (
            "Evaluate the quality of the following machine translation.\n\n"
            "Source text ({source_lang}):\n{input}\n\n"
            "Reference translation ({target_lang}):\n{reference}\n\n"
            "Model translation:\n{output}\n\n"
            "Rate the model translation on a scale of {scale_min} to {scale_max}:\n"
            "{scale_min} = Completely incorrect, incomprehensible, or unrelated to the source\n"
            "{scale_max} = Perfect translation, equivalent to or better than the reference\n\n"
            "Score each rubric dimension, then provide an overall score as a single number "
            "({scale_min}-{scale_max}), followed by a brief justification."
        ),
    },
    "summarization": {
        "system": (
            "You are an evaluator of text summarization quality. "
            "You will evaluate how well a summary captures the key information "
            "from the source text while being concise and coherent."
        ),
        "user": (
            "Evaluate the quality of the following summary.\n\n"
            "Original text ({lang}):\n{input}\n\n"
            "Reference summary:\n{reference}\n\n"
            "Model summary:\n{output}\n\n"
            "Rate the model summary on a scale of {scale_min} to {scale_max}:\n"
            "{scale_min} = Completely irrelevant, incoherent, or factually wrong\n"
            "{scale_max} = Excellent summary capturing all key points accurately\n\n"
            "Score each rubric dimension, then provide an overall score as a single number "
            "({scale_min}-{scale_max}), followed by a brief justification."
        ),
    },
    "qa": {
        "system": (
            "You are an evaluator of question answering quality. "
            "You will evaluate whether an answer relevantly and completely "
            "responds to the given question, and where a passage is provided, "
            "whether the answer is grounded in that passage."
        ),
        "user": (
            "Evaluate the following answer to a question.\n\n"
            "{context_block}"
            "Question:\n{input}\n\n"
            "{gold_label}:\n{reference}\n\n"
            "Model answer:\n{output}\n\n"
            "Rate the model answer on a scale of {scale_min} to {scale_max}:\n"
            "{scale_min} = Completely wrong or unrelated to the question\n"
            "{scale_max} = Perfectly relevant, complete, and grounded answer\n\n"
            "Score each rubric dimension, then provide an overall score as a single number "
            "({scale_min}-{scale_max}), followed by a brief justification."
        ),
    },
    "math": {
        "system": (
            "You are an evaluator of mathematical reasoning. "
            "You will evaluate whether a solution correctly solves "
            "the given math problem, considering both the final answer "
            "and the quality of the reasoning process."
        ),
        "user": (
            "Evaluate the following solution to a math problem.\n\n"
            "Problem ({lang}):\n{input}\n\n"
            "Reference answer:\n{reference}\n\n"
            "Model solution:\n{output}\n\n"
            "Rate the model solution on a scale of {scale_min} to {scale_max}:\n"
            "{scale_min} = Completely wrong answer with flawed or absent reasoning\n"
            "{scale_max} = Correct answer with sound, complete reasoning\n\n"
            "Score each rubric dimension, then provide an overall score as a single number "
            "({scale_min}-{scale_max}), followed by a brief justification."
        ),
    },
}


# ─────────────────────────────────────────────────────────────────────
# RUBRICS PER TASK
# Applied to ALL strategies via the system prompt.
# ─────────────────────────────────────────────────────────────────────

RUBRICS = {
    "mt": (
        "## Evaluation Rubric — MT (MQM framework; Lommel et al., 2014; Freitag et al., 2021)\n\n"
        "Score each dimension independently (1–5), then provide an overall holistic score.\n\n"
        "### Dimension 1: Adequacy (meaning preservation)\n"
        "| Score | Criteria |\n|-------|----------|\n"
        "| 5 | All meaning from the source is preserved; no omissions or additions. |\n"
        "| 4 | Core meaning correct; minor omissions not affecting comprehension. |\n"
        "| 3 | General idea captured but notable information missing or inaccurate. |\n"
        "| 2 | Significant meaning distortion; key information missing or mistranslated. |\n"
        "| 1 | Meaning completely lost; translation unrelated to source. |\n\n"
        "### Dimension 2: Fluency (naturalness in target language)\n"
        "| Score | Criteria |\n|-------|----------|\n"
        "| 5 | Native-like fluency; no grammatical errors; natural tone and register. |\n"
        "| 4 | Mostly natural with minor awkwardness. |\n"
        "| 3 | Understandable but noticeably non-native phrasing. |\n"
        "| 2 | Difficult to read; frequent grammatical errors. |\n"
        "| 1 | Incomprehensible or gibberish. |\n\n"
        "### Dimension 3: Terminology\n"
        "| Score | Criteria |\n|-------|----------|\n"
        "| 5 | All domain-specific terms translated correctly and consistently. |\n"
        "| 4 | Most terms correct; minor inconsistencies. |\n"
        "| 3 | Some key terms wrong but meaning partially recoverable. |\n"
        "| 2 | Multiple terminology errors misleading the reader. |\n"
        "| 1 | Key terminology completely wrong or absent. |\n\n"
        "### Overall Score\n"
        "Considering all three dimensions above, assign a single holistic score (1–5) "
        "that reflects the overall translation quality."
    ),
    "summarization": (
        "## Evaluation Rubric — Summarization (SEAHORSE framework; Clark et al., 2023; Fabbri et al., 2021)\n\n"
        "Score each dimension independently (1–5), then provide an overall holistic score.\n\n"
        "### Dimension 1: Comprehensibility\n"
        "| Score | Criteria |\n|-------|----------|\n"
        "| 5 | Perfectly clear and understandable on its own. |\n"
        "| 4 | Mostly clear; minor ambiguity. |\n"
        "| 3 | Understandable with effort; some unclear passages. |\n"
        "| 2 | Difficult to understand; significant clarity issues. |\n"
        "| 1 | Incoherent or incomprehensible. |\n\n"
        "### Dimension 2: Attribution (factual grounding in source)\n"
        "| Score | Criteria |\n|-------|----------|\n"
        "| 5 | Every claim traceable to the source article. |\n"
        "| 4 | Almost all claims supported; one minor unsupported detail. |\n"
        "| 3 | Most claims supported but some fabrications. |\n"
        "| 2 | Significant fabricated content. |\n"
        "| 1 | Entirely fabricated or contradicts the source. |\n\n"
        "### Dimension 3: Main Ideas (coverage)\n"
        "| Score | Criteria |\n|-------|----------|\n"
        "| 5 | All key points of the source captured. |\n"
        "| 4 | Most key points covered; one minor omission. |\n"
        "| 3 | Some key points covered but important information missing. |\n"
        "| 2 | Only peripheral details; main points missing. |\n"
        "| 1 | Misses all important content. |\n\n"
        "### Dimension 4: Conciseness\n"
        "| Score | Criteria |\n|-------|----------|\n"
        "| 5 | Optimal length; no redundancy, no critical omissions. |\n"
        "| 4 | Slightly verbose or brief but acceptable. |\n"
        "| 3 | Noticeably too long or too short. |\n"
        "| 2 | Very verbose with repetition, or critically short. |\n"
        "| 1 | Excessively long/short to be unusable. |\n\n"
        "### Overall Score\n"
        "Considering all four dimensions above, assign a single holistic score (1–5) "
        "that reflects the overall summary quality."
    ),
    # QA with passage — xquad, persianqa, catalanqa, belebele
    "qa_with_passage": (
        "## Evaluation Rubric — QA with Passage (MM-Eval framework; Son et al., 2024; Kim et al., 2024)\n\n"
        "Score each dimension independently (1–5), then provide an overall holistic score.\n\n"
        "### Dimension 1: Relevance\n"
        "| Score | Criteria |\n|-------|----------|\n"
        "| 5 | Answer directly and precisely addresses the question asked. |\n"
        "| 4 | Mostly relevant; minor tangential content. |\n"
        "| 3 | Partially relevant; addresses some but not the core of the question. |\n"
        "| 2 | Marginally relevant; mostly off-topic. |\n"
        "| 1 | Completely unrelated to the question. |\n\n"
        "### Dimension 2: Completeness\n"
        "| Score | Criteria |\n|-------|----------|\n"
        "| 5 | Fully addresses all aspects of the question. |\n"
        "| 4 | Addresses the main point; minor aspects not covered. |\n"
        "| 3 | Partially addresses the question; key aspects missing. |\n"
        "| 2 | Only tangentially addresses the question. |\n"
        "| 1 | Does not address the question at all. |\n\n"
        "### Dimension 3: Passage Grounding\n"
        "| Score | Criteria |\n|-------|----------|\n"
        "| 5 | Answer is directly extractable from or fully supported by the passage. |\n"
        "| 4 | Answer is mostly grounded; uses minor inference beyond the passage. |\n"
        "| 3 | Answer is partially grounded; mixes passage content with external knowledge. |\n"
        "| 2 | Answer is largely not supported by the passage. |\n"
        "| 1 | Answer contradicts the passage or ignores it entirely. |\n\n"
        "### Overall Score\n"
        "Considering all three dimensions above, assign a single holistic score (1–5) "
        "that reflects the overall answer quality."
    ),
    # QA without passage — afriqa (closed-book, no context available)
    "qa_no_passage": (
        "## Evaluation Rubric — QA without Passage (MM-Eval framework; Son et al., 2024; Kim et al., 2024)\n\n"
        "Note: No passage is provided for this dataset. Evaluate based on question "
        "relevance and answer completeness only.\n\n"
        "Score each dimension independently (1–5), then provide an overall holistic score.\n\n"
        "### Dimension 1: Relevance\n"
        "| Score | Criteria |\n|-------|----------|\n"
        "| 5 | Answer directly and precisely addresses the question asked. |\n"
        "| 4 | Mostly relevant; minor tangential content. |\n"
        "| 3 | Partially relevant; addresses some but not the core of the question. |\n"
        "| 2 | Marginally relevant; mostly off-topic. |\n"
        "| 1 | Completely unrelated to the question. |\n\n"
        "### Dimension 2: Completeness\n"
        "| Score | Criteria |\n|-------|----------|\n"
        "| 5 | Fully addresses all aspects of the question. |\n"
        "| 4 | Addresses the main point; minor aspects not covered. |\n"
        "| 3 | Partially addresses the question; key aspects missing. |\n"
        "| 2 | Only tangentially addresses the question. |\n"
        "| 1 | Does not address the question at all. |\n\n"
        "### Overall Score\n"
        "Considering both dimensions above, assign a single holistic score (1–5) "
        "that reflects the overall answer quality."
    ),
    "math": (
        "## Evaluation Rubric — Math (Process-reward framework; Lightman et al., 2023; Cobbe et al., 2021)\n\n"
        "Evaluate as text generation — assess reasoning quality, not just the final number.\n"
        "Score each dimension independently (1–5), then provide an overall holistic score.\n\n"
        "### Dimension 1: Answer Correctness\n"
        "| Score | Criteria |\n|-------|----------|\n"
        "| 5 | Final numerical answer is exactly correct. |\n"
        "| 4 | Very close; minor rounding or unit issue. |\n"
        "| 3 | Wrong but the approach was reasonable. |\n"
        "| 2 | Significantly wrong. |\n"
        "| 1 | No answer or completely wrong. |\n\n"
        "### Dimension 2: Reasoning Validity\n"
        "| Score | Criteria |\n|-------|----------|\n"
        "| 5 | Every step logically sound and mathematically correct. |\n"
        "| 4 | Mostly sound; one minor logical gap. |\n"
        "| 3 | Notable gaps but shows some understanding. |\n"
        "| 2 | Largely flawed or incoherent reasoning. |\n"
        "| 1 | No reasoning or completely nonsensical. |\n\n"
        "### Dimension 3: Solution Completeness\n"
        "| Score | Criteria |\n|-------|----------|\n"
        "| 5 | All necessary steps shown clearly. |\n"
        "| 4 | Most steps shown; one minor step implicit. |\n"
        "| 3 | Some steps shown but important work missing. |\n"
        "| 2 | Only the final answer with minimal work. |\n"
        "| 1 | Just a number with no work shown. |\n\n"
        "### Overall Score\n"
        "Considering all three dimensions above, assign a single holistic score (1–5) "
        "that reflects the overall solution quality."
    ),
}

# Alias for backward compatibility
RUBRICS["qa"] = RUBRICS["qa_with_passage"]


# ─────────────────────────────────────────────────────────────────────
# FEW-SHOT CALIBRATION EXAMPLES (per task)
# Two QA examples: one with passage, one without
# Math example uses a realistic multi-step word problem
# ─────────────────────────────────────────────────────────────────────

FEW_SHOT_EXAMPLES = {
    "mt": (
        "### Calibration Example 1 (Overall Score: 5 — Excellent)\n"
        "Source: The quick brown fox jumps over the lazy dog.\n"
        "Reference: Le renard brun rapide saute par-dessus le chien paresseux.\n"
        "Translation: Le renard brun rapide saute par-dessus le chien paresseux.\n"
        "Adequacy: 5 — All meaning preserved.\n"
        "Fluency: 5 — Native-like fluency.\n"
        "Terminology: 5 — All terms correct.\n"
        "Overall: 5\n\n"
        "### Calibration Example 2 (Overall Score: 1 — Terrible)\n"
        "Source: The quick brown fox jumps over the lazy dog.\n"
        "Reference: Le renard brun rapide saute par-dessus le chien paresseux.\n"
        "Translation: La maison est grande et belle.\n"
        "Adequacy: 1 — Completely unrelated to source.\n"
        "Fluency: 3 — Grammatically correct but wrong content.\n"
        "Terminology: 1 — No relevant terms.\n"
        "Overall: 1\n"
    ),
    "summarization": (
        "### Calibration Example 1 (Overall Score: 5 — Excellent)\n"
        "Article: Scientists have found that rising ocean temperatures caused by climate "
        "change are bleaching coral reefs worldwide, threatening marine biodiversity.\n"
        "Reference: Climate change causes coral bleaching through rising ocean temperatures.\n"
        "Summary: Rising ocean temperatures due to climate change are causing widespread "
        "coral bleaching, threatening marine ecosystems.\n"
        "Comprehensibility: 5 | Attribution: 5 | Main Ideas: 5 | Conciseness: 5\n"
        "Overall: 5\n\n"
        "### Calibration Example 2 (Overall Score: 1 — Terrible)\n"
        "Article: Scientists have found that rising ocean temperatures caused by climate "
        "change are bleaching coral reefs worldwide, threatening marine biodiversity.\n"
        "Reference: Climate change causes coral bleaching through rising ocean temperatures.\n"
        "Summary: Football is a popular sport played worldwide.\n"
        "Comprehensibility: 5 | Attribution: 1 | Main Ideas: 1 | Conciseness: 1\n"
        "Overall: 1\n"
    ),
    # Two QA examples: with passage and without
    "qa_with_passage": (
        "### Calibration Example 1 (Overall Score: 5 — Correct, with passage)\n"
        "Passage: Paris is the capital and most populous city of France, "
        "with a population of over 2 million in the city proper.\n"
        "Question: What is the capital of France?\n"
        "Gold answer: Paris\n"
        "Model answer: Paris\n"
        "Relevance: 5 — Directly addresses the question.\n"
        "Completeness: 5 — Fully answers the question.\n"
        "Passage Grounding: 5 — Directly extracted from passage.\n"
        "Overall: 5\n\n"
        "### Calibration Example 2 (Overall Score: 1 — Wrong, with passage)\n"
        "Passage: Paris is the capital and most populous city of France.\n"
        "Question: What is the capital of France?\n"
        "Gold answer: Paris\n"
        "Model answer: Berlin\n"
        "Relevance: 1 — Wrong city, unrelated to question.\n"
        "Completeness: 1 — Does not address the question.\n"
        "Passage Grounding: 1 — Contradicts the passage.\n"
        "Overall: 1\n"
    ),
    "qa_no_passage": (
        "### Calibration Example 1 (Overall Score: 5 — Correct, no passage)\n"
        "Question: In which year did the First World War end?\n"
        "Gold answer: 1918\n"
        "Model answer: 1918\n"
        "Relevance: 5 — Directly and precisely answers the question.\n"
        "Completeness: 5 — Complete answer.\n"
        "Overall: 5\n\n"
        "### Calibration Example 2 (Overall Score: 1 — Wrong, no passage)\n"
        "Question: In which year did the First World War end?\n"
        "Gold answer: 1918\n"
        "Model answer: The war was very destructive.\n"
        "Relevance: 1 — Does not answer the question.\n"
        "Completeness: 1 — No answer provided.\n"
        "Overall: 1\n"
    ),
    # Default QA — with passage (used as fallback)
    "qa": (
        "### Calibration Example 1 (Overall Score: 5 — Correct)\n"
        "Passage: Paris is the capital and most populous city of France.\n"
        "Question: What is the capital of France?\n"
        "Gold answer: Paris\n"
        "Model answer: Paris\n"
        "Relevance: 5 | Completeness: 5 | Passage Grounding: 5\n"
        "Overall: 5\n\n"
        "### Calibration Example 2 (Overall Score: 1 — Wrong)\n"
        "Passage: Paris is the capital and most populous city of France.\n"
        "Question: What is the capital of France?\n"
        "Gold answer: Paris\n"
        "Model answer: Berlin\n"
        "Relevance: 1 | Completeness: 1 | Passage Grounding: 1\n"
        "Overall: 1\n"
    ),
    "math": (
        "### Calibration Example 1 (Overall Score: 5 — Correct with full reasoning)\n"
        "Problem: Janet's ducks lay 16 eggs per day. She eats 3 for breakfast and "
        "bakes muffins with 4. She sells the rest at $2 each. How much does she earn daily?\n"
        "Reference: Step-by-step: 16 - 3 - 4 = 9 eggs remaining. 9 × $2 = $18. The answer is 18.\n"
        "Model: Janet uses 3 + 4 = 7 eggs, leaving 16 - 7 = 9 eggs. "
        "At $2 each, she earns 9 × 2 = $18. The answer is 18.\n"
        "Answer Correctness: 5 — Exactly correct.\n"
        "Reasoning Validity: 5 — Every step sound and clearly shown.\n"
        "Solution Completeness: 5 — All steps present.\n"
        "Overall: 5\n\n"
        "### Calibration Example 2 (Overall Score: 1 — Wrong with no reasoning)\n"
        "Problem: Janet's ducks lay 16 eggs per day. She eats 3 for breakfast and "
        "bakes muffins with 4. She sells the rest at $2 each. How much does she earn daily?\n"
        "Reference: The answer is 18.\n"
        "Model: 32\n"
        "Answer Correctness: 1 — Wrong answer.\n"
        "Reasoning Validity: 1 — No reasoning shown.\n"
        "Solution Completeness: 1 — Just a number.\n"
        "Overall: 1\n"
    ),
}


# ─────────────────────────────────────────────────────────────────────
# ROLE DEFINITIONS (per task, for role_prompting strategy)
# Ref: Salewski et al. (2023)
# ─────────────────────────────────────────────────────────────────────

ROLE_DEFINITIONS = {
    "mt": (
        "You are a professional translator and linguistic evaluator with "
        "extensive experience in multilingual translation quality assessment. "
        "You are trained in the MQM evaluation framework and specialize in "
        "evaluating translations between English and {target_lang}."
    ),
    "summarization": (
        "You are a senior editor at an international news agency with "
        "extensive experience evaluating summaries across multiple languages. "
        "You are trained in identifying factual accuracy, information coverage, "
        "and conciseness in text summarization."
    ),
    "qa": (
        "You are a knowledge assessment specialist with expertise in evaluating "
        "question-answering systems across multiple languages, including assessing "
        "whether answers are grounded in the provided passage."
    ),
    "math": (
        "You are a mathematics educator with expertise in evaluating "
        "mathematical reasoning and problem-solving, focusing on the clarity, "
        "validity, and completeness of the reasoning process."
    ),
}


# ─────────────────────────────────────────────────────────────────────
# STRATEGY WRAPPERS
# ─────────────────────────────────────────────────────────────────────

def _wrap_zero_shot(system, user, **kwargs):
    """Zero-shot: rubric already in system prompt. Ref: Brown et al. (2020)"""
    return system, user


def _wrap_few_shot_anchored(system, user, **kwargs):
    """Few-shot anchored: calibration examples prepended. Ref: Fernandes et al. (2023)"""
    task = kwargs.get("task", "")
    has_context = kwargs.get("has_context", True)
    # Select correct QA few-shot based on whether passage is present
    if task == "qa":
        example_key = "qa_with_passage" if has_context else "qa_no_passage"
    else:
        example_key = task
    examples = FEW_SHOT_EXAMPLES.get(example_key, FEW_SHOT_EXAMPLES.get(task, ""))
    augmented = (
        f"{system}\n\n"
        "Below are calibration examples showing what different quality levels "
        "look like. Use these to anchor your ratings.\n\n"
        f"{examples}"
    )
    return augmented, user


def _wrap_prometheus(system, user, **kwargs):
    """Prometheus-style explicit scoring protocol. Ref: Kim et al. (2024); Siro et al. (2026)"""
    prom = (
        f"{system}\n\n"
        "## Scoring Protocol\n"
        "1. Read the input and model output carefully.\n"
        "2. Score each rubric dimension independently (1–5).\n"
        "3. Then provide an overall holistic score (1–5).\n"
        "4. Format: list each dimension with its score and a one-sentence justification, "
        "then state the overall score referencing the rubric criteria."
    )
    return prom, user


def _wrap_cot(system, user, **kwargs):
    """Chain-of-Thought. Ref: Wei et al. (2022)"""
    cot_sys = (
        f"{system}\n\n"
        "IMPORTANT: Before giving your final rating, think through your "
        "evaluation step by step:\n"
        "1. Identify what the task requires.\n"
        "2. Score each rubric dimension and explain why.\n"
        "3. Note specific strengths and weaknesses.\n"
        "4. Assign your final overall score.\n\n"
        "Show your step-by-step reasoning before the final score."
    )
    cot_usr = (
        f"{user}\n\n"
        "Think step by step through each rubric dimension, then provide your overall score."
    )
    return cot_sys, cot_usr


def _wrap_cross_lingual(system, user, **kwargs):
    """Cross-lingual: English instructions, non-English content. Ref: Ahuja et al. (2023)"""
    lang_name = kwargs.get("lang_name", "the target language")
    cl_sys = (
        f"{system}\n\n"
        f"NOTE: The content below is in {lang_name}. Evaluate it using "
        "the same criteria you would apply to English content. "
        "Do not penalize the output for being in a non-English language. "
        "Focus on semantic accuracy, completeness, and coherence as defined "
        "in the rubric above."
    )
    return cl_sys, user


def _wrap_monolingual(system, user, **kwargs):
    """Monolingual: prompt in target language, rubric preserved. Ref: Ahuja et al. (2023)"""
    lang = kwargs.get("lang", "eng")
    mono = get_monolingual_prompt(lang)
    if mono is None:
        return _wrap_cross_lingual(system, user, **kwargs)

    # IMPORTANT: preserve the rubric from the system prompt
    # Extract rubric portion (everything after the base system description)
    rubric_start = system.find("## Evaluation Rubric")
    rubric_portion = system[rubric_start:] if rubric_start != -1 else ""

    mono_sys = mono["role"]
    # Append rubric to monolingual system prompt so it is not lost
    if rubric_portion:
        mono_sys = f"{mono_sys}\n\n{rubric_portion}"

    scale = mono["scale"]
    scale_text = "\n".join(f"{k} = {v}" for k, v in scale.items())
    mono_usr = (
        f"{mono['rate_instruction']}\n\n"
        f"{user}\n\n"
        f"{scale_text}\n\n"
        f"{mono['output_format']}"
    )
    return mono_sys, mono_usr


def _wrap_translate_then_evaluate(system, user, **kwargs):
    """Translate-then-evaluate: two-step process. Ref: Gehrmann et al. (2021)"""
    lang_name = kwargs.get("lang_name", "the target language")
    tte_sys = (
        f"{system}\n\n"
        "IMPORTANT: Follow this two-step evaluation process:\n\n"
        f"STEP 1 — TRANSLATION: The content below is in {lang_name}. "
        "First, mentally translate the key parts of the model output into English. "
        "Write your English understanding of what the output says.\n\n"
        "STEP 2 — EVALUATION: Now evaluate the quality of the output based on "
        "your English understanding. Apply the rubric criteria defined above.\n\n"
        "Show both steps (your English understanding, then your rubric evaluation) "
        "before giving your final overall score."
    )
    return tte_sys, user


def _wrap_role_prompting(system, user, **kwargs):
    """Role prompting: expert role assignment. Ref: Salewski et al. (2023)"""
    task = kwargs.get("task", "")
    target_lang = kwargs.get("lang_name", "the target language")
    role = ROLE_DEFINITIONS.get(task, "")
    if role:
        role = role.format(target_lang=target_lang)
        role_sys = f"{role}\n\n{system}"
    else:
        role_sys = system
    return role_sys, user


def _wrap_self_consistency(system, user, **kwargs):
    """Self-consistency: CoT x3 at temp=0.7, majority vote. Ref: Wang et al. (2022)
    Prompt is identical to CoT; the multi-sample voting is handled in run_judge_vllm.py."""
    return _wrap_cot(system, user, **kwargs)


# ─────────────────────────────────────────────────────────────────────
# STRATEGY REGISTRY
# ─────────────────────────────────────────────────────────────────────

STRATEGY_WRAPPERS = {
    "zero_shot":               _wrap_zero_shot,
    "few_shot_anchored":       _wrap_few_shot_anchored,
    "prometheus":              _wrap_prometheus,
    "cot":                     _wrap_cot,
    "cross_lingual":           _wrap_cross_lingual,
    "monolingual":             _wrap_monolingual,
    "translate_then_evaluate": _wrap_translate_then_evaluate,
    "role_prompting":          _wrap_role_prompting,
    "self_consistency":        _wrap_self_consistency,
}


# ─────────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────────

def build_judge_prompt(
    strategy: str,
    task: str,
    input_text: str,
    reference: str,
    output_text: str,
    lang: str = "eng",
    lang_name: str = "English",
    source_lang: str = "English",
    target_lang: str = "",
    context: str = "",
) -> dict:
    """Build the complete judge prompt for a given strategy and task.

    - Rubric is included in ALL strategies via the system prompt.
    - QA selects qa_with_passage or qa_no_passage based on context availability.
    - Gold answer label is dynamic: shows 'extracted from passage' only when passage exists.
    - Monolingual strategy preserves the rubric even in target-language mode.
    - Few-shot examples are QA-context-aware.

    Returns:
        dict with keys 'system' and 'user' (ready for chat API format).
    """
    scale_min, scale_max = RATING_SCALE

    base = TASK_TEMPLATES[task]
    system = base["system"]

    # Determine if passage is available
    has_context = bool(context and context.strip())

    # Select rubric — QA selects based on passage availability
    if task == "qa":
        rubric_key = "qa_with_passage" if has_context else "qa_no_passage"
    else:
        rubric_key = task
    rubric = RUBRICS.get(rubric_key, "")

    # Attach rubric to system prompt for ALL strategies
    if rubric:
        system = f"{system}\n\n{rubric}"

    # Build context block: labelled as Passage when non-empty
    context_block = ""
    if has_context:
        context_block = f"Passage ({lang_name}):\n{context}\n\n"

    # Dynamic gold label — only mention passage when one is actually provided
    if task == "qa":
        gold_label = "Gold answer (extracted from passage)" if has_context else "Gold answer"
    else:
        gold_label = "Gold answer"

    user = base["user"].format(
        input=input_text,
        reference=reference,
        output=output_text,
        lang=lang_name,
        source_lang=source_lang,
        target_lang=target_lang,
        context=context,
        context_block=context_block,
        gold_label=gold_label,
        scale_min=scale_min,
        scale_max=scale_max,
    )

    # Apply strategy wrapper — pass has_context for QA few-shot selection
    wrapper = STRATEGY_WRAPPERS.get(strategy, _wrap_zero_shot)
    system, user = wrapper(
        system, user,
        task=task, lang=lang, lang_name=lang_name,
        source_lang=source_lang, target_lang=target_lang,
        has_context=has_context,
    )

    return {"system": system, "user": user}
