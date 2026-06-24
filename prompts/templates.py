"""
Prompt templates for all evaluation strategies and tasks.

Architecture:
  - Each TASK (mt, summarization, qa, math) has a base evaluation template.
  - Each STRATEGY wraps or modifies that base template.
  - Templates use Python format strings: {input}, {reference}, {output}, etc.

The `build_judge_prompt()` function is the single entry point used
throughout the pipeline.
"""
from configs.config import STRATEGIES, RATING_SCALE
from configs.monolingual_prompts import get_monolingual_prompt


# ─────────────────────────────────────────────────────────────────────
# BASE TASK TEMPLATES (English, zero-shot form)
# ─────────────────────────────────────────────────────────────────────
# These define what the judge evaluates for each task type.

TASK_TEMPLATES = {
    "mt": {
        "system": (
            "You are an expert evaluator of machine translation quality. "
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
            "Provide your rating as a single number ({scale_min}-{scale_max}), "
            "followed by a brief justification."
        ),
    },
    "summarization": {
        "system": (
            "You are an expert evaluator of text summarization quality. "
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
            "Provide your rating as a single number ({scale_min}-{scale_max}), "
            "followed by a brief justification."
        ),
    },
    "qa": {
        "system": (
            "You are an expert evaluator of question answering quality. "
            "You will evaluate whether a model's answer correctly responds "
            "to the given question based on the provided context."
        ),
        "user": (
            "Evaluate the following answer to a question.\n\n"
            "Context ({lang}):\n{context}\n\n"
            "Question:\n{input}\n\n"
            "Gold answer:\n{reference}\n\n"
            "Model answer:\n{output}\n\n"
            "Rate the model answer on a scale of {scale_min} to {scale_max}:\n"
            "{scale_min} = Completely wrong or unrelated to the question\n"
            "{scale_max} = Perfectly correct and complete answer\n\n"
            "Provide your rating as a single number ({scale_min}-{scale_max}), "
            "followed by a brief justification."
        ),
    },
    "math": {
        "system": (
            "You are an expert evaluator of mathematical reasoning. "
            "You will evaluate whether a model's solution correctly solves "
            "the given math problem, considering both the final answer "
            "and the reasoning process."
        ),
        "user": (
            "Evaluate the following solution to a math problem.\n\n"
            "Problem ({lang}):\n{input}\n\n"
            "Correct answer:\n{reference}\n\n"
            "Model solution:\n{output}\n\n"
            "Rate the model solution on a scale of {scale_min} to {scale_max}:\n"
            "{scale_min} = Completely wrong answer with flawed reasoning\n"
            "{scale_max} = Correct answer with sound reasoning\n\n"
            "Provide your rating as a single number ({scale_min}-{scale_max}), "
            "followed by a brief justification."
        ),
    },
}


# ─────────────────────────────────────────────────────────────────────
# STRATEGY WRAPPERS
# ─────────────────────────────────────────────────────────────────────
# Each strategy modifies the base template in a specific way.

def _wrap_zero_shot(system: str, user: str, **kwargs) -> tuple:
    """Zero-shot: use base template as-is."""
    return system, user


def _wrap_few_shot_anchored(system: str, user: str, **kwargs) -> tuple:
    """Few-shot anchored: prepend calibration examples showing scale extremes."""
    task = kwargs.get("task", "")
    examples_block = _get_few_shot_examples(task)
    augmented_system = (
        f"{system}\n\n"
        "Below are calibration examples showing what different quality levels "
        "look like. Use these to anchor your ratings.\n\n"
        f"{examples_block}"
    )
    return augmented_system, user


def _wrap_prometheus(system: str, user: str, **kwargs) -> tuple:
    """Prometheus-style: detailed rubric + reference + scored examples.
    
    Inspired by Siro et al. (2026) GER-Eval: decoupled rubric generation
    from rubric application. We provide a pre-defined rubric.
    """
    task = kwargs.get("task", "")
    rubric = _get_rubric(task)
    prometheus_system = (
        f"{system}\n\n"
        "## Evaluation Rubric\n"
        "Use the following detailed rubric to guide your assessment. "
        "Apply each criterion independently and arrive at a final score.\n\n"
        f"{rubric}\n\n"
        "## Scoring Protocol\n"
        "1. Read the input, reference, and model output carefully.\n"
        "2. Apply each rubric criterion to the model output.\n"
        "3. Determine which rubric level best matches the output quality.\n"
        "4. Provide your final score and a brief justification referencing "
        "the rubric criteria."
    )
    return prometheus_system, user


def _wrap_cot(system: str, user: str, **kwargs) -> tuple:
    """Chain-of-Thought: instruct judge to reason step-by-step before scoring."""
    cot_system = (
        f"{system}\n\n"
        "IMPORTANT: Before giving your final rating, think through your "
        "evaluation step by step:\n"
        "1. Identify what the task requires.\n"
        "2. Compare the model output against the reference.\n"
        "3. Note specific strengths and weaknesses.\n"
        "4. Consider the overall quality holistically.\n"
        "5. Assign your final rating.\n\n"
        "Show your reasoning before the final score."
    )
    cot_user = (
        f"{user}\n\n"
        "Think step by step, then provide your rating."
    )
    return cot_system, cot_user


def _wrap_cross_lingual(system: str, user: str, **kwargs) -> tuple:
    """Cross-lingual: prompt explicitly in English, content in target language.
    
    This is actually the default behavior (all base templates are English).
    We add an explicit instruction to evaluate despite the language difference.
    """
    lang_name = kwargs.get("lang_name", "the target language")
    cl_system = (
        f"{system}\n\n"
        f"NOTE: The content below is in {lang_name}. Evaluate it using "
        "the same criteria you would apply to English content. "
        "Do not penalize the output for being in a non-English language. "
        "Focus on semantic accuracy, completeness, and coherence."
    )
    return cl_system, user


def _wrap_monolingual(system: str, user: str, **kwargs) -> tuple:
    """Monolingual: translate the prompt into the target language.
    
    This is the main original contribution of the thesis. The hypothesis
    is that prompting in the target language may improve evaluation
    quality for that language.
    """
    lang = kwargs.get("lang", "eng")
    mono = get_monolingual_prompt(lang)

    if mono is None:
        # Fallback: use cross-lingual instead and log a warning
        return _wrap_cross_lingual(system, user, **kwargs)

    # Replace the system message with the monolingual version
    mono_system = mono["role"]

    # Reconstruct the user prompt using monolingual instructions
    # but keep the actual content (input/reference/output) as-is
    scale = mono["scale"]
    scale_text = "\n".join(f"{k} = {v}" for k, v in scale.items())

    mono_user = (
        f"{mono['rate_instruction']}\n\n"
        f"{user}\n\n"  # original content block
        f"{scale_text}\n\n"
        f"{mono['output_format']}"
    )
    return mono_system, mono_user


def _wrap_pairwise(system: str, user: str, **kwargs) -> tuple:
    """Pairwise comparison: judge picks A or B (or tie). Position randomized."""
    task = kwargs.get("task", "")
    pw_system = (
        f"{system}\n\n"
        "You will be shown two responses (A and B) to the same input. "
        "Compare them and decide which is better. Respond with:\n"
        "- 'A' if Response A is better\n"
        "- 'B' if Response B is better\n"
        "- 'Tie' if they are equally good\n\n"
        "Do not let the order influence your judgment."
    )
    # The user template for pairwise is different — built by the caller
    return pw_system, user


def _wrap_self_consistency(system: str, user: str, **kwargs) -> tuple:
    """Self-consistency: same prompt, run multiple times.
    
    The wrapper doesn't change the prompt — the runner handles
    multiple inference passes and aggregation.
    """
    return _wrap_cot(system, user, **kwargs)  # Use CoT as the base


def _wrap_tree_of_thought(system: str, user: str, **kwargs) -> tuple:
    """Tree-of-Thought: multi-path reasoning with self-evaluation."""
    tot_system = (
        f"{system}\n\n"
        "Use the following structured reasoning approach:\n\n"
        "## Path 1: Semantic Accuracy\n"
        "Evaluate whether the output preserves the meaning of the input/reference.\n\n"
        "## Path 2: Fluency and Coherence\n"
        "Evaluate whether the output is well-formed and natural-sounding.\n\n"
        "## Path 3: Completeness\n"
        "Evaluate whether the output covers all relevant information.\n\n"
        "## Synthesis\n"
        "Considering all three paths, which quality level best describes "
        "this output? Provide your final rating."
    )
    return tot_system, user


# Strategy → wrapper function mapping
STRATEGY_WRAPPERS = {
    "zero_shot":        _wrap_zero_shot,
    "few_shot_anchored": _wrap_few_shot_anchored,
    "prometheus":       _wrap_prometheus,
    "cot":              _wrap_cot,
    "cross_lingual":    _wrap_cross_lingual,
    "monolingual":      _wrap_monolingual,
    "pairwise":         _wrap_pairwise,
    "self_consistency":  _wrap_self_consistency,
    "tree_of_thought":   _wrap_tree_of_thought,
}


# ─────────────────────────────────────────────────────────────────────
# FEW-SHOT EXAMPLES & RUBRICS (per task)
# ─────────────────────────────────────────────────────────────────────

def _get_few_shot_examples(task: str) -> str:
    """Return 2-3 calibration examples for few-shot anchored strategy."""
    examples = {
        "mt": (
            "### Example 1 (Rating: 5 — Excellent)\n"
            "Source: The quick brown fox jumps over the lazy dog.\n"
            "Reference: Le renard brun rapide saute par-dessus le chien paresseux.\n"
            "Translation: Le renard brun rapide saute par-dessus le chien paresseux.\n"
            "Justification: Perfect translation, matching the reference exactly.\n\n"
            "### Example 2 (Rating: 1 — Terrible)\n"
            "Source: The quick brown fox jumps over the lazy dog.\n"
            "Reference: Le renard brun rapide saute par-dessus le chien paresseux.\n"
            "Translation: La maison est grande et belle.\n"
            "Justification: Completely unrelated to the source text.\n"
        ),
        "summarization": (
            "### Example 1 (Rating: 5 — Excellent)\n"
            "Article: [300-word article about climate change impacts on coral reefs]\n"
            "Reference: Climate change causes coral bleaching through rising ocean temperatures.\n"
            "Summary: Rising ocean temperatures due to climate change are causing widespread "
            "coral bleaching, threatening marine ecosystems.\n"
            "Justification: Captures the key point accurately and concisely.\n\n"
            "### Example 2 (Rating: 1 — Terrible)\n"
            "Article: [Same article about climate change impacts on coral reefs]\n"
            "Reference: Climate change causes coral bleaching through rising ocean temperatures.\n"
            "Summary: Football is a popular sport played worldwide.\n"
            "Justification: Completely irrelevant to the source article.\n"
        ),
        "qa": (
            "### Example 1 (Rating: 5 — Correct)\n"
            "Question: What is the capital of France?\n"
            "Gold answer: Paris\n"
            "Model answer: Paris\n"
            "Justification: Exact match with gold answer.\n\n"
            "### Example 2 (Rating: 1 — Wrong)\n"
            "Question: What is the capital of France?\n"
            "Gold answer: Paris\n"
            "Model answer: Berlin\n"
            "Justification: Incorrect — Berlin is the capital of Germany.\n"
        ),
        "math": (
            "### Example 1 (Rating: 5 — Correct)\n"
            "Problem: If x + 3 = 7, what is x?\n"
            "Gold answer: 4\n"
            "Model: x + 3 = 7, so x = 7 - 3 = 4. The answer is 4.\n"
            "Justification: Correct answer with clear reasoning.\n\n"
            "### Example 2 (Rating: 1 — Wrong)\n"
            "Problem: If x + 3 = 7, what is x?\n"
            "Gold answer: 4\n"
            "Model: x = 10\n"
            "Justification: Incorrect answer with no reasoning shown.\n"
        ),
    }
    return examples.get(task, "")


def _get_rubric(task: str) -> str:
    """Return Prometheus-style evaluation rubric for a task."""
    rubrics = {
        "mt": (
            "| Score | Criteria |\n"
            "|-------|----------|\n"
            "| 5 | Translation is semantically equivalent to the reference. "
            "Natural fluency, no grammatical errors, preserves tone and register. |\n"
            "| 4 | Translation conveys the core meaning correctly. Minor fluency "
            "issues or slight deviations that don't affect comprehension. |\n"
            "| 3 | Translation captures the general idea but has notable errors — "
            "missing information, awkward phrasing, or moderate inaccuracies. |\n"
            "| 2 | Translation has significant errors that distort meaning. "
            "Key information missing or mistranslated. |\n"
            "| 1 | Translation is incomprehensible, unrelated to the source, "
            "or in the wrong language. |\n"
        ),
        "summarization": (
            "| Score | Criteria |\n"
            "|-------|----------|\n"
            "| 5 | Summary captures all key information accurately. Concise, "
            "coherent, and faithful to the source. No hallucinations. |\n"
            "| 4 | Summary covers most key points. Minor omissions or slight "
            "redundancy but overall accurate. |\n"
            "| 3 | Summary captures the general topic but misses important details "
            "or includes some inaccuracies. |\n"
            "| 2 | Summary is only tangentially related. Major omissions, "
            "significant inaccuracies, or largely incoherent. |\n"
            "| 1 | Summary is irrelevant, factually wrong, or incomprehensible. |\n"
        ),
    }
    return rubrics.get(task, "")


# ─────────────────────────────────────────────────────────────────────
# PAIRWISE USER TEMPLATE (special case)
# ─────────────────────────────────────────────────────────────────────

PAIRWISE_USER_TEMPLATES = {
    "mt": (
        "Compare the following two translations of the same source text.\n\n"
        "Source text ({source_lang}):\n{input}\n\n"
        "Reference translation ({target_lang}):\n{reference}\n\n"
        "Response A:\n{response_a}\n\n"
        "Response B:\n{response_b}\n\n"
        "Which translation is better? Answer 'A', 'B', or 'Tie'."
    ),
    "summarization": (
        "Compare the following two summaries of the same text.\n\n"
        "Original text ({lang}):\n{input}\n\n"
        "Reference summary:\n{reference}\n\n"
        "Response A:\n{response_a}\n\n"
        "Response B:\n{response_b}\n\n"
        "Which summary is better? Answer 'A', 'B', or 'Tie'."
    ),
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
    response_a: str = "",
    response_b: str = "",
) -> dict:
    """Build the complete judge prompt for a given strategy and task.
    
    Returns:
        dict with keys 'system' and 'user' (ready for chat API format).
    """
    scale_min, scale_max = RATING_SCALE

    # Get base task template
    base = TASK_TEMPLATES[task]
    system = base["system"]

    # Format the user template with content
    if strategy == "pairwise" and task in PAIRWISE_USER_TEMPLATES:
        user = PAIRWISE_USER_TEMPLATES[task].format(
            input=input_text, reference=reference,
            response_a=response_a, response_b=response_b,
            lang=lang_name, source_lang=source_lang, target_lang=target_lang,
        )
    else:
        user = base["user"].format(
            input=input_text, reference=reference, output=output_text,
            lang=lang_name, source_lang=source_lang, target_lang=target_lang,
            context=context, scale_min=scale_min, scale_max=scale_max,
        )

    # Apply strategy wrapper
    wrapper = STRATEGY_WRAPPERS.get(strategy, _wrap_zero_shot)
    system, user = wrapper(
        system, user,
        task=task, lang=lang, lang_name=lang_name,
        source_lang=source_lang, target_lang=target_lang,
    )

    return {"system": system, "user": user}
