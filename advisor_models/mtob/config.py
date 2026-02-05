"""Configuration for MTOB (Machine Translation from One Book) domain.

Contains system prompts and instructions for machine translation with advisor feedback.
Kalamang->English translation tasks.
"""

import evaluate

ADVISOR_PROMPT_PREFIX = "Kalamang is a language spoken on the Karas Islands in West Papua. You are an advisor tasked with helping a model translate the following sentence from Kalamang to English: {source_text}"

ADVISOR_PROMPT_SUFFIX = "\nNow write the advice. You must not provide a full translation in your advice; your advice should be helpful for a student wishing to do the translation on their own and learn from the process.\nKalamang: {source_text}"

STUDENT_PROMPT_TEMPLATE = """Kalamang is a language spoken on the Karas Islands in West Papua. Translate the following sentence from Kalamang to English: {original_question}
Here is some additional reference material:
{reference_materials}

Here is some advice from the advisor: {advisor_feedback}

Now determine the translation.
Kalamang: {original_question}
Reason over the information and end your response with 'Translation: {{translation}}'"""

BASELINE_PROMPT_TEMPLATE = """Kalamang is a language spoken on the Karas Islands in West Papua. Translate the following sentence from Kalamang to English: {original_question}
Here is some additional reference material:
{reference_materials}

Now determine the translation.
Kalamang: {original_question}
Reason over the information and end your response with 'Translation: {{translation}}'"""


def compute_translation_score(translation: str, ground_truth: str) -> float:
    """Compute translation quality score using chrF metric.

    Args:
        translation: The model's translation
        ground_truth: The reference translation

    Returns:
        chrF score (0-100 scale, normalized to 0-1)
    """
    if not translation or not translation.strip():
        return 0.0

    try:
        # Initialize chrF metric
        chrf_metric = evaluate.load("chrf")

        # Compute chrF score
        chrf_result = chrf_metric.compute(
            predictions=[translation.strip()], references=[ground_truth.strip()]
        )

        # chrF returns score on 0-100 scale, normalize to 0-1
        return chrf_result["score"] / 100.0

    except Exception as e:
        print(f"Error computing translation score: {e}")
        return 0.0


def extract_translation(response: str) -> str:
    """Extract the translation from the model response.

    For MTOB, we typically expect the entire response to be the translation,
    but we'll clean it up by removing any extra formatting.
    """
    if not response:
        return ""

    # Remove common prefixes that models might add
    response = response.strip()

    # Extract translation from response
    try:
        translation = response.split("Translation: ")[1].strip()
    except Exception:
        return ""

    # Remove quotes if the entire response is quoted
    if translation.startswith('"') and translation.endswith('"'):
        translation = translation[1:-1]
    elif translation.startswith("'") and translation.endswith("'"):
        translation = translation[1:-1]

    return translation.strip()
