"""Configuration for MTOB (Machine Translation from One Book) domain.

Contains system prompts and instructions for machine translation with advisor feedback.
Kalamang->English translation tasks.
"""

from __future__ import annotations

from collections import Counter

try:
    import evaluate  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    evaluate = None

_warned_evaluate_failure = False

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

def _character_ngrams(text: str, n: int) -> Counter[str]:
    # chrF is character n-gram based; remove whitespace to avoid scoring formatting.
    normalized = "".join(text.split())
    if n <= 0 or len(normalized) < n:
        return Counter()
    return Counter(normalized[i : i + n] for i in range(len(normalized) - n + 1))


def _simple_chrf_score(prediction: str, reference: str, *, max_order: int = 6, beta: float = 2.0) -> float:
    """Compute a lightweight chrF-like score (0-100).

    This is a fallback for environments where `evaluate` isn't installed or can't
    download metrics. It's not guaranteed to match sacrebleu/evaluate exactly,
    but it's stable and monotonic for training.
    """
    if not prediction or not reference:
        return 0.0

    beta2 = beta * beta
    f_scores = []
    for n in range(1, max_order + 1):
        pred_ngrams = _character_ngrams(prediction, n)
        ref_ngrams = _character_ngrams(reference, n)
        # If both are too short for this order, ignore this n (avoids penalizing short exact matches).
        if not pred_ngrams and not ref_ngrams:
            continue
        if not pred_ngrams or not ref_ngrams:
            f_scores.append(0.0)
            continue

        overlap = sum((pred_ngrams & ref_ngrams).values())
        pred_total = sum(pred_ngrams.values())
        ref_total = sum(ref_ngrams.values())

        precision = overlap / pred_total if pred_total else 0.0
        recall = overlap / ref_total if ref_total else 0.0
        denom = (beta2 * precision) + recall
        f = ((1.0 + beta2) * precision * recall / denom) if denom else 0.0
        f_scores.append(f)

    if not f_scores:
        return 0.0
    return (sum(f_scores) / len(f_scores)) * 100.0


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

    pred = translation.strip()
    ref = ground_truth.strip()

    # Prefer evaluate's implementation when available, but fall back if it fails
    # (e.g., missing optional deps like sacrebleu).
    if evaluate is not None:
        try:
            chrf_metric = evaluate.load("chrf")
            chrf_result = chrf_metric.compute(predictions=[pred], references=[ref])
            return chrf_result["score"] / 100.0
        except Exception as e:
            global _warned_evaluate_failure
            if not _warned_evaluate_failure:
                print(f"Error computing translation score with evaluate/chrf; falling back to local scorer: {e}")
                _warned_evaluate_failure = True

    # Fallback: local chrF-like implementation.
    try:
        return _simple_chrf_score(pred, ref) / 100.0
    except Exception as e:  # pragma: no cover
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
