"""MTOB (Machine Translation from One Book) domain environment for SkyRL training.

Provides MTOBEnv class for machine translation with advisor feedback.
Kalamang->English translation using chrF score evaluation.
"""

from typing import Dict, Any, List, Tuple
from omegaconf import DictConfig

from ..env_base import BaseAdvisorEnv

from .config import (
    STUDENT_PROMPT_TEMPLATE,
    BASELINE_PROMPT_TEMPLATE,
    compute_translation_score,
    extract_translation,
)


class MTOBEnv(BaseAdvisorEnv):
    def __init__(self, env_config: DictConfig, extras: Dict[str, Any] = {}):
        super().__init__(env_config, extras)

        # Store pre-built reference materials from dataset
        self.reference_materials = extras.get("reference_materials", [])
        assert len(self.reference_materials) > 0, "No reference materials found"

    def _build_baseline_prompt(
        self, prompt: List[Dict[str, str]]
    ) -> Tuple[List[Dict[str, str]], str]:
        formatted_prompt = BASELINE_PROMPT_TEMPLATE.format(
            original_question=self.original_question,
            reference_materials="\n".join(self.reference_materials),
        )

        return [
            {"role": "user", "content": formatted_prompt},
        ], formatted_prompt

    def _build_advisor_prompt(
        self, prompt: List[Dict[str, str]]
    ) -> Tuple[List[Dict[str, str]], str]:
        return super()._build_advisor_prompt(prompt)

    def _build_student_prompt(
        self, advisor_feedback: str
    ) -> Tuple[List[Dict[str, str]], str]:
        """Compose the prompt sent to the student model following MTOB format."""

        # Build the student prompt
        formatted_prompt = STUDENT_PROMPT_TEMPLATE.format(
            original_question=self.original_question,
            reference_materials="\n".join(self.reference_materials),
            advisor_feedback=advisor_feedback,
        )

        return [
            {"role": "user", "content": formatted_prompt},
        ], formatted_prompt

    def _compute_step(self) -> Tuple[float, bool, Dict[str, Any]]:
        translation = extract_translation(self.final_response)
        score = compute_translation_score(translation, self.ground_truth)
        return score, True, {}

    def _get_metadata(self) -> Dict[str, Any]:
        return super()._get_metadata()
