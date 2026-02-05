"""Reviews level domain environment for SkyRL training.

Provides ReviewsLevelEnv class for review writing with reading level preferences.
"""

from typing import Any, Dict, List, Tuple

from omegaconf import DictConfig
from openai import OpenAI

from ..env_base import BaseAdvisorEnv
from .config import (
    STUDENT_SYSTEM_PROMPT,
    STUDENT_INSTRUCTION,
    BASELINE_SYSTEM_PROMPT,
    BASELINE_INSTRUCTION,
)


class ReviewsLevelEnv(BaseAdvisorEnv):
    """Environment for review writing with advisor feedback using 2-step flow."""

    def __init__(self, env_config: DictConfig, extras: Dict[str, Any] = {}):
        super().__init__(env_config, extras)

        # additional required fields
        assert "person" in extras, "person field is required"
        self.person = extras["person"]

    def _build_baseline_prompt(
        self, prompt: List[Dict[str, str]]
    ) -> Tuple[List[Dict[str, str]], str]:
        user_context = BASELINE_INSTRUCTION.format(
            prompt=self.original_question,
            person=self.person,
        )

        return [
            {"role": "system", "content": BASELINE_SYSTEM_PROMPT},
            {"role": "user", "content": user_context},
        ], user_context

    def _build_advisor_prompt(
        self, prompt: List[Dict[str, str]]
    ) -> Tuple[List[Dict[str, str]], str]:
        return super()._build_advisor_prompt(prompt)

    def _build_student_prompt(
        self, advisor_feedback: str
    ) -> Tuple[List[Dict[str, str]], str]:
        """Build prompt for student model to write the review."""
        user_context = STUDENT_INSTRUCTION.format(
            prompt=self.original_question,
            advisor_feedback=advisor_feedback,
        )

        return [
            {"role": "system", "content": STUDENT_SYSTEM_PROMPT},
            {"role": "user", "content": user_context},
        ], user_context

    def _compute_step(self) -> Tuple[float, bool, Dict[str, Any]]:
        try:
            client = OpenAI()
            response = client.chat.completions.create(
                model="gpt-5-mini",
                messages=[
                    {"role": "system", "content": "You are a reading level evaluator."},
                    {
                        "role": "user",
                        "content": self.ground_truth.format(review=self.final_response),
                    },
                ],
            )
            response = response.choices[0].message.content
            if "Yes" in response and "No" not in response:
                return 1.0, True, {}
            return 0.0, True, {}
        except Exception as e:
            print(f"Error computing review score: {e}")
            return 0.0, True, {}

    def _get_metadata(self) -> Dict[str, Any]:
        metadata = super()._get_metadata()
        metadata["other_info"] = (
            f"Person: {self.person}\nLevel Criteria: {self.ground_truth}"
        )
        return metadata
