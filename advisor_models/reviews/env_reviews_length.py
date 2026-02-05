"""Reviews length domain environment for SkyRL training.

Provides ReviewsLengthEnv class for review writing with length preferences.
"""

from typing import Any, Dict, List, Tuple

from omegaconf import DictConfig

from .config import (
    STUDENT_SYSTEM_PROMPT,
    STUDENT_INSTRUCTION,
    BASELINE_SYSTEM_PROMPT,
    BASELINE_INSTRUCTION,
    compute_length_reward,
)
from ..env_base import BaseAdvisorEnv


class ReviewsLengthEnv(BaseAdvisorEnv):
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
            reward = compute_length_reward(self.final_response, self.ground_truth)
            return reward, True, {}
        except Exception as e:
            print(f"Error computing review score: {e}")
            return 0.0, True, {}

    def _get_metadata(self) -> Dict[str, Any]:
        metadata = super()._get_metadata()
        metadata["other_info"] = (
            f"{self.person} prefers reviews of length {self.ground_truth} words."
        )
        return metadata
