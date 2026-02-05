"""Rule arena (US tax) domain environment for SkyRL training.

Provides RuleArenaEnv class for US tax calculation with advisor feedback.
"""

from typing import Dict, Any, List, Tuple
from omegaconf import DictConfig

from ..env_base import BaseAdvisorEnv
from .config import STUDENT_SYSTEM_PROMPT, build_prompt, compute_score


class RuleArenaEnv(BaseAdvisorEnv):
    def __init__(self, env_config: DictConfig, extras: Dict[str, Any] = {}):
        super().__init__(env_config, extras)

        assert "info_dict" in extras, "Taxpayer dict not found in extras"
        self.taxpayer_dict = extras["info_dict"]

    def _build_baseline_prompt(
        self, prompt: List[Dict[str, str]]
    ) -> Tuple[List[Dict[str, str]], str]:
        formatted_prompt = build_prompt(self.taxpayer_dict)
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
        """Compose the prompt sent to the student model."""

        # parse out think block
        if "</think>" in advisor_feedback:
            advisor_feedback = advisor_feedback.split("</think>")[1]

        return [
            {"role": "system", "content": STUDENT_SYSTEM_PROMPT},
            {"role": "user", "content": self.original_question},
            {"role": "assistant", "content": self.original_response},
            {"role": "user", "content": advisor_feedback},
        ], "[ADVISOR FEEDBACK W/O THINK BLOCK PROVIDED AS NEXT USER TURN]"

    def _compute_step(self) -> Tuple[float, bool, Dict[str, Any]]:
        try:
            reward, info = compute_score(self.final_response, self.ground_truth)
            return reward, True, {"reward_info": info}
        except Exception as e:
            return 0.0, True, {"reward_info": f"Failed to compute reward: {e}"}

    def _get_metadata(self) -> Dict[str, Any]:
        metadata = super()._get_metadata()
        metadata["other_info"] = self.reward_info["reward_info"]
        return metadata
