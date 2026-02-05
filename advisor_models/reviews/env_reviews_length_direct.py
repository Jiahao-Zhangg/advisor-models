"""Reviews length direct domain environment for SkyRL training.

Provides ReviewsLengthDirectEnv class for direct RL training on review writing.
No advisor setup - the model directly generates reviews and is rewarded based on length matching.

This is used to demonstrate that direct RL on a task leads to performance degradation
on other tasks (e.g., training on length → degradation on math).
"""

from typing import Any, Dict, List, Tuple

from omegaconf import DictConfig
from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput

from .config_length_direct import (
    DIRECT_SYSTEM_PROMPT,
    DIRECT_INSTRUCTION,
    compute_length_reward,
)


class ReviewsLengthDirectEnv(BaseTextEnv):
    """Environment for direct RL training on review writing.

    Unlike the advisor setup, this environment directly trains the model to write reviews.
    The model receives the prompt and person info, generates a review, and gets rewarded
    based on how well the review length matches the person's preference.
    """

    def __init__(self, env_config: DictConfig, extras: Dict[str, Any] = {}):
        super().__init__()

        assert "reward_spec" in extras, "reward_spec field is required"
        assert "ground_truth" in extras["reward_spec"], (
            "ground_truth is required in reward_spec field"
        )
        assert "original_question" in extras
        assert "person" in extras, "person field is required"

        self.ground_truth = extras["reward_spec"]["ground_truth"]
        self.original_question = extras["original_question"]
        self.person = extras["person"]

        # Store for logging
        self.prompt_to_log = None
        self.action = None
        self.final_reward = None

    def init(
        self, prompt: List[Dict[str, str]]
    ) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
        """Build the prompt for direct RL training."""
        # Build direct prompt (no advisor)
        user_content = DIRECT_INSTRUCTION.format(
            person=self.person,
            prompt=self.original_question,
        )

        modified_prompt = [
            {"role": "system", "content": DIRECT_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        self.prompt_to_log = user_content

        return modified_prompt, {}

    def step(self, action: str) -> BaseTextEnvStepOutput:
        """Execute one step - the action is the generated review."""
        self.action = action

        # Compute reward based on length matching
        try:
            self.final_reward = compute_length_reward(action, self.ground_truth)
        except Exception as e:
            print(f"Error computing reward: {e}")
            self.final_reward = 0.0

        return BaseTextEnvStepOutput(
            observations=[],
            reward=self.final_reward,
            done=True,
            metadata=self._get_metadata(),
        )

    def _get_metadata(self) -> Dict[str, Any]:
        """Return metadata dict for logging."""
        word_count = len(self.action.split()) if self.action else 0

        return {
            "add_to_log": True,
            "original_question": self.original_question,
            "prompt": self.prompt_to_log,
            "response": self.action,
            "ground_truth": self.ground_truth,
            "final_reward": self.final_reward,
            "other_info": (
                f"{self.person} prefers {self.ground_truth} words. "
                f"Generated {word_count} words."
            ),
        }
