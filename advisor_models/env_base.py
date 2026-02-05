"""Base environment class for advisor-based RL training.

Provides BaseAdvisorEnv class that supports both advisor and baseline modes.
Subclasses implement domain-specific logic for different tasks.
"""

from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput
from typing import Dict, Any, List, Tuple
from omegaconf import DictConfig
from litellm import completion
import os

# ENVIRONMENT VARIABLES
ADVISOR_MODELS_MODE = os.environ.get("ADVISOR_MODELS_MODE", "advisor").lower()
"""Used to control the advisor models setting. Supported modes:
- `advisor`: Advisor model generates advice for student model to generate final response.
- `baseline`: Direct RL on the "advisor" model.
"""

STUDENT_MODEL = os.environ.get("STUDENT_MODEL", "gpt-4o-mini")
"""Used to set the student model. Overwritten if `model` is provided in dataset."""

API_BASE = os.environ.get("API_BASE", None)
"""Used to set the API base URL."""


class BaseAdvisorEnv(BaseTextEnv):
    """Base environment for Advisor Models RL training.

    Subclasses must implement:
    - `_build_baseline_prompt()`: For direct RL mode
    - `_build_student_prompt()`: For advisor mode student prompts
    - `_compute_step()`: For reward calculation

    Optionally override:
    - `_build_advisor_prompt()`: Default uses dataset prompt as-is
    - `_get_metadata()`: Default logs all standard fields
    """

    def __init__(self, env_config: DictConfig, extras: Dict[str, Any] = {}):
        """Initialize the environment.

        Args:
            env_config: The environment configuration.
            extras: Additional metadata.
        """
        super().__init__()

        assert "reward_spec" in extras, "reward_spec field is required"
        assert "ground_truth" in extras["reward_spec"], (
            "ground_truth is required in reward_spec field"
        )
        assert "original_question" in extras

        self.ground_truth = extras["reward_spec"]["ground_truth"]
        self.original_question = extras["original_question"]
        self.original_response = extras.get("original_response", "")
        self.initial_reward = extras.get("initial_reward", "")
        self.advisor_mode = ADVISOR_MODELS_MODE
        self.student_model = extras.get("model", STUDENT_MODEL)
        self.advisor_prompt_to_log = None
        self.action = None
        self.student_prompt_to_log = None
        self.final_response = None
        self.reward_info = None

    def _build_baseline_prompt(
        self, prompt: List[Dict[str, str]]
    ) -> Tuple[List[Dict[str, str]], str]:
        """Compose the prompt used for direct RL of open-source model.

        Args:
            prompt: The original prompt.

        Returns:
            List[Dict[str, str]]: The modified prompt to input to the policy model.
            str: The prompt to log.
        """
        raise NotImplementedError

    def _build_advisor_prompt(
        self, prompt: List[Dict[str, str]]
    ) -> Tuple[List[Dict[str, str]], str]:
        """Compose the prompt sent to the advisor model.

        By default uses the prompt provided in the dataset, logging
        the last turn's content.

        Args:
            prompt: The original prompt.

        Returns:
            List[Dict[str, str]]: The modified prompt to input to the policy model.
            str: The advisor prompt to log.
        """
        return prompt, prompt[-1]["content"]

    def _build_student_prompt(
        self, advisor_feedback: str
    ) -> Tuple[List[Dict[str, str]], str]:
        """Compose the prompt sent to the student model. 2-step vs 3-step flow patterns
        should be handled here by changing prompt format.

        Args:
            advisor_feedback: The advisor feedback.

        Returns:
            List[Dict[str, str]]: The modified prompt to input to the student model.
            str: The student prompt to log.
        """
        raise NotImplementedError

    def _compute_step(self) -> Tuple[float, bool, Dict[str, Any]]:
        """Computes reward.

        When this function is called, `self.final_response` is available.

        Returns:
            float: The scalar reward.
            bool: Whether the episode is done.
            Dict[str, Any]: Additional reward information saved to `self.reward_info`.
        """
        raise NotImplementedError

    def _get_metadata(self) -> Dict[str, Any]:
        """Return metadata dict for logging.

        Only fields `original_question`, `initial_response`, `advisor_prompt`,
        `advisor_response`, `student_prompt`, `student_response`, `ground_truth`,
        `initial_reward`, `final_reward`, and `other_info` are used.
        When this function is called, `self.advisor_prompt_to_log`, `self.action`,
        `self.student_prompt_to_log`, `self.final_response`, `self.final_reward`, and
        `self.reward_info` are available.
        Defaults to filling in all fields except `other_info`.

        Returns:
            Dict[str, Any]: Metadata dict for logging.
        """
        # handle student prompt/response
        if self.advisor_mode == "advisor":
            student_prompt = self.student_prompt_to_log
            student_response = self.final_response
        elif self.advisor_mode == "baseline":
            student_prompt = ""
            student_response = ""
        else:
            raise ValueError(f"Unknown advisor mode: {self.advisor_mode}")

        return {
            "add_to_log": True,
            "original_question": self.original_question,
            "initial_response": self.original_response,
            "advisor_prompt": self.advisor_prompt_to_log,
            "advisor_response": self.action,
            "student_prompt": student_prompt,
            "student_response": student_response,
            "ground_truth": self.ground_truth,
            "initial_reward": self.initial_reward,
            "final_reward": self.final_reward,
            "other_info": "",
        }

    def init(
        self, prompt: List[Dict[str, str]]
    ) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
        """SkyRL initialization for obtaining actual input prompt.

        Args:
            prompt: The original prompt.

        Returns:
            List[Dict[str, str]]: The modified prompt to input to the policy model.
            Dict[str, Any]: Additional metadata (left empty).
        """

        if self.advisor_mode == "advisor":
            modified_prompt, self.advisor_prompt_to_log = self._build_advisor_prompt(
                prompt
            )
        elif self.advisor_mode == "baseline":
            modified_prompt, self.advisor_prompt_to_log = self._build_baseline_prompt(
                prompt
            )
        else:
            raise ValueError(f"Unknown advisor mode: {self.advisor_mode}")

        return modified_prompt, {}

    def call_student(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        timeout: float = 120.0,
    ) -> str:
        """Call the chat completion endpoint using student model.

        Args:
            messages: The messages to send to the student model.
            temperature: The temperature to use for the student model.
            timeout: Timeout in seconds for the API call (default: 120.0).

        Returns:
            str: The parsed response from the student model.
        """
        try:
            response = completion(
                model=self.student_model,
                messages=messages,
                temperature=temperature,
                timeout=timeout,
                base_url=API_BASE,
            )
            return response.choices[0].message.content
        except Exception as e:  # pragma: no cover
            print(f"[BaseAdvisorEnv] LLM request failed: {e}")
            return ""

    def step(self, action: str) -> BaseTextEnvStepOutput:
        """Execute one step of the environment.

        Args:
            action: The action taken by the policy model.

        Returns:
            BaseTextEnvStepOutput: The step output.
        """
        self.action = action
        if self.advisor_mode == "advisor":
            student_prompt, self.student_prompt_to_log = self._build_student_prompt(
                self.action
            )
            self.final_response = self.call_student(student_prompt, temperature=0.0)
        elif self.advisor_mode == "baseline":
            self.final_response = action
        else:
            raise ValueError(f"Unknown advisor mode: {self.advisor_mode}")

        # Extract answer and compute reward
        self.final_reward, done, self.reward_info = self._compute_step()

        return BaseTextEnvStepOutput(
            observations=[],
            reward=self.final_reward,
            done=done,
            metadata=self._get_metadata(),
        )
