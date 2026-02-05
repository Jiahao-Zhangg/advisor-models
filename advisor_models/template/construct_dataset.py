"""Dataset construction template for advisor-based RL training.

This file should implement construction of training and validation datasets saved in parquet format.

## Required Dataset Schema

Each row should contain:
```python
{
    "prompt": List[Dict[str, str]],  # OpenAI messages format
    "env_class": str,                # Environment name (e.g., "template")
    "reward_spec": {"ground_truth": Any},  # Ground truth for scoring
    "original_question": str,        # Question text for student prompt
}
```

### Field Descriptions:
- **`prompt`**: Advisor model input in OpenAI messages format
  - Example: `[{"role": "user", "content": "Provide advice for solving: 2+2"}]`
- **`env_class`**: Environment name matching registration in `main_*.py`
  - Example: `"template"`
- **`reward_spec.ground_truth`**: Ground truth value for reward calculation
  - Example: `4` or `"The answer is 4"` depending on domain
- **`original_question`**: Question text used in student prompt formatting
  - Example: `"What is 2+2?"`

### Additional Fields (accessed via `extras` in `__init__`):
Any extra fields are passed to the environment's `__init__` method via the `extras` dict.

### For 3-Step Flow (Initial Response → Advisor Feedback → Updated Response):
Include these additional fields:
- **`initial_response`**: Initial student response before advisor feedback
- **`initial_reward`**: Reward for initial response (computed using same scoring function)
- **`model`**: Student model name (e.g., `"gpt-4o-mini"`). This overrides the `STUDENT_MODEL` environment variable.

## Example Implementation:
```python
import datasets
import pandas as pd

def build_advisor_prompt(question: str) -> List[Dict[str, str]]:
    return [{"role": "user", "content": f"Provide advice for: {question}"}]

def create_dataset():
    rows = []
    for question, answer in load_problems():
        row = {
            "prompt": build_advisor_prompt(question),
            "env_class": "template",
            "reward_spec": {"ground_truth": answer},
            "original_question": question,
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_parquet("train.parquet")
```
"""
