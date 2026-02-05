"""Configuration template for domain-specific prompts and scoring logic.

This file should contain:
1. System prompts and instruction templates
2. Scoring/reward functions
3. Domain-specific constants

Example structure:
```python
# Advisor prompts (for advisor mode)
ADVISOR_SYSTEM_PROMPT = "You are an advisor..."
ADVISOR_INSTRUCTION = "Provide advice for: {original_question}"

# Student prompts (for advisor mode)
STUDENT_SYSTEM_PROMPT = "You are a student..."
STUDENT_INSTRUCTION = "Question: {original_question}\nAdvice: {advisor_feedback}"

# Baseline prompts (for baseline mode - direct RL)
BASELINE_SYSTEM_PROMPT = "You are a helpful assistant..."
BASELINE_INSTRUCTION = "Question: {original_question}"

# Scoring functions
def compute_score(response: str, ground_truth: str) -> Tuple[float, str]:
    '''Compute reward for the response.

    Args:
        response: The model's response
        ground_truth: The ground truth answer

    Returns:
        float: Reward score (typically 0.0 to 1.0)
        str: Info string for logging
    '''
    # Extract answer from response
    extracted = extract_answer(response)

    # Compare with ground truth
    if extracted == ground_truth:
        return 1.0, "Correct answer"
    else:
        return 0.0, "Incorrect answer"

def extract_answer(response: str) -> str:
    '''Extract the final answer from the response.'''
    # Use regex, parsing, or other logic
    return response.strip()
```
"""
