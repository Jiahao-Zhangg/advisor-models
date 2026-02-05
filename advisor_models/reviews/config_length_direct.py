"""Configuration for reviews length direct domain.

Contains system prompts and reward functions for direct RL training on review writing.
No advisor setup - the model directly generates reviews and is rewarded based on length matching.
"""

# People with different length preferences (same as regular length domain)
LENGTH_PEOPLE = ["Alan", "Parth", "Matei", "Joey", "Alex"]

# Preferred word counts for each person
LENGTH_CRITERIA = {
    "Alan": 500,
    "Parth": 50,
    "Matei": 10,
    "Joey": 200,
    "Alex": 1000,
}

# System prompt for direct RL model
DIRECT_SYSTEM_PROMPT = """You are a review writer. Based on the prompt and the person's preferences, write a review that matches their preferred style. Consider the length preferences and style that would work best for the target person."""

# Instructions for direct RL model
DIRECT_INSTRUCTION = """You need to write a review for {person}.

The task is: {prompt}

Write a review that would work best for {person}. Think carefully about the length of the review."""


def compute_length_reward(review_text: str, preferred_length: int) -> float:
    """Compute reward based on how well the review matches the person's length preference."""
    word_count = len(review_text.split())

    distance = abs(word_count - preferred_length)

    # Inverse distance reward - never reaches 0, smooth gradient for learning
    # Reward approaches 1.0 as distance approaches 0
    reward = 1.0 / (1.0 + distance / preferred_length)

    return reward
