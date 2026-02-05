"""Configuration for reviews length scaled domain.

Contains system prompts, dynamically generated user personas with random length preferences,
and reward functions for review writing with scaled user count (up to 100 users).
"""

import random
from typing import Dict, List, Tuple


# Pool of 100 real names for scaled user generation
REAL_NAMES = [
    "Emma",
    "Liam",
    "Olivia",
    "Noah",
    "Ava",
    "Ethan",
    "Sophia",
    "Mason",
    "Isabella",
    "William",
    "Mia",
    "James",
    "Charlotte",
    "Benjamin",
    "Amelia",
    "Lucas",
    "Harper",
    "Henry",
    "Evelyn",
    "Alexander",
    "Abigail",
    "Michael",
    "Emily",
    "Daniel",
    "Elizabeth",
    "Jacob",
    "Sofia",
    "Logan",
    "Avery",
    "Jackson",
    "Ella",
    "Sebastian",
    "Scarlett",
    "Mateo",
    "Grace",
    "Jack",
    "Chloe",
    "Owen",
    "Victoria",
    "Theodore",
    "Riley",
    "Aiden",
    "Aria",
    "Samuel",
    "Lily",
    "Ryan",
    "Aurora",
    "John",
    "Zoey",
    "Nathan",
    "Nora",
    "Caleb",
    "Camila",
    "Christian",
    "Hannah",
    "Dylan",
    "Addison",
    "Isaac",
    "Eleanor",
    "Joshua",
    "Stella",
    "Andrew",
    "Natalie",
    "Thomas",
    "Zoe",
    "Joseph",
    "Leah",
    "David",
    "Hazel",
    "Carter",
    "Violet",
    "Luke",
    "Aurora",
    "Gabriel",
    "Savannah",
    "Anthony",
    "Audrey",
    "Jayden",
    "Brooklyn",
    "Lincoln",
    "Bella",
    "Julian",
    "Claire",
    "Christopher",
    "Skylar",
    "Jaxon",
    "Lucy",
    "Levi",
    "Paisley",
    "Isaiah",
    "Everly",
    "Grayson",
    "Anna",
    "Josiah",
    "Caroline",
    "Charles",
    "Nova",
    "Maverick",
    "Genesis",
    "Miles",
]


def generate_scaled_users(
    num_users: int = 100,
    min_length: int = 10,
    max_length: int = 1000,
    seed: int = 42,
) -> Tuple[List[str], Dict[str, int]]:
    """Generate user names and random length preferences.

    Args:
        num_users: Number of users to generate (default 100)
        min_length: Minimum word count preference
        max_length: Maximum word count preference
        seed: Random seed for reproducibility

    Returns:
        Tuple of (list of user names, dict mapping names to length preferences)
    """
    random.seed(seed)

    if num_users > len(REAL_NAMES):
        raise ValueError(
            f"num_users ({num_users}) exceeds available names ({len(REAL_NAMES)})"
        )

    # Select names from the pool
    user_names = REAL_NAMES[:num_users]

    # Generate random length preferences within the range
    length_preferences = {}
    for name in user_names:
        length_preferences[name] = random.randint(min_length, max_length)

    return user_names, length_preferences


# Default configuration with 100 users
DEFAULT_NUM_USERS = 100
DEFAULT_MIN_LENGTH = 10
DEFAULT_MAX_LENGTH = 1000

# Generate default users (can be overridden by construct_dataset)
LENGTH_SCALED_PEOPLE, LENGTH_SCALED_CRITERIA = generate_scaled_users(
    num_users=DEFAULT_NUM_USERS,
    min_length=DEFAULT_MIN_LENGTH,
    max_length=DEFAULT_MAX_LENGTH,
)

# System prompt for advisor model
LENGTH_SCALED_ADVISOR_SYSTEM_PROMPT = """You are a review writing advisor. Provide specific guidance for writing a review that matches the person's preferences. Consider the length preferences and style that would work best for the target person."""

# Instructions for advisor model
LENGTH_SCALED_ADVISOR_INSTRUCTION = """You need to provide advice for writing a review for {person}.

The task is: {prompt}

Provide specific advice about the review that would work best for {person}. Think carefully about the length of the review in your advice. Keep your advice to 3-4 sentences."""

# Prompts for student model (writes the actual review)
STUDENT_SYSTEM_PROMPT = """You are a review writer. Based on the prompt and advisor guidance, write a review that follows the guidance provided. Write a clear, well-structured review."""

STUDENT_INSTRUCTION = """Review Prompt: {prompt}

Advisor Guidance:
{advisor_feedback}

Write a review following the advisor's guidance."""

# Baseline prompts
BASELINE_SYSTEM_PROMPT = (
    """You are a review writer. Based on the prompt, write a review."""
)
BASELINE_INSTRUCTION = """Review Prompt: {prompt}"""


def compute_length_reward(review_text: str, preferred_length: int) -> float:
    """Compute reward based on how well the review matches the person's length preference."""
    word_count = len(review_text.split())

    distance = abs(word_count - preferred_length)

    # Inverse distance reward - never reaches 0, smooth gradient for learning
    # Reward approaches 1.0 as distance approaches 0
    reward = 1.0 / (1.0 + distance / preferred_length)

    return reward
