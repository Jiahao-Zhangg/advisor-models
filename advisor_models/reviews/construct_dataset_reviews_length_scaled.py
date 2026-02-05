"""Dataset construction for reviews length scaled domain.

Generates training and validation datasets for review writing with scaled user count
(configurable up to 100 users with random length preferences).

Example usage:
    python advisor_models/reviews/construct_dataset_reviews_length_scaled.py \
        --output_dir data/reviews \
        --num_users 100

The script writes ``train_length_scaled_<num_users>.parquet`` and ``validation_length_scaled_<num_users>.parquet``.
"""

from __future__ import annotations

import argparse
import os
import random
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any

import datasets
from datasets import load_dataset
from tqdm import tqdm

from config_length_scaled import (
    LENGTH_SCALED_ADVISOR_SYSTEM_PROMPT,
    LENGTH_SCALED_ADVISOR_INSTRUCTION,
    generate_scaled_users,
)


def build_advisor_prompt(
    task: Dict[str, Any],
) -> List[Dict[str, str]]:
    """Build the advisor prompt that the model will receive."""
    user_content = LENGTH_SCALED_ADVISOR_INSTRUCTION.format(
        person=task["person"],
        prompt=task["prompt"],
    )

    return [
        {"role": "system", "content": LENGTH_SCALED_ADVISOR_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def load_unique_prompts() -> tuple[List[str], List[str]]:
    """Load unique prompts from the HuggingFace dataset."""
    print("Loading dataset from HuggingFace...")
    dataset = load_dataset("Asap7772/steered_reviews_full_autolabel")

    # Get unique prompts from train and test sets
    train_prompts = list(set(dataset["train"]["prompt"]))
    test_prompts = list(set(dataset["test"]["prompt"]))

    print(f"Found {len(train_prompts)} unique prompts in train set")
    print(f"Found {len(test_prompts)} unique prompts in test set")

    return train_prompts, test_prompts


def generate_review_tasks(
    prompts: List[str],
    user_names: List[str],
    length_criteria: Dict[str, int],
    examples_per_user: int = 16,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Generate review tasks with train/test/val split per user.

    Each user samples examples_per_user unique prompts from the shared pool:
    - First 10 for train
    - Next 5 for test
    - Last 1 for val

    Returns:
        Tuple of (train_tasks, test_tasks, val_tasks)
    """
    if len(prompts) < examples_per_user:
        raise ValueError(
            f"Not enough prompts ({len(prompts)}) to sample {examples_per_user} per user"
        )

    train_tasks = []
    test_tasks = []
    val_tasks = []

    # Each user samples from the same shared pool
    for person in user_names:
        # Sample 16 unique prompts for this user from the shared pool
        user_prompts = random.sample(prompts, examples_per_user)

        # Split into train (10), test (5), val (1)
        train_prompts = user_prompts[:10]
        test_prompts = user_prompts[10:15]
        val_prompts = user_prompts[15:16]

        # Create tasks for each split
        for prompt in train_prompts:
            train_tasks.append(
                {
                    "prompt": prompt,
                    "person": person,
                    "length_preference": length_criteria[person],
                }
            )

        for prompt in test_prompts:
            test_tasks.append(
                {
                    "prompt": prompt,
                    "person": person,
                    "length_preference": length_criteria[person],
                }
            )

        for prompt in val_prompts:
            val_tasks.append(
                {
                    "prompt": prompt,
                    "person": person,
                    "length_preference": length_criteria[person],
                }
            )

    return train_tasks, test_tasks, val_tasks


def process_review_task(task: Dict[str, Any]) -> Dict[str, Any]:
    """Process a single review task to create a training example."""
    # Build the advisor prompt
    prompt = build_advisor_prompt(task)

    return {
        "prompt": prompt,
        "env_class": "reviews_length_scaled",
        "reward_spec": {
            "ground_truth": task["length_preference"],
        },
        # The following keys become ``extras`` in the env
        "original_question": task["prompt"],
        "person": task["person"],
    }


def process_tasks(tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Process review tasks in parallel to create training examples."""
    # Process tasks in parallel
    rows = []
    with ThreadPoolExecutor(max_workers=12) as executor:
        for row in tqdm(
            executor.map(process_review_task, tasks),
            desc="Processing tasks",
            total=len(tasks),
        ):
            rows.append(row)

    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Construct reviews length scaled dataset"
    )
    parser.add_argument(
        "--num_users",
        type=int,
        default=100,
        help="Number of users to generate (default: 100)",
    )
    parser.add_argument(
        "--min_length",
        type=int,
        default=10,
        help="Minimum word count preference (default: 10)",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=1000,
        help="Maximum word count preference (default: 1000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for user generation (default: 42)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/reviews",
        help="Output directory for dataset",
    )

    args = parser.parse_args()
    random.seed(args.seed)

    # Generate scaled users with random length preferences
    print(f"Generating {args.num_users} users with length preferences...")
    user_names, length_criteria = generate_scaled_users(
        num_users=args.num_users,
        min_length=args.min_length,
        max_length=args.max_length,
        seed=args.seed,
    )

    print(f"Generated {len(user_names)} users")
    print("Sample users and preferences:")
    for name in user_names[:5]:
        print(f"  {name}: {length_criteria[name]} words")
    print("  ...")

    print("\nLoading unique prompts from HuggingFace dataset...")
    hf_train_prompts, hf_test_prompts = load_unique_prompts()

    # Combine all prompts for sampling
    all_prompts = hf_train_prompts + hf_test_prompts
    print(f"Total unique prompts available: {len(all_prompts)}")
    print(f"Users: {len(user_names)}")
    print(f"Required prompts (16 per user): {len(user_names) * 16}")

    print("\nGenerating review tasks with train/test/val split...")
    # Generate review tasks with per-user split
    train_tasks, test_tasks, val_tasks = generate_review_tasks(
        all_prompts, user_names, length_criteria, examples_per_user=16
    )

    print(
        f"Processing {len(train_tasks)} train, {len(test_tasks)} test, "
        f"and {len(val_tasks)} val tasks..."
    )

    # Process tasks to create training examples
    train_rows = process_tasks(train_tasks)
    test_rows = process_tasks(test_tasks)
    val_rows = process_tasks(val_tasks)

    # Write to parquet
    os.makedirs(args.output_dir, exist_ok=True)
    suffix = f"_n={args.num_users}"
    train_parquet_path = os.path.join(
        args.output_dir, f"train_length_scaled{suffix}.parquet"
    )
    test_parquet_path = os.path.join(
        args.output_dir, f"test_length_scaled{suffix}.parquet"
    )
    val_parquet_path = os.path.join(
        args.output_dir, f"validation_length_scaled{suffix}.parquet"
    )

    datasets.Dataset.from_list(train_rows).to_parquet(train_parquet_path)
    datasets.Dataset.from_list(test_rows).to_parquet(test_parquet_path)
    datasets.Dataset.from_list(val_rows).to_parquet(val_parquet_path)

    print(
        f"\nWrote {len(train_rows)} train, {len(test_rows)} test, "
        f"and {len(val_rows)} val examples"
    )
    print(f"Train: {train_parquet_path}")
    print(f"Test: {test_parquet_path}")
    print(f"Validation: {val_parquet_path}")
