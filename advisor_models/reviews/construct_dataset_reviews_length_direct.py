"""Dataset construction for reviews length direct domain.

Generates training and validation datasets for direct RL training on review writing.
No advisor setup - the model directly generates reviews.

Example usage:
    python advisor_models/reviews/construct_dataset_reviews_length_direct.py \
        --output_dir data/reviews
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

from config_length_direct import (
    DIRECT_SYSTEM_PROMPT,
    DIRECT_INSTRUCTION,
    LENGTH_PEOPLE,
    LENGTH_CRITERIA,
)


def build_direct_prompt(task: Dict[str, Any]) -> List[Dict[str, str]]:
    """Build the direct prompt that the model will receive (no advisor)."""
    user_content = DIRECT_INSTRUCTION.format(
        person=task["person"],
        prompt=task["prompt"],
    )

    return [
        {"role": "system", "content": DIRECT_SYSTEM_PROMPT},
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


def generate_review_tasks(prompts: List[str]) -> List[Dict[str, Any]]:
    """Generate review tasks by assigning prompts to different people."""
    tasks = []

    for i, prompt in enumerate(prompts):
        # Cycle through people for each prompt
        person = LENGTH_PEOPLE[i % len(LENGTH_PEOPLE)]

        task = {
            "prompt": prompt,
            "person": person,
            "length_preference": LENGTH_CRITERIA[person],
        }

        tasks.append(task)

    return tasks


def process_review_task(task: Dict[str, Any]) -> Dict[str, Any]:
    """Process a single review task to create a training example."""
    # Build the direct prompt (no advisor)
    prompt = build_direct_prompt(task)

    return {
        "prompt": prompt,
        "env_class": "reviews_length_direct",
        "reward_spec": {
            "ground_truth": task["length_preference"],
        },
        # The following keys become ``extras`` in the env
        "original_question": task["prompt"],
        "person": task["person"],
    }


def process_tasks(tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Process review tasks in parallel to create training examples."""
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
        description="Construct reviews length direct dataset"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/reviews",
        help="Output directory for dataset",
    )

    args = parser.parse_args()
    random.seed(42)

    print("Loading unique prompts from HuggingFace dataset...")
    train_prompts, test_prompts = load_unique_prompts()

    print("Generating review tasks...")
    print(f"Train prompts: {len(train_prompts)}, Test prompts: {len(test_prompts)}")
    print(f"People: {LENGTH_PEOPLE}")

    # Generate review tasks
    train_tasks = generate_review_tasks(train_prompts)
    val_tasks = generate_review_tasks(test_prompts)

    print(
        f"Processing {len(train_tasks)} training and {len(val_tasks)} validation tasks..."
    )

    # Process tasks to create training examples
    train_rows = process_tasks(train_tasks)
    val_rows = process_tasks(val_tasks)

    # Write to parquet
    os.makedirs(args.output_dir, exist_ok=True)
    train_parquet_path = os.path.join(args.output_dir, "train_length_direct.parquet")
    val_parquet_path = os.path.join(args.output_dir, "validation_length_direct.parquet")

    datasets.Dataset.from_list(train_rows).to_parquet(train_parquet_path)
    datasets.Dataset.from_list(val_rows).to_parquet(val_parquet_path)

    print(
        f"Wrote {len(train_rows)} training and {len(val_rows)} validation examples to {args.output_dir}"
    )
