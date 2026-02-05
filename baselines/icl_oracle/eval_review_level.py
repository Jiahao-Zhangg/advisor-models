"""Evaluation script for review level domain with oracle (ground truth) preferences.

Uses ground truth reading level preferences for all users from the dataset.
Collects all user preferences and provides them in every prompt.
This provides an upper bound on performance with perfect personalization.

Example usage:
    python baselines/icl_oracle/eval_review_level.py \
        --model gpt-4o-mini \
        --data_file data/reviews/validation_level.parquet \
        --num_runs 5 \
        --max_workers 25
"""

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import litellm
import pandas as pd
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from advisor_models.reviews.config import (
    BASELINE_SYSTEM_PROMPT,
    BASELINE_INSTRUCTION,
)
from utils.eval_utils import compute_multi_run_statistics, format_ci_string


@dataclass
class EvalResult:
    """Result for a single evaluation instance."""

    person: str
    prompt: str
    response: str
    reward: float
    level_criteria: str
    all_preferences: str


def build_all_preferences_string(user_preferences: Dict[str, str]) -> str:
    """Build a string containing all user reading level preferences."""
    preference_lines = []

    # Map criteria to reading levels
    level_mapping = {
        "elementary school student": "elementary school",
        "middle school student": "middle school",
        "high school student": "high school",
        "college student": "college",
        "college professor": "college professor",
    }

    for person, criteria in sorted(user_preferences.items()):
        # Extract reading level from criteria
        level = "unknown"
        for key, value in level_mapping.items():
            if key in criteria:
                level = value
                break
        preference_lines.append(
            f"- If you are writing a review for {person}, you must write at a {level} reading level"
        )

    return "\n".join(preference_lines)


def compute_level_reward(review_text: str, level_criteria: str) -> float:
    """Compute reward based on reading level appropriateness using LLM judge."""
    try:
        litellm.drop_params = True
        response = litellm.completion(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a reading level evaluator."},
                {
                    "role": "user",
                    "content": level_criteria.format(review=review_text),
                },
            ],
        )
        response_text = response.choices[0].message.content
        if "Yes" in response_text and "No" not in response_text:
            return 1.0
        return 0.0
    except Exception as e:
        print(f"Error computing level reward: {e}")
        return 0.0


def generate_response(
    prompt: str,
    person: str,
    all_preferences_string: str,
    model: str,
) -> str:
    """Generate a response using all user preferences."""
    # Build prefix with all preferences
    prefix = f"""Here are the reading level requirements for all users:

{all_preferences_string}

"""

    # Format baseline prompt
    user_content = BASELINE_INSTRUCTION.format(
        person=person,
        prompt=prompt,
    )

    messages = [
        {"role": "system", "content": BASELINE_SYSTEM_PROMPT},
        {"role": "user", "content": prefix + user_content},
    ]

    try:
        litellm.drop_params = True
        response = litellm.completion(
            model=model,
            messages=messages,
            temperature=0.7,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error generating response: {e}")
        return ""


def evaluate_single_task(args_tuple) -> EvalResult:
    """Evaluate a single task."""
    task, model, all_preferences_string = args_tuple

    # Extract task information
    person = task["person"]
    prompt = task["original_question"]
    level_criteria = task["reward_spec"]["ground_truth"]

    # Generate response with all preferences
    response = generate_response(prompt, person, all_preferences_string, model)

    # Compute reward
    reward = compute_level_reward(response, level_criteria)

    return EvalResult(
        person=person,
        prompt=prompt,
        response=response,
        reward=reward,
        level_criteria=level_criteria,
        all_preferences=all_preferences_string,
    )


def evaluate_dataset(
    tasks: List[Dict[str, Any]],
    model: str,
    max_workers: int = 20,
) -> List[EvalResult]:
    """Evaluate all tasks in the dataset."""
    # Collect all user preferences from the dataset
    user_preferences = {}
    for task in tasks:
        person = task["person"]
        level_criteria = task["reward_spec"]["ground_truth"]
        user_preferences[person] = level_criteria

    # Build the concatenated preferences string
    all_preferences_string = build_all_preferences_string(user_preferences)

    print(f"\nCollected preferences for {len(user_preferences)} users:")
    print(all_preferences_string)
    print()

    # Create args list with all_preferences_string for each task
    args_list = [(task, model, all_preferences_string) for task in tasks]

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for result in tqdm(
            executor.map(evaluate_single_task, args_list),
            total=len(tasks),
            desc="Evaluating with Oracle",
        ):
            results.append(result)

    return results


def print_summary(
    results: List[EvalResult],
    aggregate_stats: Dict[str, Any] = None,
    num_runs: int = 1,
):
    """Print evaluation summary."""
    print("\n" + "=" * 80)
    print("ORACLE EVALUATION SUMMARY (Ground Truth Reading Level Preferences)")
    print("=" * 80)

    if aggregate_stats:
        # Multi-run statistics
        print(f"\nResults over {num_runs} runs:")
        print(f"  Average Reward: {format_ci_string(aggregate_stats['reward'])}")

        # Per-person breakdown
        print("\n  Per-person statistics:")
        for person, stats in aggregate_stats["per_person"].items():
            print(f"    {person}: {format_ci_string(stats)}")

    else:
        # Single run statistics
        avg_reward = sum(r.reward for r in results) / len(results)
        print(f"\nAverage Reward: {avg_reward:.4f}")

        # Per-person breakdown
        person_rewards = {}
        for r in results:
            if r.person not in person_rewards:
                person_rewards[r.person] = []
            person_rewards[r.person].append(r.reward)

        print("\nPer-person average rewards:")
        for person, rewards in sorted(person_rewards.items()):
            avg = sum(rewards) / len(rewards)
            print(f"  {person}: {avg:.4f} ({len(rewards)} samples)")

    print()


def run_multi_evaluation(
    tasks: List[Dict[str, Any]],
    model: str,
    num_runs: int,
    max_workers: int,
) -> Dict[str, Any]:
    """Run evaluation multiple times and compute aggregate statistics."""
    all_run_rewards = []
    all_run_person_rewards = {person: [] for person in set(t["person"] for t in tasks)}
    all_run_results = []

    for run_idx in range(num_runs):
        print(f"\n{'=' * 80}")
        print(f"ORACLE EVALUATION RUN {run_idx + 1}/{num_runs}")
        print(f"{'=' * 80}")

        results = evaluate_dataset(tasks, model, max_workers)
        all_run_results.append(results)

        # Collect rewards from this run
        run_rewards = [r.reward for r in results]
        all_run_rewards.append(run_rewards)

        # Collect per-person rewards
        for r in results:
            all_run_person_rewards[r.person].append(r.reward)

        # Print run summary
        avg_reward = sum(run_rewards) / len(run_rewards)
        print(f"Run {run_idx + 1} average reward: {avg_reward:.4f}")

    # Compute aggregate statistics
    aggregate_stats = {
        "reward": compute_multi_run_statistics(all_run_rewards),
    }

    # Compute per-person statistics
    per_person_stats = {}
    for person in all_run_person_rewards:
        # Reshape: list of runs, each containing rewards for this person
        person_run_rewards = []
        for run_results in all_run_results:
            person_rewards = [r.reward for r in run_results if r.person == person]
            if person_rewards:
                person_run_rewards.append(person_rewards)

        if person_run_rewards:
            per_person_stats[person] = compute_multi_run_statistics(person_run_rewards)

    aggregate_stats["per_person"] = per_person_stats

    return {
        "run_results": all_run_results,
        "aggregate_stats": aggregate_stats,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate model on review level tasks with oracle (ground truth) preferences"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model to evaluate (e.g., gpt-4o-mini)",
    )
    parser.add_argument(
        "--data_file",
        type=str,
        required=True,
        help="Path to validation parquet file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="baselines/icl_oracle/results",
        help="Directory to save results",
    )
    parser.add_argument(
        "--num_runs",
        type=int,
        default=1,
        help="Number of evaluation runs",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=20,
        help="Number of parallel workers",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Number of samples to evaluate (default: all)",
    )

    args = parser.parse_args()

    # Load dataset
    print(f"Loading dataset from {args.data_file}")
    df = pd.read_parquet(args.data_file)
    tasks = df.to_dict("records")

    if args.num_samples:
        tasks = tasks[: args.num_samples]

    print(f"Loaded {len(tasks)} tasks")

    # Run evaluation
    if args.num_runs > 1:
        multi_results = run_multi_evaluation(
            tasks, args.model, args.num_runs, args.max_workers
        )
        print_summary(
            multi_results["run_results"][-1],
            multi_results["aggregate_stats"],
            args.num_runs,
        )

        # Save aggregate results
        os.makedirs(args.output_dir, exist_ok=True)
        output_file = (
            Path(args.output_dir)
            / f"oracle_level_{args.model.replace('/', '_')}_{args.num_runs}runs.json"
        )
        with open(output_file, "w") as f:
            json.dump(
                {
                    "model": args.model,
                    "evaluation_type": "oracle_level",
                    "num_runs": args.num_runs,
                    "num_samples": len(tasks),
                    "aggregate_stats": multi_results["aggregate_stats"],
                },
                f,
                indent=2,
            )
        print(f"Saved aggregate results to {output_file}")

    else:
        results = evaluate_dataset(tasks, args.model, args.max_workers)
        print_summary(results)

        # Save results
        os.makedirs(args.output_dir, exist_ok=True)
        output_file = (
            Path(args.output_dir) / f"oracle_level_{args.model.replace('/', '_')}.json"
        )
        with open(output_file, "w") as f:
            json.dump(
                {
                    "model": args.model,
                    "evaluation_type": "oracle_level",
                    "num_samples": len(tasks),
                    "average_reward": sum(r.reward for r in results) / len(results),
                    "results": [
                        {
                            "person": r.person,
                            "prompt": r.prompt,
                            "response": r.response,
                            "reward": r.reward,
                            "level_criteria": r.level_criteria,
                        }
                        for r in results
                    ],
                    "all_preferences": results[0].all_preferences if results else "",
                },
                f,
                indent=2,
            )
        print(f"Saved results to {output_file}")


if __name__ == "__main__":
    main()
