"""Evaluation script for rule arena (US tax) domain without advisor.

This version first generates an initial calculation, then prompts the model to improve it.

Evaluates a single model's performance on US tax calculation tasks over multiple runs
and reports confidence intervals.

Example usage:
    python -m baselines.no_advisor.eval_rule_arena \
        --model gpt-5 \
        --data_file data/rule_arena/validation_gpt-5_0.parquet \
        --num_runs 5 \
        --max_workers 25
"""

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import litellm
import pandas as pd
from tqdm import tqdm

from advisor_models.rule_arena.config import (
    build_prompt,
    compute_score,
)
from utils.eval_utils import compute_multi_run_statistics, format_ci_string


@dataclass
class EvalResult:
    """Result for a single evaluation instance."""

    problem: str
    initial_response: str
    final_response: str
    reward: float
    ground_truth: str


def generate_improved_response(
    problem: str,
    initial_response: str,
    model: str,
) -> str:
    """Generate an improved response based on the initial response."""
    improve_prompt = """Now review your calculation carefully and provide an improved version. Consider:
- Are all form fields calculated correctly?
- Did you follow the instructions precisely?
- Are there any arithmetic errors?
- Did you apply the correct tax rules?

Provide your improved calculation with the same format, ending with:
1. "The total tax owed is $xxx." (xxx is a number) if there is tax owed.
2. "The total tax overpaid is $xxx." (xxx is a number) if there is tax overpaid (and should be refunded)."""

    messages = [
        {"role": "user", "content": problem},
        {"role": "assistant", "content": initial_response},
        {"role": "user", "content": improve_prompt},
    ]

    try:
        litellm.drop_params = True
        response = litellm.completion(
            model=model,
            messages=messages,
            temperature=1.0,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error generating improved response: {e}")
        return ""


def evaluate_single_task(args_tuple) -> EvalResult:
    """Evaluate a single task."""
    task, model = args_tuple

    # Extract task information
    info_dict = task["info_dict"]
    ground_truth = task["reward_spec"]["ground_truth"]

    # Use the original_response from the data file as the initial response
    initial_response = task.get("original_response", "")

    if not initial_response:
        print("Warning: No original_response found in task data")
        return EvalResult(
            problem="",
            initial_response="",
            final_response="",
            reward=0.0,
            ground_truth=str(ground_truth),
        )

    # Build the problem prompt
    problem = build_prompt(info_dict)

    # Generate improved response based on the original response from data
    final_response = generate_improved_response(problem, initial_response, model)

    # Compute reward on final response
    reward, _ = compute_score(final_response, ground_truth)

    return EvalResult(
        problem=problem,
        initial_response=initial_response,
        final_response=final_response,
        reward=reward,
        ground_truth=str(ground_truth),
    )


def evaluate_dataset(
    tasks: List[Dict[str, Any]],
    model: str,
    max_workers: int = 20,
) -> List[EvalResult]:
    """Evaluate all tasks in the dataset."""
    args_list = [(task, model) for task in tasks]

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for result in tqdm(
            executor.map(evaluate_single_task, args_list),
            total=len(tasks),
            desc="Evaluating",
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
    print("EVALUATION SUMMARY")
    print("=" * 80)

    if aggregate_stats:
        # Multi-run statistics
        print(f"\nResults over {num_runs} runs:")
        print(f"  Average Accuracy: {format_ci_string(aggregate_stats['reward'])}")

    else:
        # Single run statistics
        avg_reward = sum(r.reward for r in results) / len(results)
        print(f"\nAverage Accuracy: {avg_reward:.4f}")

    print()


def run_multi_evaluation(
    tasks: List[Dict[str, Any]],
    model: str,
    num_runs: int,
    max_workers: int,
) -> Dict[str, Any]:
    """Run evaluation multiple times and compute aggregate statistics."""
    all_run_rewards = []
    all_run_results = []

    for run_idx in range(num_runs):
        print(f"\n{'=' * 80}")
        print(f"EVALUATION RUN {run_idx + 1}/{num_runs}")
        print(f"{'=' * 80}")

        results = evaluate_dataset(tasks, model, max_workers)
        all_run_results.append(results)

        # Collect rewards from this run
        run_rewards = [r.reward for r in results]
        all_run_rewards.append(run_rewards)

        # Print run summary
        avg_reward = sum(run_rewards) / len(run_rewards)
        print(f"Run {run_idx + 1} average accuracy: {avg_reward:.4f}")

    # Compute aggregate statistics
    aggregate_stats = {
        "reward": compute_multi_run_statistics(all_run_rewards),
    }

    return {
        "run_results": all_run_results,
        "aggregate_stats": aggregate_stats,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate model on rule arena (US tax) tasks without advisor"
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
        default="baselines/no_advisor/results",
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
            / f"rule_arena_{args.model.replace('/', '_')}_{args.num_runs}runs.json"
        )
        with open(output_file, "w") as f:
            json.dump(
                {
                    "model": args.model,
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
            Path(args.output_dir) / f"rule_arena_{args.model.replace('/', '_')}.json"
        )
        with open(output_file, "w") as f:
            json.dump(
                {
                    "model": args.model,
                    "num_samples": len(tasks),
                    "average_accuracy": sum(r.reward for r in results) / len(results),
                    "results": [
                        {
                            "problem": r.problem,
                            "initial_response": r.initial_response,
                            "final_response": r.final_response,
                            "reward": r.reward,
                            "ground_truth": r.ground_truth,
                        }
                        for r in results
                    ],
                },
                f,
                indent=2,
            )
        print(f"Saved results to {output_file}")


if __name__ == "__main__":
    main()
