"""Evaluation script for reviews length scaled advisor model.

Supports:
- GCP model download with --gcp flag
- Automatic vLLM server management
- Multiple evaluation runs with bootstrap confidence intervals
- Both trained advisor models and untrained baselines
- Scaled user count (up to 100 users)

Example usage:
    # Evaluate a model from GCP
    python -m advisor_models.reviews.eval_reviews_length_scaled \
        --gcp \
        --model_name "reviews_length_scaled_qwen2.5_7b_10ep" \
        --dataset_path data/reviews/test_length_scaled_n=100.parquet \
        --student_model gpt-4o-mini \
        --num_runs 5

    # Evaluate a HuggingFace/local model (untrained baseline)
    python -m advisor_models.reviews.eval_reviews_length_scaled \
        --model_name "Qwen/Qwen2.5-7B-Instruct" \
        --dataset_path data/reviews/test_length_scaled_n=100.parquet \
        --student_model gpt-4o-mini \
        --num_runs 5
"""

import argparse
import json
import os
import sys
import pandas as pd
from typing import Dict, List, Any, Optional
from tqdm import tqdm
import numpy as np
import litellm
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from advisor_models.reviews.config_length_scaled import (
    STUDENT_SYSTEM_PROMPT,
    STUDENT_INSTRUCTION,
    compute_length_reward,
)
from utils.eval_utils import (
    download_model_if_needed,
    setup_vllm_server,
    cleanup_vllm_server,
    compute_multi_run_statistics,
    format_ci_string,
    add_common_eval_args,
)


class ReviewsLengthScaledEvaluator:
    """Evaluator for reviews length scaled advisor model via an OpenAI-compatible endpoint."""

    def __init__(
        self,
        advisor_model: str,
        advisor_api_base: str = "http://127.0.0.1:8000/v1",
        student_model: str = "gpt-4o-mini",
    ):
        """Initialize evaluator to call a remote advisor endpoint."""
        self.advisor_model = advisor_model
        self.advisor_api_base = advisor_api_base
        self.student_model = student_model
        self.openai_client = OpenAI()

        # Initialize Anthropic client if using Claude models
        if "openrouter" in self.student_model.lower():
            self.openrouter_client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=os.getenv("OPENROUTER_API_KEY"),
            )
        elif "claude" in self.student_model.lower():
            self.anthropic_client = OpenAI(
                base_url="https://api.anthropic.com/v1",
                api_key=os.getenv("ANTHROPIC_API_KEY"),
            )

    def generate_advisor_feedback(self, prompt: List[Dict[str, str]]) -> str:
        """Generate advisor feedback by calling the configured OpenAI-compatible endpoint."""
        try:
            response = litellm.completion(
                model=self.advisor_model,
                messages=prompt,
                temperature=0.0,
                api_base=self.advisor_api_base,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error generating advisor feedback: {e}")
            return ""

    def get_student_response(self, advisor_feedback: str, review_prompt: str) -> str:
        """Get review from student model using advisor feedback."""
        user_context = STUDENT_INSTRUCTION.format(
            prompt=review_prompt,
            advisor_feedback=advisor_feedback,
        )

        messages = [
            {"role": "system", "content": STUDENT_SYSTEM_PROMPT},
            {"role": "user", "content": user_context},
        ]

        temperature = 0
        if "gpt-5" in self.student_model:
            temperature = 1.0

        try:
            if "openrouter" in self.student_model.lower():
                response = self.openrouter_client.chat.completions.create(
                    model=self.student_model.split("openrouter/")[1],
                    messages=messages,
                    temperature=temperature,
                )
            elif "claude" in self.student_model.lower():
                # Use Anthropic client for Claude models
                response = self.anthropic_client.chat.completions.create(
                    model=self.student_model,
                    messages=messages,
                    temperature=temperature,
                )
            else:
                # Use OpenAI client for other models
                response = self.openai_client.chat.completions.create(
                    model=self.student_model,
                    messages=messages,
                    temperature=temperature,
                )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error getting student response: {e}")
            return ""

    def process_single_example(self, idx_row_tuple):
        """Process a single example - designed for multithreading."""
        idx, row = idx_row_tuple
        try:
            # Extract data from row
            prompt = row["prompt"]
            original_question = row["original_question"]
            person = row["person"]
            preferred_length = row["reward_spec"]["ground_truth"]

            # Convert prompt to proper format if needed
            if isinstance(prompt, str):
                # If prompt is a string, assume it's JSON and parse it
                try:
                    prompt = json.loads(prompt)
                except Exception:
                    # If not JSON, create a simple user message
                    prompt = [{"role": "user", "content": prompt}]
            elif hasattr(prompt, "tolist"):
                # If it's a numpy array or pandas series, convert to list
                prompt = prompt.tolist()

            # Ensure prompt is a list of dicts
            if not isinstance(prompt, list):
                prompt = [{"role": "user", "content": str(prompt)}]

            # Generate advisor feedback
            advisor_feedback = self.generate_advisor_feedback(prompt)

            # Get student response
            student_response = self.get_student_response(
                advisor_feedback, original_question
            )

            # Compute score
            score = compute_length_reward(student_response, preferred_length)

            # Store results
            result = {
                "index": idx,
                "person": person,
                "original_question": original_question,
                "advisor_feedback": advisor_feedback,
                "student_response": student_response,
                "score": score,
                "review_length": len(student_response.split())
                if student_response
                else 0,
                "preferred_length": preferred_length,
            }
            return result

        except Exception as e:
            print(f"Error processing example {idx}: {e}")
            # Return failed result
            return {
                "index": idx,
                "person": row["person"],
                "original_question": row["original_question"],
                "advisor_feedback": "",
                "student_response": "",
                "score": 0.0,
                "review_length": 0,
                "preferred_length": row["reward_spec"]["ground_truth"],
                "error": str(e),
            }

    def evaluate_dataset(
        self,
        dataset_path: str,
        max_workers: int = 12,
        max_examples: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Evaluate the model on a dataset using multithreading."""
        print(f"Loading dataset from {dataset_path}")
        df = pd.read_parquet(dataset_path)

        # Limit examples if specified
        if max_examples is not None and len(df) > max_examples:
            df = df.sample(n=max_examples, random_state=42)

        results = []
        # Get unique people from the dataset
        unique_people = df["person"].unique().tolist()
        person_scores = {person: [] for person in unique_people}

        print(f"Evaluating {len(df)} examples with {max_workers} threads...")
        print(f"Found {len(unique_people)} unique users in dataset")

        # Create list of (index, row) tuples for processing
        examples = list(df.iterrows())

        # Use ThreadPoolExecutor for concurrent processing
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_example = {
                executor.submit(self.process_single_example, example): example[0]
                for example in examples
            }

            # Process completed tasks with progress bar
            with tqdm(total=len(examples)) as pbar:
                for future in as_completed(future_to_example):
                    result = future.result()
                    if result["student_response"]:
                        results.append(result)

                        # Update person scores
                        person = result["person"]
                        if person in person_scores:
                            person_scores[person].append(result["score"])
                    else:  # if the student response errors, ignore it so it doesn't count towards stats
                        print(
                            f"Warning: Student response for example {result['index']} was empty or None"
                        )

                    pbar.update(1)

        # Compute aggregate metrics
        all_scores = [r["score"] for r in results]
        metrics = {
            "total_examples": len(results),
            "overall_accuracy": np.mean(all_scores) if all_scores else 0.0,
            "overall_std": np.std(all_scores) if all_scores else 0.0,
            "overall_sem": np.std(all_scores) / np.sqrt(len(all_scores))
            if all_scores
            else 0.0,
            "person_accuracies": {},
            "person_counts": {},
            "avg_review_length": np.mean([r["review_length"] for r in results])
            if results
            else 0.0,
            "unique_people": unique_people,
        }

        # Per-person metrics
        for person in unique_people:
            scores = person_scores[person]
            if scores:
                metrics["person_accuracies"][person] = np.mean(scores)
                metrics["person_counts"][person] = len(scores)
            else:
                metrics["person_accuracies"][person] = 0.0
                metrics["person_counts"][person] = 0

        return {
            "metrics": metrics,
            "detailed_results": results,
            "all_scores": all_scores,
        }

    def print_evaluation_report(
        self,
        evaluation_results: Dict[str, Any],
        aggregate_stats: Dict[str, Any],
    ):
        """Print a formatted evaluation report."""
        metrics = evaluation_results["metrics"]

        print("\n" + "=" * 70)
        print("REVIEWS LENGTH SCALED ADVISOR EVALUATION REPORT")
        print("=" * 70)

        print(f"\nNumber of runs: {aggregate_stats['n']}")
        print(f"Total examples per run: {metrics['total_examples']}")
        print(f"Number of unique users: {len(metrics['unique_people'])}")
        print(f"{format_ci_string(aggregate_stats, 'Overall Score')}")
        print(f"Average Review Length: {metrics['avg_review_length']:.1f} words")

        print("\nPer-Person Performance (last run, showing first 10 users):")
        unique_people = metrics["unique_people"]
        for person in unique_people[:10]:
            accuracy = metrics["person_accuracies"][person]
            count = metrics["person_counts"][person]
            # Get preferred length from first result with this person
            preferred = None
            for result in evaluation_results["detailed_results"]:
                if result["person"] == person:
                    preferred = result["preferred_length"]
                    break
            if preferred is not None:
                print(
                    f"  {person:12s}: {accuracy:.4f} ({count:2d} examples, prefers {preferred:4d} words)"
                )
            else:
                print(f"  {person:12s}: {accuracy:.4f} ({count:2d} examples)")

        if len(unique_people) > 10:
            print(f"  ... and {len(unique_people) - 10} more users")

        print("\n" + "=" * 70)


def run_multi_evaluation(
    evaluator: ReviewsLengthScaledEvaluator,
    dataset_path: str,
    num_runs: int = 5,
    max_examples: Optional[int] = None,
    max_workers: int = 12,
) -> Dict[str, Any]:
    """Run evaluation multiple times and compute aggregate statistics.

    Args:
        evaluator: The evaluator instance
        dataset_path: Path to the dataset
        num_runs: Number of evaluation runs
        max_examples: Maximum examples per run (None = all)
        max_workers: Number of parallel workers

    Returns:
        Dictionary with all run results and aggregate statistics
    """
    all_run_scores = []  # List of lists: all individual scores from each run
    all_run_results = []

    for run_idx in range(num_runs):
        print(f"\n{'=' * 70}")
        print(f"EVALUATION RUN {run_idx + 1}/{num_runs}")
        print(f"{'=' * 70}")

        results = evaluator.evaluate_dataset(
            dataset_path,
            max_workers=max_workers,
            max_examples=max_examples,
        )

        all_run_results.append(results)
        # Collect all individual scores from this run
        all_run_scores.append(results["all_scores"])
        run_mean = results["metrics"]["overall_accuracy"]
        print(f"Run {run_idx + 1} mean score: {run_mean:.4f}")

    # Compute aggregate statistics across runs using all individual scores
    aggregate_stats = compute_multi_run_statistics(all_run_scores)

    return {
        "run_results": all_run_results,
        "run_scores": all_run_scores,
        "aggregate_stats": aggregate_stats,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate reviews length scaled advisor model with GCP/vLLM support"
    )

    # Add common evaluation arguments
    add_common_eval_args(parser)

    args = parser.parse_args()

    # Setup model and vLLM server
    vllm_process = None
    temp_dir = None

    try:
        # Download model from GCP if needed
        model_path, temp_dir = download_model_if_needed(
            model_name=args.model_name,
            gcp=args.gcp,
            bucket_name=args.bucket_name,
        )

        # Start vLLM server
        served_model_name = "advisor_model"
        vllm_process = setup_vllm_server(
            model_path=model_path,
            served_model_name=served_model_name,
            tensor_parallel_size=args.tensor_parallel_size,
            max_model_len=args.max_model_len,
        )

        # Initialize evaluator
        evaluator = ReviewsLengthScaledEvaluator(
            advisor_model="hosted_vllm/" + served_model_name,
            advisor_api_base="http://127.0.0.1:8000/v1",
            student_model=args.student_model,
        )

        # Run multi-evaluation
        multi_results = run_multi_evaluation(
            evaluator=evaluator,
            dataset_path=args.dataset_path,
            num_runs=args.num_runs,
            max_examples=args.max_examples,
            max_workers=args.max_workers,
        )

        # Print final report
        evaluator.print_evaluation_report(
            multi_results["run_results"][-1],
            aggregate_stats=multi_results["aggregate_stats"],
        )

    finally:
        # Cleanup vLLM server and temp directory
        cleanup_vllm_server(vllm_process, temp_dir)


if __name__ == "__main__":
    main()
