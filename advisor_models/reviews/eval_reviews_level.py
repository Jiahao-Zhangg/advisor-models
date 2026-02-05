"""Evaluation script for reviews level advisor model.

Supports:
- GCP model download with --gcp flag
- Automatic vLLM server management
- Multiple evaluation runs with bootstrap confidence intervals
- Both trained advisor models and untrained baselines

Example usage:
    # Evaluate a model from GCP
    python -m advisor_models.reviews.eval_reviews_level \
        --gcp \
        --model_name "reviews_v0_qwen2.5_7b_10ep_hint_level" \
        --dataset_path data/reviews/validation_level.parquet \
        --student_model gpt-4o-mini \
        --num_runs 5

    # Evaluate a HuggingFace model (untrained baseline)
    python -m advisor_models.reviews.eval_reviews_level \
        --model_name "Qwen/Qwen2.5-7B-Instruct" \
        --dataset_path data/reviews/validation_level.parquet \
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

from advisor_models.reviews.config import (
    STUDENT_SYSTEM_PROMPT,
    STUDENT_INSTRUCTION,
    LEVEL_PEOPLE,
)
from utils.eval_utils import (
    download_model_if_needed,
    setup_vllm_server,
    cleanup_vllm_server,
    compute_multi_run_statistics,
    format_ci_string,
    add_common_eval_args,
)


class ReviewsLevelEvaluator:
    """Evaluator for reviews level advisor model via an OpenAI-compatible endpoint."""

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

    def compute_reading_level_score(
        self, review_text: str, level_criteria: str
    ) -> float:
        """Compute reading level appropriateness score using evaluation model."""
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-5-mini",
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
            print(f"Error computing reading level score: {e}")
            return 0.0

    def process_single_example(self, idx_row_tuple):
        """Process a single example - designed for multithreading."""
        idx, row = idx_row_tuple
        try:
            # Extract data from row
            prompt = row["prompt"]
            original_question = row["original_question"]
            person = row["person"]
            level_criteria = row["reward_spec"]["ground_truth"]

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
            score = self.compute_reading_level_score(student_response, level_criteria)

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
        person_scores = {person: [] for person in LEVEL_PEOPLE}

        print(f"Evaluating {len(df)} examples with {max_workers} threads...")

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
                    results.append(result)

                    # Update person scores
                    person = result["person"]
                    if person in person_scores:
                        person_scores[person].append(result["score"])

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
        }

        # Per-person metrics
        for person in LEVEL_PEOPLE:
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
        print("REVIEWS LEVEL ADVISOR EVALUATION REPORT")
        print("=" * 70)

        print(f"\nNumber of runs: {aggregate_stats['n']}")
        print(f"Total examples per run: {metrics['total_examples']}")
        print(f"{format_ci_string(aggregate_stats, 'Overall Score')}")
        print(f"Average Review Length: {metrics['avg_review_length']:.1f} words")

        print("\nPer-Person Performance (last run):")
        for person in LEVEL_PEOPLE:
            accuracy = metrics["person_accuracies"][person]
            count = metrics["person_counts"][person]
            print(f"  {person:8s}: {accuracy:.4f} ({count:3d} examples)")

        print("\n" + "=" * 70)


def run_multi_evaluation(
    evaluator: ReviewsLevelEvaluator,
    dataset_path: str,
    num_runs: int = 5,
    max_examples: Optional[int] = None,
    max_workers: int = 12,
) -> Dict[str, Any]:
    """Run evaluation multiple times and compute aggregate statistics."""
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
        description="Evaluate reviews level advisor model with GCP/vLLM support"
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
        evaluator = ReviewsLevelEvaluator(
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
