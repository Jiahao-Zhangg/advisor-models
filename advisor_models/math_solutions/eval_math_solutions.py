"""Evaluation script for math solutions advisor model.

Supports:
- GCP model download with --gcp flag
- Automatic vLLM server management
- Multiple evaluation runs with bootstrap confidence intervals
- Both trained advisor models and untrained baselines

Example usage:
    # Evaluate a model from GCP
    python -m advisor_models.math_solutions.eval_math_solutions \
        --gcp \
        --model_name "math_solutions_v0_qwen2.5_7b_10ep" \
        --dataset_path data/math_solutions/validation.parquet \
        --student_model gpt-4o-mini \
        --num_runs 5

    # Evaluate a HuggingFace model (untrained baseline)
    python -m advisor_models.math_solutions.eval_math_solutions \
        --model_name "Qwen/Qwen2.5-7B-Instruct" \
        --dataset_path data/math_solutions/validation.parquet \
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
import math_verify

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from advisor_models.math_solutions.config import (
    STUDENT_SYSTEM_PROMPT,
    STUDENT_INSTRUCTION,
    STUDENTS,
    STUDENT_PERSONAS,
    STYLE_JUDGE_PROMPT,
)
from utils.eval_utils import (
    download_model_if_needed,
    setup_vllm_server,
    cleanup_vllm_server,
    compute_multi_run_statistics,
    format_ci_string,
    add_common_eval_args,
)


class MathSolutionsEvaluator:
    """Evaluator for math solutions advisor model using an OpenAI-compatible endpoint."""

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
        self.judge_model = "gpt-4.1-mini"
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

    def get_student_response(self, advisor_feedback: str, math_problem: str) -> str:
        """Get math solution from student model using advisor feedback."""
        user_context = STUDENT_INSTRUCTION.format(
            problem=math_problem,
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

    def compute_correctness_score(self, solution: str, ground_truth: str) -> float:
        """Compute correctness score using math_verify."""
        try:
            # Extract answer from solution
            extracted_answer = math_verify.parse(solution, parsing_timeout=None)
            if extracted_answer is None:
                return 0.0

            # Verify against ground truth
            is_correct = math_verify.verify(
                extracted_answer, ground_truth, timeout_seconds=None
            )

            return 1.0 if is_correct else 0.0
        except Exception as e:
            print(f"Error computing correctness score: {e}")
            return 0.0

    def compute_style_score(self, solution: str, judge_criteria: str) -> float:
        """Compute style matching score using LLM-as-a-judge."""
        judge_prompt = STYLE_JUDGE_PROMPT.format(
            solution=solution, judge_criteria=judge_criteria
        )

        try:
            response = self.openai_client.chat.completions.create(
                model=self.judge_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a style evaluator for math solutions.",
                    },
                    {"role": "user", "content": judge_prompt},
                ],
                temperature=0.0,
            )
            response_text = response.choices[0].message.content.strip()

            # Parse judge response
            if "ACCEPT" in response_text:
                return 1.0
            elif "PARTIAL" in response_text:
                return 0.4
            elif "REJECT" in response_text:
                return 0.0
            else:
                print(f"Warning: Unexpected judge response: {response_text}")
                return 0.0
        except Exception as e:
            print(f"Error computing style score: {e}")
            return 0.0

    def process_single_example(self, idx_row_tuple):
        """Process a single example - designed for multithreading."""
        idx, row = idx_row_tuple
        try:
            # Extract data from row
            prompt = row["prompt"]
            original_question = row["original_question"]
            student = row["student"]
            judge_criteria = row.get("reward_spec").get("ground_truth")
            ground_truth = row.get("math_correct_answer")

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

            # Compute scores
            correctness_score = self.compute_correctness_score(
                student_response, ground_truth
            )
            style_score = self.compute_style_score(student_response, judge_criteria)
            total_score = (correctness_score + style_score) / 2.0

            # Store results
            result = {
                "index": idx,
                "student": student,
                "original_question": original_question,
                "ground_truth": ground_truth,
                "advisor_feedback": advisor_feedback,
                "student_response": student_response,
                "correctness_score": correctness_score,
                "style_score": style_score,
                "total_score": total_score,
                "solution_length": len(student_response.split())
                if student_response
                else 0,
            }
            return result

        except Exception as e:
            print(f"Error processing example {idx}: {e}")
            # Return failed result
            return {
                "index": idx,
                "student": row.get("student", "unknown"),
                "original_question": row.get("original_question", ""),
                "ground_truth": row.get("math_correct_answer"),
                "advisor_feedback": "",
                "student_response": "",
                "correctness_score": 0.0,
                "style_score": 0.0,
                "total_score": 0.0,
                "solution_length": 0,
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
        student_scores = {
            student: {"correctness": [], "style": [], "total": []}
            for student in STUDENTS
        }

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

                    # Update student scores
                    student = result["student"]
                    if student in student_scores:
                        student_scores[student]["correctness"].append(
                            result["correctness_score"]
                        )
                        student_scores[student]["style"].append(result["style_score"])
                        student_scores[student]["total"].append(result["total_score"])

                    pbar.update(1)

        # Compute aggregate metrics
        all_correctness = [r["correctness_score"] for r in results]
        all_style = [r["style_score"] for r in results]
        all_total = [r["total_score"] for r in results]

        metrics = {
            "total_examples": len(results),
            "overall_correctness": np.mean(all_correctness) if all_correctness else 0.0,
            "overall_style": np.mean(all_style) if all_style else 0.0,
            "overall_total": np.mean(all_total) if all_total else 0.0,
            "correctness_std": np.std(all_correctness) if all_correctness else 0.0,
            "style_std": np.std(all_style) if all_style else 0.0,
            "total_std": np.std(all_total) if all_total else 0.0,
            "correctness_sem": np.std(all_correctness) / np.sqrt(len(all_correctness))
            if all_correctness
            else 0.0,
            "style_sem": np.std(all_style) / np.sqrt(len(all_style))
            if all_style
            else 0.0,
            "total_sem": np.std(all_total) / np.sqrt(len(all_total))
            if all_total
            else 0.0,
            "student_metrics": {},
            "student_counts": {},
            "avg_solution_length": np.mean([r["solution_length"] for r in results])
            if results
            else 0.0,
        }

        # Per-student metrics
        for student in STUDENTS:
            scores = student_scores[student]
            if scores["total"]:
                metrics["student_metrics"][student] = {
                    "correctness": np.mean(scores["correctness"]),
                    "style": np.mean(scores["style"]),
                    "total": np.mean(scores["total"]),
                }
                metrics["student_counts"][student] = len(scores["total"])
            else:
                metrics["student_metrics"][student] = {
                    "correctness": 0.0,
                    "style": 0.0,
                    "total": 0.0,
                }
                metrics["student_counts"][student] = 0

        return {
            "metrics": metrics,
            "detailed_results": results,
            "all_total_scores": all_total,
            "all_correctness_scores": all_correctness,
            "all_style_scores": all_style,
        }

    def print_evaluation_report(
        self,
        evaluation_results: Dict[str, Any],
        correctness_aggregate_stats: Dict[str, Any],
        style_aggregate_stats: Dict[str, Any],
    ):
        """Print a formatted evaluation report."""
        metrics = evaluation_results["metrics"]

        print("\n" + "=" * 70)
        print("MATH SOLUTIONS ADVISOR EVALUATION REPORT")
        print("=" * 70)

        print(f"\nNumber of runs: {correctness_aggregate_stats['n']}")
        print(f"Total examples per run: {metrics['total_examples']}")
        print(f"{format_ci_string(correctness_aggregate_stats, 'Correctness Score')}")
        print(f"{format_ci_string(style_aggregate_stats, 'Style Score')}")
        print(f"Average Solution Length: {metrics['avg_solution_length']:.1f} words")

        print("\nPer-Student Performance (last run):")
        print(
            f"{'Student':<8} {'Count':<6} {'Correctness':<12} {'Style':<8} {'Total':<8} {'Description'}"
        )
        print("-" * 70)
        for student in STUDENTS:
            student_metrics = metrics["student_metrics"][student]
            count = metrics["student_counts"][student]
            description = STUDENT_PERSONAS[student]["style"]
            print(
                f"{student:<8} {count:<6} {student_metrics['correctness']:<12.4f} "
                f"{student_metrics['style']:<8.4f} {student_metrics['total']:<8.4f} {description}"
            )

        print("\n" + "=" * 70)


def run_multi_evaluation(
    evaluator: MathSolutionsEvaluator,
    dataset_path: str,
    num_runs: int = 5,
    max_examples: Optional[int] = None,
    max_workers: int = 12,
) -> Dict[str, Any]:
    """Run evaluation multiple times and compute aggregate statistics."""
    all_run_correctness_scores = []  # List of lists: all individual correctness scores from each run
    all_run_style_scores = []  # List of lists: all individual style scores from each run
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
        # Collect all individual correctness and style scores from this run
        all_run_correctness_scores.append(results["all_correctness_scores"])
        all_run_style_scores.append(results["all_style_scores"])

        run_correctness_mean = results["metrics"]["overall_correctness"]
        run_style_mean = results["metrics"]["overall_style"]
        print(
            f"Run {run_idx + 1} mean correctness: {run_correctness_mean:.4f}, mean style: {run_style_mean:.4f}"
        )

    # Compute aggregate statistics across runs for correctness and style separately
    correctness_aggregate_stats = compute_multi_run_statistics(
        all_run_correctness_scores
    )
    style_aggregate_stats = compute_multi_run_statistics(all_run_style_scores)

    return {
        "run_results": all_run_results,
        "correctness_aggregate_stats": correctness_aggregate_stats,
        "style_aggregate_stats": style_aggregate_stats,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate math solutions advisor model with GCP/vLLM support"
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
        evaluator = MathSolutionsEvaluator(
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
            correctness_aggregate_stats=multi_results["correctness_aggregate_stats"],
            style_aggregate_stats=multi_results["style_aggregate_stats"],
        )

    finally:
        # Cleanup vLLM server and temp directory
        cleanup_vllm_server(vllm_process, temp_dir)


if __name__ == "__main__":
    main()
