"""Evaluation script for MTOB (Machine Translation from One Book) advisor model.

Supports:
- GCP model download with --gcp flag
- Automatic vLLM server management
- Multiple evaluation runs with bootstrap confidence intervals
- Both trained advisor models and untrained baselines

Example usage:
    # Evaluate a model from GCP
    python -m advisor_models.mtob.eval_mtob \
        --gcp \
        --model_name "mtob_v0_qwen2.5_7b_10ep" \
        --dataset_path data/mtob/validation.parquet \
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
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from advisor_models.mtob.config import (
    STUDENT_PROMPT_TEMPLATE,
    compute_translation_score,
    extract_translation,
)
from utils.eval_utils import (
    download_model_if_needed,
    setup_vllm_server,
    cleanup_vllm_server,
    compute_multi_run_statistics,
    format_ci_string,
    add_common_eval_args,
)


class MTOBEvaluator:
    """Evaluator for MTOB advisor model via an OpenAI-compatible endpoint."""

    def __init__(
        self,
        advisor_model: str,
        advisor_api_base: str = "http://127.0.0.1:8000/v1",
        student_model: str = "gpt-4o-mini",
        student_api_base: Optional[str] = None,
    ):
        """Initialize evaluator to call a remote advisor endpoint."""
        self.advisor_model = advisor_model
        self.advisor_api_base = advisor_api_base
        self.student_model = student_model
        self.student_api_base = student_api_base

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

    def get_student_response(
        self,
        advisor_feedback: str,
        original_question: str,
        reference_materials: List[str],
    ) -> str:
        """Get translation from student model using advisor feedback."""
        formatted_prompt = STUDENT_PROMPT_TEMPLATE.format(
            original_question=original_question,
            reference_materials="\n".join(reference_materials),
            advisor_feedback=advisor_feedback,
        )

        messages = [
            {"role": "user", "content": formatted_prompt},
        ]

        temperature = 0
        if "gpt-5" in self.student_model:
            temperature = 1.0

        try:
            # Build kwargs for litellm.completion
            kwargs = {
                "model": self.student_model,
                "messages": messages,
                "temperature": temperature,
            }

            # Add api_base if provided
            if self.student_api_base is not None:
                kwargs["api_base"] = self.student_api_base

            response = litellm.completion(**kwargs)
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
            ground_truth = row["reward_spec"]["ground_truth"]
            reference_materials = row.get("reference_materials", [])

            # Convert prompt to proper format if needed
            if isinstance(prompt, str):
                try:
                    prompt = json.loads(prompt)
                except Exception:
                    prompt = [{"role": "user", "content": prompt}]
            elif hasattr(prompt, "tolist"):
                prompt = prompt.tolist()

            if not isinstance(prompt, list):
                prompt = [{"role": "user", "content": str(prompt)}]

            # Convert reference_materials if needed
            if hasattr(reference_materials, "tolist"):
                reference_materials = reference_materials.tolist()

            # Generate advisor feedback
            advisor_feedback = self.generate_advisor_feedback(prompt)

            # Get student response
            student_response = self.get_student_response(
                advisor_feedback, original_question, reference_materials
            )

            # Extract translation and compute score
            translation = extract_translation(student_response)
            score = compute_translation_score(translation, ground_truth)

            result = {
                "index": idx,
                "original_question": original_question,
                "ground_truth": ground_truth,
                "advisor_feedback": advisor_feedback,
                "student_response": student_response,
                "translation": translation,
                "score": score,
            }
            return result

        except Exception as e:
            print(f"Error processing example {idx}: {e}")
            return {
                "index": idx,
                "original_question": row.get("original_question", ""),
                "ground_truth": row.get("reward_spec", {}).get("ground_truth", ""),
                "advisor_feedback": "",
                "student_response": "",
                "translation": "",
                "score": 0.0,
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

        print(f"Evaluating {len(df)} examples with {max_workers} threads...")

        # Create list of (index, row) tuples for processing
        examples = list(df.iterrows())

        # Use ThreadPoolExecutor for concurrent processing
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_example = {
                executor.submit(self.process_single_example, example): example[0]
                for example in examples
            }

            with tqdm(total=len(examples)) as pbar:
                for future in as_completed(future_to_example):
                    result = future.result()
                    results.append(result)
                    pbar.update(1)

        # Compute aggregate metrics
        all_scores = [r["score"] for r in results]
        metrics = {
            "total_examples": len(results),
            "overall_chrf": np.mean(all_scores) if all_scores else 0.0,
            "overall_std": np.std(all_scores) if all_scores else 0.0,
            "overall_sem": np.std(all_scores) / np.sqrt(len(all_scores))
            if all_scores
            else 0.0,
        }

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
        print("MTOB ADVISOR EVALUATION REPORT")
        print("=" * 70)

        print(f"\nNumber of runs: {aggregate_stats['n']}")
        print(f"Total examples per run: {metrics['total_examples']}")
        print(f"{format_ci_string(aggregate_stats, 'chrF Score')}")

        print("\n" + "=" * 70)


def run_multi_evaluation(
    evaluator: MTOBEvaluator,
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
        run_mean = results["metrics"]["overall_chrf"]
        print(f"Run {run_idx + 1} mean chrF score: {run_mean:.4f}")

    # Compute aggregate statistics across runs using all individual scores
    aggregate_stats = compute_multi_run_statistics(all_run_scores)

    return {
        "run_results": all_run_results,
        "run_scores": all_run_scores,
        "aggregate_stats": aggregate_stats,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate MTOB advisor model with GCP/vLLM support"
    )

    # Add common evaluation arguments
    add_common_eval_args(parser)

    # Add student model API base argument
    parser.add_argument(
        "--student_model_api_base",
        type=str,
        default=None,
        help="API base for student model",
    )

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
        evaluator = MTOBEvaluator(
            advisor_model="hosted_vllm/" + served_model_name,
            advisor_api_base="http://127.0.0.1:8000/v1",
            student_model=args.student_model,
            student_api_base=args.student_model_api_base,
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
