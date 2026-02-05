#!/usr/bin/env python3
"""
Script to run PAG evaluations 5 times for each domain and compute statistics.

This script runs evaluations for:
- Review Level
- Review Length
- Math Solutions

For each domain, it runs 5 evaluations and computes:
- Mean score
- SEM (standard error of the mean)
- Bootstrap 95% confidence interval

Usage:
    python -m baselines.pag.run_pag_evals
"""

import argparse
import json
import subprocess
from typing import List, Dict, Any

from utils.eval_utils import compute_multi_run_statistics


def run_evaluation(
    script_name: str,
    train_file: str,
    val_file: str,
    output_file: str,
    corpus_file: str,
    model: str = "gpt-4o-mini",
    k: int = 5,
    max_workers: int = 50,
) -> Dict[str, Any]:
    """Run a single PAG evaluation.

    Args:
        script_name: Name of the PAG script (e.g., 'pag_review_level')
        train_file: Path to training data
        val_file: Path to validation data
        output_file: Path to save results
        corpus_file: Path to corpus file to load
        model: Model to use for generation
        k: Number of similar examples to retrieve
        max_workers: Number of parallel workers

    Returns:
        Dictionary with evaluation results
    """
    cmd = [
        "python",
        "-m",
        f"baselines.pag.{script_name}",
        "--train_file",
        train_file,
        "--val_file",
        val_file,
        "--output_file",
        output_file,
        "--model",
        model,
        "--k",
        str(k),
        "--max_workers",
        str(max_workers),
        "--load_corpus",
        corpus_file,
    ]

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"Evaluation failed with return code {result.returncode}")

    # Load and return results
    with open(output_file, "r") as f:
        return json.load(f)


def extract_scores(results: Dict[str, Any]) -> List[float]:
    """Extract individual scores from evaluation results.

    Args:
        results: Dictionary with evaluation results

    Returns:
        List of scores
    """
    return [item["score"] for item in results["results"]]


def run_domain_evaluations(
    domain_name: str,
    script_name: str,
    train_file: str,
    val_file: str,
    output_prefix: str,
    corpus_file: str,
    num_runs: int = 5,
    model: str = "gpt-4o-mini",
    k: int = 5,
    max_workers: int = 50,
) -> List[List[float]]:
    """Run multiple evaluations for a domain.

    Args:
        domain_name: Name of the domain (for display)
        script_name: Name of the PAG script
        train_file: Path to training data
        val_file: Path to validation data
        output_prefix: Prefix for output files
        corpus_file: Path to corpus file to load
        num_runs: Number of evaluation runs
        model: Model to use for generation
        k: Number of similar examples to retrieve
        max_workers: Number of parallel workers

    Returns:
        List of lists, where each inner list contains scores from one run
    """
    print(f"\n{'=' * 80}")
    print(f"Running {domain_name} evaluations...")
    print(f"{'=' * 80}\n")

    all_run_scores = []

    for i in range(1, num_runs + 1):
        print(f"{domain_name} - Run {i}/{num_runs}")
        output_file = f"{output_prefix}_run{i}.json"

        results = run_evaluation(
            script_name=script_name,
            train_file=train_file,
            val_file=val_file,
            output_file=output_file,
            corpus_file=corpus_file,
            model=model,
            k=k,
            max_workers=max_workers,
        )

        scores = extract_scores(results)
        all_run_scores.append(scores)

        mean_score = sum(scores) / len(scores) if scores else 0.0
        print(f"  Mean score: {mean_score:.4f}\n")

    return all_run_scores


def print_statistics(domain_name: str, all_run_scores: List[List[float]]):
    """Print statistics for a domain.

    Args:
        domain_name: Name of the domain
        all_run_scores: List of lists of scores from multiple runs
    """
    stats = compute_multi_run_statistics(all_run_scores)

    # Compute individual run means for display
    run_means = [sum(scores) / len(scores) for scores in all_run_scores if scores]

    print(f"\n{domain_name}:")
    print(f"  Individual run means: {', '.join(f'{m:.4f}' for m in run_means)}")
    print(f"  Overall mean:         {stats['mean']:.4f}")
    print(f"  SEM:                  {stats['sem']:.4f}")
    print(
        f"  Bootstrap 95% CI:     [{stats['bootstrap_ci_lower']:.4f}, {stats['bootstrap_ci_upper']:.4f}]"
    )
    print(f"  Number of runs:       {stats['n']}")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Run PAG evaluations across all domains with statistics"
    )
    parser.add_argument(
        "--num_runs",
        type=int,
        default=5,
        help="Number of evaluation runs per domain",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="Model to use for generation",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of similar examples to retrieve",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=50,
        help="Number of parallel workers",
    )

    args = parser.parse_args()

    print("Starting PAG evaluations across all domains...")
    print(f"{'=' * 80}")
    print("Configuration:")
    print(f"  Number of runs: {args.num_runs}")
    print(f"  Model: {args.model}")
    print(f"  K: {args.k}")
    print(f"  Max workers: {args.max_workers}")
    print(f"{'=' * 80}")

    # Run evaluations for each domain
    review_level_scores = run_domain_evaluations(
        domain_name="Review Level",
        script_name="pag_review_level",
        train_file="data/reviews/train_level.parquet",
        val_file="data/reviews/validation_level.parquet",
        output_prefix="baselines/pag/results/pag_review_level",
        corpus_file="baselines/pag/data/review_level_corpus.pkl",
        num_runs=args.num_runs,
        model=args.model,
        k=args.k,
        max_workers=args.max_workers,
    )

    review_length_scores = run_domain_evaluations(
        domain_name="Review Length",
        script_name="pag_review_length",
        train_file="data/reviews/train_length.parquet",
        val_file="data/reviews/validation_length.parquet",
        output_prefix="baselines/pag/results/pag_review_length",
        corpus_file="baselines/pag/data/review_length_corpus_strong.pkl",
        num_runs=args.num_runs,
        model=args.model,
        k=args.k,
        max_workers=args.max_workers,
    )

    math_solutions_scores = run_domain_evaluations(
        domain_name="Math Solutions",
        script_name="pag_math_solutions",
        train_file="data/math_solutions/train.parquet",
        val_file="data/math_solutions/validation.parquet",
        output_prefix="baselines/pag/results/pag_math_solutions",
        corpus_file="baselines/pag/data/math_solutions_corpus.pkl",
        num_runs=args.num_runs,
        model=args.model,
        k=args.k,
        max_workers=args.max_workers,
    )

    # Print final results
    print(f"\n{'=' * 80}")
    print("FINAL RESULTS")
    print(f"{'=' * 80}")

    review_level_stats = print_statistics("Review Level", review_level_scores)
    review_length_stats = print_statistics("Review Length", review_length_scores)
    math_solutions_stats = print_statistics("Math Solutions", math_solutions_scores)

    # Print summary table
    print(f"\n{'=' * 80}")
    print("Summary Table:")
    print(f"{'=' * 80}")
    print(f"{'Domain':<20} {'Mean':<12} {'SEM':<12} {'Bootstrap 95% CI':<30}")
    print(f"{'-' * 20} {'-' * 12} {'-' * 12} {'-' * 30}")

    for domain, stats in [
        ("Review Level", review_level_stats),
        ("Review Length", review_length_stats),
        ("Math Solutions", math_solutions_stats),
    ]:
        ci_str = (
            f"[{stats['bootstrap_ci_lower']:.4f}, {stats['bootstrap_ci_upper']:.4f}]"
        )
        print(f"{domain:<20} {stats['mean']:<12.4f} {stats['sem']:<12.4f} {ci_str:<30}")

    print(f"{'=' * 80}")
    print("\nAll evaluations complete!")
    print(
        f"Results saved to baselines/pag/results/pag_*_run{{1..{args.num_runs}}}.json"
    )


if __name__ == "__main__":
    main()
