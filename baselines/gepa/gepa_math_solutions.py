"""
GEPA baseline for math solutions domain

python -m baselines.gepa.gepa_math_solutions \
        --minibatch-size 8 \
        --max-calls 32000 \
        --train-size 350 \
        --val-size 50 \
        --num-threads 20 \
        --log-dir baselines/gepa/logs/math_solutions \
        --num-runs 5 \
        --output-dir baselines/gepa/results \
        --wandb-name math_solutions_gepa_paper \
        --temperature 1.0
"""

import dspy
from dspy.evaluate import Evaluate
from dspy import GEPA
import pandas as pd
import random
import os
import statistics
import json
import argparse
from pathlib import Path
from openai import OpenAI
from advisor_models.math_solutions.config import (
    STYLE_JUDGE_SYSTEM_PROMPT,
    STYLE_JUDGE_PROMPT,
    BASELINE_INSTRUCTION,
)
from utils.eval_utils import compute_multi_run_statistics


random.seed(42)
llm = dspy.LM("openai/gpt-4o-mini", cache=False, temperature=1.0)
dspy.settings.configure(lm=llm)


def load_math_solutions_data():
    """Load math solutions data from parquet files."""
    train_path = "data/math_solutions/train.parquet"
    val_path = "data/math_solutions/validation.parquet"

    if not os.path.exists(train_path) or not os.path.exists(val_path):
        raise FileNotFoundError(
            f"Math solutions data files not found at {train_path} or {val_path}"
        )

    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)

    # Convert to DSPy format
    trainset = []
    for _, row in train_df.iterrows():
        student = row["student"]
        problem = row["original_question"]
        judge_criteria = row["reward_spec"]["ground_truth"]

        full_prompt = BASELINE_INSTRUCTION.format(problem=problem, student=student)

        example = dspy.Example(
            prompt=full_prompt,
            judge_criteria=judge_criteria,
            student=student,
        ).with_inputs("prompt", "judge_criteria")
        trainset.append(example)

    valset = []
    for _, row in val_df.iterrows():
        student = row["student"]
        problem = row["original_question"]
        judge_criteria = row["reward_spec"]["ground_truth"]

        full_prompt = BASELINE_INSTRUCTION.format(problem=problem, student=student)

        example = dspy.Example(
            prompt=full_prompt,
            judge_criteria=judge_criteria,
            student=student,
        ).with_inputs("prompt", "judge_criteria")
        valset.append(example)

    return trainset, valset


class MathSolver(dspy.Signature):
    """Solve a math problem step by step."""

    prompt = dspy.InputField(desc="The math problem to solve")
    solution = dspy.OutputField(
        desc="The step-by-step solution with final answer in \\boxed{} format"
    )


class MathSolverModule(dspy.Module):
    """Math solver module."""

    def __init__(self):
        super().__init__()
        self.generate = dspy.ChainOfThought(MathSolver)

    def forward(self, prompt, judge_criteria=None):
        return self.generate(prompt=prompt)


def compute_style_reward(solution: str, judge_criteria) -> float:
    """Compute style matching reward using LLM-as-a-judge."""
    try:
        judge_prompt = STYLE_JUDGE_PROMPT.format(
            judge_criteria=judge_criteria,
            solution=solution,
        )

        client = OpenAI()
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": STYLE_JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": judge_prompt},
            ],
            temperature=0.0,
        )

        judge_response = response.choices[0].message.content.strip()

        # Parse the three-way response: ACCEPT=1.0, PARTIAL=0.4, REJECT=0.0
        if "ACCEPT" in judge_response:
            return 1.0
        elif "PARTIAL" in judge_response:
            return 0.4
        else:  # REJECT or any other response
            return 0.0

    except Exception as e:
        print(f"Error computing style reward: {e}")
        return 0.0


def compute_score_metric(example, pred, trace=None):
    """Compute the reward score for a prediction."""
    solution_text = pred.solution
    judge_criteria = example.judge_criteria
    reward = compute_style_reward(solution_text, judge_criteria)
    return reward


def feedback_metric(example, pred, trace=None, *args, **kwargs):
    """Provide scalar feedback for GEPA optimization."""
    reward = compute_score_metric(example, pred, trace)
    feedback = f"Based on the style, {example.student} assigned the solution a reward of {reward}/1.0. Your goal is to achieve 1.0 reward."
    return dspy.Prediction(score=reward, feedback=feedback)


def evaluate_model(model, dataset, model_name, num_threads=72):
    """Evaluate a model on the given dataset."""
    print(f"\n=== Evaluating {model_name} ===")

    # Evaluate on subset for faster testing
    eval_dataset = random.sample(dataset, min(100, len(dataset)))

    evaluator = Evaluate(
        devset=eval_dataset,
        metric=compute_score_metric,
        num_threads=num_threads,
        display_progress=True,
    )

    eval_result = evaluator(model)
    score = eval_result.score
    results = [entry[2] for entry in eval_result.results]

    # Calculate standard error
    reward_se = (
        statistics.stdev(results) / (len(results) ** 0.5) if len(results) > 1 else 0
    )

    print(f"Average style match: {score:.4f}±{reward_se:.4f}")

    return results


def run_multi_evaluation(model, dataset, num_runs, num_threads=72):
    """Run multiple evaluations and compute statistics."""
    all_run_scores = []

    for run_idx in range(num_runs):
        print(f"\n=== Run {run_idx + 1}/{num_runs} ===")
        scores = evaluate_model(model, dataset, f"Run {run_idx + 1}", num_threads)
        all_run_scores.append(scores)

    return all_run_scores


def save_optimized_prompt(model, output_dir, domain_name):
    """Save the optimized prompt."""
    os.makedirs(output_dir, exist_ok=True)

    # Save model
    model_path = Path(output_dir) / f"{domain_name}_optimized_model.json"
    model.save(str(model_path))
    print(f"Saved optimized model to {model_path}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="GEPA baseline for math solutions domain"
    )

    parser.add_argument(
        "--minibatch-size",
        type=int,
        default=3,
        help="Reflection minibatch size (default: 3)",
    )

    parser.add_argument(
        "--max-calls",
        type=int,
        default=64000,
        help="Maximum number of metric calls (default: 64000)",
    )

    parser.add_argument(
        "--train-size",
        type=int,
        default=300,
        help="Number of training examples to use (default: 300)",
    )

    parser.add_argument(
        "--val-size",
        type=int,
        default=100,
        help="Number of validation examples to use (default: 100)",
    )

    parser.add_argument(
        "--num-threads",
        type=int,
        default=72,
        help="Number of threads for parallel execution (default: 72)",
    )

    parser.add_argument(
        "--log-dir",
        type=str,
        default="gepa_logs_math_solutions",
        help="Directory for GEPA logs",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=5,
        help="Number of evaluation runs for final evaluation",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="baselines/gepa/results",
        help="Directory to save results and optimized prompts",
    )

    parser.add_argument(
        "--wandb-name",
        type=str,
        default="math_solutions_gepa",
        help="W&B run name (default: math_solutions_gepa)",
    )

    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B logging")

    parser.add_argument(
        "--temperature", type=float, default=1.0, help="LLM temperature (default: 1.0)"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    print("Configuration:")
    print(f"  Minibatch size: {args.minibatch_size}")
    print(f"  Max metric calls: {args.max_calls}")
    print(f"  Train size: {args.train_size}")
    print(f"  Val size: {args.val_size}")
    print(f"  Num threads: {args.num_threads}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Log directory: {args.log_dir}")
    print(f"  W&B enabled: {not args.no_wandb}")
    if not args.no_wandb:
        print(f"  W&B run name: {args.wandb_name}")
    print()

    # Configure LLM
    llm = dspy.LM("openai/gpt-4o-mini", cache=False, temperature=args.temperature)
    dspy.settings.configure(lm=llm)

    # Load data
    print("Loading math solutions data...")
    trainset, valset = load_math_solutions_data()
    print(
        f"Loaded {len(trainset)} training examples, {len(valset)} validation examples"
    )

    # Initialize model
    model = MathSolverModule()
    print("Running GEPA optimization...")

    # Prepare datasets
    random.shuffle(trainset)
    train_subset = trainset[: args.train_size]
    val_subset = trainset[args.train_size : args.train_size + args.val_size]
    eval_subset = valset

    # Configure GEPA
    gepa_kwargs = {
        "metric": feedback_metric,
        "max_metric_calls": args.max_calls,
        "num_threads": args.num_threads,
        "track_stats": True,
        "reflection_minibatch_size": args.minibatch_size,
        "reflection_lm": dspy.LM(
            model="openai/gpt-4o-mini", temperature=args.temperature, max_tokens=4000
        ),
        "log_dir": args.log_dir,
    }

    # Add W&B configuration if enabled
    if not args.no_wandb:
        gepa_kwargs.update(
            {
                "use_wandb": True,
                "wandb_init_kwargs": {
                    "entity": "bare-sky",
                    "project": "advisor-models-baselines",
                    "name": args.wandb_name,
                },
                "wandb_api_key": os.getenv("WANDB_API_KEY"),
            }
        )

    gepa = GEPA(**gepa_kwargs)

    optimized_model = gepa.compile(model, trainset=train_subset, valset=val_subset)

    print("Optimized prompt:")
    for name, pred in optimized_model.named_predictors():
        print("================================")
        print(f"Predictor: {name}")
        print("================================")
        print("Prompt:")
        print(pred.signature.instructions)
        print("*********************************")

    # Save optimized prompts
    save_optimized_prompt(optimized_model, args.output_dir, "math_solutions")

    # Run multi-run evaluation
    print(f"\nRunning {args.num_runs} evaluation runs...")
    all_run_scores = run_multi_evaluation(
        optimized_model, eval_subset, args.num_runs, args.num_threads
    )

    # Compute and report statistics
    stats = compute_multi_run_statistics(all_run_scores)
    print("\n=== Final Evaluation Statistics ===")
    print(f"Mean: {stats['mean']:.4f}")
    print(f"SEM: {stats['sem']:.4f}")
    print(
        f"95% Bootstrap CI: [{stats['bootstrap_ci_lower']:.4f}, {stats['bootstrap_ci_upper']:.4f}]"
    )
    print(f"Number of runs: {args.num_runs}")

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    results_file = (
        Path(args.output_dir) / f"math_solutions_gepa_{args.num_runs}runs.json"
    )
    with open(results_file, "w") as f:
        json.dump(
            {
                "domain": "math_solutions",
                "num_runs": args.num_runs,
                "num_samples": len(eval_subset),
                "statistics": stats,
            },
            f,
            indent=2,
        )
    print(f"Saved results to {results_file}")


if __name__ == "__main__":
    main()
