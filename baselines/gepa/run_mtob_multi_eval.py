import argparse
import json
import os
from pathlib import Path

import dspy

from baselines.gepa.gepa_mtob import (
    TranslationModule,
    load_mtob_data,
    run_multi_evaluation,
)
from utils.eval_utils import compute_multi_run_statistics


def parse_args():
    parser = argparse.ArgumentParser(
        description="Load a saved GEPA-optimized MTOB model and run multi-evaluation"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="baselines/gepa/results/mtob_optimized_model.json",
        help="Path to saved optimized model JSON",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=5,
        help="Number of evaluation runs (default: 5)",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=72,
        help="Number of threads for parallel evaluation (default: 72)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="LLM temperature (default: 1.0)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="baselines/gepa/results",
        help="Directory to save evaluation results",
    )
    return parser.parse_args()


def load_optimized_model(model_path: str) -> TranslationModule:
    model = TranslationModule()
    if hasattr(model, "load"):
        model.load(model_path)
        return model

    if hasattr(dspy, "load"):
        loaded = dspy.load(model_path)
        return loaded

    raise RuntimeError(
        "Unable to load saved model: neither model.load(...) nor dspy.load(...) exists."
    )


def main():
    args = parse_args()

    llm = dspy.LM("openai/gpt-4o-mini", cache=False, temperature=args.temperature)
    dspy.settings.configure(lm=llm)

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Saved model not found at: {args.model_path}")

    print("Loading MTOB data...")
    _, valset = load_mtob_data()

    print(f"Loading optimized model from {args.model_path}...")
    model = load_optimized_model(args.model_path)

    print(f"\nRunning {args.num_runs} evaluation runs...")
    all_run_scores = run_multi_evaluation(
        model, valset, args.num_runs, args.num_threads
    )

    stats = compute_multi_run_statistics(all_run_scores)
    print("\n=== Final Evaluation Statistics ===")
    print(f"Mean: {stats['mean']:.4f}")
    print(f"SEM: {stats['sem']:.4f}")
    print(
        f"95% Bootstrap CI: [{stats['bootstrap_ci_lower']:.4f}, {stats['bootstrap_ci_upper']:.4f}]"
    )
    print(f"Number of runs: {args.num_runs}")

    os.makedirs(args.output_dir, exist_ok=True)
    results_file = Path(args.output_dir) / f"mtob_gepa_loaded_{args.num_runs}runs.json"
    with open(results_file, "w") as f:
        json.dump(
            {
                "domain": "mtob",
                "num_runs": args.num_runs,
                "num_samples": len(valset),
                "model_path": args.model_path,
                "statistics": stats,
            },
            f,
            indent=2,
        )
    print(f"Saved results to {results_file}")


if __name__ == "__main__":
    main()
