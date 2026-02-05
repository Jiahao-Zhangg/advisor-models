"""PAG baseline evaluation for reviews length domain.

Evaluates the PAG (Personalized Agentic Generation) approach on review writing with length preferences.
Uses BM25 retrieval to find similar examples and generates personalized responses.

Debug Mode (--debug):
    When enabled, the following changes are applied:
    - Train set reduced to 3 tasks
    - Validation set reduced to 1 task
    - k (retrieval) set to 1
    - repeats set to 2
    - max_workers set to 1 (sequential execution)
    - Prints all prompts:
        * Sample question during corpus building
        * Summary prompts for each user
        * PAG generation prompts for validation
        * Generated responses and scores
    - Prints complete corpus state (all questions, responses, scores)
    - Prints user preference summaries

Example usage:
    python -m baselines.pag.pag_review_length \
        --train_file data/reviews/train_length.parquet \
        --val_file data/reviews/validation_length.parquet \
        --output_file baselines/pag/results/pag_review_length_direct.json \
        --model gpt-4o-mini \
        --repeats 10 \
        --k 5 \
        --max_workers 50 \
        --save_corpus baselines/pag/data/review_length_corpus_direct.pkl
    
    # Debug mode
    python -m baselines.pag.pag_review_length \
        --train_file data/reviews/train_length.parquet \
        --val_file data/reviews/validation_length.parquet \
        --output_file baselines/pag/results/pag_review_length_debug.json \
        --debug \
        --load_corpus baselines/pag/data/review_length_corpus_debug.pkl
"""

import argparse
import json
import os
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import datasets
import litellm

from baselines.pag.pag import Corpus
from baselines.pag.prompts import PAG_REVIEW_LENGTH_DIRECTION
from advisor_models.reviews.config import compute_length_reward


def scorer(response: str, ground_truth: int) -> float:
    """Score a review response based on length preference."""
    return compute_length_reward(response, ground_truth)


def generate_response_with_pag(
    corpus: Corpus,
    task: Dict[str, Any],
    model: str,
    k: int = 5,
    debug: bool = False,
) -> tuple[str, float, str]:
    """Generate a response using PAG prompt and score it.

    Args:
        corpus: Initialized Corpus with summaries
        task: Task dictionary with person, original_question, reward_spec
        model: Model to use for generation
        k: Number of similar examples to retrieve
        debug: Whether to print debug information

    Returns:
        Tuple of (response, score, pag_prompt)
    """
    user = task["person"]
    question = task["original_question"]
    ground_truth = task["reward_spec"]["ground_truth"]

    # Get PAG prompt
    pag_prompt = corpus.get_pag_prompt(user, question, k=k)

    if debug:
        print(f"\n{'=' * 80}")
        print(f"PAG PROMPT FOR {user} - {question[:50]}...")
        print(f"{'=' * 80}")
        print(pag_prompt)
        print(f"{'=' * 80}\n")

    # Generate response
    response = (
        litellm.completion(
            model=model, messages=[{"role": "user", "content": pag_prompt}]
        )
        .choices[0]
        .message.content
    )

    # Score response
    score = scorer(response, ground_truth)

    if debug:
        print("GENERATED RESPONSE:")
        print(response)
        print(f"\nSCORE: {score:.4f}")
        print(f"{'=' * 80}\n")

    return response, score, pag_prompt


def evaluate_validation_set(
    corpus: Corpus,
    val_tasks: List[Dict[str, Any]],
    model: str,
    k: int = 5,
    max_workers: int = 1,
    debug: bool = False,
) -> List[Dict[str, Any]]:
    """Evaluate validation set with PAG approach.

    Args:
        corpus: Initialized Corpus with summaries
        val_tasks: List of validation tasks
        model: Model to use for generation
        k: Number of similar examples to retrieve
        max_workers: Number of parallel workers
        debug: Whether to print debug information

    Returns:
        List of results with responses and scores
    """
    results = []

    if max_workers == 1 or debug:
        # Sequential execution
        for task in tqdm(val_tasks, desc="Evaluating validation set", disable=debug):
            response, score, pag_prompt = generate_response_with_pag(
                corpus, task, model, k=k, debug=debug
            )
            results.append(
                {
                    "person": task["person"],
                    "original_question": task["original_question"],
                    "ground_truth": task["reward_spec"]["ground_truth"],
                    "response": response,
                    "score": score,
                    "pag_prompt": pag_prompt,
                }
            )
    else:
        # Parallel execution
        results_list = [None] * len(val_tasks)

        def process_task(idx_task):
            idx, task = idx_task
            response, score, pag_prompt = generate_response_with_pag(
                corpus, task, model, k=k, debug=False
            )
            return idx, {
                "person": task["person"],
                "original_question": task["original_question"],
                "ground_truth": task["reward_spec"]["ground_truth"],
                "response": response,
                "score": score,
                "pag_prompt": pag_prompt,
            }

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(process_task, (idx, task)): idx
                for idx, task in enumerate(val_tasks)
            }

            with tqdm(total=len(val_tasks), desc="Evaluating validation set") as pbar:
                for future in as_completed(future_to_idx):
                    try:
                        idx, result = future.result()
                        results_list[idx] = result
                    except Exception as e:
                        print(f"\nError evaluating task: {e}")
                        import traceback

                        traceback.print_exc()
                    pbar.update(1)

        results = [r for r in results_list if r is not None]

    return results


def print_corpus_state(corpus: Corpus):
    """Print the state of the corpus for debugging."""
    print(f"\n{'=' * 80}")
    print("CORPUS STATE")
    print(f"{'=' * 80}")

    for user in corpus.user_corpus:
        print(f"\nUser: {user}")
        print(f"Summary: {corpus.user_summaries.get(user, 'None')}")
        print("Questions and responses:")

        for question, (ground_truth, responses) in corpus.user_corpus[user].items():
            print(f"  Question: {question[:80]}...")
            print(f"  Ground truth: {ground_truth}")
            print(f"  Responses: {len(responses)}")
            for idx, (response, score) in enumerate(responses, 1):
                print(f"    Response {idx} (score={score:.4f}): {response[:100]}...")
        print()

    print(f"{'=' * 80}\n")


def main():
    parser = argparse.ArgumentParser(description="PAG baseline for reviews length")
    parser.add_argument(
        "--train_file",
        type=str,
        required=True,
        help="Path to training parquet file",
    )
    parser.add_argument(
        "--val_file",
        type=str,
        required=True,
        help="Path to validation parquet file",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to save results JSON file",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="Model to use for generation",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Number of responses to generate per training question",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of similar examples to retrieve for PAG",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=1,
        help="Number of parallel workers",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode (reduces dataset sizes and prints prompts)",
    )
    parser.add_argument(
        "--save_corpus",
        type=str,
        default=None,
        help="Path to save the corpus state after building (optional)",
    )
    parser.add_argument(
        "--load_corpus",
        type=str,
        default=None,
        help="Path to load the corpus state from (skips corpus building if provided)",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to use when generating summaries (optional)",
    )
    parser.add_argument(
        "--regenerate_summaries",
        action="store_true",
        help="Regenerate summaries even when loading corpus from file",
    )

    args = parser.parse_args()

    # Load datasets
    print(f"Loading training data from {args.train_file}...")
    train_dataset = datasets.load_dataset("parquet", data_files=args.train_file)[
        "train"
    ]
    train_tasks = list(train_dataset)

    print(f"Loading validation data from {args.val_file}...")
    val_dataset = datasets.load_dataset("parquet", data_files=args.val_file)["train"]
    val_tasks = list(val_dataset)

    # Debug mode adjustments
    if args.debug:
        print("\n*** DEBUG MODE ENABLED ***")
        print("Reducing dataset sizes and enabling verbose output\n")
        train_tasks = train_tasks[:3]
        val_tasks = val_tasks[:1]
        k = 1
        max_workers = 1
        repeats = 2
    else:
        k = args.k
        max_workers = args.max_workers
        repeats = args.repeats

    print(f"Training tasks: {len(train_tasks)}")
    print(f"Validation tasks: {len(val_tasks)}")
    print(f"Model: {args.model}")
    print(f"Repeats: {repeats}")
    print(f"K: {k}")
    print(f"Max workers: {max_workers}\n")

    # Initialize or load corpus
    if args.load_corpus:
        print(f"Loading corpus state from {args.load_corpus}...")
        corpus = Corpus.load_state(
            args.load_corpus,
            scorer,
            max_workers=max_workers,
            debug=args.debug,
            direction=PAG_REVIEW_LENGTH_DIRECTION,
        )

        # Regenerate summaries if requested
        if args.regenerate_summaries:
            print("Regenerating user preference summaries...")
            corpus.generate_summaries(max_samples=args.max_samples)

            # Save corpus if requested
            if args.save_corpus:
                corpus.save_state(args.save_corpus)
    else:
        # Initialize corpus
        print("Initializing corpus...")
        corpus = Corpus(
            train_tasks,
            scorer,
            model=args.model,
            max_workers=max_workers,
            debug=args.debug,
            direction=PAG_REVIEW_LENGTH_DIRECTION,
        )

        # Build corpus (generate responses for training questions)
        print(f"Building corpus with {repeats} repeats per question...")
        if args.debug:
            print("\nBUILDING CORPUS - Sample prompts:")
            print(f"{'=' * 80}")
            # Show first prompt
            if train_tasks:
                sample_task = train_tasks[0]
                print(f"Sample question: {sample_task['original_question']}")
                print(
                    f"Ground truth length: {sample_task['reward_spec']['ground_truth']}"
                )
                print(f"{'=' * 80}\n")

        corpus.build_corpus(repeats=repeats)

        if args.debug:
            print_corpus_state(corpus)

        # Generate summaries
        print("Generating user preference summaries...")
        corpus.generate_summaries(max_samples=args.max_samples)

        # Save corpus if requested
        if args.save_corpus:
            corpus.save_state(args.save_corpus)

    if args.debug:
        print("\nUSER SUMMARIES:")
        print(f"{'=' * 80}")
        for user, summary in corpus.user_summaries.items():
            print(f"\n{user}:")
            print(summary)
        print(f"\n{'=' * 80}\n")

    # Evaluate validation set
    print(f"Evaluating validation set with k={k}...")
    results = evaluate_validation_set(
        corpus, val_tasks, args.model, k=k, max_workers=max_workers, debug=args.debug
    )

    # Compute statistics
    scores = [r["score"] for r in results]
    avg_score = sum(scores) / len(scores) if scores else 0.0

    print(f"\n{'=' * 80}")
    print("RESULTS")
    print(f"{'=' * 80}")
    print(f"Total validation tasks: {len(results)}")
    print(f"Average score: {avg_score:.4f}")
    print(f"Min score: {min(scores):.4f}")
    print(f"Max score: {max(scores):.4f}")

    # Per-person statistics
    person_scores = {}
    for r in results:
        person = r["person"]
        if person not in person_scores:
            person_scores[person] = []
        person_scores[person].append(r["score"])

    print("\nPer-person statistics:")
    for person, person_score_list in sorted(person_scores.items()):
        avg = sum(person_score_list) / len(person_score_list)
        print(f"  {person}: {avg:.4f} (n={len(person_score_list)})")
    print(f"{'=' * 80}\n")

    # Save results
    output_data = {
        "config": {
            "train_file": args.train_file,
            "val_file": args.val_file,
            "model": args.model,
            "repeats": repeats,
            "k": k,
            "max_workers": max_workers,
            "debug": args.debug,
        },
        "summary": {
            "total_tasks": len(results),
            "average_score": avg_score,
            "min_score": min(scores),
            "max_score": max(scores),
            "per_person_scores": {
                person: sum(scores) / len(scores)
                for person, scores in person_scores.items()
            },
        },
        "results": results,
    }

    print(f"Saving results to {args.output_file}...")
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w") as f:
        json.dump(output_data, f, indent=2)

    print("Done!")


if __name__ == "__main__":
    main()
