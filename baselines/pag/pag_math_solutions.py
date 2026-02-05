"""PAG (Personalized Agentic Generation) baseline for math solutions domain.

This script evaluates the PAG approach on the math solutions task, where different
students have preferences for different solution styles (multiple methods vs single,
with questions vs without).

The PAG approach:
1. Builds a corpus of student-specific questions and responses with ratings
2. Generates preference summaries for each student based on their corpus
3. Uses BM25 retrieval to find similar questions from the student's corpus
4. Generates responses using the student's summary and retrieved examples

Debug mode effects (--debug flag):
    - Reduces training dataset to 3 samples
    - Reduces validation dataset to 1 sample
    - Sets k=1 (retrieves only 1 similar example)
    - Sets max_workers=1 (sequential processing)
    - Sets repeats=2 (generates 2 responses per training question)
    - Prints verbose output:
        * Sample question during corpus building
        * Summary prompts for each student
        * PAG generation prompts for validation
        * Generated responses and scores
    - Prints complete corpus state (all questions, responses, scores)
    - Prints student preference summaries

Example usage:
    python -m baselines.pag.pag_math_solutions \
        --train_file data/math_solutions/train.parquet \
        --val_file data/math_solutions/validation.parquet \
        --output_file baselines/pag/results/pag_math_solutions.json \
        --model gpt-4o-mini \
        --repeats 10 \
        --k 5 \
        --max_workers 50 \
        --save_corpus baselines/pag/data/math_solutions_corpus.pkl
    
    # Debug mode
    python -m baselines.pag.pag_math_solutions \
        --train_file data/math_solutions/train.parquet \
        --val_file data/math_solutions/validation.parquet \
        --output_file baselines/pag/results/pag_math_solutions_debug.json \
        --debug \
        --save_corpus baselines/pag/data/math_solutions_corpus_debug.pkl
"""

import argparse
import json
import os
from typing import Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import datasets
import litellm

from baselines.pag.pag import Corpus
from baselines.pag.prompts import PAG_MATH_SOLUTIONS_DIRECTION
from advisor_models.math_solutions.config import (
    STYLE_JUDGE_SYSTEM_PROMPT,
    STYLE_JUDGE_PROMPT,
)


def compute_style_reward(
    solution: str, judge_criteria: str, model: str = "gpt-4o-mini"
) -> float:
    """Compute reward based on whether the solution matches the style criteria.

    Uses an LLM to evaluate if the solution aligns with the judge criteria.
    Returns 1.0 for ACCEPT, 0.4 for PARTIAL, 0.0 for REJECT.
    """
    try:
        judge_prompt = STYLE_JUDGE_PROMPT.format(
            judge_criteria=judge_criteria,
            solution=solution,
        )

        response = litellm.completion(
            model=model,
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


def evaluate_single_task(args_tuple) -> Dict[str, Any]:
    """Evaluate a single validation task."""
    task, corpus, k, model, debug = args_tuple

    student = task["student"]
    question = task["original_question"]
    judge_criteria = task["reward_spec"]["ground_truth"]
    correct_answer = task.get("math_correct_answer", "N/A")

    # Get PAG prompt
    pag_prompt = corpus.get_pag_prompt(student, question, k=k)

    if debug:
        print(f"\n{'=' * 80}")
        print(f"PAG PROMPT FOR {student}")
        print(f"{'=' * 80}")
        print(pag_prompt)
        print(f"{'=' * 80}\n")

    # Generate response
    try:
        response = (
            litellm.completion(
                model=model, messages=[{"role": "user", "content": pag_prompt}]
            )
            .choices[0]
            .message.content
        )

        # Score the response
        score = compute_style_reward(response, judge_criteria, model)

        if debug:
            print(f"Generated solution for {student}:")
            print(f"Score: {score:.4f}")
            print(f"Solution: {response[:200]}...")
            print()

        return {
            "student": student,
            "question": question,
            "response": response,
            "score": score,
            "judge_criteria": judge_criteria,
            "correct_answer": correct_answer,
            "pag_prompt": pag_prompt,
        }
    except Exception as e:
        print(f"Error evaluating task for {student}: {e}")
        import traceback

        traceback.print_exc()
        return {
            "student": student,
            "question": question,
            "response": "",
            "score": 0.0,
            "judge_criteria": judge_criteria,
            "correct_answer": correct_answer,
            "pag_prompt": pag_prompt,
            "error": str(e),
        }


def print_corpus_state(corpus: Corpus):
    """Print detailed corpus state for debugging."""
    print("\nCORPUS STATE:")
    print(f"{'=' * 80}")

    for user in corpus.user_corpus:
        print(f"\n{user}:")
        print(f"  Number of questions: {len(corpus.user_corpus[user])}")

        for question, (ground_truth, responses) in corpus.user_corpus[user].items():
            print(f"\n  Question: {question[:100]}...")
            print(f"  Judge criteria: {ground_truth[:100]}...")
            print(f"  Responses ({len(responses)}):")
            for idx, (response, score) in enumerate(responses, 1):
                print(f"    Response {idx} (score={score:.4f}): {response[:100]}...")
        print()

    print(f"{'=' * 80}\n")


def main():
    parser = argparse.ArgumentParser(description="PAG baseline for math solutions")
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

    # Define scorer function that captures judge_criteria from task
    def scorer(response: str, judge_criteria: str) -> float:
        return compute_style_reward(response, judge_criteria, args.model)

    # Initialize or load corpus
    if args.load_corpus:
        print(f"Loading corpus state from {args.load_corpus}...")
        corpus = Corpus.load_state(
            args.load_corpus,
            scorer,
            max_workers=max_workers,
            debug=args.debug,
            direction=PAG_MATH_SOLUTIONS_DIRECTION,
        )

        # Regenerate summaries if requested
        if args.regenerate_summaries:
            print("Regenerating student preference summaries...")
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
            direction=PAG_MATH_SOLUTIONS_DIRECTION,
        )

        # Build corpus (generate responses for training questions)
        print(f"Building corpus with {repeats} repeats per question...")
        if args.debug:
            print("\nBUILDING CORPUS - Sample prompts:")
            print(f"{'=' * 80}")
            # Show first prompt
            if train_tasks:
                sample_task = train_tasks[0]
                print(f"Sample question: {sample_task['original_question'][:100]}...")
                print(f"Student: {sample_task['student']}")
                print(
                    f"Judge criteria: {sample_task['reward_spec']['ground_truth'][:100]}..."
                )
                print(f"{'=' * 80}\n")

        corpus.build_corpus(repeats=repeats)

        if args.debug:
            print_corpus_state(corpus)

        # Generate summaries
        print("Generating student preference summaries...")
        corpus.generate_summaries(max_samples=args.max_samples)

        # Save corpus if requested
        if args.save_corpus:
            corpus.save_state(args.save_corpus)

    if args.debug:
        print("\nSTUDENT SUMMARIES:")
        print(f"{'=' * 80}")
        for user, summary in corpus.user_summaries.items():
            print(f"\n{user}:")
            print(summary)
        print(f"\n{'=' * 80}\n")

    # Evaluate validation set
    print(f"Evaluating validation set with k={k}...")

    if max_workers == 1 or args.debug:
        # Sequential evaluation
        results = []
        for task in tqdm(val_tasks, desc="Evaluating", disable=args.debug):
            result = evaluate_single_task((task, corpus, k, args.model, args.debug))
            results.append(result)
    else:
        # Parallel evaluation
        eval_args = [(task, corpus, k, args.model, False) for task in val_tasks]

        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_task = {
                executor.submit(evaluate_single_task, args): args[0]
                for args in eval_args
            }

            with tqdm(total=len(val_tasks), desc="Evaluating") as pbar:
                for future in as_completed(future_to_task):
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        print(f"\nError in evaluation: {e}")
                        import traceback

                        traceback.print_exc()
                    pbar.update(1)

    # Compute statistics
    scores = [r["score"] for r in results]
    student_scores = {}
    for result in results:
        student = result["student"]
        if student not in student_scores:
            student_scores[student] = []
        student_scores[student].append(result["score"])

    print(f"\n{'=' * 80}")
    print("RESULTS")
    print(f"{'=' * 80}")
    print(f"Total validation tasks: {len(val_tasks)}")
    print(f"Average score: {sum(scores) / len(scores):.4f}")
    print(f"Min score: {min(scores):.4f}")
    print(f"Max score: {max(scores):.4f}")
    print("\nPer-student statistics:")
    for student, scores_list in student_scores.items():
        print(
            f"  {student}: {sum(scores_list) / len(scores_list):.4f} (n={len(scores_list)})"
        )
    print(f"{'=' * 80}\n")

    # Save results
    output_data = {
        "config": {
            "model": args.model,
            "repeats": repeats,
            "k": k,
            "max_workers": max_workers,
            "train_file": args.train_file,
            "val_file": args.val_file,
        },
        "summary": {
            "num_val_tasks": len(val_tasks),
            "avg_score": sum(scores) / len(scores),
            "min_score": min(scores),
            "max_score": max(scores),
            "per_student_scores": {
                student: sum(scores_list) / len(scores_list)
                for student, scores_list in student_scores.items()
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
