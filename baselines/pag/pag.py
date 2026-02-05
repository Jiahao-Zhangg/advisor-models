from typing import List, Dict, Any, Tuple, Callable, Optional
from rank_bm25 import BM25Okapi
import tiktoken
import litellm
import pickle
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from baselines.pag.prompts import (
    PAG_SUMMARY_PROMPT,
    PAG_GENERATION_PROMPT,
    SAMPLE_TEMPLATE,
    RESPONSE_AND_RATING_TEMPLATE,
)


class Corpus:
    def __init__(
        self,
        tasks: List[Dict[str, Any]],
        scorer: Callable[[str, Any], float],
        model: str = "gpt-4o-mini",
        max_workers: int = 1,
        debug: bool = False,
        direction: str = "",
    ):
        # maps user to summary
        self.user_summaries: Dict[str, str] = dict()

        # maps user to retriever
        self.user_retrievers: Dict[str, Tuple[List[str], BM25Okapi]] = dict()

        # maps user to question map
        # question map maps question to (ground_truth, list of (response, score))
        self.user_corpus: Dict[str, Dict[str, Tuple[Any, List[Tuple[str, float]]]]] = (
            dict()
        )
        self.scorer = scorer
        self.model = model
        self.tokenizer = tiktoken.encoding_for_model(model)
        self.max_workers = max_workers
        self.debug = debug
        self.direction = direction

        # collect all tasks
        for task in tasks:
            user = task["person"] if "person" in task else task["student"]
            if user not in self.user_summaries:
                self.user_summaries[user] = None
                self.user_corpus[user] = {
                    task["original_question"]: (task["reward_spec"]["ground_truth"], [])
                }
            else:
                self.user_corpus[user][task["original_question"]] = (
                    task["reward_spec"]["ground_truth"],
                    [],
                )

        self.init_retrievers()

    def init_retrievers(self):
        # tokenize questions and init retriever
        for user in self.user_corpus:
            questions = list(self.user_corpus[user].keys())
            tokenized_questions = [q.split(" ") for q in questions]
            retriever = BM25Okapi(tokenized_questions)
            self.user_retrievers[user] = (questions, retriever)

    def build_corpus(self, repeats: int = 1):
        # generate multiple responses and ratings for each question
        if self.max_workers == 1 or self.debug:
            # Sequential execution
            total_tasks = sum(
                len(self.user_corpus[user]) * repeats for user in self.user_corpus
            )
            with tqdm(total=total_tasks, desc="Building corpus") as pbar:
                for user in self.user_corpus:
                    for question in self.user_corpus[user]:
                        ground_truth = self.user_corpus[user][question][0]
                        responses = self.user_corpus[user][question][1]
                        for _ in range(repeats):
                            responses.append(
                                self.generate_and_rate_response(question, ground_truth)
                            )
                            pbar.update(1)
        else:
            # Parallel execution
            all_tasks = []
            for user in self.user_corpus:
                for question in self.user_corpus[user]:
                    ground_truth = self.user_corpus[user][question][0]
                    for _ in range(repeats):
                        all_tasks.append((user, question, ground_truth))

            def generate_single(task):
                user, question, ground_truth = task
                return (
                    user,
                    question,
                    self.generate_and_rate_response(question, ground_truth),
                )

            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_task = {
                    executor.submit(generate_single, task): task for task in all_tasks
                }

                with tqdm(total=len(all_tasks), desc="Building corpus") as pbar:
                    for future in as_completed(future_to_task):
                        try:
                            user, question, response_score = future.result()
                            self.user_corpus[user][question][1].append(response_score)
                        except Exception as e:
                            print(f"\nError generating response: {e}")
                            import traceback

                            traceback.print_exc()
                        pbar.update(1)

    def generate_and_rate_response(
        self, question: str, ground_truth
    ) -> Tuple[str, float]:
        # use litellm and designated scorer to generate response and rate it
        response = (
            litellm.completion(
                model=self.model, messages=[{"role": "user", "content": question}]
            )
            .choices[0]
            .message.content
        )
        score = self.scorer(response, ground_truth)
        return response, score

    def generate_summaries(
        self, max_tokens: int = 128000, max_samples: Optional[int] = None
    ):
        # generate summary for each user
        for user in tqdm(
            self.user_summaries, desc="Generating summaries", disable=self.debug
        ):
            formatted_samples = []
            token_count = 0

            for idx, (question, (_, responses)) in enumerate(
                self.user_corpus[user].items(), 1
            ):
                # format one sample
                formatted_responses = []

                for resp_idx, (response, rating) in enumerate(responses, 1):
                    # format one response and rating
                    formatted_response = RESPONSE_AND_RATING_TEMPLATE.format(
                        idx=resp_idx, response=response, user=user, rating=rating
                    )
                    formatted_responses.append(formatted_response)

                sample = SAMPLE_TEMPLATE.format(
                    idx=idx,
                    question=question,
                    formatted_responses_and_ratings="\n\n".join(formatted_responses),
                )

                # check token count
                sample_tokens = len(self.tokenizer.encode(sample))
                if token_count + sample_tokens > max_tokens:
                    break

                formatted_samples.append(sample)
                token_count += sample_tokens

                # check max samples
                if max_samples is not None and len(formatted_samples) >= max_samples:
                    break

            # build summary prompt
            prompt = PAG_SUMMARY_PROMPT.format(
                user=user,
                direction=self.direction,
                formatted_samples="\n\n".join(formatted_samples),
            )

            if self.debug:
                print(f"\n{'=' * 80}")
                print(f"SUMMARY PROMPT FOR {user}")
                print(f"{'=' * 80}")
                print(prompt)
                print(f"{'=' * 80}\n")

            # generate summary
            summary = (
                litellm.completion(
                    model=self.model, messages=[{"role": "user", "content": prompt}]
                )
                .choices[0]
                .message.content
            )

            self.user_summaries[user] = summary

    def get_pag_prompt(self, user: str, question: str, k: int = 5) -> str:
        # summary
        summary = self.user_summaries[user]

        # retrieve questions
        questions, retriever = self.user_retrievers[user]
        tokenized_question = question.split(" ")
        retrieved_questions = retriever.get_top_n(tokenized_question, questions, k)

        # format samples
        formatted_samples = []
        for idx, retrieved_q in enumerate(retrieved_questions, 1):
            # format one retrieved sample
            _, responses = self.user_corpus[user][retrieved_q]

            formatted_responses = []
            for resp_idx, (response, rating) in enumerate(responses, 1):
                # format one response and rating
                formatted_response = RESPONSE_AND_RATING_TEMPLATE.format(
                    idx=resp_idx, response=response, user=user, rating=rating
                )
                formatted_responses.append(formatted_response)

            sample = SAMPLE_TEMPLATE.format(
                idx=idx,
                question=retrieved_q,
                formatted_responses_and_ratings="\n\n".join(formatted_responses),
            )
            formatted_samples.append(sample)

        # build generation prompt
        prompt = PAG_GENERATION_PROMPT.format(
            user=user,
            question=question,
            user_preferences=summary,
            formatted_samples="\n\n".join(formatted_samples),
        )

        return prompt

    def save_state(self, filepath: str):
        """Save the corpus state to a file.

        Args:
            filepath: Path to save the corpus state
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        state = {
            "user_summaries": self.user_summaries,
            "user_corpus": self.user_corpus,
            "user_retrievers": self.user_retrievers,
            "model": self.model,
        }

        with open(filepath, "wb") as f:
            pickle.dump(state, f)

        print(f"Corpus state saved to {filepath}")

    @classmethod
    def load_state(
        cls,
        filepath: str,
        scorer: Callable[[str, Any], float],
        max_workers: int = 1,
        debug: bool = False,
        direction: str = "",
    ) -> "Corpus":
        """Load corpus state from a file.

        Args:
            filepath: Path to load the corpus state from
            tasks: Tasks list (needed for initialization)
            scorer: Scorer function (needed for initialization)
            max_workers: Number of parallel workers
            debug: Debug mode flag

        Returns:
            Corpus instance with loaded state
        """
        with open(filepath, "rb") as f:
            state = pickle.load(f)

        # Create a new corpus instance
        corpus = cls.__new__(cls)

        # Restore state
        corpus.user_summaries = state["user_summaries"]
        corpus.user_corpus = state["user_corpus"]
        corpus.user_retrievers = state["user_retrievers"]
        corpus.model = state["model"]
        corpus.scorer = scorer
        corpus.tokenizer = tiktoken.encoding_for_model(corpus.model)
        corpus.max_workers = max_workers
        corpus.debug = debug
        corpus.direction = direction

        print(f"Corpus state loaded from {filepath}")
        return corpus
