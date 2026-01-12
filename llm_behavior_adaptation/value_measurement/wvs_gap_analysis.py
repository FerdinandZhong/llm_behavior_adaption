"""Gap analysis between BA dialogue results and human values with rationale generation."""

import argparse
import asyncio
import json
import logging
import os
import random
from copy import deepcopy
from functools import partial
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml
from openai import AsyncOpenAI
from pydantic import BaseModel
from tqdm.asyncio import tqdm

from llm_behavior_adaptation.dialogue_dataset_creation.generation_utils import load_json_folder
from llm_behavior_adaptation.utils import register_logger

logger = logging.getLogger(__name__)
register_logger(logger)


def setup_file_logging(output_file_path: str, logger: logging.Logger) -> None:
    """
    Add file handler to logger to save logs to a file alongside console output.

    Args:
        output_file_path: Path to the main output file (will create .log file with same name)
        logger: Logger instance to configure
    """
    # Create log file path by replacing extension
    log_file_path = output_file_path.replace(".jsonl", ".log")

    # Create file handler with detailed formatting
    file_handler = logging.FileHandler(log_file_path, mode="a", encoding="utf-8")
    file_formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s - %(message)s"
    )
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(logging.INFO)

    # Add handler to logger if not already present
    if not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
        logger.addHandler(file_handler)
        logger.info("=" * 80)
        logger.info("Gap analysis logging started - log file: %s", log_file_path)
        logger.info("=" * 80)


DATASET_DIR = "datasets/wvs_benchmarks"


def _load_yaml(path: Optional[str]) -> Dict[str, Any]:
    """Load YAML configuration file."""
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_jsonl_file(file_path: str) -> List[Dict]:
    """Load JSONL file and return list of JSON objects."""
    list_of_json_objs = []
    with open(file_path, "r", encoding="utf-8") as file:
        for json_obj in file.readlines():
            dialogue_obj = json.loads(json_obj)
            list_of_json_objs.append(dialogue_obj)
    return list_of_json_objs


class Response(BaseModel):
    """Response schema for LLM output."""

    option_id: int
    reason: str


class GapAnalysisController:
    """Controller for analyzing gaps between BA dialogue results and human values."""

    def __init__(
        self,
        evaluated_model: str,
        ba_dialogue_results_path: str,
        human_results_path: str,
        generated_dialogues_path: str,
        picked_questions_path: str,
        prompts_folder: str,
        output_file_path: str,
        openai_client: Optional[AsyncOpenAI] = None,
        verbose: int = 0,
        storage_step: Optional[int] = None,
        llm_server: str = "llm_platform",
        reasoning: bool = False,
        extra_body: Optional[Dict] = None,
        starting_row: Optional[int] = None,
        ending_row: Optional[int] = None,
        gap_threshold: float = 0.0,
    ) -> None:
        """
        Initialize the GapAnalysisController.

        Args:
            evaluated_model: The name or identifier of the model to use for rationale generation.
            ba_dialogue_results_path: Path to BA dialogue results JSONL file.
            human_results_path: Path to human results CSV file.
            generated_dialogues_path: Path to generated dialogues JSONL file.
            picked_questions_path: Path to picked questions JSON file.
            prompts_folder: Path to prompts folder.
            output_file_path: Path to output JSONL file for gap analysis results.
            openai_client: Optional OpenAI client instance.
            verbose: Verbosity level for logging.
            storage_step: Optional interval for flushing results to disk.
            llm_server: One of {"llm_platform", "gpt", "vllm", "sglang"}.
            reasoning: Whether the tested model is a reasoning model.
            extra_body: Extra parameters for API calls.
            starting_row: Optional starting row index for processing a slice (0-indexed).
            ending_row: Optional ending row index for processing a slice (exclusive, -1 for all).
            gap_threshold: Relative threshold (0.0-1.0) for querying model. 0.0 means query all gaps,
                          1.0 means only query maximum gaps. Calculated as |predicted - human| / scale_range.

        Raises:
            ValueError: If required arguments are invalid or missing.
            TypeError: If input arguments are of incorrect types.
        """
        if not evaluated_model:
            raise ValueError("evaluated_model must be provided.")
        if not isinstance(ba_dialogue_results_path, str) or not ba_dialogue_results_path:
            raise ValueError("ba_dialogue_results_path must be a non-empty string.")
        if not isinstance(human_results_path, str) or not human_results_path:
            raise ValueError("human_results_path must be a non-empty string.")
        if not isinstance(output_file_path, str) or not output_file_path:
            raise ValueError("output_file_path must be a non-empty string.")
        if not isinstance(verbose, int) or verbose < 0:
            raise ValueError("verbose must be a non-negative integer.")
        if not (0.0 <= gap_threshold <= 1.0):
            raise ValueError("gap_threshold must be between 0.0 and 1.0.")

        self.prompts = load_json_folder(prompts_folder)

        self._evaluated_model = evaluated_model
        self._ba_dialogue_results_path = ba_dialogue_results_path
        self._human_results_path = human_results_path
        self._generated_dialogues_path = generated_dialogues_path
        self._picked_questions_path = picked_questions_path
        self._output_file_path = output_file_path
        self._verbose = verbose
        self._storage_step = storage_step
        self._reasoning = reasoning
        self._starting_row = starting_row if starting_row is not None else 0
        self._ending_row = ending_row
        self._gap_threshold = gap_threshold

        # Setup file logging
        setup_file_logging(output_file_path, logger)

        # Load data
        self._load_data()

        # Setup OpenAI client
        if openai_client is None:
            if "gpt" in evaluated_model:
                self._openai_client = AsyncOpenAI(api_key=os.environ["api_key"])
            else:
                base_url = os.getenv("base_url", "http://localhost:8000/v1")
                self._openai_client = AsyncOpenAI(api_key=os.environ["api_key"], base_url=base_url)
        else:
            self._openai_client = openai_client

        # Setup query function
        if llm_server == "llm_platform":
            self.query_llm = partial(
                self.openai_client.chat.completions.create,
                model=self.evaluated_model,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "option_response",
                        "schema": Response.model_json_schema(),
                    },
                },
                extra_body=extra_body,
            )
        else:
            extra_body = extra_body or {}
            self.query_llm = partial(
                self.openai_client.chat.completions.create,
                model=self.evaluated_model,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "option_response",
                        "schema": Response.model_json_schema(),
                    },
                },
                **extra_body,
            )
        self.llm_server = llm_server

        # Setup retryable errors
        try:
            from openai import APIConnectionError, APIError, APIStatusError, APITimeoutError, RateLimitError

            self._OPENAI_ERRORS = (
                APIError,
                RateLimitError,
                APITimeoutError,
                APIConnectionError,
                APIStatusError,
            )
        except Exception:
            self._OPENAI_ERRORS = tuple()

    def _load_data(self):
        """Load all required data files and apply slicing if specified."""
        # Load BA dialogue results
        ba_results_list = load_jsonl_file(self._ba_dialogue_results_path)

        # Apply slicing to BA results
        if self._ending_row is not None and self._ending_row >= 0:
            ba_results_list = ba_results_list[self._starting_row : self._ending_row]
        else:
            ba_results_list = ba_results_list[self._starting_row :]

        self._ba_dialogue_results = self._process_model_outputs(ba_results_list)

        ending_display = self._ending_row if self._ending_row is not None and self._ending_row >= 0 else "end"
        logger.info(
            "Loaded BA dialogue results: rows %d to %s (%d users)",
            self._starting_row,
            ending_display,
            len(self._ba_dialogue_results),
        )

        # Load human results
        human_df = pd.read_csv(self._human_results_path)
        human_df = human_df.loc[:, ~human_df.columns.str.contains("^Unnamed")]
        self._human_results = (
            human_df.astype({"D_INTERVIEW": str}).groupby("D_INTERVIEW", as_index=True).last().to_dict(orient="index")
        )

        # Load generated dialogues
        with open(self._generated_dialogues_path, "r", encoding="utf-8") as f:
            all_dialogues = f.readlines()

            # Apply slicing to dialogues
            if self._ending_row is not None and self._ending_row >= 0:
                all_dialogues = all_dialogues[self._starting_row : self._ending_row]
            else:
                all_dialogues = all_dialogues[self._starting_row :]

            dialogue_list = [json.loads(dialogue) for dialogue in all_dialogues]
            self._generated_dialogues = {k: v for d in dialogue_list for k, v in d.items()}

        logger.info("Loaded %d generated dialogues", len(self._generated_dialogues))

        # Check for missing dialogues
        ba_user_ids = set(self._ba_dialogue_results.keys())
        dialogue_user_ids = set(self._generated_dialogues.keys())
        missing_dialogues = ba_user_ids - dialogue_user_ids

        if missing_dialogues:
            logger.warning(
                "Found %d user(s) in BA results but not in generated dialogues: %s",
                len(missing_dialogues),
                sorted(list(missing_dialogues)),
            )
            logger.warning(
                "These users will be skipped during gap analysis. "
                "This may indicate missing data in the dialogue generation step."
            )

        # Load picked questions
        with open(self._picked_questions_path, "r", encoding="utf-8") as f:
            self._picked_questions = json.load(f)

        # Create flat questions dict
        self._all_questions = {}
        for _, q_list in self._picked_questions.items():
            self._all_questions.update(q_list)

    @staticmethod
    def _process_model_outputs(original_answers_list: List[Dict]) -> Dict[str, Dict]:
        """Process model outputs from JSONL format to flat dict."""
        processed_answers_dict = {}
        for answer_details in original_answers_list:
            for user_id, answers in answer_details.items():
                per_user_answers = {}
                for cat_answers in list(answers.values()):
                    for answer in cat_answers:
                        per_user_answers.update(answer)
                if user_id in processed_answers_dict:
                    logger.warning("Duplicate user id: %s", user_id)
                processed_answers_dict[user_id] = per_user_answers
        return processed_answers_dict

    # Properties
    @property
    def evaluated_model(self) -> str:
        """The evaluated model identifier."""
        return self._evaluated_model

    @property
    def ba_dialogue_results(self) -> Dict[str, Dict]:
        """BA dialogue results."""
        return self._ba_dialogue_results

    @property
    def human_results(self) -> Dict[str, Dict]:
        """Human results."""
        return self._human_results

    @property
    def generated_dialogues(self) -> Dict[str, List]:
        """Generated dialogues."""
        return self._generated_dialogues

    @property
    def all_questions(self) -> Dict:
        """All questions metadata."""
        return self._all_questions

    @property
    def openai_client(self) -> AsyncOpenAI:
        """OpenAI client."""
        return self._openai_client

    @property
    def reasoning(self) -> bool:
        """Whether the tested model is a reasoning model."""
        return self._reasoning

    @property
    def verbose(self) -> int:
        """Verbosity level."""
        return self._verbose

    @property
    def storage_step(self) -> Optional[int]:
        """Interval for periodic storage."""
        return self._storage_step

    @property
    def output_file_path(self) -> str:
        """Output file path."""
        return self._output_file_path

    @property
    def gap_threshold(self) -> float:
        """Gap threshold for querying model."""
        return self._gap_threshold

    # Retry helper
    async def _retry_llm(self, call_factory, *, max_attempts: int = 3, base_delay: float = 1.0):
        """
        Retry wrapper for LLM calls with exponential backoff and jitter.

        Args:
            call_factory: zero-arg callable returning an awaitable.
            max_attempts: maximum attempts (including the first).
            base_delay: base sleep in seconds.

        Returns:
            Result of awaited call_factory()

        Raises:
            The last exception if all attempts fail.
        """
        last_err = None
        for attempt in range(1, max_attempts + 1):
            try:
                return await call_factory()
            except (
                *self._OPENAI_ERRORS,
                json.decoder.JSONDecodeError,
                TimeoutError,
            ) as e:
                last_err = e
                if attempt >= max_attempts:
                    raise
                delay = base_delay * (2 ** (attempt - 1)) + random.uniform(0, 0.25)
                if self._verbose:
                    logger.warning(
                        "Retryable error on attempt %d/%d: %s. Sleeping %.2fs",
                        attempt,
                        max_attempts,
                        repr(e),
                        delay,
                    )
                await asyncio.sleep(delay)
            except Exception as e:
                last_err = e
                if attempt >= max_attempts:
                    raise
                delay = base_delay * (2 ** (attempt - 1)) + random.uniform(0, 0.25)
                if self._verbose:
                    logger.warning(
                        "Unexpected error on attempt %d/%d: %s. Sleeping %.2fs",
                        attempt,
                        max_attempts,
                        repr(e),
                        delay,
                    )
                await asyncio.sleep(delay)
        raise last_err

    def _prompt(self, key: str) -> list[dict]:
        """Get prompt template by key."""
        if key not in self.prompts:
            raise KeyError(f"Missing prompt '{key}'")
        return deepcopy(self.prompts[key])

    async def _llm_output_processing(self, full_messages, reasoning=None):
        """Process LLM output and extract structured response."""
        try:
            if reasoning:
                full_chat_response = await self.query_llm(
                    messages=full_messages,
                    temperature=0.6,
                    max_completion_tokens=2048,
                )
            else:
                full_chat_response = await self.query_llm(messages=full_messages)
            content = full_chat_response.choices[0].message.content
            json_output = json.loads(content)
        except UnicodeDecodeError:
            logger.warning("Error decoding as json: %s", content)
            return {
                "option_id": -1,
                "reason": "Response un-decodable",
            }

        try:
            selected_option_id = json_output["option_id"]
            reason_for_selection = json_output.get("reason", "")

            if reasoning:
                if self.llm_server == "llm_platform":
                    reasoning_content = full_chat_response.choices[0].message.reasoning
                else:
                    reasoning_content = full_chat_response.choices[0].message.reasoning_content
                reason_for_selection = f"reasoning:{reasoning_content}\n\n{reason_for_selection}"

        except KeyError:
            logger.warning("Error processing decoded json: %s", content)
            return {
                "option_id": -1,
                "reason": "Wrong structured response",
            }

        return Response(option_id=int(selected_option_id), reason=reason_for_selection).model_dump()

    def _compute_correlation(
        self,
        predicted_values: List[int],
        human_values: List[int],
    ) -> Dict[str, float]:
        """
        Compute Pearson correlation between predicted and human values across all answers.

        Uses numerically stable correlation computation similar to wvs_values_comparison.py.
        Computes correlation across all (user, question) pairs to measure overall alignment.

        Args:
            predicted_values: List of predicted option IDs (all users, all questions)
            human_values: List of human option IDs (all users, all questions)

        Returns:
            Dictionary with correlation coefficient (no p-value as we use custom implementation)
        """
        import numpy as np

        if len(predicted_values) != len(human_values):
            logger.warning(
                "Mismatch in value counts: predicted=%d, human=%d", len(predicted_values), len(human_values)
            )
            return {"correlation": None, "p_value": None, "n_samples": 0}

        if len(predicted_values) < 2:
            logger.warning("Not enough samples for correlation: %d", len(predicted_values))
            return {"correlation": None, "p_value": None, "n_samples": len(predicted_values)}

        # Convert to float arrays
        try:
            x = np.array(predicted_values, dtype=float)
            y = np.array(human_values, dtype=float)
        except Exception as e:
            logger.error("Error converting values to numeric arrays: %s", str(e))
            return {"correlation": None, "p_value": None, "n_samples": len(predicted_values)}

        # Handle NaNs/inf - omit policy
        mask = np.isfinite(x) & np.isfinite(y)
        if not np.all(mask):
            n_invalid = (~mask).sum()
            logger.warning("Found %d non-finite values, omitting them", n_invalid)
            x, y = x[mask], y[mask]
            if x.size < 2:
                logger.warning("After omitting non-finite values, only %d samples remain", x.size)
                return {"correlation": None, "p_value": None, "n_samples": int(x.size)}

        # Guard against constant vectors (zero variance)
        if np.allclose(x, x[0]) or np.allclose(y, y[0]):
            logger.warning("Constant vector detected: x_std=%.6f, y_std=%.6f", np.std(x), np.std(y))
            return {"correlation": None, "p_value": None, "n_samples": len(x)}

        # Numerically stable Pearson correlation (from wvs_values_comparison.py)
        x_dev = x - x.mean()
        y_dev = y - y.mean()
        denom = np.sqrt(np.dot(x_dev, x_dev) * np.dot(y_dev, y_dev))
        if denom == 0:
            logger.warning("Denominator is zero in correlation computation")
            return {"correlation": None, "p_value": None, "n_samples": len(x)}

        r = float(np.dot(x_dev, y_dev) / denom)
        # Clamp to valid range due to floating point errors
        r = max(-1.0, min(1.0, r))

        # Compute p-value using scipy if available
        try:
            from scipy.stats import pearsonr as scipy_pearsonr

            _, p_value = scipy_pearsonr(x, y)
            p_value = float(p_value)
        except ImportError:
            # If scipy not available, don't compute p-value
            p_value = None
        except Exception as e:
            logger.warning("Could not compute p-value: %s", str(e))
            p_value = None

        return {"correlation": r, "p_value": p_value, "n_samples": len(x)}

    async def _generate_gap_rationale(
        self,
        question_id: str,
        dialogue_history: List[Dict],
        full_question: str,
        ba_answer: Dict,
        human_option_id: int,
    ) -> Dict:
        """
        Generate rationale for the gap between predicted and human values.

        This method tests if the model can adapt to user feedback by presenting
        the dialogue context and having the user express preference for the human option.

        Args:
            question_id: Question identifier.
            dialogue_history: List of dialogue messages.
            full_question: Full question text.
            ba_answer: Complete BA answer dict with option_id and reason.
            human_option_id: Actual option ID chosen by human.

        Returns:
            Dict with question_id and rationale response.
        """
        gap_rationale_prompt = self._prompt("gap_rationale")

        # Prepare dialogue context - convert chatbot role to assistant
        dialogue_based_msgs = deepcopy(dialogue_history)
        dialogue_based_msgs = [
            {**m, "role": "assistant"} if m.get("role") == "chatbot" else m for m in dialogue_based_msgs
        ]

        # Insert the full dialogue context before the user's feedback
        # The dialogue provides the context for the conversation
        full_messages = [gap_rationale_prompt[0]]  # System message
        full_messages.extend(dialogue_based_msgs)  # Full dialogue history

        # Add the values question and BA result as assistant message
        # Format: {"role": "assistant", "content": {"option_id": X, "reason": "..."}}
        assistant_response = f"For this question:\n\n{full_question}\n\n"
        assistant_response += json.dumps(ba_answer, ensure_ascii=False)

        full_messages.append({"role": "assistant", "content": assistant_response})

        # Add user's feedback (hint at preferring the human option)
        full_messages.append(
            {"role": "user", "content": gap_rationale_prompt[1]["content"].format(human_option_id=human_option_id)}
        )

        structured_output = await self._retry_llm(
            lambda: self._llm_output_processing(full_messages=full_messages, reasoning=self.reasoning),
            max_attempts=3,
            base_delay=1.0,
        )

        return {
            question_id: {
                "predicted_option_id": ba_answer.get("option_id"),
                "human_option_id": human_option_id,
                "gap_rationale": structured_output,
            }
        }

    async def run_gap_analysis(self) -> None:
        """
        Main method to run gap analysis.

        For each user in BA dialogue results:
        1. Find matching human results
        2. For each question, compare predicted vs human option_id
        3. If there's a gap, generate rationale using the model
        4. Track statistics: original gaps vs remaining gaps after adaptation
        """
        gap_analysis_results = []

        # Statistics tracking
        total_questions = 0
        original_gaps = 0  # All gaps (predicted != human)
        queried_gaps = 0  # Gaps that exceeded threshold and were queried
        remaining_gaps = 0  # Gaps that remain after adaptation
        skipped_gaps = 0  # Gaps below threshold that were not queried
        skipped_users = 0  # Users skipped due to missing data

        # Correlation tracking - collect all predicted and human values
        before_adaptation_predicted = []
        before_adaptation_human = []
        after_adaptation_predicted = []
        after_adaptation_human = []

        try:
            with tqdm(
                total=len(self.ba_dialogue_results),
                desc="Analyzing gaps and generating rationales",
                unit="user",
            ) as pbar:
                for user_id, ba_answers in self.ba_dialogue_results.items():
                    # Find matching human results
                    if user_id not in self.human_results:
                        logger.warning("No human results found for user: %s", user_id)
                        skipped_users += 1
                        pbar.update(1)
                        continue

                    human_answers = self.human_results[user_id]

                    # Get dialogue history
                    if user_id not in self.generated_dialogues:
                        logger.warning("No dialogue found for user: %s (skipping)", user_id)
                        skipped_users += 1
                        pbar.update(1)
                        continue

                    dialogue_history = self.generated_dialogues[user_id]

                    # Analyze gaps for this user
                    user_gaps = {}
                    gap_queries = []

                    for question_id, ba_answer in ba_answers.items():
                        if question_id not in human_answers:
                            if self.verbose:
                                logger.warning(
                                    "Question %s not found in human results for user %s",
                                    question_id,
                                    user_id,
                                )
                            continue

                        if question_id not in self.all_questions:
                            if self.verbose:
                                logger.warning("Question %s not found in question metadata", question_id)
                            continue

                        predicted_option = ba_answer.get("option_id")
                        human_option = human_answers[question_id]

                        # Count total questions processed
                        total_questions += 1

                        # Collect values for correlation (before adaptation)
                        before_adaptation_predicted.append(predicted_option)
                        before_adaptation_human.append(human_option)

                        # Check if there's a gap
                        if predicted_option != human_option:
                            original_gaps += 1

                            # Calculate relative gap
                            question_meta = self.all_questions[question_id]
                            scale_min = question_meta.get("answer_scale_min", 1)
                            scale_max = question_meta.get("answer_scale_max", 10)
                            scale_range = scale_max - scale_min

                            # Relative gap: |predicted - human| / scale_range
                            if scale_range > 0:
                                relative_gap = abs(predicted_option - human_option) / scale_range
                            else:
                                relative_gap = 1.0  # If scale_range is 0, consider it maximum gap

                            # Only query if relative gap exceeds threshold
                            if relative_gap >= self.gap_threshold:
                                queried_gaps += 1
                                full_question = question_meta["question"]
                                gap_queries.append(
                                    {
                                        "question_id": question_id,
                                        "dialogue_history": dialogue_history,
                                        "full_question": full_question,
                                        "ba_answer": ba_answer,
                                        "human_option_id": human_option,
                                    }
                                )
                            else:
                                skipped_gaps += 1

                    # Generate rationales for all gaps for this user
                    if gap_queries:
                        gap_rationales = await asyncio.gather(
                            *[self._generate_gap_rationale(**kwargs) for kwargs in gap_queries]
                        )

                        for rationale in gap_rationales:
                            user_gaps.update(rationale)

                        # Count remaining gaps (where model didn't change to human option)
                        for _question_id, gap_info in user_gaps.items():
                            adapted_option = gap_info["gap_rationale"]["option_id"]
                            human_option = gap_info["human_option_id"]
                            if adapted_option != human_option:
                                remaining_gaps += 1

                    # Collect after-adaptation values for correlation
                    # For each question, use adapted value if available, otherwise original predicted value
                    for question_id, ba_answer in ba_answers.items():
                        if question_id not in human_answers or question_id not in self.all_questions:
                            continue

                        human_option = human_answers[question_id]

                        # Use adapted value if this question had a gap and was queried
                        if question_id in user_gaps:
                            adapted_option = user_gaps[question_id]["gap_rationale"]["option_id"]
                            after_adaptation_predicted.append(adapted_option)
                        else:
                            # No gap or gap was skipped - use original prediction
                            predicted_option = ba_answer.get("option_id")
                            after_adaptation_predicted.append(predicted_option)

                        after_adaptation_human.append(human_option)

                    if user_gaps:
                        gap_analysis_results.append({user_id: user_gaps})

                    # Store results periodically
                    if self.storage_step and (pbar.n + 1) % self.storage_step == 0:
                        self.append_to_file(gap_analysis_results, self.output_file_path)
                        gap_analysis_results.clear()

                    pbar.update(1)

                # Store any remaining results
                if gap_analysis_results:
                    self.append_to_file(gap_analysis_results, self.output_file_path)

            # Calculate and log statistics
            adapted_gaps = queried_gaps - remaining_gaps  # Gaps that were queried and adapted
            adaptation_rate = (adapted_gaps / queried_gaps * 100) if queried_gaps > 0 else 0
            remaining_rate = (remaining_gaps / queried_gaps * 100) if queried_gaps > 0 else 0
            total_remaining_gaps = skipped_gaps + remaining_gaps  # Total gaps still present

            # Compute Pearson correlations
            logger.info("Computing Pearson correlations...")
            before_correlation = self._compute_correlation(before_adaptation_predicted, before_adaptation_human)
            after_correlation = self._compute_correlation(after_adaptation_predicted, after_adaptation_human)

            stats = {
                "summary": {
                    "gap_threshold": self.gap_threshold,
                    "total_users": len(self.ba_dialogue_results),
                    "processed_users": len(self.ba_dialogue_results) - skipped_users,
                    "skipped_users": skipped_users,
                    "total_questions": total_questions,
                    "original_gaps": original_gaps,
                    "queried_gaps": queried_gaps,
                    "skipped_gaps": skipped_gaps,
                    "adapted_gaps": adapted_gaps,
                    "remaining_gaps_after_adaptation": remaining_gaps,
                    "total_remaining_gaps": total_remaining_gaps,
                    "adaptation_rate_percent": round(adaptation_rate, 2),
                    "remaining_gap_rate_percent": round(remaining_rate, 2),
                    "original_accuracy_percent": (
                        round((total_questions - original_gaps) / total_questions * 100, 2)
                        if total_questions > 0
                        else 0
                    ),
                    "post_adaptation_accuracy_percent": (
                        round((total_questions - total_remaining_gaps) / total_questions * 100, 2)
                        if total_questions > 0
                        else 0
                    ),
                },
                "correlation": {
                    "before_adaptation": before_correlation,
                    "after_adaptation": after_correlation,
                    "improvement": {
                        "correlation_delta": (
                            round(after_correlation["correlation"] - before_correlation["correlation"], 4)
                            if before_correlation["correlation"] is not None
                            and after_correlation["correlation"] is not None
                            else None
                        )
                    },
                },
            }

            # Save statistics to a separate JSON file
            stats_file_path = self.output_file_path.replace(".jsonl", "_statistics.json")
            with open(stats_file_path, "w", encoding="utf-8") as f:
                json.dump(stats, f, ensure_ascii=False, indent=2)

            # Log statistics
            logger.info("=" * 80)
            logger.info("GAP ANALYSIS STATISTICS")
            logger.info("=" * 80)
            logger.info("Configuration:")
            logger.info("  Gap threshold: %.2f", self.gap_threshold)
            logger.info("  Starting row: %d", self._starting_row)
            logger.info("  Ending row: %s", self._ending_row if self._ending_row is not None else "end")
            logger.info("")
            logger.info("User Statistics:")
            logger.info("  Total users: %d", len(self.ba_dialogue_results))
            logger.info("  Successfully processed users: %d", len(self.ba_dialogue_results) - skipped_users)
            logger.info("  Skipped users (missing data): %d", skipped_users)
            logger.info("")
            logger.info("Question Statistics:")
            logger.info("  Total questions processed: %d", total_questions)
            logger.info(
                "  Original gaps (model != human): %d (%.2f%%)",
                original_gaps,
                (original_gaps / total_questions * 100) if total_questions > 0 else 0,
            )
            logger.info(
                "    - Queried gaps (>= threshold): %d (%.2f%% of gaps)",
                queried_gaps,
                (queried_gaps / original_gaps * 100) if original_gaps > 0 else 0,
            )
            logger.info(
                "    - Skipped gaps (< threshold): %d (%.2f%% of gaps)",
                skipped_gaps,
                (skipped_gaps / original_gaps * 100) if original_gaps > 0 else 0,
            )
            logger.info("")
            logger.info("Adaptation Results:")
            logger.info(
                "  Gaps adapted (model changed to human): %d (%.2f%% of queried)", adapted_gaps, adaptation_rate
            )
            logger.info("  Remaining gaps after adaptation: %d (%.2f%% of queried)", remaining_gaps, remaining_rate)
            logger.info("  Total remaining gaps (skipped + unadapted): %d", total_remaining_gaps)
            logger.info("")
            logger.info("Accuracy Metrics:")
            logger.info("  Original accuracy: %.2f%%", stats["summary"]["original_accuracy_percent"])
            logger.info("  Post-adaptation accuracy: %.2f%%", stats["summary"]["post_adaptation_accuracy_percent"])
            logger.info(
                "  Accuracy improvement: %.2f%%",
                stats["summary"]["post_adaptation_accuracy_percent"] - stats["summary"]["original_accuracy_percent"],
            )
            logger.info("")
            logger.info("Pearson Correlation (Predicted vs Human):")
            if before_correlation["correlation"] is not None:
                logger.info(
                    "  Before adaptation: r = %.4f (p = %.4e, n = %d)",
                    before_correlation["correlation"],
                    before_correlation["p_value"] if before_correlation["p_value"] is not None else 0,
                    before_correlation["n_samples"],
                )
            else:
                logger.info("  Before adaptation: N/A (insufficient data)")

            if after_correlation["correlation"] is not None:
                logger.info(
                    "  After adaptation:  r = %.4f (p = %.4e, n = %d)",
                    after_correlation["correlation"],
                    after_correlation["p_value"] if after_correlation["p_value"] is not None else 0,
                    after_correlation["n_samples"],
                )
            else:
                logger.info("  After adaptation: N/A (insufficient data)")

            if before_correlation["correlation"] is not None and after_correlation["correlation"] is not None:
                delta = after_correlation["correlation"] - before_correlation["correlation"]
                logger.info("  Correlation improvement: %.4f", delta)
            logger.info("=" * 80)
            logger.info("Statistics saved to: %s", stats_file_path)

        except Exception as e:
            logger.error("An error occurred during gap analysis: %s", str(e))
            raise

    def append_to_file(self, data: List[Dict], output_file_path: str):
        """Append data to the specified JSONL file."""
        with open(output_file_path, "a", encoding="utf-8") as jsonl_file:
            for entry in data:
                jsonl_file.write(json.dumps(entry, ensure_ascii=False) + "\n")

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """Add CLI arguments."""
        parser.add_argument(
            "--config",
            type=str,
            default=None,
            help="Path to YAML config (recommended). CLI flags override YAML.",
        )
        return parser

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace):
        """Create instance from CLI arguments."""
        cfg = _load_yaml(args.config)

        # Required keys
        required = [
            "evaluated_model",
            "ba_dialogue_results_path",
            "human_results_path",
            "generated_dialogues_path",
            "picked_questions_path",
            "prompts_folder",
            "output_file_path",
        ]
        for k in required:
            if not cfg.get(k):
                raise ValueError(f"Missing required config key: {k}")

        # Setup OpenAI client
        api_key = cfg.get("openai_api_key") or os.environ.get("api_key") or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("Missing OpenAI API key.")

        evaluated_model = cfg["evaluated_model"]
        llm_server = cfg.get("llm_server", "llm_platform")

        if "gpt" in evaluated_model and "oss" not in evaluated_model:
            openai_client = AsyncOpenAI(api_key=api_key)
        else:
            base_url = cfg.get("model_base_url") or os.environ.get("base_url") or "http://localhost:8000/v1"
            openai_client = AsyncOpenAI(api_key=api_key, base_url=base_url)

        return cls(
            evaluated_model=evaluated_model,
            ba_dialogue_results_path=cfg["ba_dialogue_results_path"],
            human_results_path=cfg["human_results_path"],
            generated_dialogues_path=cfg["generated_dialogues_path"],
            picked_questions_path=cfg["picked_questions_path"],
            prompts_folder=cfg["prompts_folder"],
            output_file_path=cfg["output_file_path"],
            openai_client=openai_client,
            storage_step=cfg.get("storage_step"),
            llm_server=llm_server,
            verbose=int(cfg.get("verbose", 0) or 0),
            reasoning=bool(cfg.get("reasoning", False)),
            extra_body=cfg.get("extra_body"),
            starting_row=cfg.get("starting_row"),
            ending_row=cfg.get("ending_row"),
            gap_threshold=float(cfg.get("gap_threshold", 0.0)),
        )


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Gap analysis between BA dialogue and human values")
    parser = GapAnalysisController.add_cli_args(parser=parser)
    args = parser.parse_args()
    controller = GapAnalysisController.from_cli_args(args=args)

    await controller.run_gap_analysis()


if __name__ == "__main__":
    asyncio.run(main())
