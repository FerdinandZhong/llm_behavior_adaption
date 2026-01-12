"""WVS Implicit Adaptation - Testing implicit feedback-based adaptation with gap analysis"""

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

from llm_behavior_adaptation.dialogue_dataset_creation.generation_utils import (
    load_json_folder,
    render_json,
    retrieve_user_profile_wvs,
)
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
        logger.info("Implicit adaptation logging started - log file: %s", log_file_path)
        logger.info("=" * 80)


def _load_yaml(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


class Response(BaseModel):
    option_id: int
    reason: str


class ImplicitAdaptationResponse(BaseModel):
    option_id: int
    reason: str
    confidence: str


class ImplicitAdaptationController:
    """Controller for testing implicit adaptation via feedback."""

    def __init__(
        self,
        evaluated_model: str,
        output_file_path: str,
        user_profile_dataset: pd.DataFrame,
        picked_questions: Dict,
        prompts_folder: str,
        human_results_path: str,
        feedback_type: str = "implicit_feedback",
        openai_client: Optional[AsyncOpenAI] = None,
        verbose: int = 0,
        storage_step: Optional[int] = None,
        llm_server: str = "llm_platform",
        reasoning: bool = False,
        extra_body: Dict = None,
        gap_threshold: float = 0.0,
    ) -> None:
        """
        Initialize the ImplicitAdaptationController.

        Args:
            evaluated_model: The name or identifier of the evaluated model.
            output_file_path: Path to the output file for results.
            user_profile_dataset: DataFrame containing user profile data.
            picked_questions: Mapping of question_id -> question metadata for evaluation.
            prompts_folder: Path to folder containing prompt templates.
            human_results_path: Path to human results CSV file.
            feedback_type: Type of implicit feedback - "implicit_feedback" or "willing_to_change".
            openai_client: Optional OpenAI client instance.
            verbose: Verbosity level for logging (non-negative integer).
            storage_step: Optional interval for flushing results to disk.
            llm_server: One of {"llm_platform", "gpt", "vllm", "sglang"}.
            reasoning: Whether the tested model is a reasoning model.
            extra_body: Additional parameters for API calls.
            gap_threshold: Relative threshold (0.0-1.0) for querying model. 0.0 means query all gaps,
                          1.0 means only query maximum gaps. Calculated as |predicted - human| / scale_range.

        Raises:
            ValueError: If required arguments are invalid or missing.
            TypeError: If input arguments are of incorrect types.
        """
        if not evaluated_model:
            raise ValueError("evaluated_model must be provided.")
        if not isinstance(user_profile_dataset, pd.DataFrame):
            raise TypeError("user_profile_dataset must be a pandas DataFrame.")
        if not isinstance(output_file_path, str) or not output_file_path:
            raise ValueError("output_file_path must be a non-empty string.")
        if not isinstance(picked_questions, dict):
            raise TypeError("picked_questions must be a Dict.")
        if not isinstance(verbose, int) or verbose < 0:
            raise ValueError("verbose must be a non-negative integer.")
        if feedback_type not in {"implicit_feedback", "willing_to_change"}:
            raise ValueError("feedback_type must be 'implicit_feedback' or 'willing_to_change'.")
        if not isinstance(human_results_path, str) or not human_results_path:
            raise ValueError("human_results_path must be a non-empty string.")
        if not (0.0 <= gap_threshold <= 1.0):
            raise ValueError("gap_threshold must be between 0.0 and 1.0.")

        self.prompts = load_json_folder(prompts_folder)

        self._evaluated_model = evaluated_model
        self._user_profile_dataset = user_profile_dataset
        self._output_file_path = output_file_path
        self._picked_questions = picked_questions
        self._verbose = verbose
        self._storage_step = storage_step
        self._reasoning = reasoning
        self._feedback_type = feedback_type
        self._gap_threshold = gap_threshold

        # Setup file logging
        setup_file_logging(output_file_path, logger)

        # Load human results
        self._load_human_results(human_results_path)

        # Create flat questions dict for easy lookup
        self._all_questions = {}
        for _, q_list in self._picked_questions.items():
            self._all_questions.update(q_list)

        print(f"reasoning: {reasoning}, feedback_type: {feedback_type}, gap_threshold: {gap_threshold}")

        # Client selection
        if openai_client is None:
            if "gpt" in evaluated_model:
                self._openai_client = AsyncOpenAI(api_key=os.environ["api_key"])
            else:
                base_url = os.getenv("base_url", "http://localhost:8000/v1")
                self._openai_client = AsyncOpenAI(api_key=os.environ["api_key"], base_url=base_url)
        else:
            self._openai_client = openai_client

        # Query function wiring
        if feedback_type == "implicit_feedback":
            response_schema = ImplicitAdaptationResponse.model_json_schema()
        else:
            response_schema = Response.model_json_schema()

        if llm_server == "llm_platform":
            self.query_llm = partial(
                self.openai_client.chat.completions.create,
                model=self.evaluated_model,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "option_response",
                        "schema": response_schema,
                    },
                },
                extra_body=extra_body,
            )
        else:
            self.query_llm = partial(
                self.openai_client.chat.completions.create,
                model=self.evaluated_model,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "option_response",
                        "schema": response_schema,
                    },
                },
                **extra_body if extra_body else {},
            )
        self.llm_server = llm_server

        # Retryable errors tuple
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

    def _load_human_results(self, human_results_path: str):
        """Load human results from CSV file."""
        human_df = pd.read_csv(human_results_path)
        human_df = human_df.loc[:, ~human_df.columns.str.contains("^Unnamed")]
        self._human_results = (
            human_df.astype({"D_INTERVIEW": str}).groupby("D_INTERVIEW", as_index=True).last().to_dict(orient="index")
        )
        logger.info("Loaded human results for %d users", len(self._human_results))

    # Properties
    @property
    def evaluated_model(self) -> str:
        return self._evaluated_model

    @property
    def user_profile_dataset(self) -> pd.DataFrame:
        return self._user_profile_dataset

    @property
    def output_file_path(self) -> str:
        return self._output_file_path

    @property
    def verbose(self) -> int:
        return self._verbose

    @property
    def storage_step(self) -> Optional[int]:
        return self._storage_step

    @property
    def picked_questions(self) -> Dict:
        return self._picked_questions

    @property
    def openai_client(self) -> AsyncOpenAI:
        return self._openai_client

    @property
    def reasoning(self) -> bool:
        return self._reasoning

    @property
    def feedback_type(self) -> str:
        return self._feedback_type

    @property
    def human_results(self) -> Dict[str, Dict]:
        return self._human_results

    @property
    def all_questions(self) -> Dict:
        return self._all_questions

    @property
    def gap_threshold(self) -> float:
        return self._gap_threshold

    # Retry helper
    async def _retry_llm(self, call_factory, *, max_attempts: int = 3, base_delay: float = 1.0):
        """Retry wrapper for LLM calls with exponential backoff and jitter."""
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

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser.add_argument(
            "--config",
            type=str,
            default=None,
            help="Path to YAML config (recommended). CLI flags override YAML.",
        )
        return parser

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace):
        cfg = _load_yaml(args.config)

        # Required keys
        required = [
            "user_profile_dataset_path",
            "picked_questions_path",
            "evaluated_model",
            "output_file_path",
            "prompts_folder",
            "human_results_path",
        ]
        for k in required:
            if not cfg.get(k):
                raise ValueError(f"Missing required config key: {k}")

        # Dataset slice
        full_dataset = pd.read_csv(cfg["user_profile_dataset_path"])
        full_dataset = full_dataset.loc[:, ~full_dataset.columns.str.contains("^Unnamed")]
        start = int(cfg.get("starting_row", 0) or 0)
        end = cfg.get("ending_row", -1)
        if end is None or int(end) < 0:
            user_profile_dataset = full_dataset.iloc[start:]
        else:
            user_profile_dataset = full_dataset.iloc[start : int(end)]

        # Picked questions
        with open(cfg["picked_questions_path"], "r", encoding="utf-8") as question_file:
            picked_questions = json.load(question_file)

        # OpenAI client
        api_key = cfg.get("openai_api_key") or os.environ.get("api_key") or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("Missing OpenAI API key (YAML 'openai_api_key' or env 'api_key'/'OPENAI_API_KEY').")

        evaluated_model = cfg["evaluated_model"]
        llm_server = cfg.get("llm_server", "llm_platform")

        if "gpt" in evaluated_model and "oss" not in evaluated_model:
            openai_client = AsyncOpenAI(api_key=api_key)
        else:
            base_url = cfg.get("model_base_url") or os.environ.get("base_url") or "http://localhost:8000/v1"
            openai_client = AsyncOpenAI(api_key=api_key, base_url=base_url)

        return cls(
            evaluated_model=evaluated_model,
            output_file_path=cfg["output_file_path"],
            user_profile_dataset=user_profile_dataset,
            picked_questions=picked_questions,
            prompts_folder=cfg["prompts_folder"],
            human_results_path=cfg["human_results_path"],
            feedback_type=cfg.get("feedback_type", "implicit_feedback"),
            openai_client=openai_client,
            storage_step=cfg.get("storage_step"),
            llm_server=llm_server,
            verbose=int(cfg.get("verbose", 0) or 0),
            reasoning=bool(cfg.get("reasoning", False)),
            extra_body=cfg.get("extra_body"),
            gap_threshold=float(cfg.get("gap_threshold", 0.0)),
        )

    def _prompt(self, key: str) -> list[dict]:
        if key not in self.prompts:
            raise KeyError(f"Missing prompt '{key}'")
        return deepcopy(self.prompts[key])

    async def _llm_output_processing(self, full_messages, reasoning=None):
        """Process LLM output and return structured response."""
        try:
            if reasoning:
                full_chat_response = await self.query_llm(
                    messages=full_messages,
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
            confidence = json_output.get("confidence", "")

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

        result = {
            "option_id": int(selected_option_id),
            "reason": reason_for_selection,
        }
        if confidence:
            result["confidence"] = confidence

        return result

    async def _initial_value_query(self, question_id, user_profile, full_question):
        """Make initial value prediction."""
        direct_value_selection_prompt = self._prompt("direct_question")
        direct_value_selection_prompt[1]["content"] = direct_value_selection_prompt[1]["content"].format(
            user_details=user_profile
        )
        direct_value_selection_prompt[2]["content"] = direct_value_selection_prompt[2]["content"].format(
            values_question=full_question
        )

        structured_output = await self._retry_llm(
            lambda: self._llm_output_processing(full_messages=direct_value_selection_prompt, reasoning=self.reasoning),
            max_attempts=3,
            base_delay=1.0,
        )

        return {question_id: structured_output}

    def _get_alternative_option(self, initial_option_id: int, question_details: Dict) -> int:
        """Get a random alternative option ID different from the initial one."""
        num_options = question_details.get("num_options", 10)
        available_options = [i for i in range(1, num_options + 1) if i != initial_option_id]
        if available_options:
            return random.choice(available_options)
        return initial_option_id

    async def _adapted_value_query(
        self,
        question_id,
        user_profile,
        full_question,
        initial_response,
        question_details,
    ):
        """Make adapted value prediction after implicit feedback."""
        # Start with initial conversation
        initial_prompt = self._prompt("direct_question")
        initial_prompt[1]["content"] = initial_prompt[1]["content"].format(user_details=user_profile)
        initial_prompt[2]["content"] = initial_prompt[2]["content"].format(values_question=full_question)

        # Add initial response as assistant message
        conversation = deepcopy(initial_prompt)
        conversation.append(
            {
                "role": "assistant",
                "content": json.dumps(
                    {
                        "option_id": initial_response["option_id"],
                        "reason": initial_response["reason"],
                    }
                ),
            }
        )

        # Add implicit feedback
        feedback_prompt = self._prompt(self.feedback_type)

        if self.feedback_type == "willing_to_change":
            # For willing_to_change, suggest a random alternative option
            alternative_option = self._get_alternative_option(initial_response["option_id"], question_details)
            feedback_prompt[1]["content"] = feedback_prompt[1]["content"].format(
                alternative_option_id=alternative_option
            )
            conversation.extend(feedback_prompt)
        else:
            # For implicit_feedback, just add the uncertainty feedback
            conversation.extend(feedback_prompt)

        structured_output = await self._retry_llm(
            lambda: self._llm_output_processing(full_messages=conversation, reasoning=self.reasoning),
            max_attempts=3,
            base_delay=1.0,
        )

        return {question_id: structured_output}

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
            Dictionary with correlation coefficient and p-value
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

    @staticmethod
    def append_to_file(list_of_json_objs: List[Dict[str, Any]], output_path: str) -> None:
        """Append JSON objects to file in JSONL format."""
        with open(output_path, "a", encoding="utf-8") as f:
            for obj in list_of_json_objs:
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    async def run_implicit_adaptation(self):
        """
        Run implicit adaptation experiment with gap-based querying:
        1. Make initial predictions for all questions
        2. Compare with human values to identify gaps
        3. For gaps exceeding threshold, provide implicit feedback and get adapted predictions
        4. Track statistics and compute correlations
        """
        list_user_results = []

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
                total=len(self.user_profile_dataset),
                desc="Running implicit adaptation with gap analysis",
                unit="user",
            ) as pbar:
                for index, row in self.user_profile_dataset.iterrows():
                    row_dict = row.to_dict()
                    user_id = str(row_dict["D_INTERVIEW"])
                    user_profile = render_json(retrieve_user_profile_wvs(row_dict))

                    # Check if human results exist for this user
                    if user_id not in self.human_results:
                        logger.warning("No human results found for user: %s", user_id)
                        skipped_users += 1
                        pbar.update(1)
                        continue

                    human_answers = self.human_results[user_id]

                    if self._verbose == 1:
                        logger.info("Processing user %s (row %s)", user_id, index)

                    one_user_results = {
                        "initial": {},
                        "adapted": {},
                        "gaps_info": {},
                    }

                    for question_category, question_dict in self.picked_questions.items():
                        # Step 1: Get initial predictions for all questions
                        initial_kwargs = []
                        for question_id, question_details in question_dict.items():
                            full_question = question_details["question"]
                            initial_kwargs.append(
                                {
                                    "question_id": question_id,
                                    "user_profile": user_profile,
                                    "full_question": full_question,
                                }
                            )

                        initial_responses = await asyncio.gather(
                            *[self._initial_value_query(**kwargs) for kwargs in initial_kwargs]
                        )

                        # Step 2: Identify gaps and decide which to query
                        adapted_kwargs = []
                        gap_info_list = []

                        for i, (question_id, question_details) in enumerate(question_dict.items()):
                            if question_id not in human_answers:
                                if self.verbose:
                                    logger.warning(
                                        "Question %s not found in human results for user %s",
                                        question_id,
                                        user_id,
                                    )
                                continue

                            initial_response = initial_responses[i][question_id]
                            predicted_option = initial_response.get("option_id")
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
                                scale_min = question_details.get("answer_scale_min", 1)
                                scale_max = question_details.get("answer_scale_max", 10)
                                scale_range = scale_max - scale_min

                                # Relative gap: |predicted - human| / scale_range
                                if scale_range > 0:
                                    relative_gap = abs(predicted_option - human_option) / scale_range
                                else:
                                    relative_gap = 1.0  # If scale_range is 0, consider it maximum gap

                                # Only query if relative gap exceeds threshold
                                if relative_gap >= self.gap_threshold:
                                    queried_gaps += 1
                                    full_question = question_details["question"]
                                    adapted_kwargs.append(
                                        {
                                            "question_id": question_id,
                                            "user_profile": user_profile,
                                            "full_question": full_question,
                                            "initial_response": initial_response,
                                            "question_details": question_details,
                                        }
                                    )
                                    gap_info_list.append(
                                        {
                                            "question_id": question_id,
                                            "predicted_option": predicted_option,
                                            "human_option": human_option,
                                            "relative_gap": relative_gap,
                                            "queried": True,
                                        }
                                    )
                                else:
                                    skipped_gaps += 1
                                    gap_info_list.append(
                                        {
                                            "question_id": question_id,
                                            "predicted_option": predicted_option,
                                            "human_option": human_option,
                                            "relative_gap": relative_gap,
                                            "queried": False,
                                        }
                                    )

                        # Step 3: Get adapted predictions only for queried gaps
                        adapted_responses = []
                        if adapted_kwargs:
                            adapted_responses = await asyncio.gather(
                                *[self._adapted_value_query(**kwargs) for kwargs in adapted_kwargs]
                            )

                        # Store results
                        one_user_results["initial"][question_category] = initial_responses
                        one_user_results["adapted"][question_category] = adapted_responses
                        one_user_results["gaps_info"][question_category] = gap_info_list

                        # Count remaining gaps (where model didn't change to human option)
                        adapted_dict = {list(r.keys())[0]: list(r.values())[0] for r in adapted_responses}
                        for gap_info in gap_info_list:
                            if gap_info["queried"]:
                                question_id = gap_info["question_id"]
                                if question_id in adapted_dict:
                                    adapted_option = adapted_dict[question_id]["option_id"]
                                    human_option = gap_info["human_option"]
                                    if adapted_option != human_option:
                                        remaining_gaps += 1

                        if self._verbose == 1:
                            logger.info(
                                "Processed %d questions for category %s (%d gaps, %d queried)",
                                len(initial_responses),
                                question_category,
                                len(gap_info_list),
                                len(adapted_kwargs),
                            )

                    # Collect after-adaptation values for correlation
                    # For each question, use adapted value if available, otherwise original predicted value
                    for question_category, question_dict in self.picked_questions.items():
                        adapted_dict = {}
                        if question_category in one_user_results["adapted"]:
                            for response in one_user_results["adapted"][question_category]:
                                adapted_dict.update({list(response.keys())[0]: list(response.values())[0]})

                        for i, (question_id, _) in enumerate(question_dict.items()):
                            if question_id not in human_answers:
                                continue

                            human_option = human_answers[question_id]

                            # Use adapted value if this question was queried
                            if question_id in adapted_dict:
                                adapted_option = adapted_dict[question_id]["option_id"]
                                after_adaptation_predicted.append(adapted_option)
                            else:
                                # No gap or gap was skipped - use original prediction
                                if question_category in one_user_results["initial"]:
                                    initial_response = one_user_results["initial"][question_category][i]
                                    if question_id in initial_response:
                                        predicted_option = initial_response[question_id]["option_id"]
                                        after_adaptation_predicted.append(predicted_option)

                            after_adaptation_human.append(human_option)

                    list_user_results.append({user_id: one_user_results})

                    # Store results periodically if storage_step is defined
                    if self._storage_step and (pbar.n + 1) % self._storage_step == 0:
                        self.append_to_file(list_user_results, self._output_file_path)
                        list_user_results.clear()

                    pbar.update(1)

                # Store any remaining results
                if list_user_results:
                    self.append_to_file(list_user_results, self._output_file_path)

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
                    "total_users": len(self.user_profile_dataset),
                    "processed_users": len(self.user_profile_dataset) - skipped_users,
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
            logger.info("IMPLICIT ADAPTATION STATISTICS")
            logger.info("=" * 80)
            logger.info("Configuration:")
            logger.info("  Feedback type: %s", self.feedback_type)
            logger.info("  Gap threshold: %.2f", self.gap_threshold)
            logger.info("")
            logger.info("User Statistics:")
            logger.info("  Total users: %d", len(self.user_profile_dataset))
            logger.info("  Successfully processed users: %d", len(self.user_profile_dataset) - skipped_users)
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
            logger.error("An error occurred during implicit adaptation: %s", str(e))
            raise


def main():
    parser = argparse.ArgumentParser(description="Run implicit adaptation experiment for WVS values")
    ImplicitAdaptationController.add_cli_args(parser)
    args = parser.parse_args()

    try:
        controller = ImplicitAdaptationController.from_cli_args(args)
        logger.info("Starting implicit adaptation experiment...")
        asyncio.run(controller.run_implicit_adaptation())
        logger.info("Completed! Results written to %s", controller.output_file_path)
    except Exception:
        logger.exception("Implicit adaptation experiment failed.")
        return 1
    return 0


if __name__ == "__main__":
    exit(main())
