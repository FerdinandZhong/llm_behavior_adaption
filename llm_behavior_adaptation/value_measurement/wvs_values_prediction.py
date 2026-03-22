"""values prediction"""

import argparse
import asyncio
import json
import logging
import os
import random  # <-- added
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


def _load_yaml(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


class Response(BaseModel):
    option_id: int
    reason: str


class ValuesPredictionController:
    """Values prediction controller."""

    def __init__(
        self,
        evaluated_model: str,
        direct_output_file_path: str,
        dialogue_output_file_path: str,
        user_profile_dataset: pd.DataFrame,
        generated_dialogues: List[Dict],
        picked_questions: Dict,
        prompts_folder: str,
        openai_client: Optional[AsyncOpenAI] = None,
        verbose: int = 0,
        storage_step: Optional[int] = None,
        llm_server: str = "llm_platform",
        reasoning: bool = False,
        run_mode: str = "both",
        extra_body: Dict = None,
        translated_questions: Optional[Dict] = None,
        uid_to_language: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Initialize the ValuesPredictionController.

        Args:
            evaluated_model: The name or identifier of the evaluated model.
            direct_output_file_path: Path to the output file for direct value questions.
            dialogue_output_file_path: Path to the output file for dialogue-based questions.
            user_profile_dataset: DataFrame containing user profile data.
            generated_dialogues: List of dicts representing generated dialogues.
            picked_questions: Mapping of question_id -> question metadata for evaluation.
            openai_client: Optional OpenAI client instance.
            verbose: Verbosity level for logging (non-negative integer).
            storage_step: Optional interval for flushing results to disk.
            llm_server: One of {"llm_platform", "gpt", "vllm", "sglang"}.
            reasoning: Whether the tested model is a reasoning model.
            run_mode: Which passes to run. One of {"profiles", "dialogue", "both"}.

        Raises:
            ValueError: If required arguments are invalid or missing.
            TypeError: If input arguments are of incorrect types.
        """
        # --- basic validations ---
        if not evaluated_model:
            raise ValueError("evaluated_model must be provided.")
        if not isinstance(user_profile_dataset, pd.DataFrame):
            raise TypeError("user_profile_dataset must be a pandas DataFrame.")
        if not isinstance(direct_output_file_path, str) or not direct_output_file_path:
            raise ValueError("direct_output_file_path must be a non-empty string.")
        if not isinstance(dialogue_output_file_path, str) or not dialogue_output_file_path:
            raise ValueError("dialogue_output_file_path must be a non-empty string.")
        if not isinstance(generated_dialogues, dict):
            raise TypeError("generated_dialogues must be a Dict.")
        if not isinstance(picked_questions, dict):
            raise TypeError("picked_questions must be a Dict.")
        if not isinstance(verbose, int) or verbose < 0:
            raise ValueError("verbose must be a non-negative integer.")
        if run_mode not in {"profiles", "dialogue", "both"}:
            raise ValueError("run_mode must be one of {'profiles','dialogue','both'}.")

        self.prompts = load_json_folder(prompts_folder)

        # --- assign core fields ---
        self._evaluated_model = evaluated_model
        self._user_profile_dataset = user_profile_dataset
        self._direct_output_file_path = direct_output_file_path
        self._dialogue_output_file_path = dialogue_output_file_path
        self._generated_dialogues = generated_dialogues
        self._picked_questions = picked_questions
        self._verbose = verbose
        self._storage_step = storage_step
        self._reasoning = reasoning
        self._run_mode = run_mode
        # Optional: per-language translated questions and instruction
        # {language: {qid: translated_text, "_instruction": translated_instruction}}
        self._translated_questions: Dict = translated_questions or {}
        # {uid: target_language}
        self._uid_to_language: Dict[str, str] = uid_to_language or {}

        print(f"reasoning: {reasoning}")

        # --- client selection ---
        if openai_client is None:
            if "gpt" in evaluated_model:
                self._openai_client = AsyncOpenAI(api_key=os.environ["api_key"])
            else:
                base_url = os.getenv("base_url", "http://localhost:8000/v1")
                self._openai_client = AsyncOpenAI(api_key=os.environ["api_key"], base_url=base_url)
        else:
            self._openai_client = openai_client

        # --- query function wiring ---
        if llm_server == "llm_platform":
            self.query_llm = partial(
                self.openai_client.chat.completions.create,
                model=self.evaluated_model,
                response_format=(
                    {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "option_response",
                            "schema": Response.model_json_schema(),  # noqa: F821
                        },
                    }
                ),
                extra_body=extra_body,
            )
        else:
            self.query_llm = partial(
                self.openai_client.chat.completions.create,
                model=self.evaluated_model,
                response_format=(
                    {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "option_response",
                            "schema": Response.model_json_schema(),  # noqa: F821
                        },
                    }
                ),
                **extra_body,
            )
        self.llm_server = llm_server

        # --- retryable errors tuple (best-effort across versions) ---
        try:
            from openai import APIConnectionError, APIError, APIStatusError, APITimeoutError, RateLimitError  # noqa

            self._OPENAI_ERRORS = (
                APIError,
                RateLimitError,
                APITimeoutError,
                APIConnectionError,
                APIStatusError,
            )
        except Exception:  # pragma: no cover
            self._OPENAI_ERRORS = tuple()

    # -------------------- properties --------------------

    @property
    def evaluated_model(self) -> str:
        """The evaluated model identifier."""
        return self._evaluated_model

    @property
    def user_profile_dataset(self) -> pd.DataFrame:
        """The user profile dataset."""
        return self._user_profile_dataset

    @property
    def direct_output_file_path(self) -> str:
        """Output file path for direct questions."""
        return self._direct_output_file_path

    @property
    def dialogue_output_file_path(self) -> str:
        """Output file path for dialogue-based questions."""
        return self._dialogue_output_file_path

    @property
    def verbose(self) -> int:
        """Verbosity level."""
        return self._verbose

    @property
    def storage_step(self) -> Optional[int]:
        """Interval for periodic storage."""
        return self._storage_step

    @property
    def generated_dialogues(self) -> List[Dict]:
        """List of generated dialogues."""
        return self._generated_dialogues

    @property
    def picked_questions(self) -> Dict:
        """Mapping of question_id -> question metadata used for evaluation."""
        return self._picked_questions

    @property
    def openai_client(self) -> AsyncOpenAI:
        """OpenAI client."""
        return self._openai_client

    @property
    def reasoning(self) -> bool:
        """Whether the tested model is a reasoning model."""
        return self._reasoning

    @property
    def run_mode(self) -> str:
        """Which passes to run: 'profiles', 'dialogue', or 'both'."""
        return self._run_mode

    # -------------------- retry helper --------------------
    async def _retry_llm(self, call_factory, *, max_attempts: int = 3, base_delay: float = 1.0):
        """
        Retry wrapper for LLM calls with exponential backoff and jitter.

        Args:
            call_factory: zero-arg callable returning an awaitable (e.g., lambda: self.query_llm(...))
            max_attempts: maximum attempts (including the first)
            base_delay: base sleep in seconds; actual delay = base_delay * 2^(attempt-1) + jitter

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
                # Optional: retry unexpected exceptions too (can be narrowed)
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
        raise last_err  # safety

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
        # Load YAML config
        cfg = _load_yaml(args.config)

        # -------- Required keys (YAML or env-composed) --------
        required = [
            "user_profile_dataset_path",
            "dialogue_file",
            "picked_questions_path",
            "evaluated_model",
            "direct_output_file_path",
            "dialogue_output_file_path",
            "prompts_folder",
        ]
        for k in required:
            if not cfg.get(k):
                raise ValueError(f"Missing required config key: {k}")

        # -------- Dataset slice --------
        full_dataset = pd.read_csv(cfg["user_profile_dataset_path"])
        full_dataset = full_dataset.loc[:, ~full_dataset.columns.str.contains("^Unnamed")]
        start = int(cfg.get("starting_row", 0) or 0)
        end = cfg.get("ending_row", -1)
        if end is None or int(end) < 0:
            user_profile_dataset = full_dataset.iloc[start:]
        else:
            user_profile_dataset = full_dataset.iloc[start : int(end)]

        # -------- Dialogues --------
        logger.info(cfg["dialogue_file"])
        with open(cfg["dialogue_file"], "r", encoding="utf-8") as d_f:
            all_dialogues = d_f.readlines()
            generated_dialogues = [json.loads(dialogue) for dialogue in all_dialogues]
            generated_dialogues = {k: v for d in generated_dialogues for k, v in d.items()}
        # -------- Picked questions --------
        with open(cfg["picked_questions_path"], "r", encoding="utf-8") as question_file:
            picked_questions = json.load(question_file)

        # -------- OpenAI / backend client --------
        api_key = cfg.get("openai_api_key") or os.environ.get("api_key") or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("Missing OpenAI API key (YAML 'openai_api_key' or env 'api_key'/'OPENAI_API_KEY').")

        evaluated_model = cfg["evaluated_model"]
        llm_server = cfg.get("llm_server", "llm_platform")

        # Preserve your previous heuristic for base_url selection
        if "gpt" in evaluated_model and "oss" not in evaluated_model:
            openai_client = AsyncOpenAI(api_key=api_key)
        else:
            base_url = cfg.get("model_base_url") or os.environ.get("base_url") or "http://localhost:8000/v1"
            openai_client = AsyncOpenAI(api_key=api_key, base_url=base_url)

        # -------- Optional: translated questions --------
        translated_questions = None
        if cfg.get("translated_questions_path"):
            with open(cfg["translated_questions_path"], "r", encoding="utf-8") as f:
                translated_questions = json.load(f)
            logger.info("Loaded translated questions for %d languages", len(translated_questions))

        # -------- Optional: uid → language map (from translated dialogues JSONL) --------
        uid_to_language = None
        if cfg.get("uid_language_map_file"):
            uid_to_language = {}
            with open(cfg["uid_language_map_file"], "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    entry = json.loads(line)
                    uid = str(entry.get("user_profile", {}).get("D_INTERVIEW", ""))
                    lang = entry.get("target_language", "")
                    if uid and lang:
                        uid_to_language[uid] = lang
            logger.info("Loaded language map for %d users", len(uid_to_language))

        # -------- Instantiate controller --------
        return cls(
            evaluated_model=evaluated_model,
            direct_output_file_path=cfg["direct_output_file_path"],
            dialogue_output_file_path=cfg["dialogue_output_file_path"],
            user_profile_dataset=user_profile_dataset,
            generated_dialogues=generated_dialogues,
            picked_questions=picked_questions,
            prompts_folder=cfg["prompts_folder"],
            openai_client=openai_client,
            storage_step=cfg.get("storage_step"),
            llm_server=llm_server,
            verbose=int(cfg.get("verbose", 0) or 0),
            reasoning=bool(cfg.get("reasoning", False)),
            run_mode=(cfg.get("run_mode") or "both"),
            extra_body=(cfg.get("extra_body") or None),
            translated_questions=translated_questions,
            uid_to_language=uid_to_language,
        )

    # ---------- Orchestration ----------

    async def run(self) -> None:
        """
        Run values generation according to `self.run_mode`.

        - 'profiles': run profile-only pass
        - 'dialogue': run dialogue-context pass
        - 'both': run profiles first, then dialogue
        """
        if self._run_mode == "profiles":
            await self.get_values_for_user_profiles()
        elif self._run_mode == "dialogue":
            await self.get_values_for_dialogue()
        else:  # both
            await self.get_values_for_user_profiles()
            await self.get_values_for_dialogue()

    # ---------- Prompt & message utils ----------
    def _prompt(self, key: str) -> list[dict]:
        if key not in self.prompts:
            raise KeyError(f"Missing prompt '{key}'")
        return deepcopy(self.prompts[key])

    async def _llm_output_processing(self, full_messages, reasoning=None):
        # 1) Try to parse JSON
        try:
            if reasoning:
                full_chat_response = await self.query_llm(
                    messages=full_messages,
                    # temperature=0.6,  # default setting for reasoning model
                    max_completion_tokens=2048,
                )

            else:
                full_chat_response = await self.query_llm(messages=full_messages)
            content = full_chat_response.choices[0].message.content
            json_output = json.loads(content)
        except UnicodeDecodeError:
            logger.warning("Error decoding as json: %s", content)
            return (
                -1,
                "Response un-decodable",
            )

        # 2) Extract fields
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
            return (
                -1,
                "Wrong structured response",
            )

        return Response(option_id=int(selected_option_id), reason=reason_for_selection).model_dump()

    async def _direct_value_query(self, question_id, user_profile, full_question):
        """
        Query the LLM to answer a single values question given a rendered user profile.
        """
        direct_value_selection_prompt = self._prompt("direct_question")
        direct_value_selection_prompt[1]["content"] = direct_value_selection_prompt[1]["content"].format(
            user_details=user_profile
        )
        direct_value_selection_prompt[2]["content"] = direct_value_selection_prompt[2]["content"].format(
            values_question=full_question
        )

        # 🔁 Auto-retry the network call only
        structured_output = await self._retry_llm(
            lambda: self._llm_output_processing(full_messages=direct_value_selection_prompt, reasoning=self.reasoning),
            max_attempts=3,
            base_delay=1.0,
        )

        return {question_id: structured_output}

    async def _dialogue_continue_value_query(
        self, question_id, dialogue_history, full_question, target_language: str = None
    ):
        """
        Query the LLM to answer a values question using prior dialogue context.

        If *target_language* is provided and translated questions are available,
        both the question text and the instruction are substituted with their
        translated versions so the entire final turn matches the dialogue language.
        """
        dialogue_continue_prompt = self._prompt("dialogue_followup")
        dialogue_based_msgs = deepcopy(dialogue_history)
        dialogue_based_msgs = [
            {**m, "role": "assistant"} if m.get("role") == "chatbot" else m for m in dialogue_based_msgs
        ]
        dialogue_based_msgs.append(dialogue_continue_prompt[0])

        # Resolve translated question and instruction if available
        if target_language and target_language != "English" and self._translated_questions.get(target_language):
            lang_translations = self._translated_questions[target_language]
            translated_q = lang_translations.get(question_id, full_question)
            translated_instruction = lang_translations.get(
                "_instruction",
                "Assume you are me, help me find the most suitable answer to the following question:",
            )
            user_content = dialogue_continue_prompt[1]["content"].replace(
                "Assume you are me, help me find the most suitable answer to the following question:\n\n{values_question}",
                f"{translated_instruction}\n\n{{values_question}}",
            )
            dialogue_continue_prompt[1]["content"] = user_content.format(values_question=translated_q)
        else:
            dialogue_continue_prompt[1]["content"] = dialogue_continue_prompt[1]["content"].format(
                values_question=full_question
            )

        dialogue_based_msgs.append(dialogue_continue_prompt[1])

        structured_output = await self._retry_llm(
            lambda: self._llm_output_processing(full_messages=dialogue_based_msgs, reasoning=self.reasoning),
            max_attempts=3,
            base_delay=1.0,
        )

        return {question_id: structured_output}

    async def get_values_for_user_profiles(self):
        """
        Generate values predictions for each user profile (no dialogue context) and
        append results to the configured JSONL output file.
        """
        list_user_selections = []
        try:
            with tqdm(
                total=len(self.user_profile_dataset),
                desc="Generating values output",
                unit="dialogue",
            ) as pbar:
                for index, row in self.user_profile_dataset.iterrows():
                    row_dict = row.to_dict()
                    one_user_selections = {}

                    user_profile = render_json(retrieve_user_profile_wvs(row_dict))
                    user_id = row_dict["D_INTERVIEW"]
                    if self._verbose == 1:
                        logger.info("Processing row %s: %s", index, row_dict)

                    for (
                        question_category,
                        question_dict,
                    ) in self.picked_questions.items():
                        list_kwargs = []
                        for question_id, question_details in question_dict.items():
                            full_question = question_details["question"]

                            list_kwargs.append(
                                {
                                    "question_id": question_id,
                                    "user_profile": user_profile,
                                    "full_question": full_question,
                                }
                            )

                        one_user_one_category_selections = await asyncio.gather(
                            *[self._direct_value_query(**kwargs) for kwargs in list_kwargs]
                        )

                        one_user_selections[question_category] = one_user_one_category_selections

                        if self._verbose == 1:
                            logger.info(
                                "Current user selections for %s is %d",
                                question_category,
                                len(one_user_one_category_selections),
                            )

                    list_user_selections.append({user_id: one_user_selections})

                    # Store results periodically if storage_step is defined
                    if self._storage_step and (index + 1) % self._storage_step == 0:
                        self.append_to_file(list_user_selections, self._direct_output_file_path)
                        list_user_selections.clear()

                    pbar.update(1)

                # Store any remaining results
                if list_user_selections:
                    self.append_to_file(list_user_selections, self._direct_output_file_path)

        except Exception as e:
            logger.error(
                "An error occurred in the value selection given user profile: %s",
                str(e),
            )
            raise

    async def get_values_for_dialogue(self):
        """
        Generate values predictions for each user using their dialogue history and
        append results to the configured JSONL output file.
        """
        list_user_selections = []
        try:
            with tqdm(
                total=len(self.user_profile_dataset),
                desc="Generating values output",
                unit="dialogue",
            ) as pbar:
                for index, row in self.user_profile_dataset.iterrows():
                    row_dict = row.to_dict()
                    one_user_selections = {}

                    user_id = str(row_dict["D_INTERVIEW"])

                    print(len(self.generated_dialogues))
                    try:
                        user_dialogue = self.generated_dialogues[user_id]
                    except KeyError:
                        logger.warning("Error finding the dialogue for user: %s", user_id)

                    if self._verbose == 1:
                        logger.info("Processing row %s: %s", index, row_dict)

                    for (
                        question_category,
                        question_dict,
                    ) in self.picked_questions.items():
                        one_user_selections[question_category] = {}
                        list_kwargs = []
                        target_language = self._uid_to_language.get(user_id)

                        for question_id, question_details in question_dict.items():
                            full_question = question_details["question"]

                            list_kwargs.append(
                                {
                                    "question_id": question_id,
                                    "dialogue_history": user_dialogue,
                                    "full_question": full_question,
                                    "target_language": target_language,
                                }
                            )

                        one_user_one_category_selections = await asyncio.gather(
                            *[self._dialogue_continue_value_query(**kwargs) for kwargs in list_kwargs]
                        )

                        one_user_selections[question_category] = one_user_one_category_selections

                        if self._verbose == 1:
                            logger.info(
                                "Current user selections for %s is %d",
                                question_category,
                                len(one_user_one_category_selections),
                            )

                    list_user_selections.append({user_id: one_user_selections})

                    # Store results periodically if storage_step is defined
                    if self._storage_step and (index + 1) % self._storage_step == 0:
                        self.append_to_file(list_user_selections, self._dialogue_output_file_path)
                        list_user_selections.clear()

                    pbar.update(1)

                # Store any remaining results
                if list_user_selections:
                    self.append_to_file(list_user_selections, self._dialogue_output_file_path)

        except Exception as e:
            logger.error(
                "An error occurred in the value selection given dialogue context: %s",
                str(e),
            )
            raise

    def append_to_file(self, data, output_file_path):
        """Append data to the specified JSONL file."""
        with open(output_file_path, "a", encoding="utf-8") as jsonl_file:
            for entry in data:
                jsonl_file.write(json.dumps(entry, ensure_ascii=False) + "\n")


async def main():
    """
    Parse CLI args, construct the controller, then run the generation passes
    according to --run-mode (profiles | dialogue | both).
    """
    parser = argparse.ArgumentParser()
    parser = ValuesPredictionController.add_cli_args(parser=parser)
    values_prediction_args = parser.parse_args()
    prediction_controller = ValuesPredictionController.from_cli_args(args=values_prediction_args)

    await prediction_controller.run()


if __name__ == "__main__":
    asyncio.run(main())
