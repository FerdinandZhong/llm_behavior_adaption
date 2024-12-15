"""values prediction
"""

import argparse
import asyncio
import json
import logging
import math
import os
from copy import deepcopy
from typing import Dict, List

import pandas as pd
from openai import AsyncOpenAI
from pydantic import BaseModel
from tqdm.asyncio import tqdm

from persona_understanding.dialogue_dataset_creation.dialogue_controller import (
    DialogueRun,
)
from persona_understanding.dialogue_dataset_creation.generation_utils import (
    render_template,
    retrieve_user_profile,
)
from persona_understanding.value_measurement.constant import (
    CONVERSATION_HISTORY_PROMPT,
    DEFAULT_OPTION_IDS,
    DIALOGUE_CONTINUE_VALUE_QUESTIONS_CSV,
    DIRECT_VALUE_QUESTIONS_CSV,
    DIRECT_VALUE_SELECTION_PROMPT,
    OPTIONS_TEMPLATE,
    PROFILE_TEMPLATE,
)

logger = logging.getLogger(__name__)


class Response(BaseModel):
    option_id: int
    reason: str


class ValuesPredictionController:
    """Values prediction class"""

    def __init__(
        self,
        evaluated_model: str,
        direct_output_file_path: str,
        dialogue_output_file_path: str,
        user_profile_dataset: pd.DataFrame,
        generated_dialogues: List[Dict],
        direct_value_questions: pd.DataFrame,
        dialogue_continue_value_questions: pd.DataFrame,
        openai_client=None,
        verbose: int = 0,
        storage_step: int = None,
    ) -> None:
        """
        Initializes the ValuesPredictionController class with the specified parameters.

        Args:
            evaluated_model (str): The name or identifier of the evaluated model.
            direct_output_file_path (str): Path to the output file for direct value questions.
            dialogue_output_file_path (str): Path to the output file for dialogue-based questions.
            user_profile_dataset (pd.DataFrame): DataFrame containing user profile data.
            generated_dialogues (List[Dict]): List of dictionaries representing generated dialogues.
            direct_value_questions (pd.DataFrame): DataFrame with direct value questions.
            dialogue_continue_value_questions (pd.DataFrame): DataFrame with continuation value questions.
            openai_client: Client for interfacing with OpenAI API.
            verbose (int, optional): Verbosity level for logging. Defaults to 0.
            storage_step (int, optional): Interval to store results to file. Defaults to None.

        Raises:
            ValueError: If required arguments are invalid or missing.
            TypeError: If input arguments are of incorrect types.
        """
        if evaluated_model is None:
            raise ValueError("dialogue_generator cannot be None.")
        if not isinstance(user_profile_dataset, pd.DataFrame):
            raise TypeError("seed_dataset must be a pandas DataFrame.")
        if not isinstance(direct_output_file_path, str) or not direct_output_file_path:
            raise ValueError("output_file_path must be a non-empty string.")
        if not isinstance(verbose, int) or verbose < 0:
            raise ValueError("verbose must be a non-negative integer.")

        self._evaluated_model = evaluated_model
        self._user_profile_dataset = user_profile_dataset
        self._direct_output_file_path = direct_output_file_path
        self._dialogue_output_file_path = dialogue_output_file_path
        self._verbose = verbose
        self._generated_dialogues = generated_dialogues
        self._direct_value_questions = direct_value_questions
        self._dialogue_continue_value_questions = dialogue_continue_value_questions
        self._storage_step = storage_step
        if openai_client is None:
            if "gpt" in evaluated_model:
                self._openai_client = AsyncOpenAI(api_key=os.environ["api_key"])
            else:
                base_url = os.getenv("base_url", "http://localhost:8000/v1")
                self._openai_client = AsyncOpenAI(
                    api_key=os.environ["api_key"],
                    base_url=base_url
                )
        else:
            self._openai_client = openai_client
        
        if "gpt" in evaluated_model:
            self.response_json_schema = {
                "type": "json_schema",
                "json_schema": {
                    "name": "math_response",
                    "schema": Response.model_json_schema(),
                },
            }
        else:
            self.response_json_schema = Response.model_json_schema()

    @property
    def evaluated_model(self) -> str:
        """
        Getter for the evaluated model name.

        Returns:
            str: The evaluated model identifier.
        """
        return self._evaluated_model

    @property
    def user_profile_dataset(self) -> pd.DataFrame:
        """
        Getter for the user profile dataset.

        Returns:
            pd.DataFrame: The user profile dataset.
        """
        return self._user_profile_dataset

    @property
    def direct_output_file_path(self) -> str:
        """
        Getter for the direct output file path.

        Returns:
            str: The output file path for direct questions.
        """
        return self._direct_output_file_path

    @property
    def dialogue_output_file_path(self) -> str:
        """
        Getter for the dialogue output file path.

        Returns:
            str: The output file path for dialogue-based questions.
        """
        return self._dialogue_output_file_path

    @property
    def verbose(self) -> int:
        """
        Getter for the verbosity level.

        Returns:
            int: The verbosity level.
        """
        return self._verbose

    @property
    def generated_dialogues(self) -> List[Dict]:
        """
        Getter for the generated dialogues.

        Returns:
            List[Dict]: List of generated dialogues.
        """
        return self._generated_dialogues

    @property
    def openai_client(self) -> AsyncOpenAI:
        """
        Getter for the OpenAI client.

        Returns:
            AsyncOpenAI: The OpenAI client.
        """
        return self._openai_client

    @property
    def direct_value_questions(self) -> pd.DataFrame:
        """
        Getter for direct value questions.

        Returns:
            pd.DataFrame: The direct value questions DataFrame.
        """
        return self._direct_value_questions

    @property
    def dialogue_continue_value_questions(self) -> pd.DataFrame:
        """
        Getter for dialogue continuation value questions.

        Returns:
            pd.DataFrame: The continuation value questions DataFrame.
        """
        return self._dialogue_continue_value_questions

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """
        Add command-line arguments for the DatasetGeneration class.

        Args:
            parser (argparse.ArgumentParser): The argument parser to which CLI arguments will be added.

        Returns:
            argparse.ArgumentParser: The updated argument parser.
        """
        parser.add_argument(
            "--openai-api-key",
            type=str,
            default=os.environ.get("api_key"),
            help="OpenAI API key. Defaults to the value of the environment variable 'api_key'.",
        )
        parser.add_argument(
            "--model-base-url",
            type=str,
            default=os.environ.get("base_url")
        )
        parser.add_argument(
            "--user-profile-dataset",
            type=str,
            required=True,
            help="Path to the seed dataset file.",
        )
        parser.add_argument(
            "--starting-row",
            type=int,
            default=0,
            help="The starting row of the seed dataset",
        )
        parser.add_argument(
            "--ending-row",
            type=int,
            default=-1,
            help="The ending row of the seed dataset",
        )
        parser.add_argument(
            "--dialogue-file",
            type=str,
            required=True,
            help="Dialogue file",
        )
        parser.add_argument(
            "--evaluated_model",
            type=str,
            required=True,
            help="The model identifier for the user simulator. Defaults to 'gpt-4o'.",
        )
        parser.add_argument(
            "--direct_output_file_path",
            type=str,
            help="The path for the output for direct questions",
            required=True,
        )
        parser.add_argument(
            "--dialogue_output_file_path",
            type=str,
            help="The path for the output for dialogues based questions",
            required=True,
        )
        parser.add_argument(
            "--verbose",
            type=int,
            choices=[0, 1],
            default=0,
            help="Verbosity level: 0 = Errors only, 1 = Detailed logs. Defaults to 0.",
        )
        parser.add_argument(
            "--storage-step",
            type=int,
            default=None,
            help="Interval at which results are stored to the file.",
        )
        return parser

    @classmethod
    def from_cli_args(
        cls,
        args: argparse.Namespace,
    ):
        """
        Create an instance of the DatasetGeneration class using parsed CLI arguments.

        Args:
            args (argparse.Namespace): The parsed arguments from argparse.

        Returns:
            DatasetGeneration: An instance of the class populated with CLI argument values.
        """

        def _read_csv(csv_path):
            df = pd.read_csv(csv_path)
            df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
            return df

        full_user_dataset = _read_csv(args.user_profile_dataset)
        user_profile_dataset = full_user_dataset[args.starting_row : args.ending_row]
        direct_value_questions = _read_csv(DIRECT_VALUE_QUESTIONS_CSV)
        dialogue_continue_value_questions = _read_csv(
            DIALOGUE_CONTINUE_VALUE_QUESTIONS_CSV
        )

        generated_dialogues = []
        with open(args.dialogue_file, "r", encoding="utf-8") as dialogue_file:
            for dialogue in dialogue_file:
                generated_dialogues.append(json.loads(dialogue))
                if len(generated_dialogues) >= args.ending_row:
                    break
        
        if "gpt" in args.evaluated_model:
            openai_client = AsyncOpenAI(api_key=args.openai_api_key)
        else:
            openai_client = AsyncOpenAI(
                api_key=args.openai_api_key,
                base_url=args.model_base_url
            )

        return cls(
            evaluated_model=args.evaluated_model,
            direct_output_file_path=args.direct_output_file_path,
            dialogue_output_file_path=args.dialogue_output_file_path,
            user_profile_dataset=user_profile_dataset,
            generated_dialogues=generated_dialogues,
            direct_value_questions=direct_value_questions,
            dialogue_continue_value_questions=dialogue_continue_value_questions,
            openai_client=openai_client,
            storage_step=args.storage_step,
            verbose=args.verbose,
        )

    def _normalize_logprobs(self, logprobs, all_tokens):
        """Normalize log probabilities and handle missing tokens."""
        # Convert log probabilities to probabilities
        probs_dict = {key: math.exp(value) for key, value in logprobs.items()}

        # Ensure all tokens are present
        probs = [probs_dict.get(token, 0.0) for token in all_tokens]

        # Normalize probabilities to sum to 1
        prob_sum = sum(probs)
        return [p / prob_sum if prob_sum > 0 else 0.0 for p in probs]

    def _llm_output_processing(self, full_chat_response):
        json_output = json.loads(full_chat_response.choices[0].message.content)
        selected_option_id = json_output["option_id"]
        reason_for_selection = json_output["reason"]
        token_logprobs_mapping = {
            token_obj.token: token_obj
            for token_obj in full_chat_response.choices[0].logprobs.content
        }
        option_id_logprobs = token_logprobs_mapping[
            str(selected_option_id)
        ].top_logprobs
        option_id_logprobs = {
            prob_item.token: prob_item.logprob for prob_item in option_id_logprobs
        }
        normalized_probs = self._normalize_logprobs(
            option_id_logprobs, DEFAULT_OPTION_IDS
        )

        return selected_option_id, normalized_probs, reason_for_selection

    async def _direct_value_query(self, user_profile, full_question, options_str):
        direct_value_selection_prompt = deepcopy(DIRECT_VALUE_SELECTION_PROMPT)
        direct_value_selection_prompt[1]["content"] = direct_value_selection_prompt[1][
            "content"
        ].format(user_details=user_profile)
        direct_value_selection_prompt[2]["content"] = direct_value_selection_prompt[2][
            "content"
        ].format(question=full_question, option_list=options_str)
        if "gpt" in self.evaluated_model:
            full_chat_response = await self.openai_client.chat.completions.create(
                model=self.evaluated_model,
                messages=direct_value_selection_prompt,
                response_format=self.response_json_schema,
                logprobs=True,
                top_logprobs=5,
            )
        else:
            full_chat_response = await self.openai_client.chat.completions.create(
                model=self.evaluated_model,
                messages=direct_value_selection_prompt,
                response_format={"type": "json_schema"},
                logprobs=True,
                top_logprobs=5,
                extra_body={"guided_json": self.response_json_schema},
            )

        (
            selected_option_id,
            normalized_probs,
            reason_for_selection,
        ) = self._llm_output_processing(full_chat_response)

        return selected_option_id, normalized_probs, reason_for_selection

    async def _dialogue_continue_value_query(
        self, dialogue_history, full_question, options_str
    ):
        direct_value_selection_prompt = deepcopy(CONVERSATION_HISTORY_PROMPT)
        dialogue_history.append(direct_value_selection_prompt[0])
        direct_value_selection_prompt[1]["content"] = direct_value_selection_prompt[2][
            "content"
        ].format(question=full_question, option_list=options_str)
        dialogue_history.append(direct_value_selection_prompt[1])
        if "gpt" in self.evaluated_model:
            full_chat_response = await self.openai_client.chat.completions.create(
                model=self.evaluated_model,
                messages=dialogue_history,
                response_format=self.response_json_schema,
                logprobs=True,
                top_logprobs=5,
            )
        else:
            full_chat_response = await self.openai_client.chat.completions.create(
                model=self.evaluated_model,
                messages=dialogue_history,
                response_format={"type": "json_schema"},
                logprobs=True,
                top_logprobs=5,
                extra_body={"guided_json": self.response_json_schema},
            )

        (
            selected_option_id,
            normalized_probs,
            reason_for_selection,
        ) = self._llm_output_processing(full_chat_response)

        return selected_option_id, normalized_probs, reason_for_selection

    def _generate_question_options(self, question_row_dict):
        full_question_str = deepcopy(question_row_dict["full_question"])
        full_question_str.format(question=question_row_dict["questions"])

        options = []
        for option_col in [f"option_{i+1}" for i in range(5)]:
            options.append(question_row_dict[option_col])

        options_str = render_template(OPTIONS_TEMPLATE, option_list=options)

        return full_question_str, options_str

    async def get_values_for_user_profiles(self):
        list_user_selections = []
        try:
            with tqdm(
                total=len(self.user_profile_dataset),
                desc="Generating Dialogues",
                unit="dialogue",
            ) as pbar:
                for index, row in self.user_profile_dataset.iterrows():
                    row_dict = row.to_dict()
                    one_user_selections = []

                    user_profile = render_template(
                        PROFILE_TEMPLATE, profile_data=retrieve_user_profile(row_dict)
                    )

                    if self._verbose == 1:
                        logger.info(f"Processing row {index}: {row_dict}")

                    for (
                        question_index,
                        question_row,
                    ) in self.direct_value_questions.iterrows():
                        question_row_dict = question_row.to_dict()
                        (
                            full_question_str,
                            options_str,
                        ) = self._generate_question_options(question_row_dict)

                        (
                            selected_option_id,
                            normalized_probs,
                            reason_for_selection,
                        ) = await self._direct_value_query(
                            user_profile=user_profile,
                            full_question=full_question_str,
                            options_str=options_str,
                        )
                        one_user_selections.append(
                            {
                                "question_idx": question_index,
                                "selected_option_id": selected_option_id,
                                "normalized_probs": normalized_probs,
                                "reason_for_selection": reason_for_selection,
                            }
                        )

                    list_user_selections.append(
                        {"user_idx": index, "value_selections": one_user_selections}
                    )

                    # Store results periodically if storage_step is defined
                    if self._storage_step and (index + 1) % self._storage_step == 0:
                        self.append_to_file(
                            list_user_selections, self._direct_output_file_path
                        )
                        list_user_selections.clear()

                    pbar.update(1)

                # Store any remaining results
                if list_user_selections:
                    self.append_to_file(
                        list_user_selections, self._direct_output_file_path
                    )

        except Exception as e:
            logger.error(
                f"An error occurred in the value selection given user profile: {e}"
            )
            raise

    async def get_values_for_dialogue(self):
        list_user_selections = []
        try:
            with tqdm(
                total=len(self.generated_dialogues),
                desc="Generating Dialogues",
                unit="dialogue",
            ) as pbar:
                for dialogue_details in self.generated_dialogues:
                    user_index = dialogue_details["index"]
                    one_user_selections = []

                    generated_dialogue_runs = []
                    for run in dialogue_details["generated_dialogue"]:
                        generated_dialogue_runs += DialogueRun.model_validate(
                            run
                        ).convert_to_openai_history()

                    if self._verbose == 1:
                        logger.info(
                            f"Processing row {user_index}: {generated_dialogue_runs}"
                        )

                    for (
                        question_index,
                        question_row,
                    ) in self.dialogue_continue_value_questions.iterrows():
                        question_row_dict = question_row.to_dict()
                        (
                            full_question_str,
                            options_str,
                        ) = self._generate_question_options(question_row_dict)

                        (
                            selected_option_id,
                            normalized_probs,
                            reason_for_selection,
                        ) = await self._dialogue_continue_value_query(
                            dialogue_history=generated_dialogue_runs,
                            full_question=full_question_str,
                            options_str=options_str,
                        )
                        one_user_selections.append(
                            {
                                "question_idx": question_index,
                                "selected_option_id": selected_option_id,
                                "normalized_probs": normalized_probs,
                                "reason_for_selection": reason_for_selection,
                            }
                        )

                    list_user_selections.append(
                        {
                            "user_idx": user_index,
                            "value_selections": one_user_selections,
                        }
                    )

                    # Store results periodically if storage_step is defined
                    if self._storage_step and (user_index + 1) % self._storage_step == 0:
                        self.append_to_file(
                            list_user_selections, self._dialogue_output_file_path
                        )
                        list_user_selections.clear()

                    pbar.update(1)

                # Store any remaining results
                if list_user_selections:
                    self.append_to_file(
                        list_user_selections, self._dialogue_output_file_path
                    )

        except Exception as e:
            logger.error(
                f"An error occurred in the value selection given dialogue context: {e}"
            )
            raise

    def append_to_file(self, data, output_file_path):
        """Append data to the specified JSONL file."""
        with open(output_file_path, "a", encoding="utf-8") as jsonl_file:
            for entry in data:
                jsonl_file.write(json.dumps(entry) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = ValuesPredictionController.add_cli_args(parser=parser)
    values_prediction_args = parser.parse_args()
    prediction_controller = ValuesPredictionController.from_cli_args(
        args=values_prediction_args
    )
    values_for_user_profiles = asyncio.run(
        prediction_controller.get_values_for_user_profiles()
    )
    values_for_dialogue = asyncio.run(prediction_controller.get_values_for_dialogue())
