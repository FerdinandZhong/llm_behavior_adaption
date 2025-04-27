"""values prediction
"""

import argparse
import asyncio
import json
import logging
import math
import os
from copy import deepcopy
from functools import partial
from typing import Dict, List, Union

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
    EXTRA_FORMAT,
    OPTIONS_TEMPLATE,
    PROFILE_TEMPLATE,
)

logger = logging.getLogger(__name__)


class Response(BaseModel):
    option_id: int
    reason: str


class QuestionnaireOutput(BaseModel):
    question_index: int
    selected_option_id: int
    normalized_probs: List[float]
    log_probs: Dict
    reason_for_selection: Union[str, None] = None


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
        llm_server: str = "llm_platform",
        reasoning: bool = False,
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
            llm_server (str, optional): llm_server, gpt, vllm, sglang
            reasoning (bool, optional): whether the tested model is a reasoning model.
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
        self._reasoning = reasoning
        if openai_client is None:
            if "gpt" in evaluated_model:
                self._openai_client = AsyncOpenAI(api_key=os.environ["api_key"])
            else:
                base_url = os.getenv("base_url", "http://localhost:8000/v1")
                self._openai_client = AsyncOpenAI(
                    api_key=os.environ["api_key"], base_url=base_url
                )
        else:
            self._openai_client = openai_client

        self.prompt_append_format = False
        if llm_server == "llm_platform":
            self.query_llm = partial(
                self.openai_client.chat.completions.create,
                model=self.evaluated_model,
                response_format={"type": "json_object"},
                logprobs=True,
                top_logprobs=5,
            )
            self.prompt_append_format = True
        elif llm_server == "gpt" or llm_server == "sglang":
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
                logprobs=True,
                top_logprobs=5,
            )
            self.prompt_append_format = True
        elif llm_server == "vllm":
            # use vllm
            self.query_llm = partial(
                self.openai_client.chat.completions.create,
                model=self.evaluated_model,
                logprobs=True,
                top_logprobs=5,
                response_format={"type": "json_object"},
                # extra_body={"guided_json": Response.model_json_schema()},
            )
            self.prompt_append_format = True
        else:
            raise ValueError("invalid llm server type")
        self.llm_server = llm_server

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

    @property
    def reasoning(self) -> bool:
        """Gettter for the reasoning attribute"""
        return self._reasoning

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
            "--model-base-url", type=str, default=os.environ.get("base_url")
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
            "--llm_server",
            type=str,
            help="The type of llm_server",
            default="llm_platform",
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
        parser.add_argument(
            "--reasoning",
            action="store_true",
            help="Define doing the reasoning",
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
                dialogue_obj = json.loads(dialogue)
                if dialogue_obj["index"] < args.starting_row:
                    continue
                generated_dialogues.append(dialogue_obj)
                if len(generated_dialogues) >= args.ending_row:
                    break

        if "gpt" in args.evaluated_model:
            openai_client = AsyncOpenAI(api_key=args.openai_api_key)
        else:
            openai_client = AsyncOpenAI(
                api_key=args.openai_api_key, base_url=args.model_base_url
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
            llm_server=args.llm_server,
            verbose=args.verbose,
            reasoning=args.reasoning,
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

    def _llm_output_processing(self, full_chat_response, reasoning=None):
        try:
            json_output = json.loads(full_chat_response.choices[0].message.content)
        except Exception:
            logger.warning(
                f"Error decoding as json: {full_chat_response.choices[0].message.content}"
            )
            return (
                0,
                [0, 0, 0, 0, 0],
                {"1": 0, "2": 0, "3": 0, "4": 0, "5": 0},
                "Response un-decodable",
            )
        try:
            selected_option_id = json_output["option_id"]
            reason_for_selection = json_output["reason"]
            if reasoning is not None:
                reason_for_selection = (
                    f"<think>{reasoning}</think>\n" + reason_for_selection
                )
            option_id_logprobs = None
            for token_obj in full_chat_response.choices[0].logprobs.content:
                if token_obj.token == str(selected_option_id):
                    option_id_logprobs = token_obj.top_logprobs
            option_id_logprobs_dict = {}
            if option_id_logprobs is None:
                logger.warning(f"{str(selected_option_id)} not in map")
                normalized_probs = [0, 0, 0, 0, 0]  # invalid probs
            for prob_item in option_id_logprobs:
                try:
                    option_id_logprobs_dict[int(prob_item.token.strip())] = (
                        prob_item.logprob
                    )
                except ValueError as e:
                    logger.warning(
                        f"Can't have {prob_item.token} casted into int: {str(e)}"
                    )
                    option_id_logprobs_dict[prob_item.token] = prob_item.logprob
            normalized_probs = self._normalize_logprobs(
                option_id_logprobs_dict, DEFAULT_OPTION_IDS
            )
        except Exception:
            logger.warning(
                f"Error decoding as json: {full_chat_response.choices[0].message.content}"
            )
            return (
                0,
                [0, 0, 0, 0, 0],
                {"1": 0, "2": 0, "3": 0, "4": 0, "5": 0},
                "Response un-decodable",
            )

        return (
            selected_option_id,
            normalized_probs,
            option_id_logprobs_dict,
            reason_for_selection,
        )

    async def _direct_value_query(
        self, question_index, user_profile, full_question, options_str
    ):
        direct_value_selection_prompt = deepcopy(DIRECT_VALUE_SELECTION_PROMPT)
        direct_value_selection_prompt[1]["content"] = direct_value_selection_prompt[1][
            "content"
        ].format(user_details=user_profile)
        direct_value_selection_prompt[2]["content"] = direct_value_selection_prompt[2][
            "content"
        ].format(question=full_question, option_list=options_str)

        if self.reasoning:
            reasoning_response = await self.openai_client.chat.completions.create(
                model=self.evaluated_model,
                messages=direct_value_selection_prompt,
                temperature=0.6,  # default setting for reasoning model
                max_tokens=4096,  # larger window
            )

            reasoning_output = reasoning_response.choices[0].message.content.split(
                "</think>"
            )[0]

            direct_value_selection_prompt.append(
                {
                    "role": "assistant",
                    "content": f"<think>\n{reasoning_output}\n</think>\n",
                }
            )

            # direct_value_selection_prompt.append({
            #     "role": "user",
            #     "content": "Select the option for the question, given the reasoning."
            # })
        else:
            reasoning_output = None

        if self.prompt_append_format:
            direct_value_selection_prompt.append(EXTRA_FORMAT)

        full_chat_response = await self.query_llm(
            messages=direct_value_selection_prompt
        )

        (
            selected_option_id,
            normalized_probs,
            option_id_logprobs_dict,
            reason_for_selection,
        ) = self._llm_output_processing(full_chat_response, reasoning=reasoning_output)

        return QuestionnaireOutput(
            question_index=question_index,
            selected_option_id=selected_option_id,
            normalized_probs=normalized_probs,
            log_probs=option_id_logprobs_dict,
            reason_for_selection=reason_for_selection,
        )

    async def _dialogue_continue_value_query(
        self, question_index, dialogue_history, full_question, options_str
    ):
        dialogue_continue_prompt = deepcopy(CONVERSATION_HISTORY_PROMPT)
        dialogue_based_msgs = deepcopy(dialogue_history)
        dialogue_based_msgs.append(dialogue_continue_prompt[0])
        dialogue_based_msgs.append(dialogue_continue_prompt[1])
        dialogue_continue_prompt[2]["content"] = dialogue_continue_prompt[2][
            "content"
        ].format(question=full_question, option_list=options_str)
        dialogue_based_msgs.append(dialogue_continue_prompt[2])

        if self.reasoning:
            reasoning_response = await self.openai_client.chat.completions.create(
                model=self.evaluated_model,
                messages=dialogue_based_msgs,
                temperature=0.6,  # default setting for reasoning model
                max_tokens=4096,  # larger window
            )

            reasoning_output = reasoning_response.choices[0].message.content.split(
                "</think>"
            )[0]

            dialogue_based_msgs.append(
                {
                    "role": "assistant",
                    "content": f"<think>\n{reasoning_output}\n</think>\n",
                }
            )
        else:
            reasoning_output = None

        if self.prompt_append_format:
            dialogue_based_msgs.append(EXTRA_FORMAT)

        full_chat_response = await self.query_llm(messages=dialogue_based_msgs)

        (
            selected_option_id,
            normalized_probs,
            option_id_logprobs_dict,
            reason_for_selection,
        ) = self._llm_output_processing(full_chat_response, reasoning=reasoning_output)

        return QuestionnaireOutput(
            question_index=question_index,
            selected_option_id=selected_option_id,
            normalized_probs=normalized_probs,
            log_probs=option_id_logprobs_dict,
            reason_for_selection=reason_for_selection,
        )

    def _generate_question_options(self, question_row_dict):
        full_question_str = deepcopy(question_row_dict["full_question"]).format(
            question=question_row_dict["questions"]
        )

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
                desc="Generating values output",
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

                    list_kwargs = []

                    for (
                        question_index,
                        question_row,
                    ) in self.direct_value_questions.iterrows():
                        question_row_dict = question_row.to_dict()
                        (
                            full_question_str,
                            options_str,
                        ) = self._generate_question_options(question_row_dict)

                        list_kwargs.append(
                            {
                                "question_index": question_index,
                                "user_profile": user_profile,
                                "full_question": full_question_str,
                                "options_str": options_str,
                            }
                        )

                    one_user_selections = await asyncio.gather(
                        *[self._direct_value_query(**kwargs) for kwargs in list_kwargs]
                    )

                    list_user_selections.append(
                        {
                            "user_idx": index,
                            "value_selections": [
                                each_question.model_dump()
                                for each_question in one_user_selections
                            ],
                        }
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
                desc="Generation Dialogue based values",
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

                    list_kwargs = []

                    for (
                        question_index,
                        question_row,
                    ) in self.dialogue_continue_value_questions.iterrows():
                        question_row_dict = question_row.to_dict()
                        (
                            full_question_str,
                            options_str,
                        ) = self._generate_question_options(question_row_dict)

                        list_kwargs.append(
                            {
                                "question_index": question_index,
                                "dialogue_history": generated_dialogue_runs,
                                "full_question": full_question_str,
                                "options_str": options_str,
                            }
                        )

                    one_user_selections = await asyncio.gather(
                        *[
                            self._dialogue_continue_value_query(**kwargs)
                            for kwargs in list_kwargs
                        ]
                    )

                    list_user_selections.append(
                        {
                            "user_idx": user_index,
                            "value_selections": [
                                each_question.model_dump()
                                for each_question in one_user_selections
                            ],
                        }
                    )

                    # Store results periodically if storage_step is defined
                    if (
                        self._storage_step
                        and (user_index + 1) % self._storage_step == 0
                    ):
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


async def main():
    parser = argparse.ArgumentParser()
    parser = ValuesPredictionController.add_cli_args(parser=parser)
    values_prediction_args = parser.parse_args()
    prediction_controller = ValuesPredictionController.from_cli_args(
        args=values_prediction_args
    )
    
    values_for_user_profiles = await prediction_controller.get_values_for_user_profiles()
    values_for_dialogue = await prediction_controller.get_values_for_dialogue()

    # You can print or return these if needed
    print(values_for_user_profiles)
    print(values_for_dialogue)


if __name__ == "__main__":
    asyncio.run(main())