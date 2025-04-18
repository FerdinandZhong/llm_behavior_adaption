"""Generated Questions Evaluation
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

from persona_understanding.dialogue_dataset_creation.generation_utils import (
    render_template,
    retrieve_user_profile,
)

from persona_understanding.dialogue_dataset_creation.constant import PROFILE_TEMPLATE, PROFILE_KEYS, LLM_JUDGE_DICT


logger = logging.getLogger(__name__)


class Rating(BaseModel):
    rating: int
    reason: str


class GenerationJudgeController:
    """LLM Judge
    """

    def __init__(
        self,
        output_file_path: str,
        user_profile_dataset: pd.DataFrame,
        generated_dialogues: List[Dict],
        judge_dim_index: int,
        openai_client=None,
        verbose: int = 0,
        storage_step: int = None,
    ):
        """Initialization

        Args:
            output_file_path (str): Output file path
            user_profile_dataset (pd.DataFrame): DataFrame containing user profile data.
            generated_dialogues (List[Dict]): List of dictionaries representing generated dialogues.
            openai_client (optional): Client for interfacing with OpenAI API. Defaults to None.
            verbose (int, optional): Verbosity level for logging. Defaults to 0.
            storage_step (int, optional): Interval to store results to file. Defaults to None.
        """
        if not isinstance(output_file_path, str) or not output_file_path:
            raise ValueError("output_file_path must be a non-empty string.")
        if not isinstance(verbose, int) or verbose < 0:
            raise ValueError("verbose must be a non-negative integer.")
        if judge_dim_index not in [1,2,3,4]:
            raise ValueError("judge_dim_index must be in 1 to 4")

        self._user_profile_dataset = user_profile_dataset
        self._output_file_path = output_file_path
        self._judge_prompt_template = LLM_JUDGE_DICT[f"judge_dim_{judge_dim_index}"]
        self._verbose = verbose
        self._generated_dialogues = generated_dialogues
        self._storage_step = storage_step
        if openai_client is None:
            base_url = os.getenv("base_url", "http://localhost:8000/v1")
            self._openai_client = AsyncOpenAI(
                api_key=os.environ["api_key"], base_url=base_url
            )
        else:
            self._openai_client = openai_client
        # self._profile_keys = deepcopy(PROFILE_KEYS)
        # self._profile_keys.pop(4)

    @property
    def user_profile_dataset(self) -> pd.DataFrame:
        """
        Getter for the user profile dataset.

        Returns:
            pd.DataFrame: The user profile dataset.
        """
        return self._user_profile_dataset
    
    @property
    def output_file_path(self) -> str:
        """
        Getter for the output file path.

        Returns:
            str: The output file path.
        """
        return self._output_file_path

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
    def judge_prompt_template(self) -> AsyncOpenAI:
        """
        Getter for the Prompt Template

        Returns:
            The Template
        """
        return self._judge_prompt_template

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
            "--output_file_path",
            type=str,
            help="The path for the outputs",
            required=True,
        )
        parser.add_argument(
            "--judge-dim-index",
            type=int,
            choices=[1,2,3,4],
            help="Judge dim index to get the template"
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
        print(user_profile_dataset.head())

        generated_dialogues = []
        with open(args.dialogue_file, "r", encoding="utf-8") as dialogue_file:
            for dialogue in dialogue_file:
                dialogue_obj = json.loads(dialogue)
                # if dialogue_obj["index"] < args.starting_row:
                #     continue
                generated_dialogues.append(dialogue_obj)
                if len(generated_dialogues) >= args.ending_row:
                    break
        
        openai_client = AsyncOpenAI(
            api_key=args.openai_api_key, base_url=args.model_base_url
        )
        
        return cls(
            openai_client=openai_client,
            output_file_path=args.output_file_path,
            user_profile_dataset=user_profile_dataset,
            generated_dialogues=generated_dialogues,
            judge_dim_index=args.judge_dim_index,
            storage_step=args.storage_step,
            verbose=args.verbose,
        )
    
    def _llm_output_processing(self, full_chat_response):
        try:
            json_output = json.loads(full_chat_response.choices[0].message.content)
            return json_output
        except Exception:
            logger.warning(
                f"Error decoding as json: {full_chat_response.choices[0].message.content}"
            )
            return {
                "rating": 3,
                "reason": "Default value"
            }
    
    async def _judge_generated_questions(
        self, user_profile, questions_str, seed=1
    ):
        messages = deepcopy(self._judge_prompt_template)
        messages[1]["content"] = messages[1]["content"].format(user_details=user_profile)
        messages[2]["content"] = messages[2]["content"].format(question_str=questions_str)
        judge_response = await self.openai_client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=4096, # larger window
            seed=seed,
            # response_format={
            #     'type': 'json_object'
            # }
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "option_response",
                    "schema": Rating.model_json_schema(),
                },
            },
        )

        json_output = self._llm_output_processing(judge_response)
        json_output["seed"] = seed

        return json_output
    
    def _render_questions(self, dialogue_runs):
        dialogue_strs = []
        for idx, dialogue_run in enumerate(dialogue_runs):
            dialogue_str = f'Question {idx+1}: {dialogue_run["user_content"]}\n\n'
            dialogue_strs.append(dialogue_str)
        
        return "".join(dialogue_strs)

    async def get_judgements_for_generations(self):
        """Get judgements for all generated datasets

        """
        list_judgements = []
        try:
            with tqdm(
                total=len(self.user_profile_dataset),
                desc="Judgement",
                unit="dialogue",
            ) as pbar:
                for user_idx, row in self.user_profile_dataset.iterrows():
                    row_dict = row.to_dict()
                    one_profile_judgements = []

                    user_profile = render_template(
                        PROFILE_TEMPLATE, profile_data=retrieve_user_profile(row_dict)
                    )
                    dialogue_details = self.generated_dialogues[user_idx]["generated_dialogue"]

                    if self._verbose == 1:
                        logger.info(f"Processing row {user_idx}: {row_dict}")

                    list_kwargs = []
                    questions_str = self._render_questions(dialogue_details)
                    for seed_value in range(10):
                        list_kwargs.append(
                            {
                                "seed": seed_value,
                                "user_profile": user_profile,
                                "questions_str": questions_str
                            }
                        )
                    one_profile_judgements = await asyncio.gather(
                        *[
                            self._judge_generated_questions(**kwargs)
                            for kwargs in list_kwargs
                        ]
                    )

                    final_rating = sum([result["rating"] for result in one_profile_judgements])/len(one_profile_judgements)
                    list_judgements.append({
                        f"generation_{user_idx}": {
                            "overall_rating": final_rating,
                            "all_results": one_profile_judgements,
                        }
                    })

                    if (
                        self._storage_step
                        and user_idx % self._storage_step == 0
                    ):
                        self.append_to_file(
                            list_judgements, self.output_file_path
                        )
                        list_judgements.clear()

                    pbar.update(1)
            if len(list_judgements) > 0:
                self.append_to_file(
                    list_judgements, self.output_file_path
                )
                list_judgements.clear()
        except Exception as e:
            logger.error(
                f"An error occurred in the value selection given dialogue context: {e}"
            )
            raise e
    
    def append_to_file(self, data, output_file_path):
        """Append data to the specified JSONL file."""
        with open(output_file_path, "a", encoding="utf-8") as jsonl_file:
            for entry in data:
                jsonl_file.write(json.dumps(entry) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = GenerationJudgeController.add_cli_args(parser=parser)
    values_prediction_args = parser.parse_args()
    judge_controller = GenerationJudgeController.from_cli_args(
        args=values_prediction_args
    )
    values_for_user_profiles = asyncio.run(
        judge_controller.get_judgements_for_generations()
    )
