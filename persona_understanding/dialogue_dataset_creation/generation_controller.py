# The controller for controling the whole dialogue dataset generation process

import argparse
import asyncio
import json
import logging
import os

import pandas as pd
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm

from persona_understanding.dialogue_dataset_creation.constant import (
    DIALOGUE_RUNS_THRESHOLD,
)
from persona_understanding.dialogue_dataset_creation.dialogue_controller import (
    DialogueGenerator,
)

logger = logging.getLogger(__name__)


class DatasetGenerationController:
    def __init__(
        self,
        dialogue_generator,
        seed_dataset: pd.DataFrame,
        output_file_path: str,
        verbose: int = 0,
        storage_step: int = None,
    ):
        """
        Initialize the DatasetGenerationController.

        Args:
            dialogue_generator: An instance of the DialogueGenerator class.
            seed_dataset (pd.DataFrame): The seed dataset for dialogue generation.
            output_file_path (str): Path where the output will be saved.
            verbose (int): Verbosity level (0 = Errors only, 1 = Detailed logs). Defaults to 0.
            storage_step (int): Interval to store results to the file. Defaults to None.
        """
        if dialogue_generator is None:
            raise ValueError("dialogue_generator cannot be None.")
        if not isinstance(seed_dataset, pd.DataFrame):
            raise TypeError("seed_dataset must be a pandas DataFrame.")
        if not isinstance(output_file_path, str) or not output_file_path:
            raise ValueError("output_file_path must be a non-empty string.")
        if not isinstance(verbose, int) or verbose < 0:
            raise ValueError("verbose must be a non-negative integer.")

        self._dialogue_generator = dialogue_generator
        self._seed_dataset = seed_dataset
        self._output_file_path = output_file_path
        self._verbose = verbose
        self._storage_step = storage_step

    # Getter for dialogue_generator
    @property
    def dialogue_generator(self):
        return self._dialogue_generator

    # Getter for seed_dataset
    @property
    def seed_dataset(self):
        return self._seed_dataset

    # Getter for output_file_path
    @property
    def output_file_path(self):
        return self._output_file_path

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
            "--seed-dataset-path",
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
            "--user-simulator",
            type=str,
            default="gpt-4o",
            help="The model identifier for the user simulator. Defaults to 'gpt-4o'.",
        )
        parser.add_argument(
            "--chatbot",
            type=str,
            default="gpt-4o",
            help="The model identifier for the chatbot. Defaults to 'gpt-4o'.",
        )
        parser.add_argument(
            "--ooc-detector-name",
            type=str,
            default=None,
            help="The name of the out-of-character (OOC) detector. Defaults to None.",
        )
        parser.add_argument(
            "--ooc-detector-type",
            type=str,
            default="llm",
            help="The type of the out-of-character (OOC) detector. Defaults to None.",
        )
        parser.add_argument(
            "--user-simulator-generation-parameters",
            type=json.loads,
            default={},
            help="JSON string of parameters for user simulator generation. Defaults to an empty dictionary.",
        )
        parser.add_argument(
            "--chatbot-generation-parameters",
            type=json.loads,
            default={},
            help="JSON string of parameters for chatbot generation. Defaults to an empty dictionary.",
        )
        parser.add_argument(
            "--dialogue-runs-threshold",
            type=int,
            default=DIALOGUE_RUNS_THRESHOLD,
            help=f"Maximum number of dialogue runs before termination. Defaults to {DIALOGUE_RUNS_THRESHOLD}.",
        )
        parser.add_argument(
            "--output_file_path",
            type=str,
            help="The path for the output generated dialogues",
            required=True,
        )
        parser.add_argument(
            "--storage-step",
            type=int,
            default=None,
            help="Interval at which results are stored to the file.",
        )
        parser.add_argument(
            "--verbose",
            type=int,
            choices=[0, 1],
            default=0,
            help="Verbosity level: 0 = Errors only, 1 = Detailed logs. Defaults to 0.",
        )
        return parser

    def _get_ooc_detector(self, ooc_detector_name):
        # TODO: add a real detector
        return None

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
        full_dataset = pd.read_csv(args.seed_dataset_path)
        full_dataset = full_dataset.loc[
            :, ~full_dataset.columns.str.contains("^Unnamed")
        ]
        seed_dataset = full_dataset[args.starting_row : args.ending_row]

        openai_client = AsyncOpenAI(api_key=args.openai_api_key)
        # ooc_detector = cls._get_ooc_detector(args.ooc_detector_name)
        dialogue_generator = DialogueGenerator(
            user_simulator=args.user_simulator,
            chatbot=args.chatbot,
            ooc_detector=args.ooc_detector_name,
            ooc_detector_type=args.ooc_detector_type,
            openai_client=openai_client,
            user_simulator_generation_parameters=args.user_simulator_generation_parameters,
            chatbot_generation_parameters=args.chatbot_generation_parameters,
            dialogue_runs_threshold=args.dialogue_runs_threshold,
            verbose=args.verbose,
        )

        return cls(
            seed_dataset=seed_dataset,
            dialogue_generator=dialogue_generator,
            verbose=args.verbose,
            output_file_path=args.output_file_path,
            storage_step=args.storage_step,
        )

    async def generation_main(self):
        """
        Main method to generate dialogues for each row in the seed dataset, with progress tracking and logging.

        Tracks progress using `tqdm` and logs information based on verbosity settings:
        - Verbose 0: Only logs errors.
        - Verbose 1: Logs progress and summarizes each generated dialogue (truncated to 20 tokens).
        """
        all_generated_dialogues = []

        try:
            with tqdm(
                total=len(self._seed_dataset),
                desc="Generating Dialogues",
                unit="dialogue",
            ) as pbar:
                for index, row in self.seed_dataset.iterrows():
                    row_dict = row.to_dict()

                    if self._verbose == 1:
                        logger.info(f"Processing row {index}: {row_dict}")

                    # Generate dialogue for the current seed row
                    try:
                        generated_dialogue = (
                            await self.dialogue_generator.dialogue_generation(
                                seed_row=row_dict
                            )
                        )

                        all_generated_dialogues.append(
                            {
                                "index": index,
                                "generated_dialogue": [
                                    run.model_dump() for run in generated_dialogue
                                ],
                            }
                        )

                        if self._verbose == 1:
                            logger.info(
                                f"Generated dialogue for row {index}: "
                                f"{str(generated_dialogue)[:100]}..."  # Truncated to 100 chars
                            )

                        # Store results periodically if storage_step is defined
                        if self._storage_step and (index + 1) % self._storage_step == 0:
                            self._append_to_file(all_generated_dialogues)
                            all_generated_dialogues.clear()

                    except Exception as e:
                        logger.error(f"Error generating dialogue for row {index}: {e}")

                    # Update progress bar
                    pbar.update(1)

            # Store any remaining results
            if all_generated_dialogues:
                self._append_to_file(all_generated_dialogues)

            if self._verbose == 1:
                logger.info("Dialogue generation completed successfully.")

        except Exception as e:
            logger.error(f"An error occurred in the dialogue generation process: {e}")
            raise

    def _append_to_file(self, data):
        """Append data to the specified JSONL file."""
        with open(self._output_file_path, "a", encoding="utf-8") as jsonl_file:
            for entry in data:
                jsonl_file.write(json.dumps(entry) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = DatasetGenerationController.add_cli_args(parser)
    generation_args = parser.parse_args()
    controller = DatasetGenerationController.from_cli_args(args=generation_args)
    generated_dialogues = asyncio.run(controller.generation_main())
