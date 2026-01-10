"""
Translation controller for dialogue datasets.

This controller orchestrates the translation of generated dialogues based on YAML configuration.
Supports both career and investment topic dialogues.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm

from llm_behavior_adaptation.dialogue_dataset_creation.dialogue_translator import (
    DialogueTranslator,
    load_user_profiles_from_csv,
)
from llm_behavior_adaptation.utils import register_logger

logger = logging.getLogger(__name__)
register_logger(logger)


def _load_yaml(path: Optional[str]) -> Dict[str, Any]:
    """Load YAML configuration file."""
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


class TranslationController:
    """
    Controller for translating dialogue datasets with user profile-based language selection.
    """

    def __init__(
        self,
        translator: DialogueTranslator,
        input_file_path: str,
        output_file_path: str,
        user_profiles: Optional[Dict[int, Dict[str, Any]]] = None,
        verbose: int = 0,
        max_dialogues: Optional[int] = None,
        batch_size: int = 1,
        starting_row: int = 0,
        ending_row: Optional[int] = None,
    ):
        """
        Initialize TranslationController.

        Args:
            translator: DialogueTranslator instance
            input_file_path: Path to input JSONL file with dialogues
            output_file_path: Path to output JSONL file for translated dialogues
            user_profiles: Optional mapping of dialogue index to user profile
            verbose: Verbosity level for logging
            max_dialogues: Optional limit on number of dialogues to translate
            batch_size: Number of dialogues to process before flushing to disk
            starting_row: Row index to start from (0-indexed, inclusive)
            ending_row: Row index to end at (exclusive, None means till end)
        """
        if not isinstance(input_file_path, str) or not input_file_path:
            raise ValueError("input_file_path must be a non-empty string.")
        if not isinstance(output_file_path, str) or not output_file_path:
            raise ValueError("output_file_path must be a non-empty string.")
        if not isinstance(verbose, int) or verbose < 0:
            raise ValueError("verbose must be a non-negative integer.")

        self._translator = translator
        self._input_file_path = input_file_path
        self._output_file_path = output_file_path
        self._user_profiles = user_profiles or {}
        self._verbose = verbose
        self._max_dialogues = max_dialogues
        self._batch_size = batch_size
        self._starting_row = starting_row
        self._ending_row = ending_row

    @property
    def translator(self):
        return self._translator

    @property
    def input_file_path(self):
        return self._input_file_path

    @property
    def output_file_path(self):
        return self._output_file_path

    # ---------------- CLI ----------------
    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """Add CLI arguments for translation controller."""
        parser.add_argument(
            "--config",
            type=str,
            default=None,
            help="Path to YAML config (recommended). CLI flags override YAML.",
        )
        parser.add_argument(
            "--input",
            type=str,
            help="Override input dialogue file path",
        )
        parser.add_argument(
            "--output",
            type=str,
            help="Override output translated file path",
        )
        parser.add_argument(
            "--profiles",
            type=str,
            help="Override user profiles CSV path",
        )
        parser.add_argument(
            "--max",
            type=int,
            help="Override max dialogues to translate",
        )
        parser.add_argument(
            "--verbose",
            type=int,
            help="Override verbosity level",
        )
        parser.add_argument(
            "--starting-row",
            type=int,
            help="Starting row index (0-indexed, inclusive)",
        )
        parser.add_argument(
            "--ending-row",
            type=int,
            help="Ending row index (exclusive, omit for end of file)",
        )
        return parser

    # ---------------- Builders ----------------
    @classmethod
    def from_cli_args(cls, args: argparse.Namespace):
        """Build TranslationController from CLI arguments and YAML config."""
        # Load base config from YAML
        yaml_cfg = _load_yaml(args.config)

        # Compose with CLI overrides
        cfg: Dict[str, Any] = dict(yaml_cfg)

        # CLI overrides
        if args.input:
            cfg["input_file_path"] = args.input
        if args.output:
            cfg["output_file_path"] = args.output
        if args.profiles:
            cfg["user_profiles_path"] = args.profiles
        if args.max is not None:
            cfg["max_dialogues"] = args.max
        if args.verbose is not None:
            cfg["verbose"] = args.verbose
        if hasattr(args, "starting_row") and args.starting_row is not None:
            cfg["starting_row"] = args.starting_row
        if hasattr(args, "ending_row") and args.ending_row is not None:
            cfg["ending_row"] = args.ending_row

        # Validate required fields
        if not cfg.get("input_file_path"):
            raise ValueError("input_file_path must be provided (YAML or --input).")
        if not cfg.get("output_file_path"):
            raise ValueError("output_file_path must be provided (YAML or --output).")

        # Load user profiles if provided
        user_profiles = None
        if cfg.get("user_profiles_path"):
            profiles_path = cfg["user_profiles_path"]
            logger.info("Loading user profiles from: %s", profiles_path)
            user_profiles = load_user_profiles_from_csv(profiles_path)
            logger.info("Loaded %d user profiles", len(user_profiles))

        # Build OpenAI client
        api_key = (
            cfg.get("openai_api_key")
            or os.environ.get("OPENAI_API_KEY")
        )
        if not api_key:
            raise RuntimeError(
                "Missing OpenAI API key (YAML 'openai_api_key' or env 'api_key'/'OPENAI_API_KEY')."
            )
        openai_client = AsyncOpenAI(api_key=api_key)

        # Build translator
        translator = DialogueTranslator(
            client=openai_client,
            model=cfg.get("translation_model", "gpt-4.1-mini"),
            temperature=cfg.get("translation_temperature", 0.3),
            verbose=int(cfg.get("verbose", 0)),
        )

        return cls(
            translator=translator,
            input_file_path=cfg["input_file_path"],
            output_file_path=cfg["output_file_path"],
            user_profiles=user_profiles,
            verbose=int(cfg.get("verbose", 0)),
            max_dialogues=cfg.get("max_dialogues"),
            batch_size=int(cfg.get("batch_size", 1)),
            starting_row=int(cfg.get("starting_row", 0)),
            ending_row=cfg.get("ending_row"),
        )

    @staticmethod
    def _normalize_dialogue_format(dialogue_data: Dict[str, Any], index: int) -> Dict[str, Any]:
        """
        Normalize dialogue format to expected structure.

        Handles two formats:
        1. Standard: {"index": 0, "generated_dialogue": [{"user_content": "...", "chatbot_content": "..."}]}
        2. WVS format: {"688070395": [{"role": "user", "content": "..."}, {"role": "chatbot", "content": "..."}]}

        Returns normalized format.
        """
        # Check if already in standard format
        if "index" in dialogue_data and "generated_dialogue" in dialogue_data:
            return dialogue_data

        # Handle WVS format: dictionary with single key (interview ID)
        if len(dialogue_data) == 1:
            interview_id = list(dialogue_data.keys())[0]
            turns = dialogue_data[interview_id]

            # Convert role/content format to user_content/chatbot_content pairs
            generated_dialogue = []
            i = 0
            while i < len(turns):
                # Expect pairs of user/chatbot turns
                if i + 1 < len(turns):
                    user_turn = turns[i]
                    chatbot_turn = turns[i + 1]

                    if user_turn.get("role") == "user" and chatbot_turn.get("role") in ["chatbot", "assistant"]:
                        generated_dialogue.append({
                            "user_content": user_turn.get("content", ""),
                            "chatbot_content": chatbot_turn.get("content", "")
                        })
                        i += 2
                    else:
                        # Skip malformed turn
                        i += 1
                else:
                    # Odd number of turns, skip last one
                    break

            return {
                "index": index,
                "generated_dialogue": generated_dialogue,
                "interview_id": interview_id
            }

        # Fallback: return as-is
        return dialogue_data

    # ---------------- Main ----------------
    async def translation_main(self):
        """
        Main translation loop.
        Reads dialogues from input JSONL, translates them, and writes to output JSONL.
        """
        input_path = Path(self._input_file_path)
        output_path = Path(self._output_file_path)

        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        # Create output directory if needed
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Count total dialogues in file
        total_in_file = sum(1 for _ in open(input_path, "r", encoding="utf-8"))

        # Calculate range to process
        start_idx = self._starting_row
        if self._ending_row is not None:
            end_idx = min(self._ending_row, total_in_file)
        else:
            end_idx = total_in_file

        # Apply max_dialogues limit if specified
        if self._max_dialogues:
            end_idx = min(start_idx + self._max_dialogues, end_idx)

        total_dialogues = end_idx - start_idx

        logger.info("Total dialogues in file: %d", total_in_file)
        logger.info("Processing range: [%d, %d) (%d dialogues)", start_idx, end_idx, total_dialogues)
        logger.info("Input: %s", input_path)
        logger.info("Output: %s", output_path)

        translated_count = 0
        pending_lines: List[str] = []

        try:
            with open(input_path, "r", encoding="utf-8") as f_in:
                with tqdm(
                    total=total_dialogues,
                    desc="Translating Dialogues",
                    unit="dialogue",
                    initial=0,
                ) as pbar:
                    for i, line in enumerate(f_in):
                        # Skip rows before starting_row
                        if i < start_idx:
                            continue

                        # Stop at ending_row
                        if i >= end_idx:
                            break

                        try:
                            raw_dialogue_data = json.loads(line)

                            # Normalize format
                            dialogue_data = self._normalize_dialogue_format(raw_dialogue_data, i)
                            idx = dialogue_data.get("index", i)

                            # Extract interview ID for profile matching if available
                            interview_id = dialogue_data.get("interview_id")
                            user_profile = None

                            if interview_id:
                                # Try interview_id as-is first
                                user_profile = self._user_profiles.get(interview_id)

                                # If not found and it's a string, try converting to int
                                if not user_profile and isinstance(interview_id, str):
                                    try:
                                        int_id = int(interview_id)
                                        user_profile = self._user_profiles.get(int_id)
                                    except (ValueError, TypeError):
                                        pass

                            # Fallback to index-based matching
                            if not user_profile:
                                user_profile = self._user_profiles.get(idx)

                            if self._verbose >= 1:
                                logger.info("Translating dialogue %d", idx)

                            # Translate dialogue
                            translated = await self.translator.translate_dialogue(
                                dialogue_data=dialogue_data,
                                user_profile=user_profile,
                            )

                            # Serialize and add to pending
                            pending_lines.append(translated.model_dump_json())
                            translated_count += 1

                            # Batch flush
                            if len(pending_lines) >= self._batch_size:
                                self._append_lines(output_path, pending_lines)
                                pending_lines.clear()

                        except Exception as e:
                            logger.error("Error translating dialogue %d: %s", i, e)
                            if self._verbose >= 2:
                                import traceback
                                traceback.print_exc()

                        pbar.update(1)

            # Final flush
            if pending_lines:
                self._append_lines(output_path, pending_lines)

            logger.info(
                "Translation completed successfully. Translated %d dialogues.",
                translated_count,
            )

        except Exception as e:
            logger.error("An error occurred in the translation process: %s", e)
            raise

    def _append_lines(self, output_path: Path, lines: List[str]):
        """Append JSONL lines to output file."""
        with open(output_path, "a", encoding="utf-8") as f:
            for line in lines:
                f.write(line + "\n")


def create_sample_config(
    output_path: str,
    topic: str = "career",
):
    """
    Create a sample YAML configuration file for translation.

    Args:
        output_path: Path where to save the config file
        topic: Topic type (career or investment)
    """
    sample_config = {
        "# Translation Configuration": None,
        "": None,
        "# Input/Output paths": None,
        "input_file_path": f"datasets/generated_dialogues/{topic}_dialogues.jsonl",
        "output_file_path": f"datasets/translated_dialogues/{topic}_translated.jsonl",
        " ": None,
        "# User profiles for language selection": None,
        "user_profiles_path": "datasets/wvs_benchmarks/sampled_demographic_features.csv",
        "  ": None,
        "# Translation settings": None,
        "translation_model": "gpt-4.1-mini",
        "translation_temperature": 0.3,
        "   ": None,
        "# Processing options": None,
        "max_dialogues": None,  # null means translate all
        "batch_size": 10,
        "verbose": 1,
        "    ": None,
        "# OpenAI API key (or use environment variable)": None,
        "openai_api_key": "${OPENAI_API_KEY}",
    }

    # Remove comment-only keys for actual YAML
    clean_config = {k: v for k, v in sample_config.items() if not k.startswith("#")}

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(clean_config, f, default_flow_style=False, allow_unicode=True)

    logger.info("Sample config created at: %s", output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Translate dialogue datasets based on user profiles"
    )
    parser.add_argument(
        "--create-config",
        type=str,
        help="Create a sample YAML config file at the specified path",
    )
    parser.add_argument(
        "--topic",
        type=str,
        choices=["career", "investment"],
        default="career",
        help="Topic type for sample config",
    )

    # Add translation controller args
    parser = TranslationController.add_cli_args(parser)

    args = parser.parse_args()

    # Handle config creation
    if args.create_config:
        create_sample_config(args.create_config, args.topic)
    else:
        # Run translation
        controller = TranslationController.from_cli_args(args=args)
        asyncio.run(controller.translation_main())
