# dialogue_generation_controller.py
# YAML-first controller, CLI overrides, Turn-only JSONL output.
# Logging uses lazy formatting (no f-strings inside logger calls).

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm

from llm_behavior_adaptation.dialogue_dataset_creation.langgraph_dialogue_controller import DialogueAgent
from llm_behavior_adaptation.utils import register_logger

logger = logging.getLogger(__name__)
register_logger(logger)


def _load_yaml(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


class DatasetGenerationController:
    def __init__(
        self,
        dialogue_generator: DialogueAgent,
        seed_dataset: pd.DataFrame,
        output_file_path: str,
        verbose: int = 0,
        storage_step: Optional[int] = None,
    ):
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

    @property
    def dialogue_generator(self):
        return self._dialogue_generator

    @property
    def seed_dataset(self):
        return self._seed_dataset

    @property
    def output_file_path(self):
        return self._output_file_path

    # ---------------- CLI ----------------
    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser.add_argument(
            "--config",
            type=str,
            default=None,
            help="Path to YAML config (recommended). CLI flags override YAML.",
        )
        return parser

    # ---------------- Builders ----------------
    @classmethod
    def from_cli_args(cls, args: argparse.Namespace):
        # Base config from YAML
        yaml_cfg = _load_yaml(args.config)

        # Compose with CLI overrides
        cfg: Dict[str, Any] = dict(yaml_cfg)

        # Requireds
        if not cfg.get("seed_dataset_path"):
            raise ValueError("seed_dataset_path must be provided (YAML or CLI).")
        if not cfg.get("output_file_path"):
            raise ValueError("output_file_path must be provided (YAML or CLI).")
        if not cfg.get("prompts_folder"):
            raise ValueError("prompts_folder must be provided (YAML or CLI).")

        # Load dataset slice
        full_dataset = pd.read_csv(cfg["seed_dataset_path"])
        full_dataset = full_dataset.loc[:, ~full_dataset.columns.str.contains("^Unnamed")]
        start = int(cfg.get("starting_row", 0) or 0)
        end = cfg.get("ending_row", -1)
        if end is None or int(end) < 0:
            seed_dataset = full_dataset.iloc[start:]
        else:
            seed_dataset = full_dataset.iloc[start : int(end)]

        # Build OpenAI client
        api_key = cfg.get("openai_api_key") or os.environ.get("api_key") or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("Missing OpenAI API key (YAML 'openai_api_key' or env 'api_key'/'OPENAI_API_KEY').")
        openai_client = AsyncOpenAI(api_key=api_key)

        dialogue_generator = DialogueAgent(
            client=openai_client,
            prompts_folder=cfg["prompts_folder"],
            models=cfg["models"],
            params=cfg["params"],
            threshold=cfg["dialogue_runs_threshold"],
            verbose=int(cfg["verbose"]),
        )

        return cls(
            dialogue_generator=dialogue_generator,
            seed_dataset=seed_dataset,
            output_file_path=cfg["output_file_path"],
            verbose=int(cfg["verbose"]),
            storage_step=cfg.get("storage_step"),
        )

    # ---------------- Main ----------------
    async def generation_main(self):
        """
        Iterate seed rows, generate dialogues, and write JSONL.
        Each JSONL line is a list[Turn] in your canonical format:
        [{"speaker":"user"|"chatbot","text":"..."}]
        """
        pending_lines: List[Dict[str, List]] = []
        wrote = 0

        try:
            with tqdm(
                total=len(self._seed_dataset),
                desc="Generating Dialogues",
                unit="dialogue",
            ) as pbar:
                for index, row in self.seed_dataset.iterrows():
                    row_dict = row.to_dict()

                    if self._verbose == 1:
                        logger.info("Processing row %s: %s", index, row_dict)

                    try:
                        start_state = self.dialogue_generator.make_start_state_from_seed(seed_row=row_dict)
                        final_state = await self.dialogue_generator.run(start_state)

                        # runs is expected to be a list of Turn-like pydantic instances
                        history = final_state.get("history", [])
                        turns_serialized = [
                            (turn.model_dump() if hasattr(turn, "model_dump") else dict(turn)) for turn in history
                        ]
                        # Enforce Turn-only shape (speaker,text) in case extras exist
                        turns_openai_format = [{"role": t["speaker"], "content": t["text"]} for t in turns_serialized]
                        pending_lines.append(
                            json.dumps(
                                {row_dict["D_INTERVIEW"]: turns_openai_format},
                                ensure_ascii=False,
                            )
                        )

                        if self._verbose == 1:
                            if turns_openai_format:
                                first_turn_preview = "%s: %s..." % (
                                    turns_openai_format[0]["role"],
                                    turns_openai_format[0]["content"][:80],
                                )
                            else:
                                first_turn_preview = "EMPTY"
                            logger.info("Row %s OK. First turn: %s", index, first_turn_preview)

                        # Periodic flush
                        if self._storage_step and ((wrote + 1) % int(self._storage_step) == 0):
                            self._append_lines(pending_lines)
                            pending_lines.clear()
                        wrote += 1

                    except Exception as e:
                        logger.error("Error generating dialogue for row %s: %s", index, e)

                    pbar.update(1)

            # Final flush
            if pending_lines:
                self._append_lines(pending_lines)

            if self._verbose == 1:
                logger.info("Dialogue generation completed successfully.")

        except Exception as e:
            logger.error("An error occurred in the dialogue generation process: %s", e)
            raise

    def _append_lines(self, lines: List[str]):
        """Append raw JSONL lines (each already serialized)."""
        os.makedirs(os.path.dirname(self._output_file_path) or ".", exist_ok=True)
        with open(self._output_file_path, "a", encoding="utf-8") as f:
            for line in lines:
                f.write(line + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = DatasetGenerationController.add_cli_args(parser)
    args = parser.parse_args()
    controller = DatasetGenerationController.from_cli_args(args=args)
    asyncio.run(controller.generation_main())
