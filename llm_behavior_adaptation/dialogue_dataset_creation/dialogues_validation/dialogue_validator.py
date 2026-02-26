"""
Dialogue Validator - LLM-based evaluation of generated dialogues.

This module provides functionality to evaluate generated dialogues using
configurable LLM models against a five-criterion scoring rubric.

Supports two execution modes:
1. Real-time API calls (default): Sequential/parallel async requests
2. Batch API (OpenAI): Submit all requests as a batch job for 50% cost savings

Supports configuration via:
- Command-line arguments
- YAML/JSON configuration file (--config)
- Environment variables for API credentials

Priority order (highest to lowest):
1. CLI arguments
2. Config file values
3. Environment variables
4. Default values

Example configuration file (YAML):
```yaml
# Required paths
user_profile_dataset: "path/to/user_profiles.csv"
dialogue_file: "path/to/dialogues.jsonl"
output_file_path: "path/to/validation_results.jsonl"

# API configuration
openai_api_key: "sk-..."  # Or use OPENAI_API_KEY env var
model_base_url: "https://api.openai.com/v1"  # Optional
model_name: "gpt-4.1"

# Evaluation settings
dialogue_topic: "career advice"  # or "investment advice"
evaluation_mode: "combined"  # or "separate"
num_seeds: 3

# Batch API settings (OpenAI only)
use_batch_api: true  # Use OpenAI Batch API for 50% cost savings
batch_poll_interval: 30  # Seconds between status checks

# Dataset range
starting_row: 0
ending_row: -1  # -1 for all rows

# Output settings
verbose: 0  # 0 or 1
storage_step: 50
```

Usage:
    python dialogue_validator.py --config config.yaml
    python dialogue_validator.py --config config.yaml --model-name gpt-4.1 --use-batch-api
"""

import argparse
import asyncio
import json
import logging
import os
import re
import tempfile
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import yaml
from openai import AsyncOpenAI, OpenAI
from pydantic import BaseModel
from tqdm.asyncio import tqdm

from llm_behavior_adaptation.dialogue_dataset_creation.dialogues_validation.validation_constants import (
    SCORE_RANGES,
    VALIDATION_CRITERIA,
)

logger = logging.getLogger(__name__)


def load_config_file(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a YAML or JSON file.

    Args:
        config_path: Path to the configuration file (.yaml, .yml, or .json)

    Returns:
        Dictionary containing configuration values

    Raises:
        ValueError: If file format is not supported
        FileNotFoundError: If config file does not exist
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    suffix = config_path.suffix.lower()

    with open(config_path, "r", encoding="utf-8") as f:
        if suffix in [".yaml", ".yml"]:
            config = yaml.safe_load(f)
        elif suffix == ".json":
            config = json.load(f)
        else:
            raise ValueError(f"Unsupported config file format: {suffix}. Use .yaml, .yml, or .json")

    return config or {}


class ValidationScore(BaseModel):
    """Pydantic model for validation scores."""

    coverage: int
    correctness: int
    diversity: int
    relevance: int
    naturalness: int
    reasoning: Optional[Dict[str, str]] = None


class DialogueValidator:
    """
    LLM-based dialogue quality validator.

    Evaluates generated dialogues against a five-criterion rubric:
    1. Attribute Coverage (0-5)
    2. Attribute Correctness (0-5)
    3. Question Diversity (1-5)
    4. Conversational Relevance (1-5)
    5. Naturalness (1-5)
    """

    def __init__(
        self,
        output_file_path: str,
        user_profile_dataset: pd.DataFrame,
        generated_dialogues: List[Dict],
        model_name: str = "gpt-4.1",
        dialogue_topic: str = "career advice",
        evaluation_mode: str = "combined",
        openai_client: Optional[AsyncOpenAI] = None,
        openai_sync_client: Optional[OpenAI] = None,
        num_seeds: int = 3,
        verbose: int = 0,
        storage_step: Optional[int] = None,
        use_seed: bool = True,
        use_batch_api: bool = False,
        batch_poll_interval: int = 30,
        request_delay: float = 0.0,
    ):
        """
        Initialize the DialogueValidator.

        Args:
            output_file_path: Path to save validation results (JSONL format)
            user_profile_dataset: DataFrame containing user profile data
            generated_dialogues: List of generated dialogue dictionaries
            model_name: LLM model to use for evaluation (e.g., "gpt-4.1", "gpt-4o")
            dialogue_topic: Topic of dialogues (e.g., "career advice", "investment advice")
            evaluation_mode: "combined" (single call) or "separate" (per-criterion calls)
            openai_client: Optional pre-configured AsyncOpenAI client (for real-time mode)
            openai_sync_client: Optional pre-configured OpenAI client (for batch mode)
            num_seeds: Number of evaluation seeds for averaging scores
            verbose: Verbosity level (0=errors only, 1=detailed logs)
            storage_step: Interval to flush results to file
            use_seed: Whether to use seed parameter in API calls (not supported by all models)
            use_batch_api: Whether to use OpenAI Batch API (50% cost savings)
            batch_poll_interval: Seconds between batch status checks
            request_delay: Delay in seconds between API requests (for rate limiting)
        """
        if not isinstance(output_file_path, str) or not output_file_path:
            raise ValueError("output_file_path must be a non-empty string.")
        if evaluation_mode not in ["combined", "separate"]:
            raise ValueError("evaluation_mode must be 'combined' or 'separate'")

        self._output_file_path = output_file_path
        self._user_profile_dataset = user_profile_dataset
        self._generated_dialogues = generated_dialogues
        self._model_name = model_name
        self._dialogue_topic = dialogue_topic
        self._evaluation_mode = evaluation_mode
        self._num_seeds = num_seeds
        self._verbose = verbose
        self._storage_step = storage_step
        self._use_seed = use_seed
        self._use_batch_api = use_batch_api
        self._batch_poll_interval = batch_poll_interval
        self._request_delay = request_delay

        api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("api_key")
        base_url = os.environ.get("OPENAI_BASE_URL") or os.environ.get("base_url")

        if openai_client is None:
            self._openai_client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        else:
            self._openai_client = openai_client

        # Synchronous client for batch API operations
        if openai_sync_client is None:
            self._openai_sync_client = OpenAI(api_key=api_key, base_url=base_url)
        else:
            self._openai_sync_client = openai_sync_client

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def output_file_path(self) -> str:
        return self._output_file_path

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """Add command-line arguments for DialogueValidator."""
        parser.add_argument(
            "--config",
            type=str,
            default=None,
            help="Path to configuration file (YAML or JSON). CLI args override config values.",
        )
        parser.add_argument(
            "--openai-api-key",
            type=str,
            default=None,
            help="OpenAI API key (default: from OPENAI_API_KEY env var)",
        )
        parser.add_argument(
            "--model-base-url",
            type=str,
            default=None,
            help="Base URL for OpenAI-compatible API (default: from OPENAI_BASE_URL env var)",
        )
        parser.add_argument(
            "--model-name",
            type=str,
            default=None,
            help="LLM model name for evaluation (e.g., gpt-4o-mini, gpt-4o, claude-3-sonnet)",
        )
        parser.add_argument(
            "--user-profile-dataset",
            type=str,
            default=None,
            help="Path to user profile CSV file",
        )
        parser.add_argument(
            "--dialogue-file",
            type=str,
            default=None,
            help="Path to generated dialogues JSONL file",
        )
        parser.add_argument(
            "--output-file-path",
            type=str,
            default=None,
            help="Output path for validation results (JSONL)",
        )
        parser.add_argument(
            "--dialogue-topic",
            type=str,
            default=None,
            choices=["career advice", "investment advice"],
            help="Topic of the dialogues being evaluated",
        )
        parser.add_argument(
            "--evaluation-mode",
            type=str,
            default=None,
            choices=["combined", "separate"],
            help="Evaluation mode: 'combined' (single LLM call) or 'separate' (per-criterion)",
        )
        parser.add_argument(
            "--num-seeds",
            type=int,
            default=None,
            help="Number of evaluation seeds for score averaging",
        )
        parser.add_argument(
            "--starting-row",
            type=int,
            default=None,
            help="Starting row index in the dataset",
        )
        parser.add_argument(
            "--ending-row",
            type=int,
            default=None,
            help="Ending row index (-1 for all)",
        )
        parser.add_argument(
            "--verbose",
            type=int,
            choices=[0, 1],
            default=None,
            help="Verbosity level",
        )
        parser.add_argument(
            "--storage-step",
            type=int,
            default=None,
            help="Interval to flush results to file",
        )
        parser.add_argument(
            "--use-seed",
            type=lambda x: x.lower() in ("true", "1", "yes"),
            default=None,
            help="Use seed parameter in API calls (default: true, set to false for Gemini)",
        )
        parser.add_argument(
            "--use-batch-api",
            type=lambda x: x.lower() in ("true", "1", "yes"),
            default=None,
            help="Use OpenAI Batch API for 50%% cost savings (default: false)",
        )
        parser.add_argument(
            "--batch-poll-interval",
            type=int,
            default=None,
            help="Seconds between batch status checks (default: 30)",
        )
        parser.add_argument(
            "--request-delay",
            type=float,
            default=None,
            help="Delay in seconds between API requests for rate limiting (default: 0)",
        )
        return parser

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace) -> "DialogueValidator":
        """
        Create DialogueValidator instance from CLI arguments.

        Supports loading configuration from a YAML/JSON file via --config.
        CLI arguments take precedence over config file values.
        """
        # Load config file if provided
        config = {}
        if args.config:
            logger.info(f"Loading configuration from: {args.config}")
            config = load_config_file(args.config)

        # Apply config values for args that are None (not provided via CLI)
        config_key_mapping = {
            "openai_api_key": "openai_api_key",
            "model_base_url": "model_base_url",
            "model_name": "model_name",
            "user_profile_dataset": "user_profile_dataset",
            "dialogue_file": "dialogue_file",
            "output_file_path": "output_file_path",
            "dialogue_topic": "dialogue_topic",
            "evaluation_mode": "evaluation_mode",
            "num_seeds": "num_seeds",
            "starting_row": "starting_row",
            "ending_row": "ending_row",
            "verbose": "verbose",
            "storage_step": "storage_step",
            "use_seed": "use_seed",
            "use_batch_api": "use_batch_api",
            "batch_poll_interval": "batch_poll_interval",
            "request_delay": "request_delay",
        }

        for config_key, arg_attr in config_key_mapping.items():
            if getattr(args, arg_attr, None) is None and config_key in config:
                setattr(args, arg_attr, config[config_key])

        # Apply defaults for any remaining None values
        defaults = {
            "openai_api_key": os.environ.get("OPENAI_API_KEY") or os.environ.get("api_key"),
            "model_base_url": os.environ.get("OPENAI_BASE_URL") or os.environ.get("base_url"),
            "model_name": "gpt-4.1",
            "dialogue_topic": "career advice",
            "evaluation_mode": "combined",
            "num_seeds": 3,
            "starting_row": 0,
            "ending_row": -1,
            "verbose": 0,
            "storage_step": 50,
            "use_seed": True,
            "use_batch_api": False,
            "batch_poll_interval": 30,
            "request_delay": 0.0,
        }

        for arg_attr, default_value in defaults.items():
            if getattr(args, arg_attr, None) is None:
                setattr(args, arg_attr, default_value)

        # Validate required arguments
        required_args = ["user_profile_dataset", "dialogue_file", "output_file_path"]
        missing_args = [arg for arg in required_args if getattr(args, arg, None) is None]
        if missing_args:
            raise ValueError(
                f"Missing required arguments: {missing_args}. "
                "Provide them via CLI, config file, or environment variables."
            )

        def _read_csv(csv_path: str) -> pd.DataFrame:
            df = pd.read_csv(csv_path)
            df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
            return df

        full_dataset = _read_csv(args.user_profile_dataset)
        ending_row = args.ending_row if args.ending_row > 0 else len(full_dataset)
        user_profile_dataset = full_dataset.iloc[args.starting_row : ending_row]

        # Load dialogues - supports both legacy and WVS formats
        generated_dialogues = []
        with open(args.dialogue_file, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if idx < args.starting_row:
                    continue
                data = json.loads(line)
                # Check if this is WVS format (dict with interview_id as key)
                # WVS format: {"688070395": [{"role": "user", "content": ...}, ...]}
                if isinstance(data, dict) and "index" not in data and "generated_dialogue" not in data:
                    # WVS format - convert to standard format
                    for interview_id, dialogue in data.items():
                        generated_dialogues.append(
                            {
                                "interview_id": int(interview_id) if interview_id.isdigit() else interview_id,
                                "dialogue": dialogue,
                            }
                        )
                else:
                    # Legacy format: {"index": 0, "generated_dialogue": [...]}
                    generated_dialogues.append(data)
                if len(generated_dialogues) >= (ending_row - args.starting_row):
                    break

        openai_client = AsyncOpenAI(
            api_key=args.openai_api_key,
            base_url=args.model_base_url,
        )
        openai_sync_client = OpenAI(
            api_key=args.openai_api_key,
            base_url=args.model_base_url,
        )

        return cls(
            output_file_path=args.output_file_path,
            user_profile_dataset=user_profile_dataset,
            generated_dialogues=generated_dialogues,
            model_name=args.model_name,
            dialogue_topic=args.dialogue_topic,
            evaluation_mode=args.evaluation_mode,
            openai_client=openai_client,
            openai_sync_client=openai_sync_client,
            num_seeds=args.num_seeds,
            verbose=args.verbose,
            storage_step=args.storage_step,
            use_seed=args.use_seed,
            use_batch_api=args.use_batch_api,
            batch_poll_interval=args.batch_poll_interval,
            request_delay=args.request_delay,
        )

    def _format_user_profile(self, row_dict: Dict) -> str:
        """Format user profile dictionary as string."""
        profile_lines = []
        for key, value in row_dict.items():
            if pd.notna(value):
                profile_lines.append(f"{key}: {value}")
        return "\n".join(profile_lines)

    def _format_dialogue(self, dialogue_data: List[Dict]) -> str:
        """Format dialogue turns as string.

        Supports two formats:
        1. Legacy: [{"user_content": ..., "chatbot_content": ...}, ...]
        2. WVS: [{"role": "user", "content": ...}, {"role": "chatbot", "content": ...}, ...]
        """
        dialogue_lines = []

        # Check if this is WVS format (role/content pairs)
        if dialogue_data and "role" in dialogue_data[0]:
            for turn in dialogue_data:
                role = turn.get("role", "").lower()
                content = turn.get("content", "")
                if role == "user":
                    dialogue_lines.append(f"User: {content}")
                elif role in ["chatbot", "assistant"]:
                    dialogue_lines.append(f"Assistant: {content}")
        else:
            # Legacy format
            for turn in dialogue_data:
                if "user_content" in turn:
                    dialogue_lines.append(f"User: {turn['user_content']}")
                if "assistant_content" in turn or "chatbot_content" in turn:
                    content = turn.get("assistant_content") or turn.get("chatbot_content")
                    dialogue_lines.append(f"Assistant: {content}")

        return "\n\n".join(dialogue_lines)

    def _validate_and_clamp_scores(self, scores: Dict[str, Any]) -> Dict[str, int]:
        """Validate and clamp scores to valid ranges."""
        validated = {}
        for criterion, (min_val, max_val) in SCORE_RANGES.items():
            raw_score = scores.get(criterion, min_val)
            try:
                score = int(raw_score)
                validated[criterion] = max(min_val, min(max_val, score))
            except (TypeError, ValueError):
                logger.warning(f"Invalid score for {criterion}: {raw_score}, using minimum")
                validated[criterion] = min_val
        return validated

    def _parse_llm_response(self, response_content: str) -> Dict[str, Any]:
        """Parse LLM response JSON."""
        try:
            return json.loads(response_content)
        except json.JSONDecodeError:
            # Try to extract JSON from response
            json_match = re.search(r"\{.*\}", response_content, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except json.JSONDecodeError:
                    pass
            logger.warning(f"Failed to parse LLM response: {response_content[:200]}...")
            return {
                "coverage": 0,
                "correctness": 0,
                "diversity": 1,
                "relevance": 1,
                "naturalness": 1,
                "reasoning": {"error": "Failed to parse response"},
            }

    async def _evaluate_single_dialogue(
        self,
        user_profile: str,
        dialogue: str,
        seed: int = 0,
    ) -> Dict[str, Any]:
        """Evaluate a single dialogue using combined evaluation."""
        messages = deepcopy(VALIDATION_CRITERIA["combined"])

        # Fill in placeholders
        messages[1]["content"] = messages[1]["content"].format(user_profile=user_profile)
        messages[2]["content"] = messages[2]["content"].format(dialogue_topic=self._dialogue_topic)
        messages[3]["content"] = messages[3]["content"].format(dialogue=dialogue)

        try:
            # Build API call parameters
            api_params = {
                "model": self._model_name,
                "messages": messages,
                "max_tokens": 2048,
                "temperature": 0.1,
                "response_format": {"type": "json_object"},
            }
            # Only include seed if supported (OpenAI supports it, Gemini doesn't)
            if self._use_seed:
                api_params["seed"] = seed

            response = await self._openai_client.chat.completions.create(**api_params)
            content = response.choices[0].message.content
            if content is None:
                logger.warning("Received None content from API response")
                return {
                    "coverage": 0,
                    "correctness": 0,
                    "diversity": 1,
                    "relevance": 1,
                    "naturalness": 1,
                    "reasoning": {"error": "API returned None content"},
                    "seed": seed if self._use_seed else None,
                }
            result = self._parse_llm_response(content)
            result["seed"] = seed if self._use_seed else None
            return result
        except Exception as e:
            logger.error(f"Error evaluating dialogue: {e}")
            return {
                "coverage": 0,
                "correctness": 0,
                "diversity": 1,
                "relevance": 1,
                "naturalness": 1,
                "reasoning": {"error": str(e)},
                "seed": seed,
            }

    async def _evaluate_dialogue_with_seeds(
        self,
        user_profile: str,
        dialogue: str,
    ) -> Dict[str, Any]:
        """Evaluate dialogue multiple times with different seeds and average."""
        # Run sequentially with delay if rate limiting is needed
        if self._request_delay > 0:
            results = []
            for seed in range(self._num_seeds):
                result = await self._evaluate_single_dialogue(user_profile, dialogue, seed=seed)
                results.append(result)
                if seed < self._num_seeds - 1:  # Don't delay after last request
                    await asyncio.sleep(self._request_delay)
        else:
            # Run in parallel for faster processing
            tasks = [
                self._evaluate_single_dialogue(user_profile, dialogue, seed=seed) for seed in range(self._num_seeds)
            ]
            results = await asyncio.gather(*tasks)

        # Average scores across seeds
        avg_scores = {}
        for criterion in SCORE_RANGES.keys():
            scores = [r.get(criterion, 0) for r in results]
            avg_scores[criterion] = sum(scores) / len(scores)

        # Round to integers for final scores
        final_scores = self._validate_and_clamp_scores({k: round(v) for k, v in avg_scores.items()})

        return {
            "scores": final_scores,
            "avg_scores": avg_scores,
            "all_evaluations": results,
        }

    # =========================================================================
    # Batch API Methods
    # =========================================================================

    def _prepare_batch_request(
        self,
        custom_id: str,
        user_profile: str,
        dialogue: str,
        seed: int = 0,
    ) -> Dict[str, Any]:
        """Prepare a single batch request in OpenAI Batch API format."""
        messages = deepcopy(VALIDATION_CRITERIA["combined"])

        # Fill in placeholders
        messages[1]["content"] = messages[1]["content"].format(user_profile=user_profile)
        messages[2]["content"] = messages[2]["content"].format(dialogue_topic=self._dialogue_topic)
        messages[3]["content"] = messages[3]["content"].format(dialogue=dialogue)

        body = {
            "model": self._model_name,
            "messages": messages,
            "max_tokens": 2048,
            "temperature": 0.1,
            "response_format": {"type": "json_object"},
        }
        if self._use_seed:
            body["seed"] = seed

        return {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": body,
        }

    def _prepare_all_batch_requests(self) -> Tuple[List[Dict[str, Any]], Dict[str, Tuple[Any, int]]]:
        """
        Prepare all batch requests for dialogues.

        Returns:
            Tuple of (list of batch requests, mapping from custom_id to (dialogue_id, seed))
        """
        batch_requests = []
        id_mapping = {}  # custom_id -> (dialogue_id, seed)

        # Check if using WVS format
        is_wvs_format = "D_INTERVIEW" in self._user_profile_dataset.columns
        if is_wvs_format:
            profile_lookup = self._user_profile_dataset.set_index("D_INTERVIEW").to_dict("index")

        logger.info(f"Preparing batch requests for {len(self._generated_dialogues)} dialogues")
        logger.info(f"Seeds per dialogue: {self._num_seeds}")
        logger.info(f"Total requests: {len(self._generated_dialogues) * self._num_seeds}")

        for idx, dialogue_obj in enumerate(self._generated_dialogues):
            # Get dialogue data and ID based on format
            if "interview_id" in dialogue_obj:
                dialogue_id = dialogue_obj["interview_id"]
                dialogue_data = dialogue_obj.get("dialogue", [])
                if is_wvs_format and dialogue_id in profile_lookup:
                    row_dict = profile_lookup[dialogue_id]
                else:
                    row_dict = {}
            else:
                dialogue_id = dialogue_obj.get("index", idx)
                dialogue_data = dialogue_obj.get("generated_dialogue", [])
                if idx < len(self._user_profile_dataset):
                    row_dict = self._user_profile_dataset.iloc[idx].to_dict()
                else:
                    row_dict = {}

            user_profile = self._format_user_profile(row_dict)
            dialogue_str = self._format_dialogue(dialogue_data)

            # Create request for each seed
            for seed in range(self._num_seeds):
                custom_id = f"dialogue_{dialogue_id}_seed_{seed}"
                request = self._prepare_batch_request(custom_id, user_profile, dialogue_str, seed)
                batch_requests.append(request)
                id_mapping[custom_id] = (dialogue_id, seed)

        return batch_requests, id_mapping

    def _write_batch_file(self, batch_requests: List[Dict[str, Any]]) -> str:
        """Write batch requests to a temporary JSONL file."""
        batch_file = tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".jsonl",
            delete=False,
            encoding="utf-8",
        )
        for request in batch_requests:
            batch_file.write(json.dumps(request) + "\n")
        batch_file.close()
        logger.info(f"Batch file written: {batch_file.name}")
        return batch_file.name

    def _submit_batch_job(self, batch_file_path: str) -> str:
        """
        Upload batch file and create batch job.

        Returns:
            Batch job ID
        """
        logger.info("Uploading batch file to OpenAI...")
        with open(batch_file_path, "rb") as f:
            file_response = self._openai_sync_client.files.create(
                file=f,
                purpose="batch",
            )
        file_id = file_response.id
        logger.info(f"File uploaded: {file_id}")

        logger.info("Creating batch job...")
        batch_response = self._openai_sync_client.batches.create(
            input_file_id=file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={
                "description": f"Dialogue validation - {self._dialogue_topic}",
                "model": self._model_name,
            },
        )
        batch_id = batch_response.id
        logger.info(f"Batch job created: {batch_id}")
        return batch_id

    def _poll_batch_status(self, batch_id: str) -> Dict[str, Any]:
        """
        Poll batch job status until completion.

        Returns:
            Final batch status object
        """
        logger.info(f"Polling batch status (interval: {self._batch_poll_interval}s)...")

        while True:
            batch = self._openai_sync_client.batches.retrieve(batch_id)
            status = batch.status

            # Log progress
            completed = batch.request_counts.completed if batch.request_counts else 0
            total = batch.request_counts.total if batch.request_counts else 0
            failed = batch.request_counts.failed if batch.request_counts else 0

            logger.info(f"Batch status: {status} | " f"Progress: {completed}/{total} completed, {failed} failed")

            if status in ["completed", "failed", "expired", "cancelled"]:
                return {
                    "status": status,
                    "output_file_id": batch.output_file_id,
                    "error_file_id": batch.error_file_id,
                    "request_counts": {
                        "total": total,
                        "completed": completed,
                        "failed": failed,
                    },
                }

            time.sleep(self._batch_poll_interval)

    def _download_batch_results(self, output_file_id: str) -> List[Dict[str, Any]]:
        """Download and parse batch results."""
        logger.info(f"Downloading batch results: {output_file_id}")

        response = self._openai_sync_client.files.content(output_file_id)
        content = response.text

        results = []
        for line in content.strip().split("\n"):
            if line:
                results.append(json.loads(line))

        logger.info(f"Downloaded {len(results)} results")
        return results

    def _process_batch_results(
        self,
        raw_results: List[Dict[str, Any]],
        id_mapping: Dict[str, Tuple[Any, int]],
    ) -> List[Dict]:
        """
        Process batch results and aggregate by dialogue.

        Args:
            raw_results: Raw batch API response items
            id_mapping: Mapping from custom_id to (dialogue_id, seed)

        Returns:
            List of validation results aggregated by dialogue
        """
        # Group results by dialogue_id
        dialogue_results: Dict[Any, List[Dict]] = {}

        for result in raw_results:
            custom_id = result.get("custom_id")
            if custom_id not in id_mapping:
                logger.warning(f"Unknown custom_id: {custom_id}")
                continue

            dialogue_id, seed = id_mapping[custom_id]

            # Parse response
            response_body = result.get("response", {}).get("body", {})
            choices = response_body.get("choices", [])

            if choices:
                content = choices[0].get("message", {}).get("content")
                if content:
                    parsed = self._parse_llm_response(content)
                    parsed["seed"] = seed if self._use_seed else None
                else:
                    parsed = {
                        "coverage": 0,
                        "correctness": 0,
                        "diversity": 1,
                        "relevance": 1,
                        "naturalness": 1,
                        "reasoning": {"error": "Empty content"},
                        "seed": seed if self._use_seed else None,
                    }
            else:
                # Handle error response
                error = result.get("error", {})
                parsed = {
                    "coverage": 0,
                    "correctness": 0,
                    "diversity": 1,
                    "relevance": 1,
                    "naturalness": 1,
                    "reasoning": {"error": str(error)},
                    "seed": seed if self._use_seed else None,
                }

            if dialogue_id not in dialogue_results:
                dialogue_results[dialogue_id] = []
            dialogue_results[dialogue_id].append(parsed)

        # Aggregate results by dialogue
        all_results = []
        for dialogue_id, seed_results in dialogue_results.items():
            # Average scores across seeds
            avg_scores = {}
            for criterion in SCORE_RANGES.keys():
                scores = [r.get(criterion, 0) for r in seed_results]
                avg_scores[criterion] = sum(scores) / len(scores) if scores else 0

            final_scores = self._validate_and_clamp_scores({k: round(v) for k, v in avg_scores.items()})

            validation_entry = {
                "dialogue_index": dialogue_id,
                "model": self._model_name,
                "timestamp": datetime.now().isoformat(),
                "scores": final_scores,
                "avg_scores": avg_scores,
                "all_evaluations": seed_results,
            }
            all_results.append(validation_entry)

        return all_results

    def validate_all_dialogues_batch(self) -> List[Dict]:
        """
        Validate all dialogues using OpenAI Batch API.

        This method:
        1. Prepares batch requests for all dialogues
        2. Uploads and submits the batch job
        3. Polls for completion
        4. Downloads and processes results
        5. Saves results to output file

        Returns:
            List of validation results for each dialogue
        """
        logger.info("=" * 60)
        logger.info("BATCH API VALIDATION")
        logger.info("=" * 60)
        logger.info(f"Model: {self._model_name}")
        logger.info(f"Dialogues: {len(self._generated_dialogues)}")
        logger.info(f"Seeds per dialogue: {self._num_seeds}")
        logger.info("=" * 60)

        # Step 1: Prepare batch requests
        batch_requests, id_mapping = self._prepare_all_batch_requests()

        # Step 2: Write to file
        batch_file_path = self._write_batch_file(batch_requests)

        try:
            # Step 3: Submit batch job
            batch_id = self._submit_batch_job(batch_file_path)

            # Step 4: Poll for completion
            batch_status = self._poll_batch_status(batch_id)

            if batch_status["status"] != "completed":
                logger.error(f"Batch job failed with status: {batch_status['status']}")
                # Try to get error details
                if batch_status.get("error_file_id"):
                    error_results = self._download_batch_results(batch_status["error_file_id"])
                    logger.error(f"Errors: {error_results[:5]}")  # Log first 5 errors
                return []

            # Step 5: Download results
            raw_results = self._download_batch_results(batch_status["output_file_id"])

            # Step 6: Process results
            all_results = self._process_batch_results(raw_results, id_mapping)

            # Step 7: Save results
            self._append_to_file(all_results)
            logger.info(f"Results saved to: {self._output_file_path}")

            return all_results

        finally:
            # Clean up temp file
            try:
                os.unlink(batch_file_path)
            except Exception:
                pass

    async def validate_all_dialogues(self) -> List[Dict]:
        """
        Validate all generated dialogues.

        Supports both legacy format (index-based matching) and WVS format
        (interview_id-based matching with D_INTERVIEW column).

        Returns:
            List of validation results for each dialogue
        """
        all_results = []
        buffer = []

        logger.info(f"Starting validation of {len(self._generated_dialogues)} dialogues")
        logger.info(f"Using model: {self._model_name}")
        logger.info(f"Evaluation mode: {self._evaluation_mode}")

        # Check if using WVS format (profiles have D_INTERVIEW column)
        is_wvs_format = "D_INTERVIEW" in self._user_profile_dataset.columns

        # Create profile lookup for WVS format
        if is_wvs_format:
            profile_lookup = self._user_profile_dataset.set_index("D_INTERVIEW").to_dict("index")
            logger.info("Using WVS format: matching dialogues by interview ID")

        with tqdm(
            total=len(self._generated_dialogues),
            desc="Validating dialogues",
            unit="dialogue",
        ) as pbar:
            for idx, dialogue_obj in enumerate(self._generated_dialogues):
                # Get dialogue data and ID based on format
                if "interview_id" in dialogue_obj:
                    # WVS format
                    dialogue_id = dialogue_obj["interview_id"]
                    dialogue_data = dialogue_obj.get("dialogue", [])

                    # Look up profile by interview ID
                    if is_wvs_format and dialogue_id in profile_lookup:
                        row_dict = profile_lookup[dialogue_id]
                    else:
                        logger.warning(f"Profile not found for interview_id: {dialogue_id}")
                        row_dict = {}
                else:
                    # Legacy format
                    dialogue_id = dialogue_obj.get("index", idx)
                    dialogue_data = dialogue_obj.get("generated_dialogue", [])

                    # Get profile by index
                    if idx < len(self._user_profile_dataset):
                        row_dict = self._user_profile_dataset.iloc[idx].to_dict()
                    else:
                        row_dict = {}

                user_profile = self._format_user_profile(row_dict)
                dialogue_str = self._format_dialogue(dialogue_data)

                if self._verbose:
                    logger.info(f"Processing dialogue {idx} (ID: {dialogue_id})")

                result = await self._evaluate_dialogue_with_seeds(user_profile, dialogue_str)

                validation_entry = {
                    "dialogue_index": dialogue_id,
                    "model": self._model_name,
                    "timestamp": datetime.now().isoformat(),
                    **result,
                }

                buffer.append(validation_entry)
                all_results.append(validation_entry)

                # Flush to file periodically
                if self._storage_step and len(buffer) >= self._storage_step:
                    self._append_to_file(buffer)
                    buffer.clear()

                pbar.update(1)

        # Flush remaining results
        if buffer:
            self._append_to_file(buffer)

        logger.info(f"Validation complete. Results saved to {self._output_file_path}")
        return all_results

    def _append_to_file(self, data: List[Dict]) -> None:
        """Append validation results to JSONL file."""
        with open(self._output_file_path, "a", encoding="utf-8") as f:
            for entry in data:
                f.write(json.dumps(entry) + "\n")

    def compute_summary_statistics(self, results: List[Dict]) -> Dict[str, Any]:
        """Compute summary statistics from validation results."""
        if not results:
            return {}

        criteria = list(SCORE_RANGES.keys())
        stats = {}

        for criterion in criteria:
            scores = [r["scores"][criterion] for r in results if "scores" in r]
            if scores:
                stats[criterion] = {
                    "mean": sum(scores) / len(scores),
                    "min": min(scores),
                    "max": max(scores),
                    "std": (sum((s - sum(scores) / len(scores)) ** 2 for s in scores) / len(scores)) ** 0.5,
                }

        # Overall score (average of all criteria)
        overall_scores = []
        for r in results:
            if "scores" in r:
                overall_scores.append(sum(r["scores"].values()) / len(r["scores"]))

        if overall_scores:
            stats["overall"] = {
                "mean": sum(overall_scores) / len(overall_scores),
                "min": min(overall_scores),
                "max": max(overall_scores),
            }

        return stats


async def main():
    """Main entry point for dialogue validation."""
    parser = argparse.ArgumentParser(
        description="Validate generated dialogues using LLM-based evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  # Using config file
  python dialogue_validator.py --config config.yaml

  # Using CLI arguments
  python dialogue_validator.py --user-profile-dataset data.csv --dialogue-file dialogues.jsonl --output-file-path results.jsonl

  # Mix: config file with CLI overrides
  python dialogue_validator.py --config config.yaml --model-name gpt-4o --num-seeds 5
        """,
    )
    parser = DialogueValidator.add_cli_args(parser)
    args = parser.parse_args()

    # Set up logging early with a default, will reconfigure after args are processed
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    validator = DialogueValidator.from_cli_args(args)

    # Reconfigure logging based on final verbose setting
    logging.getLogger().setLevel(logging.INFO if args.verbose else logging.WARNING)

    # Log effective configuration
    logger.info("=" * 60)
    logger.info("EFFECTIVE CONFIGURATION")
    logger.info("=" * 60)
    if args.config:
        logger.info(f"Config file: {args.config}")
    logger.info(f"Model: {args.model_name}")
    logger.info(f"Dialogue topic: {args.dialogue_topic}")
    logger.info(f"Evaluation mode: {args.evaluation_mode}")
    logger.info(f"Num seeds: {args.num_seeds}")
    logger.info(f"Row range: {args.starting_row} to {args.ending_row}")
    logger.info(f"Batch API: {args.use_batch_api}")
    logger.info("=" * 60)

    # Choose validation method based on batch API flag
    if args.use_batch_api:
        logger.info("Using OpenAI Batch API for validation")
        results = validator.validate_all_dialogues_batch()
    else:
        logger.info("Using real-time API calls for validation")
        results = await validator.validate_all_dialogues()

    # Print summary statistics
    stats = validator.compute_summary_statistics(results)
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    for criterion, values in stats.items():
        if criterion != "overall":
            print(f"{criterion}: mean={values['mean']:.2f}, std={values['std']:.2f}")
    if "overall" in stats:
        print(f"\nOverall: mean={stats['overall']['mean']:.2f}")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
