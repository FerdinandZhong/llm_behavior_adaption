"""Dialogue Dataset Validation Module

This module provides tools for validating generated dialogue datasets using
LLM-based evaluation and computing alignment with human annotations.

Components:
- DialogueValidator: LLM-based dialogue quality evaluation
- generate_human_annotation_samples: Generate samples for human annotation
- validation_constants: Scoring rubrics and criteria definitions
"""

from llm_behavior_adaptation.dialogue_dataset_creation.dialogues_validation.dialogue_validator import (
    DialogueValidator,
    load_config_file,
)
from llm_behavior_adaptation.dialogue_dataset_creation.dialogues_validation.generate_human_annotation_samples import (
    SCORING_RUBRIC,
    create_annotation_dataframe,
    generate_rubric_text,
)
from llm_behavior_adaptation.dialogue_dataset_creation.dialogues_validation.validation_constants import (
    SCORE_RANGES,
    VALIDATION_CRITERIA,
)

__all__ = [
    "DialogueValidator",
    "load_config_file",
    "SCORING_RUBRIC",
    "create_annotation_dataframe",
    "generate_rubric_text",
    "SCORE_RANGES",
    "VALIDATION_CRITERIA",
]
