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
from .formulas import  jensen_shannon_divergence, hellinger_distance


logger = logging.getLogger(__name__)


class ValuesComparison:
    def __init__(
        self,
        user_profile_dataset: pd.DataFrame,
        direct_values_predictions: List[Dict],
        generated_dialogues: List[Dict] = None,
        dialogue_values_predictions: List[Dict] = None,
        verbose: int = 0,
    ) -> None:
        """
        Initializes the ValuesComparison class with user profile data, direct predictions, and optional dialogue data.
        
        Args:
            user_profile_dataset (pd.DataFrame): A pandas DataFrame containing the user profile data.
            direct_values_predictions (List[Dict]): A list of dictionaries containing the direct predictions for values.
            generated_dialogues (List[Dict], optional): A list of generated dialogues. Defaults to None.
            dialogue_values_predictions (List[Dict], optional): A list of predicted values based on dialogues. Defaults to None.
            verbose (int, optional): A verbosity level for logging. Defaults to 0.
        
        Raises:
            TypeError: If `user_profile_dataset` is not a pandas DataFrame.
            TypeError: If `direct_values_predictions` is not a list of dictionaries.
        """
        if not isinstance(user_profile_dataset, pd.DataFrame):
            raise TypeError("user_profile_dataset must be a pandas DataFrame.")
        if not isinstance(direct_values_predictions, list):
            raise TypeError("direct_values_predictions must be a list of dictionaries.")

        self._user_profile_dataset = user_profile_dataset
        self._direct_values_predictions = direct_values_predictions
        self._generated_dialogues = generated_dialogues
        self._dialogue_values_predictions = dialogue_values_predictions
        self._verbose = verbose

    @property
    def user_profile_dataset(self) -> pd.DataFrame:
        """Returns the user profile dataset."""
        return self._user_profile_dataset

    @property
    def direct_values_predictions(self) -> List[Dict]:
        """Returns the list of direct values predictions."""
        return self._direct_values_predictions

    @property
    def generated_dialogues(self) -> List[Dict]:
        """Returns the list of generated dialogues."""
        return self._generated_dialogues

    @property
    def dialogue_values_predictions(self) -> List[Dict]:
        """Returns the list of predicted values based on dialogues."""
        return self._dialogue_values_predictions

    @property
    def verbose(self) -> int:
        """Returns the verbosity level."""
        return self._verbose

    def _get_index_list_for_groups(self, df, target_col):
        grouped_indices = df.groupby(target_col).apply(lambda x: x.index.tolist())    

        return grouped_indices

    

        