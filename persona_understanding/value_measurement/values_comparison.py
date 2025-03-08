import argparse
import json
import logging
import random
from itertools import combinations
from typing import Dict, List

import numpy as np
import pandas as pd
from tqdm import tqdm

from persona_understanding.dialogue_dataset_creation.generation_utils import (
    calculate_age,
)
from persona_understanding.value_measurement.formulas import (
    compute_js_centroid,
    compute_js_centroid_and_avg,
    filter_rows,
    hellinger_distance,
    jensen_shannon_divergence,
    compute_emd
)
from persona_understanding.value_measurement.measurement_utils import (
    JobClassifier,
    get_continent,
    get_culture,
    get_development_level,
)

logger = logging.getLogger(__name__)

formula_mapping = {"JSD": jensen_shannon_divergence, "HD": hellinger_distance}


def load_jsonl_file(file_path, index_name="index", starting_index=0, ending_index=-1):
    list_of_json_objs = []
    with open(file_path, "r", encoding="utf-8") as file:
        for json_obj in file:
            dialogue_obj = json.loads(json_obj)
            if dialogue_obj[index_name] < starting_index:
                continue
            list_of_json_objs.append(dialogue_obj)
            if ending_index != -1 and len(list_of_json_objs) >= ending_index:
                break
    return list_of_json_objs


class ValuesComparison:
    """class for doing the values comparison"""

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
            "--user-profile-dataset",
            type=str,
            required=True,
            help="Path to the seed dataset file.",
        )
        parser.add_argument(
            "--dialogue-file",
            type=str,
            required=True,
            help="Dialogue file",
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
            "--verbose",
            type=int,
            choices=[0, 1],
            default=0,
            help="Verbosity level: 0 = Errors only, 1 = Detailed logs. Defaults to 0.",
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

        generated_dialogues = load_jsonl_file(
            args.dialogue_file, args.starting_row, args.ending_row
        )
        direct_values_prediction = load_jsonl_file(
            args.direct_output_file_path, args.starting_row, args.ending_row
        )
        dialogue_values_prediction = load_jsonl_file(
            args.dialogue_output_file_path, args.starting_row, args.ending_row
        )

        return cls(
            user_profile_dataset=user_profile_dataset,
            direct_values_predictions=direct_values_prediction,
            generated_dialogues=generated_dialogues,
            dialogue_values_predictions=dialogue_values_prediction,
            verbose=args.verbose,
        )

    def _get_index_list_for_groups(self, target_col):
        """
        Get the grouped indices and the target column's values from a DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame to process.
            target_col (str): The column to group by.

        Returns:
            dict: A dictionary where keys are group values from the target column,
                and values are lists of indices for each group.
        """
        if target_col == "Age":
            self.user_profile_dataset["Age"] = self.user_profile_dataset.apply(
                lambda x: calculate_age(x["Date of Birth"]), axis=1
            )

            grouped_data = self._get_index_list_for_groups_age()
            return grouped_data
        elif target_col == "Continent":
            self.user_profile_dataset["Continent"] = self.user_profile_dataset.apply(
                lambda x: get_continent(x["Country"]), axis=1
            )
        elif target_col == "Development":
            self.user_profile_dataset["Development"] = self.user_profile_dataset.apply(
                lambda x: get_development_level(x["Country"]), axis=1
            )
        elif target_col == "Culture":
            self.user_profile_dataset["Culture"] = self.user_profile_dataset.apply(
                lambda x: get_culture(x["Country"]), axis=1
            )
        elif target_col == "Position_Level":
            job_classifier = JobClassifier()
            self.user_profile_dataset["Position_Level"] = (
                self.user_profile_dataset.apply(
                    lambda x: job_classifier.get_position_level(x["Job Title"]), axis=1
                )
            )

        grouped_data = self.user_profile_dataset.groupby(target_col).apply(
            lambda x: x.index.tolist()
        )
        return grouped_data.to_dict()

    def _get_index_list_for_groups_age(self, target_col="Age"):
        """
        Get grouped indices and target column's values based on specified ranges.

        Args:
            target_col (str): The column to group by.

        Returns:
            dict: A dictionary where keys are range labels, and values are lists of indices for each group.
        """
        # Define range bins and labels
        bins = [-float("inf"), 30, 40, 50, 60, float("inf")]
        labels = ["<30", "30-40", "40-50", "50-60", ">60"]

        # Add a temporary column for grouping ranges
        self.user_profile_dataset["range_group"] = pd.cut(
            self.user_profile_dataset[target_col], bins=bins, labels=labels, right=False
        )

        # Group indices by the range_group column
        grouped_data = self.user_profile_dataset.groupby("range_group").apply(
            lambda x: x.index.tolist()
        )

        # Convert to a dictionary and remove empty groups
        grouped_data_dict = {
            group: indices for group, indices in grouped_data.items() if indices
        }

        return grouped_data_dict

    def _calculate_divergence_for_groups_with_custom_function(
        self, formula, group1, group2
    ):
        """
        Calculate the average Divergence between two groups of users' probability distributions,
        handling failed distributions.

        Args:
            group1 (list of list): First group of users' outputs (each user is a list of 17-probability distributions).
            group2 (list of list): Second group of users' outputs (each user is a list of 17-probability distributions).
            formula (callable): A function to compute the divergence (e.g., JSD or Hellinger distance).

        Returns:
            dict: A dictionary containing:
                - average_user_divergence: Average divergence across user pairs.
                - pairwise_user_divergence: List of divergences for each user pair.
                - std_user_divergence: Standard deviation of user divergences.
                - failed_distributions_count: Number of failed distributions encountered.
        """
        pairwise_user_divergences = []
        failed_distributions_count = 0

        with tqdm(
            total=len(group1) * len(group2), desc="Pairwise User Comparisons"
        ) as pbar:
            # Pairwise divergence for each user in group1 and group2
            for user1 in group1:
                for user2 in group2:
                    user_divergences = []
                    for dist1, dist2 in zip(user1, user2):
                        # Check if distributions are valid
                        if not np.allclose(np.sum(dist1), 1) or not np.allclose(
                            np.sum(dist2), 1
                        ):
                            failed_distributions_count += 1
                            user_divergences.append(
                                1.0
                            )  # Assign a score of 1 for failed distributions
                        else:
                            divergence = formula(dist1, dist2)
                            user_divergences.append(divergence)

                    # Compute average divergence for the user pair
                    user_avg_divergence = np.mean(user_divergences)
                    pairwise_user_divergences.append(user_avg_divergence)
                    pbar.update(1)

        # Aggregate Results
        average_user_divergence = np.mean(pairwise_user_divergences)
        std_user_divergence = np.std(pairwise_user_divergences)

        return {
            "average_user_divergence": average_user_divergence,
            "pairwise_user_divergence": pairwise_user_divergences,
            "std_user_divergence": std_user_divergence,
            "failed_distributions_count": failed_distributions_count,
        }

    def inner_dataset_comparison(
        self, formula, target_col, values_selection_results, groups=None
    ):
        """
        Efficiently compare distributions across groups in a dataset, avoiding redundant comparisons.

        Args:
            formula (str): The name of the divergence formula to use.
            target_col (str): The target column to group by.
            values_selection_results (list): List of values selections for users.
            groups (Dict): Pre-defined groups

        Returns:
            dict: Comparison results with divergence scores between groups.
        """

        # Map user indices to value selections
        values_selections_dict = {
            value_selections["user_idx"]: value_selections["value_selections"]
            for value_selections in values_selection_results
        }

        # Group user indices by target column
        if groups is None:
            groups = self._get_index_list_for_groups(target_col)
        formula_func = formula_mapping[formula]

        # Precompute distributions for each group
        group_distributions = {
            group: [
                [
                    question_selection["normalized_probs"]
                    for question_selection in values_selections_dict[user_idx]
                ]
                for user_idx in idx_list
            ]
            for group, idx_list in groups.items()
        }

        # Initialize results and progress bar
        comparison_results = []
        total_comparisons = len(groups) * (len(groups) - 1) // 2
        with tqdm(total=total_comparisons, desc="Group Comparisons") as pbar:
            # Iterate over unique group pairs
            for base_group, target_group in combinations(groups.keys(), 2):
                # Calculate cross-group divergence
                cross_group_divergence = (
                    self._calculate_divergence_for_groups_with_custom_function(
                        formula_func,
                        group_distributions[base_group],
                        group_distributions[target_group],
                    )
                )

                # Store the result
                comparison_results.append(
                    {
                        "compared_groups": f"{base_group.replace(' ', '_')}--{target_group.replace(' ', '_')}",
                        "compared_details": cross_group_divergence,
                    }
                )

                # Update progress bar
                pbar.update(1)

        return comparison_results

    def cross_dataset_comparison(self, formula):
        """
        Compare distributions across groups in a dataset, avoiding redundant comparisons.

        Args:
            formula (str): The name of the divergence formula to use.
            target_col (str): The target column to group by.
            values_selection_results (list): List of values selections for users.

        Returns:
            dict: Comparison results with divergence scores between groups.
        """
        formula_func = formula_mapping[formula]

        # Track completed comparisons to avoid duplicates
        total_comparisons = len(self.direct_values_predictions)

        pairwise_user_divergences = []
        failed_distributions_count = 0
        with tqdm(total=total_comparisons, desc="Cross Datasets Comparisons") as pbar:

            for direct_user_values, dialogue_user_values in zip(
                self.direct_values_predictions, self.dialogue_values_predictions
            ):
                user_divergences = []

                direct_values_group = [
                    question_selection["normalized_probs"]
                    for question_selection in direct_user_values["value_selections"]
                ]
                dialogue_values_group = [
                    question_selection["normalized_probs"]
                    for question_selection in dialogue_user_values["value_selections"]
                ]

                for dist1, dist2 in zip(direct_values_group, dialogue_values_group):
                    # Check if distributions are valid
                    if not np.allclose(np.sum(dist1), 1) or not np.allclose(
                        np.sum(dist2), 1
                    ):
                        failed_distributions_count += 1
                        user_divergences.append(
                            1.0
                        )  # Assign a score of 1 for failed distributions
                    else:
                        divergence = formula_func(dist1, dist2)
                        user_divergences.append(divergence)

                user_avg_divergence = np.mean(user_divergences)
                pairwise_user_divergences.append(user_avg_divergence)

                # Update progress bar
                pbar.update(1)

        average_user_divergence = np.mean(pairwise_user_divergences)
        std_user_divergence = np.std(pairwise_user_divergences)

        return {
            "average_user_divergence": average_user_divergence,
            "pairwise_user_divergence": pairwise_user_divergences,
            "std_user_divergence": std_user_divergence,
            "failed_distributions_count": failed_distributions_count,
        }

    #
    def compute_baseline_pairwise(self, formula, values_selection_results):
        """Compute the benchmark using pairwise user comparisons"""
        formula_func = formula_mapping[formula]

        # Create a dictionary of user_idx to value selections
        values_selections_dict = {
            value_selections["user_idx"]: np.array(
                [
                    question["normalized_probs"]
                    for question in value_selections["value_selections"]
                ]
            )
            for value_selections in values_selection_results
        }

        distributions_arry = np.array(list(values_selections_dict.values())).swapaxes(
            0, 1
        )

        user_indices = list(values_selections_dict.keys())
        user_pairs = list(combinations(user_indices, 2))  # All unique pairs

        all_questions_divergences = []
        failed_distributions_count = 0

        per_question_divergences = {}
        num_questions = distributions_arry.shape[0]
        for q_idx in tqdm(range(num_questions), desc="Questions", leave=True):
            question_divergences = 0

            question_all_distributions = distributions_arry[q_idx]  # [1000, 5]
            with tqdm(
                total=len(user_pairs), desc="Pairwise Baseline Computation"
            ) as pbar:
                for user1_idx, user2_idx in user_pairs:
                    user1_distribution = question_all_distributions[user1_idx]  # [5]
                    user2_distribution = question_all_distributions[user2_idx]

                    if not np.allclose(np.sum(user1_distribution, axis=0), 1):
                        failed_distributions_count += 1
                        question_divergences += 1.0
                    elif not np.allclose(np.sum(user2_distribution, axis=0), 1):
                        failed_distributions_count += 1
                        question_divergences += 1.0
                    else:
                        # Compute divergence using vectorized operations
                        question_divergences += formula_func(
                            user1_distribution, user2_distribution
                        )
                    pbar.update(1)

                avg_divergence = question_divergences / len(user_pairs)
                all_questions_divergences.append(avg_divergence)
                per_question_divergences[f"question_{q_idx}"] = {
                    "average_divergence": avg_divergence,
                    "failed_count": failed_distributions_count,
                }

        # Compute final statistics
        average_question_divergence = np.mean(all_questions_divergences)
        std_questions_divergence = np.std(all_questions_divergences)

        return {
            "average_user_divergence": average_question_divergence,
            "std_user_divergence": std_questions_divergence,
            "per_question_details": per_question_divergences,
        }

    def cross_datasets_benchmark(self, formula):
        formula_func = formula_mapping[formula]

        # Track completed comparisons to avoid duplicates
        total_comparisons = len(self.direct_values_predictions)

        direct_values_selections_dict = {
            value_selections["user_idx"]: np.array(
                [
                    question["normalized_probs"]
                    for question in value_selections["value_selections"]
                ]
            )
            for value_selections in self.direct_values_predictions
        }

        dialogue_values_selection_dict = {
            value_selections["user_idx"]: np.array(
                [
                    question["normalized_probs"]
                    for question in value_selections["value_selections"]
                ]
            )
            for value_selections in self.dialogue_values_predictions
        }

        # Precompute valid distributions for all users
        direct_valid_distributions = {
            user_idx: np.allclose(np.sum(selections, axis=1), 1)
            for user_idx, selections in direct_values_selections_dict.items()
        }

        dialogue_valid_distributions = {
            user_idx: np.allclose(np.sum(selections, axis=1), 1)
            for user_idx, selections in dialogue_values_selection_dict.items()
        }

        user_indices = list(direct_values_selections_dict.keys())

        all_users_divergence = []
        failed_distributions_count = 0

        with tqdm(
            total=total_comparisons, desc="Cross Datasets Baseline Computation"
        ) as pbar:
            for current_user_idx in user_indices:
                current_direct_user_distributions = direct_values_selections_dict[
                    current_user_idx
                ]
                current_dialogue_user_distributions = dialogue_values_selection_dict[
                    current_user_idx
                ]
                current_direct_valid = direct_valid_distributions[current_user_idx]
                current_dialogue_valid = dialogue_valid_distributions[current_user_idx]

                user_divergences = []

                random_selected_user_idx_list = random.choices(user_indices, k=2)
                while current_user_idx in random_selected_user_idx_list:
                    random_selected_user_idx_list = random.choices(user_indices, k=2)

                for compared_user_idx in random_selected_user_idx_list:
                    compared_user_direct_distributions = direct_values_selections_dict[
                        compared_user_idx
                    ]
                    compared_direct_valid = direct_valid_distributions[
                        compared_user_idx
                    ]

                    compared_user_dialogue_distributions = (
                        dialogue_values_selection_dict[compared_user_idx]
                    )
                    compared_dialogue_valid = dialogue_valid_distributions[
                        compared_user_idx
                    ]

                    if not current_direct_valid or not compared_dialogue_valid:
                        failed_distributions_count += len(
                            current_direct_user_distributions
                        )
                        user_divergences.extend(
                            [1.0] * len(current_direct_user_distributions)
                        )
                        continue

                    if not current_dialogue_valid or not compared_direct_valid:
                        failed_distributions_count += len(
                            current_dialogue_user_distributions
                        )
                        user_divergences.extend(
                            [1.0] * len(current_dialogue_user_distributions)
                        )
                        continue

                    # Compute divergence using vectorized operations
                    divergences = np.array(
                        [
                            formula_func(dist1, dist2)
                            for dist1, dist2 in zip(
                                current_direct_user_distributions,
                                compared_user_dialogue_distributions,
                            )
                        ]
                    )
                    user_divergences.extend(divergences)
                    divergences = np.array(
                        [
                            formula_func(dist1, dist2)
                            for dist1, dist2 in zip(
                                current_dialogue_user_distributions,
                                compared_user_direct_distributions,
                            )
                        ]
                    )
                    user_divergences.extend(divergences)

                # For one user
                user_avg_divergence = np.mean(user_divergences)
                all_users_divergence.append(user_avg_divergence)

                # Update progress bar
                pbar.update(1)

        average_user_divergence = np.mean(all_users_divergence)
        std_user_divergence = np.std(all_users_divergence)

        return {
            "average_user_divergence": average_user_divergence,
            "all_users_divergence": all_users_divergence,
            "std_user_divergence": std_user_divergence,
            "failed_distributions_count": failed_distributions_count,
        }

    def inner_dataset_groups_comparison(
        self, target_col, values_selection_results, groups=None
    ):
        """
        Compute per group divergences with Jenson-shannon centroids of each group.

        Args:
            target_col (str): The target column to group by.
            values_selection_results (list): List of values selections for users.
            groups (Dict): Pre-defined groups

        Returns:
            dict: Comparison results with divergence scores between groups.
        """

        # Map user indices to value selections
        values_selections_dict = {
            value_selections["user_idx"]: value_selections["value_selections"]
            for value_selections in values_selection_results
        }

        # Group user indices by target column
        if groups is None:
            groups = self._get_index_list_for_groups(target_col)

        # Precompute distributions for each group
        all_group_distributions = {
            group: [
                [
                    question_selection["normalized_probs"]
                    for question_selection in values_selections_dict[user_idx]
                ]
                for user_idx in idx_list
            ]
            for group, idx_list in groups.items()
        }

        distributions_arry = np.array(
            [
                np.array(
                    [
                        question["normalized_probs"]
                        for question in value_selections["value_selections"]
                    ]
                )
                for value_selections in values_selection_results
            ]
        ).swapaxes(0, 1)

        num_questions = distributions_arry.shape[0]
        per_question_centroids = {}
        per_question_baseline = {}
        for q_idx in tqdm(range(num_questions), desc="Compute Centroid Globally"):
            per_question_centroids[q_idx] = compute_js_centroid(
                distributions_arry[q_idx]
            )[0]
            per_question_baseline[q_idx] = {
                "avg_baseline_value": 0,
                "baseline_std": 0,
                "baseline_values": [],
            }

        groups_details = {}

        for group in tqdm(groups.keys(), desc="Baseline Computation"):
            group_distributions = np.array(all_group_distributions[group]).swapaxes(
                0, 1
            )
            group_details = {}
            for q_idx in range(num_questions):
                (
                    group_centroid,
                    failed_count,
                    inner_group_divergence,
                ) = compute_js_centroid_and_avg(group_distributions[q_idx])
                to_centroid_divergence = jensen_shannon_divergence(
                    group_centroid, per_question_centroids[q_idx]
                )
                per_question_baseline[q_idx]["baseline_values"].append(
                    to_centroid_divergence
                )
                group_details[q_idx] = {
                    "centroid": group_centroid,
                    "to_centroid_divergence": to_centroid_divergence,
                    "failed_count": failed_count,
                    "inner_group_divergence": inner_group_divergence,
                }
            groups_details[group] = group_details

        cross_questions_divergence_total = 0
        for q_idx in range(num_questions):
            baseline_avg = np.mean(per_question_baseline[q_idx]["baseline_values"])
            per_question_baseline[q_idx]["avg_baseline_value"] = baseline_avg
            per_question_baseline[q_idx]["baseline_std"] = np.std(
                per_question_baseline[q_idx]["baseline_values"]
            )
            cross_questions_divergence_total += baseline_avg

        # Initialize results and progress bar
        comparison_results = []
        total_comparisons = len(groups) * (len(groups) - 1) // 2

        with tqdm(total=total_comparisons, desc="Group Comparisons") as pbar:
            # Iterate over unique group pairs
            for base_group, target_group in combinations(groups.keys(), 2):
                base_group_details = groups_details[base_group]
                target_group_details = groups_details[target_group]
                with tqdm(
                    total=num_questions,
                    desc=f"Questions {base_group} vs {target_group}",
                    position=1,
                    leave=False,
                ) as inner_pbar:
                    per_question_divergence = {}
                    total_divergence = 0
                    for question_idx in range(num_questions):
                        base_group_centroid = base_group_details[question_idx][
                            "centroid"
                        ]
                        target_group_centroid = target_group_details[question_idx][
                            "centroid"
                        ]
                        # Calculate cross-group divergence
                        cross_group_divergence = jensen_shannon_divergence(
                            base_group_centroid, target_group_centroid
                        )
                        total_divergence += cross_group_divergence

                        per_question_divergence[f"question_{question_idx}"] = {
                            "divergence": cross_group_divergence
                        }

                        # Update inner progress bar
                        inner_pbar.update(1)
                        inner_pbar.refresh()  # Force immediate update

                    # Store the result
                    comparison_results.append(
                        {
                            "compared_groups": f"{base_group.replace(' ', '_')}--{target_group.replace(' ', '_')}",
                            "compared_details": {
                                "average_divergence": total_divergence / num_questions,
                                "per_question_details": per_question_divergence,
                            },
                        }
                    )

                    # Update progress bar
                    pbar.update(1)

        return (
            comparison_results,
            groups_details,
            per_question_baseline,
            cross_questions_divergence_total / num_questions,
        )

    def ba_scenarios_baseline_pairwise(self, values_selection_results):
        """Compute baseline for BA_user and BA_dialogue with global Jenson-shannon centroids.

        Args:
            values_selection_results (list): List of values selections for users.
        """

        distributions_arry = np.array(
            [
                np.array(
                    [
                        question["normalized_probs"]
                        for question in value_selections["value_selections"]
                    ]
                )
                for value_selections in values_selection_results
            ]
        ).swapaxes(0, 1)

        num_question = distributions_arry.shape[0]
        total_divergence = 0
        per_question_results = {}
        with tqdm(total=num_question, desc="Questions Comparison") as pbar:
            for question_idx in range(num_question):
                centroid, failed_count, avg_divergence = compute_js_centroid_and_avg(
                    distributions_arry[question_idx]
                )
                per_question_results[f"question_{question_idx}"] = {
                    "average_divergence": avg_divergence,
                    "centroid": centroid.tolist(),
                    "failed_count": failed_count,
                }
                total_divergence += avg_divergence

                # Update inner progress bar
                pbar.update(1)
                pbar.refresh()  # Force immediate update

        baseline_divergence = total_divergence / num_question

        return {
            "baseline_divergence": baseline_divergence,
            "per_question_details": per_question_results,
        }

    def cross_datasets_centroids_divergences(self):
        """Compute the cross dataset centroids divergences"""
        user_distributions_arry = np.array(
            [
                np.array(
                    [
                        question["normalized_probs"]
                        for question in value_selections["value_selections"]
                    ]
                )
                for value_selections in self.direct_values_predictions
            ]
        ).swapaxes(0, 1)

        dialogue_distributions_arry = np.array(
            [
                np.array(
                    [
                        question["normalized_probs"]
                        for question in value_selections["value_selections"]
                    ]
                )
                for value_selections in self.dialogue_values_predictions
            ]
        ).swapaxes(0, 1)

        num_questions = user_distributions_arry.shape[0]
        per_question_centroids = {}
        overall_divergences = []
        for q_idx in tqdm(range(num_questions), desc="Compute Centroid Globally"):
            user_centroid = compute_js_centroid(user_distributions_arry[q_idx])[0]
            dialogue_centroid = compute_js_centroid(dialogue_distributions_arry[q_idx])[
                0
            ]
            divergence = jensen_shannon_divergence(user_centroid, dialogue_centroid)
            overall_divergences.append(divergence)
            per_question_centroids[q_idx] = {
                "BA_user_centroid": user_centroid,
                "BA_dialogue_centroid": dialogue_centroid,
                "divergence": divergence,
            }

        return {
            "avg_divergence": np.mean(overall_divergences),
            "std_divergence": np.std(overall_divergences),
            "per_question_details": per_question_centroids,
        }

    def cross_datasets_divergences(self):
        """Compute the cross dataset pairwise divergences"""
        user_distributions_arry = np.array(
            [
                np.array(
                    [
                        question["normalized_probs"]
                        for question in value_selections["value_selections"]
                    ]
                )
                for value_selections in self.direct_values_predictions
            ]
        ).swapaxes(0, 1)

        dialogue_distributions_arry = np.array(
            [
                np.array(
                    [
                        question["normalized_probs"]
                        for question in value_selections["value_selections"]
                    ]
                )
                for value_selections in self.dialogue_values_predictions
            ]
        ).swapaxes(0, 1)

        num_questions = user_distributions_arry.shape[0]
        per_question_divergences = {}
        overall_divergences = []
        for q_idx in tqdm(range(num_questions), desc="Compute Centroid Globally"):
            user_valid_distributions, u_failed_count = filter_rows(
                user_distributions_arry[q_idx]
            )
            dialogue_valid_distributions, d_failed_count = filter_rows(
                dialogue_distributions_arry[q_idx]
            )
            per_question_divergences_list = []

            for user_distribution, dialogue_distribution in zip(
                user_valid_distributions, dialogue_valid_distributions
            ):
                per_question_divergences_list.append(
                    jensen_shannon_divergence(user_distribution, dialogue_distribution)
                )

            avg_divergence = np.mean(per_question_divergences_list)
            per_question_divergences[q_idx] = {
                "avg_divergence": avg_divergence,
                "std_divergence": np.mean(per_question_divergences_list),
                "failed_count": u_failed_count + d_failed_count,
            }
            overall_divergences.append(avg_divergence)

        return {
            "avg_divergence": np.mean(overall_divergences),
            "std_divergence": np.std(overall_divergences),
            "per_question_divergences": per_question_divergences,
        }

    def cross_datasets_divergences_baseline(self):
        """Compute the cross dataset divergences baseline"""
        user_distributions_arry = np.array(
            [
                np.array(
                    [
                        question["normalized_probs"]
                        for question in value_selections["value_selections"]
                    ]
                )
                for value_selections in self.direct_values_predictions
            ]
        ).swapaxes(0, 1)

        dialogue_distributions_arry = np.array(
            [
                np.array(
                    [
                        question["normalized_probs"]
                        for question in value_selections["value_selections"]
                    ]
                )
                for value_selections in self.dialogue_values_predictions
            ]
        ).swapaxes(0, 1)

        num_questions = user_distributions_arry.shape[0]
        per_question_divergences = {}
        overall_divergences = []
        for q_idx in tqdm(range(num_questions), desc="Compute Centroid Globally"):
            user_valid_distributions, u_failed_count = filter_rows(
                user_distributions_arry[q_idx]
            )
            dialogue_valid_distributions, d_failed_count = filter_rows(
                dialogue_distributions_arry[q_idx]
            )
            per_question_divergences_list = []

            for _ in range(
                len(user_valid_distributions)
            ):  # Loop over the desired number of comparisons
                # Randomly sample a user distribution and a dialogue distribution
                random_user_distribution = random.choice(user_valid_distributions)
                random_dialogue_distribution = random.choice(
                    dialogue_valid_distributions
                )

                # Compute the divergence and append to the list
                per_question_divergences_list.append(
                    jensen_shannon_divergence(
                        random_user_distribution, random_dialogue_distribution
                    )
                )

            avg_divergence = np.mean(per_question_divergences_list)
            per_question_divergences[q_idx] = {
                "avg_divergence": avg_divergence,
                "std_divergence": np.mean(per_question_divergences_list),
                "failed_count": u_failed_count + d_failed_count,
            }
            overall_divergences.append(avg_divergence)

        return {
            "avg_divergence": np.mean(overall_divergences),
            "std_divergence": np.std(overall_divergences),
            "per_question_divergences": per_question_divergences,
        }

    def cross_datasets_divergences_id_based(self):
        """Compute the cross dataset pairwise divergences"""
        user_distributions_arry = np.array(
            [
                np.array(
                    [
                        question["selected_option_id"]
                        for question in value_selections["value_selections"]
                    ]
                )
                for value_selections in self.direct_values_predictions
            ]
        )

        dialogue_distributions_arry = np.array(
            [
                np.array(
                    [
                        question["selected_option_id"]
                        for question in value_selections["value_selections"]
                    ]
                )
                for value_selections in self.dialogue_values_predictions
            ]
        )

        overall_divergences = []
       

        for user_distribution, dialogue_distribution in zip(
            user_distributions_arry, dialogue_distributions_arry
        ):
            overall_divergences.append(
                compute_emd(user_distribution, dialogue_distribution)
            )

        return {
            "avg_divergence": np.mean(overall_divergences),
            "std_divergence": np.std(overall_divergences),
        }
    
    def cross_datasets_divergences_baseline_id_based(self):
        """Compute the cross dataset divergences baseline"""
        user_distributions_arry = np.array(
            [
                np.array(
                    [
                        question["selected_option_id"]
                        for question in value_selections["value_selections"]
                    ]
                )
                for value_selections in self.direct_values_predictions
            ]
        )

        dialogue_distributions_arry = np.array(
            [
                np.array(
                    [
                        question["selected_option_id"]
                        for question in value_selections["value_selections"]
                    ]
                )
                for value_selections in self.dialogue_values_predictions
            ]
        )

        overall_divergences = []

        for idx, user_distribution in enumerate(user_distributions_arry):  # Loop over the desired number of comparisons
            # Randomly sample a user distribution and a dialogue distribution
            random_idx = random.randint(0, user_distributions_arry.shape[0]-1)
            while random_idx == idx:
                random_idx = random.randint(0, user_distributions_arry.shape[0]-1)
            random_dialogue_distribution = dialogue_distributions_arry[random_idx]

            # Compute the divergence and append to the list
            overall_divergences.append(
                compute_emd(
                    user_distribution, random_dialogue_distribution
                )
            )

        return {
            "avg_divergence": np.mean(overall_divergences),
            "std_divergence": np.std(overall_divergences)
        }