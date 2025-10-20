import argparse
import json
import logging
import os
import random
import sys
from collections import OrderedDict
from copy import deepcopy
from typing import Any, Dict, Iterable, List, Mapping, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from llm_behavior_adaptation.utils import register_logger
from llm_behavior_adaptation.value_measurement.formulas import (
    componentwise_centroid,
    emd_distance,
)

logger = logging.getLogger(__name__)
register_logger(logger)


DATASET_DIR = "datasets/wvs_benchmarks"
ATTRIBUTES = [
    "age",
    "continent_of_residence",
    "immigration_status",
    "highest_level_of_education",
    "socioeconomic_status",
    "occupation_group",
]


def load_jsonl_file(file_path):
    """Load jsonl file"""
    list_of_json_objs = []
    with open(file_path, "r", encoding="utf-8") as file:
        for json_obj in file.readlines():
            dialogue_obj = json.loads(json_obj)
            list_of_json_objs.append(dialogue_obj)
    return list_of_json_objs


class ValuesComparison:
    """class for doing the values comparison"""

    def __init__(
        self,
        user_profile_dataset: pd.DataFrame,
        user_value_dataset: pd.DataFrame,
        ba_user_results: Dict,
        ba_dialogue_career_results: Dict,
        ba_dialogue_investment_results: Dict,
        picked_questions: Dict,
        results_output_path: str,
        verbose: int = 0,
    ) -> None:
        """
        Initializes the ValuesComparison class with user/profile/value datasets and
        behavior-adaptation results from user and dialogue tasks.

        Args:
            user_profile_dataset (pd.DataFrame): DataFrame of user profiles.
            user_value_dataset (pd.DataFrame): DataFrame of user values (ground truth or labels).
            ba_user_results (List[Dict]): BA outputs for direct user-level predictions.
            ba_dialogue_career_results (List[Dict]): BA outputs from career dialogues.
            ba_dialogue_investment_results (List[Dict]): BA outputs from investment dialogues.
            picked_questions (Dict): Mapping of selected question IDs/metadata.
            verbose (int, optional): Verbosity level. Defaults to 0.

        Raises:
            TypeError: If inputs are of incorrect types.
            ValueError: If verbose is not 0 or 1.
        """
        if not isinstance(user_profile_dataset, pd.DataFrame):
            raise TypeError("user_profile_dataset must be a pandas DataFrame.")
        if not isinstance(user_value_dataset, pd.DataFrame):
            raise TypeError("user_value_dataset must be a pandas DataFrame.")
        if not isinstance(picked_questions, dict):
            raise TypeError("picked_questions must be a dictionary.")
        if verbose not in (0, 1):
            raise ValueError("verbose must be 0 or 1.")

        self._user_profile_dataset = user_profile_dataset
        self._ba_user_results = ba_user_results
        self._ba_dialogue_career_results = ba_dialogue_career_results
        self._ba_dialogue_investment_results = ba_dialogue_investment_results
        self._picked_questions = picked_questions
        self._results_output_path = results_output_path
        self._verbose = verbose

        self._all_questions = {}
        for _, q_list in picked_questions.items():
            self._all_questions.update(q_list)

        self._user_value_dataset = user_value_dataset[
            ["D_INTERVIEW"] + list(self.all_questions.keys())
        ]

    # ----------------------
    # Properties
    # ----------------------
    @property
    def user_profile_dataset(self) -> pd.DataFrame:
        """Returns the user profile dataset."""
        return self._user_profile_dataset

    @property
    def user_value_dataset(self) -> pd.DataFrame:
        """Returns the user value dataset."""
        return self._user_value_dataset

    @property
    def ba_user_results(self) -> List[Dict]:
        """Returns BA results for direct user-level predictions."""
        return self._ba_user_results

    @property
    def ba_dialogue_career_results(self) -> List[Dict]:
        """Returns BA results from career dialogues."""
        return self._ba_dialogue_career_results

    @property
    def ba_dialogue_investment_results(self) -> List[Dict]:
        """Returns BA results from investment dialogues."""
        return self._ba_dialogue_investment_results

    @property
    def picked_questions(self) -> Dict:
        """Returns the mapping of selected/picked questions."""
        return self._picked_questions

    @property
    def all_questions(self) -> Dict:
        """Returns questions without categories"""
        return self._all_questions

    @property
    def results_output_path(self) -> str:
        """Returns the results output path."""
        return self._results_output_path

    @property
    def verbose(self) -> int:
        """Returns the verbosity level."""
        return self._verbose

    # ----------------------
    # CLI args (unchanged)
    # ----------------------
    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """
        Add command-line arguments for ValuesComparison.

        Args:
            parser (argparse.ArgumentParser): Parser to add args to.

        Returns:
            argparse.ArgumentParser: The updated parser.
        """
        parser.add_argument(
            "--user-profile-dataset",
            type=str,
            default=f"{DATASET_DIR}/sampled_demographic_features.csv",
            help="Path to the user profile dataset file (e.g., CSV/Parquet).",
        )
        parser.add_argument(
            "--user-value-dataset",
            type=str,
            default=f"{DATASET_DIR}/sampled_values_df.csv",
            help="Path to the user value dataset file (e.g., CSV/Parquet).",
        )
        parser.add_argument(
            "--ba-user-results",
            type=str,
            required=True,
            help="Path to BA user-level results (e.g., JSON/JSONL).",
        )
        parser.add_argument(
            "--ba-dialogue-career-results",
            type=str,
            required=True,
            help="Path to BA dialogue results for career domain (e.g., JSON/JSONL).",
        )
        parser.add_argument(
            "--ba-dialogue-investment-results",
            type=str,
            required=True,
            help="Path to BA dialogue results for investment domain (e.g., JSON/JSONL).",
        )
        parser.add_argument(
            "--results-output-path",
            type=str,
            required=True,
            help="Output path for results",
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

        def _process_model_outputs(original_answers_list):
            processed_answers_dict = {}
            for answer_details in original_answers_list:
                per_user_answers = {}
                for user_id, answers in answer_details.items():
                    for cat_answers in list(answers.values()):
                        for answer in cat_answers:
                            per_user_answers.update(answer)
                    if user_id in processed_answers_dict:
                        print(user_id)
                    processed_answers_dict[user_id] = per_user_answers
            return processed_answers_dict

        user_profile_dataset = _read_csv(args.user_profile_dataset)
        user_value_dataset = _read_csv(args.user_value_dataset)

        with open(
            f"{DATASET_DIR}/picked_questions.json", "r", encoding="utf-8"
        ) as picked_question_f:
            picked_questions = json.load(picked_question_f)

        ba_user_results = _process_model_outputs(load_jsonl_file(args.ba_user_results))
        ba_dialogue_career_results = _process_model_outputs(
            load_jsonl_file(args.ba_dialogue_career_results)
        )

        ba_dialogue_investment_results = _process_model_outputs(
            load_jsonl_file(args.ba_dialogue_investment_results)
        )

        return cls(
            user_profile_dataset=user_profile_dataset,
            user_value_dataset=user_value_dataset,
            ba_user_results=ba_user_results,
            ba_dialogue_career_results=ba_dialogue_career_results,
            ba_dialogue_investment_results=ba_dialogue_investment_results,
            picked_questions=picked_questions,
            results_output_path=args.results_output_path,
            verbose=args.verbose,
        )

    def _get_index_list_for_groups(
        self, target_col: str, include_unknown: bool = False
    ) -> Dict[str, List[int]]:
        """
        Get grouped indices for a target column from self.user_profile_dataset.

        Args:
            target_col (str): The column to group by.
            include_unknown (bool): If True, unrecognized education labels are put into an 'Unknown' bin.

        Returns:
            dict: {group_name: [row_index, ...]}
        """
        if target_col not in self.user_profile_dataset.columns:
            raise KeyError(f"Column '{target_col}' not found in user_profile_dataset.")
        if "D_INTERVIEW" not in self.user_profile_dataset.columns:
            raise KeyError("Column 'D_INTERVIEW' not found in user_profile_dataset.")

        # Base copy and drop rows where target_col is exactly "not sure" (normalized)
        df = deepcopy(self.user_profile_dataset)
        norm_col = df[target_col].astype(str).str.strip().str.lower()
        df = df[norm_col != "not sure"]

        # --- Age binning branch ---
        if target_col.lower() == "age":
            bins = [-float("inf"), 30, 40, 50, 60, float("inf")]
            labels = ["<30", "30-40", "40-50", "50-60", ">60"]

            tmp_col = "_range_group"
            df[tmp_col] = pd.cut(df[target_col], bins=bins, labels=labels, right=False)

            grouped = (
                df.dropna(subset=[tmp_col])
                .groupby(tmp_col, sort=False)
                .apply(lambda x: x.index.tolist())
                .to_dict()
            )
            # Keep label order
            return {lab: grouped[lab] for lab in labels if lab in grouped}

        # --- Education binning branch (exact-match to 5 bins) ---
        if target_col == "highest_level_of_education":
            bins_to_labels = OrderedDict(
                {
                    "Basic education": [
                        "Early childhood education",
                        "Primary education",
                        "Lower secondary education",
                    ],
                    "High school & equivalent": [
                        "Upper secondary education",
                        "Post-secondary non-tertiary education",
                    ],
                    "Short-cycle tertiary": [
                        "Short-cycle tertiary education",
                    ],
                    "Bachelor": [
                        "Bachelor or equivalent",
                    ],
                    "Master’s & Doctoral": [
                        "Master or equivalent",
                        "Doctoral or equivalent",
                    ],
                }
            )
            # Reverse lookup: raw label -> bin name (exact match)
            label_to_bin = {
                lbl: bin_name
                for bin_name, labels in bins_to_labels.items()
                for lbl in labels
            }

            tmp_col = "_edu_bin"
            df = self.user_profile_dataset.copy()
            df[tmp_col] = df[target_col].map(label_to_bin)

            if include_unknown:
                df[tmp_col] = df[tmp_col].fillna("Unknown")

            grouped = (
                df.dropna(subset=[tmp_col])
                .groupby(tmp_col, sort=False)
                .apply(lambda x: x.index.tolist())
                .to_dict()
            )

            # Keep bins in the desired order, append Unknown last (if any)
            ordered = {k: grouped[k] for k in bins_to_labels.keys() if k in grouped}
            if include_unknown and "Unknown" in grouped:
                ordered["Unknown"] = grouped["Unknown"]
            return ordered

        # --- Fallback: group by the raw values of the target column ---
        grouped = (
            df.groupby(target_col, sort=False)
            .apply(lambda x: x.index.tolist())
            .to_dict()
        )
        return grouped

    def _get_user_id_list_for_groups(
        self, target_col: str, include_unknown: bool = False
    ) -> Dict[str, List[str]]:
        """
        Group user IDs (from D_INTERVIEW) by a target column.

        Special handling:
        - Age: bucket into fixed ranges.
        - highest_level_of_education: map original labels into 5 consolidated bins.

        Args:
            target_col (str): Column to group by.
            include_unknown (bool): If True, unrecognized education labels are put into an 'Unknown' bin.

        Returns:
            dict: {group_label: [user_id_str, ...]}
        """
        if target_col not in self.user_profile_dataset.columns:
            raise KeyError(f"Column '{target_col}' not found in user_profile_dataset.")
        if "D_INTERVIEW" not in self.user_profile_dataset.columns:
            raise KeyError("Column 'D_INTERVIEW' not found in user_profile_dataset.")

        # Base copy and drop rows where target_col is exactly "not sure" (normalized)
        df = deepcopy(self.user_profile_dataset)
        norm_col = df[target_col].astype(str).str.strip().str.lower()
        df = df[norm_col != "not sure"]

        # --- Age binning branch ---
        if target_col.lower() == "age":
            bins = [-float("inf"), 30, 40, 50, 60, float("inf")]
            labels = ["<30", "30-40", "40-50", "50-60", ">60"]

            tmp_col = "_range_group"
            df[tmp_col] = pd.cut(df[target_col], bins=bins, labels=labels, right=False)

            grouped = (
                df.dropna(subset=[tmp_col])
                .groupby(tmp_col, sort=False)
                .apply(lambda x: [str(d_id) for d_id in x["D_INTERVIEW"].tolist()])
                .to_dict()
            )
            # Keep label order
            return {lab: grouped[lab] for lab in labels if lab in grouped}

        # --- Education binning branch (exact-match to 5 bins) ---
        if target_col == "highest_level_of_education":
            bins_to_labels = OrderedDict(
                {
                    "Basic education": [
                        "Early childhood education",
                        "Primary education",
                        "Lower secondary education",
                    ],
                    "High school & equivalent": [
                        "Upper secondary education",
                        "Post-secondary non-tertiary education",
                    ],
                    "Short-cycle tertiary": [
                        "Short-cycle tertiary education",
                    ],
                    "Bachelor": [
                        "Bachelor or equivalent",
                    ],
                    "Master’s & Doctoral": [
                        "Master or equivalent",
                        "Doctoral or equivalent",
                    ],
                }
            )
            # Reverse lookup: raw label -> bin name (exact match)
            label_to_bin = {
                lbl: bin_name
                for bin_name, labels in bins_to_labels.items()
                for lbl in labels
            }

            tmp_col = "_edu_bin"
            df = self.user_profile_dataset.copy()
            df[tmp_col] = df[target_col].map(label_to_bin)

            if include_unknown:
                df[tmp_col] = df[tmp_col].fillna("Unknown")

            grouped = (
                df.dropna(subset=[tmp_col])
                .groupby(tmp_col, sort=False)
                .apply(lambda x: [str(d_id) for d_id in x["D_INTERVIEW"].tolist()])
                .to_dict()
            )

            # Keep bins in desired order; append Unknown last (if any)
            ordered = {k: grouped[k] for k in bins_to_labels.keys() if k in grouped}
            if include_unknown and "Unknown" in grouped:
                ordered["Unknown"] = grouped["Unknown"]
            return ordered

        # --- Fallback: group by raw values in target_col ---
        grouped = (
            df.dropna(subset=[target_col])
            .groupby(target_col, sort=False)
            .apply(lambda x: [str(d_id) for d_id in x["D_INTERVIEW"].tolist()])
            .to_dict()
        )
        return grouped

    def _map_human_values_to_groups(
        self,
        group_dict: Dict[str, List[int]],
        user_values_df: pd.DataFrame,
    ) -> Dict[str, pd.DataFrame]:
        """Map each group to its corresponding subset of rows in `user_values_df`.

        Args:
            group_dict:
                Dictionary where:
                - key (str): Group name or label.
                - value (List[int]): List of row indices belonging to that group.
                Example:
                    {
                        "group_A": [0, 1, 5],
                        "group_B": [2, 3],
                        "group_C": [4]
                    }
            user_values_df:
                Pandas DataFrame containing user values.
                Must share the same index as the indices listed in `group_dict`.

        Returns:
            Dict[str, pd.DataFrame]:
                Dictionary mapping each group name to a DataFrame containing
                only the rows of `user_values_df` corresponding to that group's indices.

        Example:
            >>> group_dict = {"A": [0, 2], "B": [1, 3]}
            >>> df = pd.DataFrame({"x": [10, 20, 30, 40]}, index=[0, 1, 2, 3])
            >>> map_user_values_to_groups(group_dict, df)
            {
                "A":     x
                        0  10
                        2  30,
                "B":     x
                        1  20
                        3  40
            }
        """
        grouped_values = {}
        for group_name, idx_list in group_dict.items():
            # Use .loc to preserve labels and handle non-integer indexes safely
            sub = user_values_df[user_values_df.index.isin(idx_list)]
            grouped_values[group_name] = sub.to_dict(orient="index")
        return grouped_values

    def _map_model_results_to_groups(
        self,
        group_dict: Dict[str, List[int]],
        user_values_dict: Dict[str, Dict],
    ) -> Dict[str, List[Dict]]:
        """Map each group to its corresponding subset of rows in `user_values_df`.

        Args:
            group_dict:
                Dictionary where:
                - key (str): Group name or label.
                - value (List[int]): List of row indices belonging to that group.
                Example:
                    {
                        "group_A": [0, 1, 5],
                        "group_B": [2, 3],
                        "group_C": [4]
                    }
            user_values_dict:
                Must share the same index as the indices listed in `group_dict`.

        Returns:
            Dict[str, List[Dict]]:
                Dictionary mapping each group name to a list of
                answers corresponding to that group's indices.

        """
        group_answers = {}
        for group_name, key_list in group_dict.items():
            # Use .loc to preserve labels and handle non-integer indexes safely
            group_answers[group_name] = [
                {
                    q_key: q_selection["option_id"]
                    for q_key, q_selection in user_values_dict[k].items()
                }
                for k in key_list
                if k in user_values_dict
            ]
        return group_answers

    def pairwise_group_emd_list(
        self,
        group_centroids: Dict[str, Dict[str, int]],
        question_metadata: Mapping[str, Mapping[str, Any]],
        *,
        normalize: bool = True,
        skip_out_of_range: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Compute pairwise EMD between group centroids and return:
        [{"compared_groups": "<g1--g2>", "distance": int}, ...]

        Notes:
        - Only unique unordered pairs are included (g1 != g2, with i < j).
        - Distance is rounded to nearest int (int(round(...))).
        """
        groups = list(group_centroids.keys())

        # (Optional) restrict metadata to qids present in any centroid
        union_qids = set().union(*[c.keys() for c in group_centroids.values()])
        meta = {q: question_metadata[q] for q in union_qids if q in question_metadata}

        out: List[Dict[str, Any]] = []
        for i, gi in enumerate(groups):
            ai = group_centroids[gi]
            for j in range(i + 1, len(groups)):
                gj = groups[j]
                bj = group_centroids[gj]
                d = emd_distance(
                    ai,
                    bj,
                    meta,
                    normalize=normalize,
                    skip_out_of_range=skip_out_of_range,
                )
                out.append(
                    {
                        "compared_groups": f"{gi}--{gj}",
                        "compared_details": {"average_divergence": float(d)},
                    }
                )
        return out

    def baseline_emd(
        self,
        global_centroid: Dict[str, int],
        group_centroids: Dict[str, Dict[str, int]],
        question_metadata: Mapping[str, Mapping[str, Any]],
        *,
        normalize: bool = True,
        skip_out_of_range: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Compute pairwise EMD between group centroids and return:
        [{"compared_groups": "<g1--g2>", "distance": int}, ...]

        Notes:
        - Only unique unordered pairs are included (g1 != g2, with i < j).
        - Distance is rounded to nearest int (int(round(...))).
        """
        groups = list(group_centroids.keys())

        # (Optional) restrict metadata to qids present in any centroid
        union_qids = set().union(*[c.keys() for c in group_centroids.values()])
        meta = {q: question_metadata[q] for q in union_qids if q in question_metadata}

        overall_distances = 0
        for gi in groups:
            group_centroid = group_centroids[gi]

            overall_distances += emd_distance(
                group_centroid,
                global_centroid,
                meta,
                normalize=normalize,
                skip_out_of_range=skip_out_of_range,
            )

        return overall_distances / len(groups)

    def _pick_model_results_option_id(self, model_results):
        results_only_dict = {}
        for user_id, user_answers in model_results.items():
            if user_id in results_only_dict:
                logger.warning("Duplicate user id: %s", user_id)
            results_only_dict[user_id] = {
                q_key: q_selection["option_id"]
                for q_key, q_selection in user_answers.items()
            }
        return results_only_dict

    def _compute_id_matched_divergences(
        self,
        dist_a: Mapping[str, Mapping[str, int]],
        dist_b: Mapping[str, Mapping[str, int]],
        ids: Optional[Iterable[str]] = None,
    ) -> Dict[str, object]:
        """
        Core computation for ID-matched divergences between two datasets A and B.
        Always shows a tqdm progress bar.
        """
        from tqdm import tqdm

        a_ids = set(dist_a.keys())
        b_ids = set(dist_b.keys())
        common_ids = list(set(ids) if ids is not None else (a_ids & b_ids))

        divergences: List[float] = []

        for uid in tqdm(
            common_ids,
            total=len(common_ids),
            desc="EMD (ID-matched A↔B)",
            unit="id",
            leave=False,
        ):
            a = dist_a.get(uid)
            b = dist_b.get(uid)
            if a is None or b is None:
                continue
            divergences.append(emd_distance(a, b, self.all_questions))

        return {
            "divergences": divergences,
            "n_pairs": len(common_ids),
            "n_users_only": len(a_ids - b_ids),  # kept for backward compat
            "n_dialogues_only": len(b_ids - a_ids),  # kept for backward compat
        }

    def _compute_baseline_two_random_matches(
        self,
        dist_a: Mapping[str, Mapping[str, int]],
        dist_b: Mapping[str, Mapping[str, int]],
        *,
        picks: int = 2,
        exclude_self: bool = True,
        rng: Optional[random.Random] = None,
    ) -> Dict[str, object]:
        """
        Baseline: for each id in A, compare to `picks` random ids from B (prefer non-self),
        average the distances per id, then return per-id means and counts.
        Always shows a tqdm progress bar.
        """
        from tqdm import tqdm

        if rng is None:
            rng = random

        a_ids = list(dist_a.keys())
        b_ids_all = list(dist_b.keys())

        per_user_means: List[float] = []

        # Show an immediate (possibly zero-length) bar for clarity
        progress_iter = tqdm(
            a_ids,
            total=len(a_ids),
            desc="Baseline EMD (A vs random B)",
            unit="id",
            leave=False,
        )

        if not b_ids_all:
            # still consume the iterator to render a completed bar
            for _ in progress_iter:
                pass
            return {
                "per_user_means": per_user_means,
                "n_users": len(a_ids),
                "n_dialogue_pool": 0,
                "n_effective_users": 0,
            }

        for uid in progress_iter:
            a_dist = dist_a[uid]

            # Prefer non-self candidates from B
            candidates = [bid for bid in b_ids_all if (bid != uid) or not exclude_self]
            if exclude_self and not candidates:
                if uid in dist_b:
                    candidates = [uid]
                else:
                    # No candidate at all for this uid
                    continue

            # Sample B ids
            if len(candidates) >= picks:
                match_ids = rng.sample(candidates, picks)  # without replacement
            else:
                match_ids = [
                    rng.choice(candidates) for _ in range(picks)
                ]  # with replacement

            dists = []
            for bid in match_ids:
                b_dist = dist_b[bid]
                dists.append(emd_distance(a_dist, b_dist, self.all_questions))

            per_user_means.append(float(np.mean(dists)))

        return {
            "per_user_means": per_user_means,
            "n_users": len(a_ids),
            "n_dialogue_pool": len(b_ids_all),
            "n_effective_users": len(per_user_means),
        }

    def cross_datasets_divergences_id_based(
        self,
        dialogue_type: str,
    ):
        """Compute the cross dataset pairwise divergences with a progress bar (id-matched)."""
        user_distributions = self._pick_model_results_option_id(self.ba_user_results)
        dialogue_attr = f"ba_dialogue_{dialogue_type}_results"
        dialogue_distributions = self._pick_model_results_option_id(
            getattr(self, dialogue_attr)
        )

        # Compute core results first (no I/O)
        core = self._compute_id_matched_divergences(
            user_distributions, dialogue_distributions
        )

        divergences: List[float] = core["divergences"]
        if not divergences:
            return {
                "avg_divergence": float("nan"),
                "std_divergence": float("nan"),
                "n_pairs": core["n_pairs"],
                "n_users_only": core["n_users_only"],
                "n_dialogues_only": core["n_dialogues_only"],
            }

        return {
            "avg_divergence": float(np.mean(divergences)),
            "std_divergence": float(np.std(divergences)),
            "n_pairs": core["n_pairs"],
            "n_users_only": core["n_users_only"],
            "n_dialogues_only": core["n_dialogues_only"],
        }

    def cross_datasets_divergences_baseline_id_based(
        self,
        dialogue_type: str,
    ):
        """
        Baseline: for each user, compare to 2 random dialogue users (preferring j != i),
        average the two EMDs, then aggregate mean/std across users. Includes optional progress bar.
        """
        # Build id -> {qid: option_id} dicts
        user_distributions = self._pick_model_results_option_id(self.ba_user_results)
        dialogue_attr = f"ba_dialogue_{dialogue_type}_results"
        dialogue_distributions = self._pick_model_results_option_id(
            getattr(self, dialogue_attr)
        )

        # Core computation (no I/O)
        core = self._compute_baseline_two_random_matches(
            user_distributions,
            dialogue_distributions,
            picks=2,
            exclude_self=True,
            rng=random,
        )

        per_user_means: List[float] = core["per_user_means"]
        if not per_user_means:
            return {
                "avg_divergence": float("nan"),
                "std_divergence": float("nan"),
                "n_users": core["n_users"],
                "n_dialogue_pool": core["n_dialogue_pool"],
                "n_effective_users": core["n_effective_users"],
            }

        return {
            "avg_divergence": float(np.mean(per_user_means)),
            "std_divergence": float(np.std(per_user_means)),
            "n_users": core["n_users"],
            "n_dialogue_pool": core["n_dialogue_pool"],
            "n_effective_users": core["n_effective_users"],
        }

    def compute_results_against_human(self):
        human_results_distributions = (
            self.user_value_dataset.astype({"D_INTERVIEW": str})
            .groupby("D_INTERVIEW", as_index=True)  # handles duplicates explicitly
            .last()[list(self.all_questions.keys())]
            .to_dict(orient="index")
        )

        results_dict = {}
        for results_type in ["ba_user", "ba_dialogue_career", "ba_dialogue_investment"]:
            core = self._compute_id_matched_divergences(
                human_results_distributions,
                self._pick_model_results_option_id(
                    getattr(self, f"{results_type}_results")
                ),
            )
            divergences: List[float] = core["divergences"]
            baseline_core = self._compute_baseline_two_random_matches(
                human_results_distributions,
                self._pick_model_results_option_id(
                    getattr(self, f"{results_type}_results")
                ),
            )
            baseline_per_user_means: List[float] = baseline_core["per_user_means"]

            results_dict[f"{results_type}_against_human"] = {
                "distance": {
                    "avg_divergence": float(np.mean(divergences)),
                    "std_divergence": float(np.std(divergences)),
                    "n_users": core["n_pairs"],
                    "n_dialogue_pool": core["n_users_only"],
                    "n_effective_users": core["n_dialogues_only"],
                },
                "baseline": {
                    "avg_divergence": float(np.mean(baseline_per_user_means)),
                    "std_divergence": float(np.std(baseline_per_user_means)),
                    "n_users": baseline_core["n_users"],
                    "n_dialogue_pool": baseline_core["n_dialogue_pool"],
                    "n_effective_users": baseline_core["n_effective_users"],
                },
            }

        return results_dict

    def _calculate_centroids_among_groups(self, grouped_output_values):
        group_centroids = {}
        for group_name, group_q_ans_maps in grouped_output_values.items():
            if isinstance(group_q_ans_maps, dict):
                group_q_ans_maps = list(group_q_ans_maps.values())
            g_centroid = componentwise_centroid(group_q_ans_maps, self.all_questions)
            logger.info("for group: %s, centroid: %s", group_name, g_centroid)
            group_centroids[group_name] = g_centroid

        return group_centroids

    def compute_attributes_groups_distances(
        self, results_attribute, show_progress: bool = True
    ):
        """compute group distances for attributes"""
        computed_results = {}
        user_values_dict = getattr(self, results_attribute)

        all_samples = []
        for _, user_answers in user_values_dict.items():
            one_user_answers = {}
            for q_key, q_selection in user_answers.items():
                one_user_answers[q_key] = q_selection["option_id"]
            all_samples.append(one_user_answers)

        print(all_samples[:2])

        global_centroid = componentwise_centroid(all_samples, self.all_questions)

        attributes_iter = (
            tqdm(ATTRIBUTES, desc="Attributes", unit="attr")
            if show_progress
            else ATTRIBUTES
        )
        for attribute in attributes_iter:
            id_based_group_dict = self._get_user_id_list_for_groups(attribute)
            grouped_values = self._map_model_results_to_groups(
                group_dict=id_based_group_dict,
                user_values_dict=user_values_dict,
            )
            group_centroids = self._calculate_centroids_among_groups(
                grouped_output_values=grouped_values
            )
            group_distances = self.pairwise_group_emd_list(
                group_centroids, self.all_questions, normalize=True
            )
            bassline = {
                "overall_baseline": self.baseline_emd(
                    global_centroid, group_centroids, self.all_questions
                )
            }
            computed_results[attribute] = {
                "baseline": bassline,
                "group_distances": group_distances,
            }

        return computed_results

    def compute_human_groups_distances(self, show_progress: bool = True):
        """compute group distances for attributes"""
        computed_results = {}

        all_samples = list(self.user_value_dataset.to_dict(orient="index").values())

        global_centroid = componentwise_centroid(all_samples, self.all_questions)

        attributes_iter = (
            tqdm(ATTRIBUTES, desc="Attributes", unit="attr")
            if show_progress
            else ATTRIBUTES
        )
        for attribute in attributes_iter:
            human_group_dict = self._get_index_list_for_groups(attribute)
            grouped_values = self._map_human_values_to_groups(
                group_dict=human_group_dict,
                user_values_df=self.user_value_dataset,
            )
            group_centroids = self._calculate_centroids_among_groups(
                grouped_output_values=grouped_values
            )
            group_distances = self.pairwise_group_emd_list(
                group_centroids, self.all_questions, normalize=True
            )
            bassline = {
                "overall_baseline": self.baseline_emd(
                    global_centroid, group_centroids, self.all_questions
                )
            }
            computed_results[attribute] = {
                "baseline": bassline,
                "group_distances": group_distances,
            }

        return computed_results


if __name__ == "__main__":

    def _ensure_parent_dir(path: str):
        parent = os.path.dirname(os.path.abspath(path))
        if parent and not os.path.exists(parent):
            os.makedirs(parent, exist_ok=True)

    # 1) Parse CLI
    parser = argparse.ArgumentParser(description="Run ValuesComparison pipeline")
    ValuesComparison.add_cli_args(parser)
    parser.add_argument(
        "--compute-human",
        action="store_true",
        default=False,
        help="If set, compute human-based comparison in addition to model results. Defaults to False.",
    )
    args = parser.parse_args()

    try:
        # 2) Instantiate from args
        vc = ValuesComparison.from_cli_args(args)
        logger.info("Loaded datasets and model outputs.")

        if args.compute_human:
            human_group_results = vc.compute_human_groups_distances()

            with open(args.results_output_path, "w", encoding="utf-8") as f:
                json.dump(human_group_results, f, ensure_ascii=False, indent=2)
            logger.info("Wrote results to %s", args.results_output_path)

        else:
            final_outputs = {}
            result_attributes = [
                "ba_user_results",
                "ba_dialogue_career_results",
                "ba_dialogue_investment_results",
            ]

            for attr_name in result_attributes:
                logger.info("Computing attribute-group distances for: %s", attr_name)
                res = vc.compute_attributes_groups_distances(attr_name)
                final_outputs[attr_name] = res

            cross_datasets_results = {}
            for topic in ["career", "investment"]:
                cross_datasets_results[topic] = {
                    "distance": vc.cross_datasets_divergences_id_based(topic),
                    "baseline": vc.cross_datasets_divergences_baseline_id_based(topic),
                }
                cross_datasets_results[topic] = {
                    "distance": vc.cross_datasets_divergences_id_based(topic),
                    "baseline": vc.cross_datasets_divergences_baseline_id_based(topic),
                }

            final_outputs["cross_datasets_results"] = cross_datasets_results

            final_outputs["against_human"] = vc.compute_results_against_human()

            # Save to JSON
            _ensure_parent_dir(args.results_output_path)
            with open(args.results_output_path, "w", encoding="utf-8") as f:
                json.dump(final_outputs, f, ensure_ascii=False, indent=2)
            logger.info("Wrote results to %s", args.results_output_path)

    except Exception as e:
        logger.exception("ValuesComparison run failed.")
        # Non-zero exit for CI/automation visibility
        sys.exit(1)
