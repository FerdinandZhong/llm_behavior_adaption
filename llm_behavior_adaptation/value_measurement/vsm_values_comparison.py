import argparse
import json
import logging
import os
import random
import sys
from typing import Any, Dict, Iterable, List, Mapping, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from llm_behavior_adaptation.dialogue_dataset_creation.generation_utils import calculate_age
from llm_behavior_adaptation.utils import register_logger
from llm_behavior_adaptation.value_measurement.formulas import (
    componentwise_centroid_vsm,
    compute_emd,
    emd_distance_vsm,
)
from llm_behavior_adaptation.value_measurement.measurement_utils import get_development_level

logger = logging.getLogger(__name__)
register_logger(logger)


DATASET_DIR = "values_results"
ATTRIBUTES = ["Age", "Education Level", "Development"]


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
        ba_user_results: Dict,
        ba_dialogue_career_results: Dict,
        results_output_path: str,
        verbose: int = 0,
    ) -> None:
        """
        Initializes the ValuesComparison class with user/profile/value datasets and
        behavior-adaptation results from user and dialogue tasks.

        Args:
            user_profile_dataset (pd.DataFrame): DataFrame of user profiles.
            ba_user_results (List[Dict]): BA outputs for direct user-level predictions.
            ba_dialogue_career_results (List[Dict]): BA outputs from career dialogues.
            verbose (int, optional): Verbosity level. Defaults to 0.

        Raises:
            TypeError: If inputs are of incorrect types.
            ValueError: If verbose is not 0 or 1.
        """
        if not isinstance(user_profile_dataset, pd.DataFrame):
            raise TypeError("user_profile_dataset must be a pandas DataFrame.")
        if verbose not in (0, 1):
            raise ValueError("verbose must be 0 or 1.")

        self._user_profile_dataset = user_profile_dataset
        self._ba_user_results = ba_user_results
        self._ba_dialogue_career_results = ba_dialogue_career_results

        self._results_output_path = results_output_path
        self._verbose = verbose

    # ----------------------
    # Properties
    # ----------------------
    @property
    def user_profile_dataset(self) -> pd.DataFrame:
        """Returns the user profile dataset."""
        return self._user_profile_dataset

    @property
    def ba_user_results(self) -> List[Dict]:
        """Returns BA results for direct user-level predictions."""
        return self._ba_user_results

    @property
    def ba_dialogue_career_results(self) -> List[Dict]:
        """Returns BA results from career dialogues."""
        return self._ba_dialogue_career_results

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
            default="datasets/seed_dataset.csv",
            help="Path to the user profile dataset file (e.g., CSV/Parquet).",
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

        user_profile_dataset = _read_csv(args.user_profile_dataset)

        def _process_model_outputs(original_model_results: List[Dict]):
            model_results = {
                value_selections["user_idx"]: value_selections["value_selections"]
                for value_selections in original_model_results
            }

            return model_results

        ba_user_results = _process_model_outputs(load_jsonl_file(args.ba_user_results))
        ba_dialogue_career_results = _process_model_outputs(load_jsonl_file(args.ba_dialogue_career_results))

        return cls(
            user_profile_dataset=user_profile_dataset,
            ba_user_results=ba_user_results,
            ba_dialogue_career_results=ba_dialogue_career_results,
            results_output_path=args.results_output_path,
            verbose=args.verbose,
        )

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
        grouped_data = self.user_profile_dataset.groupby("range_group").apply(lambda x: x.index.tolist())

        # Convert to a dictionary and remove empty groups
        grouped_data_dict = {group: indices for group, indices in grouped_data.items() if indices}

        return grouped_data_dict

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
        elif target_col == "Development":
            self.user_profile_dataset["Development"] = self.user_profile_dataset.apply(
                lambda x: get_development_level(x["Country"]), axis=1
            )

        grouped_data = self.user_profile_dataset.groupby(target_col).apply(lambda x: x.index.tolist())
        return grouped_data.to_dict()

    def _map_model_results_to_groups(
        self,
        group_dict: Dict[str, List[int]],
        values_selection_results: Dict[str, Dict],
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
        for group_name, idx_list in group_dict.items():
            group_answers[group_name] = [
                [question_selection["selected_option_id"] for question_selection in values_selection_results[user_idx]]
                for user_idx in idx_list
            ]

        return group_answers

    def pairwise_group_emd_list(
        self,
        group_centroids: Dict[str, Dict[str, int]],
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

        out: List[Dict[str, Any]] = []
        for i, gi in enumerate(groups):
            ai = group_centroids[gi]
            for j in range(i + 1, len(groups)):
                gj = groups[j]
                bj = group_centroids[gj]
                d = emd_distance_vsm(
                    ai,
                    bj,
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

        overall_distances = 0
        for gi in groups:
            group_centroid = group_centroids[gi]

            overall_distances += emd_distance_vsm(
                group_centroid,
                global_centroid,
                normalize=normalize,
                skip_out_of_range=skip_out_of_range,
            )

        return overall_distances / len(groups)

    def _pick_model_results_option_id(self, model_results):
        results_only_dict = {}
        for user_id, user_answers in model_results.items():
            if user_id in results_only_dict:
                logger.warning("Duplicate user id: %s", user_id)
            results_only_dict[user_id] = [
                question_selection["selected_option_id"] for question_selection in user_answers
            ]

        return results_only_dict

    def _compute_id_matched_divergences(
        self,
        dist_a: Mapping[str, Mapping[str, int]],
        dist_b: Mapping[str, Mapping[str, int]],
        ids: Optional[Iterable[str]] = None,
        use_old_emd: bool = True,
    ) -> Dict[str, object]:
        """
        Core computation for ID-matched divergences between two datasets A and B.
        Always shows a tqdm progress bar.

        Returns:
            {
            "user_ids": [uid, ...],                       # deterministic order
            "per_user_divergences": [float, ...],          # aligned with user_ids
            "per_user_map": {uid: float, ...},             # convenience mapping
            "divergences": [float, ...],                   # same as per_user_divergences
            "n_pairs": int,
            "n_users_only": int,
            "n_dialogues_only": int,
            }
        """
        a_ids = set(dist_a.keys())
        b_ids = set(dist_b.keys())
        base_ids = set(ids) if ids is not None else (a_ids & b_ids)
        common_ids = sorted(base_ids)  # <- deterministic ordering

        per_user_divergences: List[float] = []
        per_user_map: Dict[str, float] = {}
        callable = emd_distance_vsm if not use_old_emd else compute_emd

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
            d = callable(a, b)
            per_user_divergences.append(d)
            per_user_map[uid] = d

        return {
            "user_ids": common_ids,
            "per_user_divergences": per_user_divergences,
            "per_user_map": per_user_map,
            "divergences": per_user_divergences,  # kept for backward compat
            "n_pairs": len(common_ids),
            "n_users_only": len(a_ids - b_ids),
            "n_dialogues_only": len(b_ids - a_ids),
        }

    def _compute_baseline_two_random_matches(
        self,
        dist_a: Mapping[str, Mapping[str, int]],
        dist_b: Mapping[str, Mapping[str, int]],
        *,
        picks: int = 2,
        exclude_self: bool = True,
        seed: int = 42,  # <- NEW: deterministic seed
        rng: Optional[random.Random] = None,
        use_old_emd: bool = True,
    ) -> Dict[str, object]:
        """
        Baseline: for each id in A, compare to `picks` random ids from B (prefer non-self),
        average the distances per id, then return per-id means and counts.
        Always shows a tqdm progress bar.

        Returns:
            {
            "user_ids": [uid, ...],                        # users for which baseline was computed
            "per_user_means": [float, ...],                # aligned with user_ids
            "per_user_map": {uid: float, ...},             # convenience mapping
            "n_users": int,
            "n_dialogue_pool": int,
            "n_effective_users": int,
            }
        """
        # deterministic RNG + deterministic ordering of IDs/candidates
        # Using random.Random for reproducible scientific experiments, not cryptography
        if rng is None:
            rng = random.Random(seed)  # noqa: S311

        a_ids_all = sorted(dist_a.keys())
        b_ids_all = sorted(dist_b.keys())

        per_user_means: List[float] = []
        per_user_map: Dict[str, float] = []
        user_ids_effective: List[str] = []

        callable = emd_distance_vsm if not use_old_emd else compute_emd

        progress_iter = tqdm(
            a_ids_all,
            total=len(a_ids_all),
            desc="Baseline EMD (A vs random B)",
            unit="id",
            leave=False,
        )

        if not b_ids_all:
            for _ in progress_iter:
                pass
            return {
                "user_ids": [],
                "per_user_means": [],
                "per_user_map": {},
                "n_users": len(a_ids_all),
                "n_dialogue_pool": 0,
                "n_effective_users": 0,
            }

        for uid in progress_iter:
            a_selections = dist_a[uid]

            # Prefer non-self candidates from B
            candidates = [bid for bid in b_ids_all if (bid != uid) or not exclude_self]
            if exclude_self and not candidates:
                if uid in dist_b:
                    candidates = [uid]
                else:
                    # No candidate at all for this uid
                    continue

            # Sample B ids (deterministic given seed + sorted candidates)
            if len(candidates) >= picks:
                match_ids = rng.sample(candidates, picks)  # without replacement
            else:
                match_ids = [rng.choice(candidates) for _ in range(picks)]  # with replacement

            dists = []
            for bid in match_ids:
                b_selections = dist_b[bid]
                dists.append(callable(a_selections, b_selections))

            mean_d = float(np.mean(dists))
            per_user_means.append(mean_d)
            user_ids_effective.append(uid)

        per_user_map = {uid: m for uid, m in zip(user_ids_effective, per_user_means)}

        return {
            "user_ids": user_ids_effective,
            "per_user_means": per_user_means,
            "per_user_map": per_user_map,
            "n_users": len(a_ids_all),
            "n_dialogue_pool": len(b_ids_all),
            "n_effective_users": len(per_user_means),
        }

    def cross_datasets_divergences_id_based(
        self,
    ):
        """Compute the cross dataset pairwise divergences with a progress bar (id-matched)."""
        user_distributions = self._pick_model_results_option_id(self.ba_user_results)
        dialogue_distributions = self._pick_model_results_option_id(self.ba_dialogue_career_results)

        core = self._compute_id_matched_divergences(user_distributions, dialogue_distributions)

        divergences: List[float] = core["per_user_divergences"]
        if not divergences:
            return {
                "avg_divergence": float("nan"),
                "std_divergence": float("nan"),
                "n_pairs": core["n_pairs"],
                "n_users_only": core["n_users_only"],
                "n_dialogues_only": core["n_dialogues_only"],
                "user_ids": core["user_ids"],
                # "per_user_divergences": [],
                "per_user_map": {},
            }

        return {
            "avg_divergence": float(np.mean(divergences)),
            "std_divergence": float(np.std(divergences)),
            "n_pairs": core["n_pairs"],
            "n_users_only": core["n_users_only"],
            "n_dialogues_only": core["n_dialogues_only"],
            # NEW: expose per-user
            "user_ids": core["user_ids"],
            # "per_user_divergences": divergences,
            "per_user_map": core["per_user_map"],
        }

    def cross_datasets_divergences_baseline_id_based(
        self,
        *,
        seed: int = 42,  # <- NEW: pass a fixed seed so it’s consistent across models
    ):
        """
        Baseline: for each user, compare to 2 random dialogue users (preferring j != i),
        average the two EMDs, then aggregate mean/std across users. Includes progress bar.
        """
        user_distributions = self._pick_model_results_option_id(self.ba_user_results)
        dialogue_distributions = self._pick_model_results_option_id(self.ba_dialogue_career_results)

        core = self._compute_baseline_two_random_matches(
            user_distributions,
            dialogue_distributions,
            picks=2,
            exclude_self=True,
            seed=seed,  # <- deterministic random baseline
            rng=None,  # let function build rng from seed
        )

        per_user_means: List[float] = core["per_user_means"]
        if not per_user_means:
            return {
                "avg_divergence": float("nan"),
                "std_divergence": float("nan"),
                "n_users": core["n_users"],
                "n_dialogue_pool": core["n_dialogue_pool"],
                "n_effective_users": core["n_effective_users"],
                "user_ids": core["user_ids"],
                # "per_user_baselines": [],
                "per_user_map": {},
            }

        return {
            "avg_divergence": float(np.mean(per_user_means)),
            "std_divergence": float(np.std(per_user_means)),
            "n_users": core["n_users"],
            "n_dialogue_pool": core["n_dialogue_pool"],
            "n_effective_users": core["n_effective_users"],
            # NEW: expose per-user
            "user_ids": core["user_ids"],
            # "per_user_baselines": per_user_means,
            "per_user_map": core["per_user_map"],
        }

    def _calculate_centroids_among_groups(self, grouped_output_values):
        group_centroids = {}
        for group_name, group_q_ans_maps in grouped_output_values.items():
            if isinstance(group_q_ans_maps, dict):
                group_q_ans_maps = list(group_q_ans_maps.values())
            g_centroid = componentwise_centroid_vsm(group_q_ans_maps)
            logger.info("for group: %s, centroid: %s", group_name, g_centroid)
            group_centroids[group_name] = g_centroid

        return group_centroids

    def compute_attributes_groups_distances(self, results_attribute, show_progress: bool = True):
        """compute group distances for attributes"""
        computed_results = {}
        user_values_dict = getattr(self, results_attribute)

        all_samples = []
        for _, user_answers in user_values_dict.items():
            all_samples.append([one_user_answer["selected_option_id"] for one_user_answer in user_answers])

        global_centroid = componentwise_centroid_vsm(all_samples)

        attributes_iter = tqdm(ATTRIBUTES, desc="Attributes", unit="attr") if show_progress else ATTRIBUTES
        for attribute in attributes_iter:
            index_based_group_dict = self._get_index_list_for_groups(attribute)
            grouped_values = self._map_model_results_to_groups(
                group_dict=index_based_group_dict,
                values_selection_results=user_values_dict,
            )
            group_centroids = self._calculate_centroids_among_groups(grouped_output_values=grouped_values)
            group_distances = self.pairwise_group_emd_list(group_centroids, normalize=True)
            bassline = {
                "overall_baseline": self.baseline_emd(
                    global_centroid,
                    group_centroids,
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

    args = parser.parse_args()

    try:
        # 2) Instantiate from args
        vc = ValuesComparison.from_cli_args(args)
        logger.info("Loaded datasets and model outputs.")

        final_outputs = {}
        result_attributes = [
            "ba_user_results",
            "ba_dialogue_career_results",
        ]

        for attr_name in result_attributes:
            logger.info("Computing attribute-group distances for: %s", attr_name)
            res = vc.compute_attributes_groups_distances(attr_name)
            final_outputs[attr_name] = res

        dist_res = vc.cross_datasets_divergences_id_based()
        base_res = vc.cross_datasets_divergences_baseline_id_based(seed=42)

        # Build per-user ratios aligned by intersection of ids
        ids_dist = set(dist_res.pop("user_ids"))
        ids_base = set(base_res.pop("user_ids"))
        common = sorted(ids_dist & ids_base)

        dist_user_map = dist_res.pop("per_user_map")
        per_user_div = [dist_user_map[uid] for uid in common]

        base_user_map = base_res.pop("per_user_map")
        per_user_base = [base_user_map[uid] for uid in common]
        per_user_ratio = [d / b if b != 0 else float("nan") for d, b in zip(per_user_div, per_user_base)]

        cross_datasets_results = {
            "distance": dist_res,
            "baseline": base_res,
            "ratio": round(dist_res["avg_divergence"] / base_res["avg_divergence"], 3),
            # NEW: per-user outputs for downstream stats (paired t-test etc.)
            "per_user": {
                "user_ids": common,
                "divergences": per_user_div,
                # "baselines": per_user_base,
                "ratios": per_user_ratio,
            },
        }

        final_outputs["cross_datasets_results"] = cross_datasets_results

        # Save to JSON
        _ensure_parent_dir(args.results_output_path)
        with open(args.results_output_path, "w", encoding="utf-8") as f:
            json.dump(final_outputs, f, ensure_ascii=False, indent=2)
        logger.info("Wrote results to %s", args.results_output_path)

    except Exception:
        logger.exception("ValuesComparison run failed.")
        # Non-zero exit for CI/automation visibility
        sys.exit(1)
