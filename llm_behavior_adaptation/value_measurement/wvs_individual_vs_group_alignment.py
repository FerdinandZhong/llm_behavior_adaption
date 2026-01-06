"""
Analysis script to measure whether a model's predictions are more aligned with
individual users or their demographic group medians.

For each (user, question) pair, we determine if the model prediction is closer to:
1. The individual user's actual answer
2. The demographic group median answer

This reveals whether the model exhibits stereotypical thinking (group bias) or
personalizes to individuals.

Updated:
- Support demographic_attributes as a list
- JSONL output includes per-attribute results for each (user, question)
- Statistics file includes overall + per-attribute summaries
"""

import argparse
import json
import logging
from collections import OrderedDict, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from llm_behavior_adaptation.utils import register_logger

logger = logging.getLogger(__name__)
register_logger(logger)


class IndividualVsGroupAlignment:
    """
    Analyze whether model predictions align more with individuals or demographic groups.
    Now supports multiple demographic attributes in one run.
    """

    def __init__(
        self,
        ba_dialogue_results_path: str,
        human_results_path: str,
        user_profile_path: str,
        picked_questions_path: str,
        output_file_path: str,
        demographic_attributes: Optional[List[str]] = None,
        starting_row: Optional[int] = None,
        ending_row: Optional[int] = None,
        verbose: int = 1,
    ):
        """
        Args:
            ba_dialogue_results_path: Path to model BA dialogue results (JSONL)
            human_results_path: Path to human results CSV
            user_profile_path: Path to user demographic profiles CSV
            picked_questions_path: Path to question metadata JSON
            output_file_path: Path to save analysis results (JSONL)
            demographic_attributes: List of demographic attributes to group by (age, education, etc.)
            starting_row: Start from this row index (optional)
            ending_row: End at this row index (optional)
            verbose: Logging verbosity (0 or 1)
        """
        self.ba_dialogue_results_path = ba_dialogue_results_path
        self.human_results_path = human_results_path
        self.user_profile_path = user_profile_path
        self.picked_questions_path = picked_questions_path
        self.output_file_path = output_file_path

        if demographic_attributes is None or len(demographic_attributes) == 0:
            demographic_attributes = ["age"]
        self.demographic_attributes = demographic_attributes

        self._starting_row = starting_row if starting_row is not None else 0
        self._ending_row = ending_row
        self.verbose = verbose

        # Load data
        self._load_data()

    def _load_data(self):
        """Load all required datasets."""
        logger.info("Loading datasets...")

        # Load BA dialogue results
        logger.info(f"Loading BA dialogue results from: {self.ba_dialogue_results_path}")
        self.ba_dialogue_results = self._load_jsonl(self.ba_dialogue_results_path)
        logger.info(f"Loaded {len(self.ba_dialogue_results)} BA dialogue results")

        # Apply row slicing
        if self._ending_row is not None:
            self.ba_dialogue_results = self.ba_dialogue_results[self._starting_row : self._ending_row]
        else:
            self.ba_dialogue_results = self.ba_dialogue_results[self._starting_row :]
        logger.info(
            f"Processing rows {self._starting_row} to {self._ending_row if self._ending_row else 'end'}: "
            f"{len(self.ba_dialogue_results)} users"
        )

        # Load human results
        logger.info(f"Loading human results from: {self.human_results_path}")
        human_df = pd.read_csv(self.human_results_path)
        logger.info(f"Loaded human results: {human_df.shape}")

        # Convert human results to dict: {user_id: {question_id: option_id}}
        self.human_results: Dict[str, Dict[str, int]] = {}
        for _, row in human_df.iterrows():
            user_id = str(int(row["D_INTERVIEW"]))  # avoid .0 suffix
            self.human_results[user_id] = {}
            for col in human_df.columns:
                if col.startswith("Q") and pd.notna(row[col]):
                    self.human_results[user_id][col] = int(row[col])

        # Load user profiles
        logger.info(f"Loading user profiles from: {self.user_profile_path}")
        self.user_profile_df = pd.read_csv(self.user_profile_path)
        logger.info(f"Loaded user profiles: {self.user_profile_df.shape}")

        # Load question metadata
        logger.info(f"Loading question metadata from: {self.picked_questions_path}")
        with open(self.picked_questions_path, "r", encoding="utf-8") as f:
            picked_questions = json.load(f)

        # Flatten question metadata
        self.all_questions: Dict[str, Any] = {}
        for category, questions in picked_questions.items():
            self.all_questions.update(questions)
        logger.info(f"Loaded {len(self.all_questions)} questions")

        logger.info("All datasets loaded successfully")

    def _load_jsonl(self, file_path: str) -> List[Dict]:
        """Load JSONL file into list of dicts."""
        results = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    results.append(json.loads(line))
        return results

    # ----------------------------
    # Grouping helpers (attribute-aware)
    # ----------------------------
    def _get_user_groups(self, demographic_attribute: str) -> Dict[str, List[int]]:
        """
        Group users by one demographic attribute.

        Returns:
            Dict mapping group name to list of row indices in user_profile_df
        """
        target_col = demographic_attribute

        if target_col not in self.user_profile_df.columns:
            raise KeyError(f"Column '{target_col}' not found in user_profile_df")

        # Remove "not sure" entries
        df = deepcopy(self.user_profile_df)
        norm_col = df[target_col].astype(str).str.strip().str.lower()
        df = df[norm_col != "not sure"]

        # Age binning
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
            return {lab: grouped[lab] for lab in labels if lab in grouped}

        # Education binning
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
                    "Master's & Doctoral": [
                        "Master or equivalent",
                        "Doctoral or equivalent",
                    ],
                }
            )
            label_to_bin = {lbl: bin_name for bin_name, labels in bins_to_labels.items() for lbl in labels}

            tmp_col = "_edu_bin"
            df[tmp_col] = df[target_col].map(label_to_bin)

            grouped = (
                df.dropna(subset=[tmp_col])
                .groupby(tmp_col, sort=False)
                .apply(lambda x: x.index.tolist())
                .to_dict()
            )
            ordered = {k: grouped[k] for k in bins_to_labels.keys() if k in grouped}
            return ordered

        # Occupation binning
        if target_col.lower() == "occupation_group":
            bins_to_labels = OrderedDict(
                {
                    "Clerical & Sales": ["Clerical", "Sales"],
                    "Skilled & Semi-Skilled": ["Skilled worker", "Semi-skilled worker"],
                    "Service & Labor": ["Service", "Unskilled worker"],
                    "Managerial / Professional": ["Higher administrative", "Professional and technical"],
                    "Agricultural Related": ["Farm worker", "Farm owner, farm manager"],
                    "No Job": ["Never had a job"],
                }
            )
            label_to_bin = {lbl: bin_name for bin_name, labels in bins_to_labels.items() for lbl in labels}

            tmp_col = "_occupation_bin"
            df[tmp_col] = df[target_col].map(label_to_bin)

            grouped = (
                df.dropna(subset=[tmp_col])
                .groupby(tmp_col, sort=False)
                .apply(lambda x: x.index.tolist())
                .to_dict()
            )
            ordered = {k: grouped[k] for k in bins_to_labels.keys() if k in grouped}
            return ordered

        # Fallback: group by raw values
        grouped = df.groupby(target_col, sort=False).apply(lambda x: x.index.tolist()).to_dict()
        return grouped

    def _compute_group_medians(
        self, group_indices: Dict[str, List[int]]
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute median response for each question within each demographic group.

        Returns:
            Dict mapping group name to dict of {question_id: median_value}
        """
        group_medians: Dict[str, Dict[str, float]] = {}

        for group_name, indices in group_indices.items():
            # Get user IDs for this group
            group_user_ids = []
            for idx in indices:
                if idx < len(self.user_profile_df):
                    user_id = str(self.user_profile_df.iloc[idx]["D_INTERVIEW"])
                    group_user_ids.append(user_id)

            # Collect all responses for each question from users in this group
            question_responses: Dict[str, List[int]] = {}
            for user_id in group_user_ids:
                if user_id in self.human_results:
                    for question_id, option_id in self.human_results[user_id].items():
                        question_responses.setdefault(question_id, []).append(option_id)

            # Compute median for each question
            group_medians[group_name] = {}
            for question_id, responses in question_responses.items():
                if responses:
                    group_medians[group_name][question_id] = float(np.median(responses))

        return group_medians

    def _get_user_group(self, user_id: str, group_indices: Dict[str, List[int]]) -> Optional[str]:
        """Find which demographic group a user belongs to (given group_indices for one attribute)."""
        user_rows = self.user_profile_df[self.user_profile_df["D_INTERVIEW"] == int(user_id)]
        if user_rows.empty:
            return None
        user_idx = user_rows.index[0]
        for group_name, indices in group_indices.items():
            if user_idx in indices:
                return group_name
        return None

    # ----------------------------
    # Core computation: one attribute at a time
    # ----------------------------
    def _run_one_attribute(self, demographic_attribute: str) -> Tuple[Dict[str, Any], Dict[Tuple[str, str], Dict[str, Any]]]:
        """
        Run alignment analysis for a single demographic attribute.

        Returns:
            stats: summary stats dict for this attribute
            per_pair: mapping (user_id, question_id) -> per-attribute result dict
        """
        logger.info("-" * 80)
        logger.info(f"Running attribute: {demographic_attribute}")

        # Get demographic groups
        group_indices = self._get_user_groups(demographic_attribute)

        # Compute group medians
        group_medians = self._compute_group_medians(group_indices)

        # Counters
        individual_wins = 0
        group_wins = 0
        ties = 0
        skipped = 0

        per_pair: Dict[Tuple[str, str], Dict[str, Any]] = {}

        # Iterate users
        for user_result in self.ba_dialogue_results:
            user_id = list(user_result.keys())[0]
            ba_answers_nested = user_result[user_id]

            # Flatten BA answers
            ba_answers: Dict[str, Dict[str, Any]] = {}
            for category, question_list in ba_answers_nested.items():
                if isinstance(question_list, list):
                    for q_dict in question_list:
                        ba_answers.update(q_dict)
                elif isinstance(question_list, dict):
                    ba_answers[category] = question_list

            if user_id not in self.human_results:
                skipped += len(ba_answers)
                continue

            human_answers = self.human_results[user_id]

            user_group = self._get_user_group(user_id, group_indices)
            if user_group is None or user_group not in group_medians:
                skipped += len(ba_answers)
                continue

            group_median_dict = group_medians[user_group]

            for question_id, ba_answer in ba_answers.items():
                if question_id not in human_answers:
                    skipped += 1
                    continue
                if question_id not in group_median_dict:
                    skipped += 1
                    continue

                model_option = ba_answer.get("option_id")
                individual_option = human_answers[question_id]
                group_median = group_median_dict[question_id]

                if model_option is None or model_option < 0:
                    skipped += 1
                    continue

                dist_to_individual = abs(model_option - individual_option)
                dist_to_group = abs(model_option - group_median)

                if dist_to_individual < dist_to_group:
                    winner = "individual"
                    individual_wins += 1
                elif dist_to_group < dist_to_individual:
                    winner = "group"
                    group_wins += 1
                else:
                    winner = "tie"
                    ties += 1

                per_pair[(user_id, question_id)] = {
                    "demographic_attribute": demographic_attribute,
                    "user_group": user_group,
                    "model_option": model_option,
                    "individual_option": individual_option,
                    "group_median": group_median,
                    "dist_to_individual": dist_to_individual,
                    "dist_to_group": dist_to_group,
                    "winner": winner,
                }

        total_comparisons = individual_wins + group_wins + ties
        individual_rate = (individual_wins / total_comparisons * 100) if total_comparisons > 0 else 0.0
        group_rate = (group_wins / total_comparisons * 100) if total_comparisons > 0 else 0.0
        tie_rate = (ties / total_comparisons * 100) if total_comparisons > 0 else 0.0
        bias_score = (individual_wins - group_wins) / total_comparisons if total_comparisons > 0 else 0.0

        stats = {
            "demographic_attribute": demographic_attribute,
            "total_users": len(self.ba_dialogue_results),
            "num_groups": len(group_indices),
            "groups": {name: len(indices) for name, indices in group_indices.items()},
            "alignment_summary": {
                "total_comparisons": total_comparisons,
                "individual_wins": individual_wins,
                "group_wins": group_wins,
                "ties": ties,
                "skipped": skipped,
            },
            "alignment_rates": {
                "individual_alignment_rate": round(individual_rate, 2),
                "group_alignment_rate": round(group_rate, 2),
                "tie_rate": round(tie_rate, 2),
            },
            "bias_score": round(bias_score, 4),
            "bias_interpretation": self._interpret_bias(bias_score),
        }
        return stats, per_pair

    def run_alignment_analysis(self) -> Dict[str, Any]:
        """
        Run alignment analysis for all demographic_attributes.

        Output JSONL:
            One line per (user_id, question_id) with:
              - base fields
              - attribute_results: {attr -> per-attr result dict}

        Output statistics:
            - overall aggregates across attributes
            - per-attribute stats
        """
        logger.info("=" * 80)
        logger.info("INDIVIDUAL VS GROUP ALIGNMENT ANALYSIS (MULTI-ATTRIBUTE)")
        logger.info("=" * 80)
        logger.info(f"Demographic attributes: {self.demographic_attributes}")
        logger.info(f"Processing {len(self.ba_dialogue_results)} users")

        per_attribute_stats: Dict[str, Any] = {}
        combined_pairs: Dict[Tuple[str, str], Dict[str, Any]] = defaultdict(dict)

        # Run each attribute
        for attr in self.demographic_attributes:
            stats, per_pair = self._run_one_attribute(attr)
            per_attribute_stats[attr] = stats

            # Merge per-pair results
            for (user_id, question_id), res in per_pair.items():
                combined_pairs[(user_id, question_id)][attr] = res

        # Build JSONL lines: one per (user, question)
        detailed_results: List[Dict[str, Any]] = []
        for (user_id, question_id), attr_results in combined_pairs.items():
            detailed_results.append(
                {
                    "user_id": user_id,
                    "question_id": question_id,
                    "attribute_results": attr_results,  # attr -> {user_group, distances, winner, ...}
                }
            )

        # Overall aggregates across attributes (sum wins/ties/skip; bias computed from totals)
        total_individual_wins = 0
        total_group_wins = 0
        total_ties = 0
        total_skipped = 0
        total_groups_union = {}

        # Build a clean, explicit per-attribute summary list/dict for the stats file
        attribute_results = {}
        for attr, st in per_attribute_stats.items():
            s = st["alignment_summary"]
            total_individual_wins += int(s["individual_wins"])
            total_group_wins += int(s["group_wins"])
            total_ties += int(s["ties"])
            total_skipped += int(s["skipped"])
            total_groups_union[attr] = st.get("groups", {})
            
            attribute_results[attr] = {
                "num_groups": st.get("num_groups", 0),
                "groups": st.get("groups", {}),
                "alignment_summary": st.get("alignment_summary", {}),
                "alignment_rates": st.get("alignment_rates", {}),
                "bias_score": st.get("bias_score", 0.0),
                "bias_interpretation": st.get("bias_interpretation", ""),
            }

        # Overall aggregates across attributes
        total_comparisons = total_individual_wins + total_group_wins + total_ties
        overall_bias_score = (
            (total_individual_wins - total_group_wins) / total_comparisons if total_comparisons > 0 else 0.0
        )

        overall_stats = {
            "demographic_attributes": self.demographic_attributes,
            "total_users": len(self.ba_dialogue_results),

            # ✅ Explicit per-attribute results (what you asked for)
            "attribute_results": attribute_results,

            "overall_alignment_summary": {
                "total_comparisons": total_comparisons,
                "individual_wins": total_individual_wins,
                "group_wins": total_group_wins,
                "ties": total_ties,
                "skipped": total_skipped,
            },
            "overall_bias_score": round(overall_bias_score, 4),
            "overall_bias_interpretation": self._interpret_bias(overall_bias_score),

            # group sizes by attribute (already present before, keep if you want)
            "groups_by_attribute": total_groups_union,
        }

        # Save statistics
        stats_file = self.output_file_path.replace(".jsonl", "_statistics.json")
        Path(stats_file).parent.mkdir(parents=True, exist_ok=True)
        with open(stats_file, "w", encoding="utf-8") as f:
            json.dump(overall_stats, f, ensure_ascii=False, indent=2)

        # Save detailed results
        Path(self.output_file_path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_file_path, "w", encoding="utf-8") as f:
            for result in detailed_results:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")

        logger.info("=" * 80)
        logger.info("DONE")
        logger.info("=" * 80)
        logger.info(f"Statistics saved to: {stats_file}")
        logger.info(f"Detailed results saved to: {self.output_file_path}")

        return overall_stats

    def _interpret_bias(self, bias_score: float) -> str:
        """Interpret the bias score."""
        if bias_score > 0.3:
            return "Strong individual alignment - model personalizes well to individual users"
        elif bias_score > 0.1:
            return "Moderate individual alignment - model somewhat personalizes to individuals"
        elif bias_score > -0.1:
            return "Neutral - model shows no clear bias toward individuals or groups"
        elif bias_score > -0.3:
            return "Moderate group alignment - model exhibits some stereotypical thinking"
        else:
            return "Strong group alignment - model relies heavily on demographic stereotypes"


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Analyze whether model predictions align more with individuals or demographic groups"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file",
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Backward compatible: accept either demographic_attribute or demographic_attributes
    demographic_attributes = config.get("demographic_attributes")
    if demographic_attributes is None:
        single = config.get("demographic_attribute", "age")
        demographic_attributes = [single] if isinstance(single, str) else list(single)

    analyzer = IndividualVsGroupAlignment(
        ba_dialogue_results_path=config["ba_dialogue_results_path"],
        human_results_path=config["human_results_path"],
        user_profile_path=config["user_profile_path"],
        picked_questions_path=config["picked_questions_path"],
        output_file_path=config["output_file_path"],
        demographic_attributes=demographic_attributes,
        starting_row=config.get("starting_row"),
        ending_row=config.get("ending_row"),
        verbose=config.get("verbose", 1),
    )

    analyzer.run_alignment_analysis()


if __name__ == "__main__":
    main()
