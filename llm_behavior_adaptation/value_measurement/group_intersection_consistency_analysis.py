import argparse
import json
import logging
import os
from itertools import combinations
from typing import Dict, Iterable, List, Mapping

import numpy as np
import pandas as pd
from tqdm import tqdm

from llm_behavior_adaptation.utils import register_logger
from llm_behavior_adaptation.value_measurement.formulas import componentwise_centroid, emd_distance
from llm_behavior_adaptation.value_measurement.wvs_values_comparison import (
    ATTRIBUTES,
    DATASET_DIR,
    ValuesComparison,
    load_jsonl_file,
)

logger = logging.getLogger(__name__)
register_logger(logger)


def _process_model_outputs(original_answers_list: List[Dict]) -> Dict[str, Dict[str, Dict]]:
    processed_answers_dict: Dict[str, Dict[str, Dict]] = {}
    for answer_details in original_answers_list:
        per_user_answers: Dict[str, Dict] = {}
        for user_id, answers in answer_details.items():
            for cat_answers in list(answers.values()):
                for answer in cat_answers:
                    per_user_answers.update(answer)
            processed_answers_dict[user_id] = per_user_answers
    return processed_answers_dict


def _group_centroids_from_lists(
    group_values: Dict[str, List[Dict[str, int]]],
    all_questions: Mapping[str, Mapping[str, int]],
) -> Dict[str, Dict[str, int]]:
    centroids: Dict[str, Dict[str, int]] = {}
    for group_name, answers_list in group_values.items():
        if not answers_list:
            continue
        centroids[group_name] = componentwise_centroid(answers_list, all_questions)
    return centroids


def _within_group_distance(
    group_values: Dict[str, List[Dict[str, int]]],
    all_questions: Mapping[str, Mapping[str, int]],
) -> float:
    distances: List[float] = []
    for _, answers_list in group_values.items():
        if not answers_list:
            continue
        centroid = componentwise_centroid(answers_list, all_questions)
        for user_answers in answers_list:
            distances.append(emd_distance(user_answers, centroid, all_questions))
    return float(np.mean(distances)) if distances else float("nan")


def _between_group_distance(
    group_centroids: Dict[str, Dict[str, int]],
    all_questions: Mapping[str, Mapping[str, int]],
) -> float:
    groups = list(group_centroids.keys())
    if len(groups) < 2:
        return float("nan")
    distances: List[float] = []
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            gi = groups[i]
            gj = groups[j]
            distances.append(emd_distance(group_centroids[gi], group_centroids[gj], all_questions))
    return float(np.mean(distances)) if distances else float("nan")


def _group_ratio(
    group_values: Dict[str, List[Dict[str, int]]],
    all_questions: Mapping[str, Mapping[str, int]],
) -> Dict[str, float]:
    centroids = _group_centroids_from_lists(group_values, all_questions)
    within_mean = _within_group_distance(group_values, all_questions)
    between_mean = _between_group_distance(centroids, all_questions)
    ratio = (
        float(between_mean) / float(within_mean)
        if np.isfinite(between_mean) and np.isfinite(within_mean) and within_mean != 0
        else float("nan")
    )
    return {
        "within_mean": float(within_mean),
        "between_mean": float(between_mean),
        "between_over_within": float(ratio),
    }


def _human_group_values(vc: ValuesComparison, attrs: Iterable[str]) -> Dict[str, List[Dict[str, int]]]:
    if len(list(attrs)) == 1:
        group_dict = vc._get_index_list_for_groups(list(attrs)[0])
    else:
        group_dict = vc._get_index_list_for_groups_multi(attrs)
    grouped = vc._map_human_values_to_groups(group_dict, vc.user_value_dataset)
    out: Dict[str, List[Dict[str, int]]] = {}
    for group_name, group_df in grouped.items():
        out[group_name] = list(group_df.values())
    return out


def _model_group_values(
    vc: ValuesComparison,
    attrs: Iterable[str],
    model_results: Dict[str, Dict],
) -> Dict[str, List[Dict[str, int]]]:
    if len(list(attrs)) == 1:
        group_dict = vc._get_user_id_list_for_groups(list(attrs)[0])
    else:
        group_dict = vc._get_user_id_list_for_groups_multi(attrs)
    return vc._map_model_results_to_groups(group_dict, model_results)


def main() -> None:
    parser = argparse.ArgumentParser(description="Consistency under subgroup intersection")
    parser.add_argument(
        "--user-profile-dataset",
        type=str,
        default=f"{DATASET_DIR}/sampled_demographic_features.csv",
    )
    parser.add_argument(
        "--user-value-dataset",
        type=str,
        default=f"{DATASET_DIR}/sampled_values_df.csv",
    )
    parser.add_argument("--ba-user-results", type=str, required=True)
    parser.add_argument("--ba-dialogue-career-results", type=str, required=True)
    parser.add_argument("--ba-dialogue-investment-results", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument(
        "--group-sizes",
        type=int,
        nargs="+",
        default=[1, 2],
        help="Group sizes to evaluate (e.g., 1 2 or 1 2 3).",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        default=False,
        help="Disable tqdm progress bars.",
    )
    args = parser.parse_args()

    user_profile = pd.read_csv(args.user_profile_dataset)
    user_value = pd.read_csv(args.user_value_dataset)

    with open(f"{DATASET_DIR}/picked_questions.json", "r", encoding="utf-8") as f:
        picked_questions = json.load(f)

    vc = ValuesComparison(
        user_profile_dataset=user_profile,
        user_value_dataset=user_value,
        ba_user_results=_process_model_outputs(load_jsonl_file(args.ba_user_results)),
        ba_dialogue_career_results=_process_model_outputs(load_jsonl_file(args.ba_dialogue_career_results)),
        ba_dialogue_investment_results=_process_model_outputs(load_jsonl_file(args.ba_dialogue_investment_results)),
        picked_questions=picked_questions,
        results_output_path="",
        verbose=0,
    )

    models = {
        "ba_user": vc.ba_user_results,
        "ba_dialogue_career": vc.ba_dialogue_career_results,
        "ba_dialogue_investment": vc.ba_dialogue_investment_results,
    }

    results: Dict[str, Dict] = {"human": {}, "models": {}}

    sizes = sorted(set(args.group_sizes))
    for size in sizes:
        if size == 1:
            combos = [(a,) for a in ATTRIBUTES]
        else:
            combos = list(combinations(ATTRIBUTES, size))

        combos_iter = tqdm(combos, desc=f"Groups n={size}", unit="combo") if not args.no_progress else combos

        for combo in combos_iter:
            combo_key = "+".join(combo)
            human_values = _human_group_values(vc, combo)
            human_stats = _group_ratio(human_values, vc.all_questions)
            results["human"].setdefault(str(size), {})[combo_key] = human_stats

            for model_name, model_results in models.items():
                model_values = _model_group_values(vc, combo, model_results)
                model_stats = _group_ratio(model_values, vc.all_questions)
                model_stats["vs_human_ratio"] = (
                    model_stats["between_over_within"] / human_stats["between_over_within"]
                    if np.isfinite(model_stats["between_over_within"])
                    and np.isfinite(human_stats["between_over_within"])
                    and human_stats["between_over_within"] != 0
                    else float("nan")
                )
                results["models"].setdefault(model_name, {}).setdefault(str(size), {})[combo_key] = model_stats

            # Intersection consistency vs singles (if size > 1)
            if size > 1:
                single_keys = [c for c in combo]
                human_single = [results["human"]["1"][k]["between_over_within"] for k in single_keys]
                human_expected = float(np.mean(human_single))
                human_consistency = (
                    human_stats["between_over_within"] / human_expected
                    if np.isfinite(human_expected) and human_expected != 0
                    else float("nan")
                )
                results["human"][str(size)][combo_key]["consistency_vs_singles"] = float(human_consistency)

                for model_name in models.keys():
                    model_single = [results["models"][model_name]["1"][k]["between_over_within"] for k in single_keys]
                    model_expected = float(np.mean(model_single))
                    model_ratio = results["models"][model_name][str(size)][combo_key]["between_over_within"]
                    model_consistency = (
                        model_ratio / model_expected
                        if np.isfinite(model_expected) and model_expected != 0
                        else float("nan")
                    )
                    results["models"][model_name][str(size)][combo_key]["consistency_vs_singles"] = float(
                        model_consistency
                    )

    output_path = os.path.abspath(args.output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    logger.info("Wrote results to %s", output_path)


if __name__ == "__main__":
    main()
