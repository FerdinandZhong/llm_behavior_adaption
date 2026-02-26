import argparse
import json
import logging
import os
from typing import Dict, List, Mapping

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


def _human_distributions(vc: ValuesComparison) -> Dict[str, Dict[str, int]]:
    return (
        vc.user_value_dataset.astype({"D_INTERVIEW": str})
        .groupby("D_INTERVIEW", as_index=True)
        .last()[list(vc.all_questions.keys())]
        .to_dict(orient="index")
    )


def _model_distributions(vc: ValuesComparison, model_results: Dict[str, Dict]) -> Dict[str, Dict[str, int]]:
    return vc._pick_model_results_option_id(model_results)


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


def _group_values_from_uids(
    group_dict: Dict[str, List[str]],
    answers_map: Mapping[str, Mapping[str, int]],
) -> Dict[str, List[Dict[str, int]]]:
    out: Dict[str, List[Dict[str, int]]] = {}
    for group_name, uid_list in group_dict.items():
        out[group_name] = [answers_map[uid] for uid in uid_list if uid in answers_map]
    return out


def _counterfactual_swap_stats(
    group_dict: Dict[str, List[str]],
    answers_map: Mapping[str, Mapping[str, int]],
    all_questions: Mapping[str, Mapping[str, int]],
) -> Dict[str, object]:
    group_values = _group_values_from_uids(group_dict, answers_map)
    centroids = _group_centroids_from_lists(group_values, all_questions)

    groups = [g for g in group_dict.keys() if g in centroids]
    if len(groups) < 2:
        return {"overall": {"mean_delta": float("nan"), "n_pairs": 0}, "per_pair": {}}

    per_pair: Dict[str, Dict[str, float]] = {}
    deltas: List[float] = []

    for g in groups:
        users = [uid for uid in group_dict[g] if uid in answers_map]
        if not users:
            continue
        c_self = centroids[g]
        for h in groups:
            if h == g:
                continue
            c_other = centroids[h]
            pair_key = f"{g}--{h}"
            pair_deltas: List[float] = []
            for uid in users:
                user_ans = answers_map[uid]
                d_self = emd_distance(user_ans, c_self, all_questions)
                d_swap = emd_distance(user_ans, c_other, all_questions)
                pair_deltas.append(float(d_swap - d_self))
            if pair_deltas:
                per_pair[pair_key] = {
                    "mean_delta": float(np.mean(pair_deltas)),
                    "n": int(len(pair_deltas)),
                }
                deltas.extend(pair_deltas)

    overall = {
        "mean_delta": float(np.mean(deltas)) if deltas else float("nan"),
        "median_delta": float(np.median(deltas)) if deltas else float("nan"),
        "n_pairs": int(len(deltas)),
    }
    return {"overall": overall, "per_pair": per_pair}


def main() -> None:
    parser = argparse.ArgumentParser(description="Counterfactual group-swap analysis")
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

    human_dist = _human_distributions(vc)
    results: Dict[str, Dict] = {"human": {}, "models": {}}

    attributes_iter = tqdm(ATTRIBUTES, desc="Attributes", unit="attr") if not args.no_progress else ATTRIBUTES
    for attribute in attributes_iter:
        # Human
        human_group_dict = vc._get_user_id_list_for_groups(attribute)
        results["human"][attribute] = _counterfactual_swap_stats(human_group_dict, human_dist, vc.all_questions)

        # Models
        for model_name, model_results in models.items():
            model_dist = _model_distributions(vc, model_results)
            model_group_dict = vc._get_user_id_list_for_groups(attribute)
            results["models"].setdefault(model_name, {})[attribute] = _counterfactual_swap_stats(
                model_group_dict, model_dist, vc.all_questions
            )

    output_path = os.path.abspath(args.output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    logger.info("Wrote results to %s", output_path)


if __name__ == "__main__":
    main()
