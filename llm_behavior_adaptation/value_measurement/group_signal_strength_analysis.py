import argparse
import json
import logging
import os
from typing import Dict, Iterable, List, Mapping, Tuple

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


def _build_group_values(
    uid_to_group: Dict[str, str],
    answers_map: Mapping[str, Mapping[str, int]],
) -> Dict[str, List[Dict[str, int]]]:
    grouped: Dict[str, List[Dict[str, int]]] = {}
    for uid, group in uid_to_group.items():
        if uid not in answers_map:
            continue
        grouped.setdefault(group, []).append(answers_map[uid])
    return grouped


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
) -> Tuple[float, int]:
    distances: List[float] = []
    for _, answers_list in group_values.items():
        if not answers_list:
            continue
        centroid = componentwise_centroid(answers_list, all_questions)
        for user_answers in answers_list:
            distances.append(emd_distance(user_answers, centroid, all_questions))
    return (float(np.mean(distances)) if distances else float("nan")), len(distances)


def _between_group_distance(
    group_centroids: Dict[str, Dict[str, int]],
    all_questions: Mapping[str, Mapping[str, int]],
) -> Tuple[float, int]:
    groups = list(group_centroids.keys())
    if len(groups) < 2:
        return float("nan"), 0
    distances: List[float] = []
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            gi = groups[i]
            gj = groups[j]
            distances.append(emd_distance(group_centroids[gi], group_centroids[gj], all_questions))
    return float(np.mean(distances)), len(distances)


def _total_distance_to_global_centroid(
    answers_map: Mapping[str, Mapping[str, int]],
    all_questions: Mapping[str, Mapping[str, int]],
    user_ids: Iterable[str],
) -> float:
    samples = [answers_map[uid] for uid in user_ids if uid in answers_map]
    if not samples:
        return float("nan")
    global_centroid = componentwise_centroid(samples, all_questions)
    dists = [emd_distance(answers_map[uid], global_centroid, all_questions) for uid in user_ids if uid in answers_map]
    return float(np.mean(dists)) if dists else float("nan")


def _group_ratio(
    group_values: Dict[str, List[Dict[str, int]]],
    all_questions: Mapping[str, Mapping[str, int]],
) -> Dict[str, float]:
    centroids = _group_centroids_from_lists(group_values, all_questions)
    within_mean, within_n = _within_group_distance(group_values, all_questions)
    between_mean, between_n = _between_group_distance(centroids, all_questions)
    ratio = (
        float(between_mean) / float(within_mean)
        if np.isfinite(between_mean) and np.isfinite(within_mean) and within_mean != 0
        else float("nan")
    )
    return {
        "within_mean": float(within_mean),
        "within_n": int(within_n),
        "between_mean": float(between_mean),
        "between_n": int(between_n),
        "between_over_within": float(ratio),
    }


def _uid_group_map(vc: ValuesComparison, attribute: str, valid_uids: Iterable[str]) -> Dict[str, str]:
    df = vc.user_profile_dataset.copy()
    df["D_INTERVIEW"] = df["D_INTERVIEW"].astype(str)
    df = df.set_index("D_INTERVIEW")
    label_series = vc._group_label_series(df.reset_index(), attribute)
    label_series.index = df.index
    label_series = label_series.dropna()
    uid_set = set(valid_uids)
    return {uid: str(label_series.loc[uid]) for uid in label_series.index if uid in uid_set}


def _permutation_p_value(
    uid_to_group: Dict[str, str],
    answers_map: Mapping[str, Mapping[str, int]],
    all_questions: Mapping[str, Mapping[str, int]],
    *,
    n_permutations: int,
    seed: int,
) -> Dict[str, object]:
    uids = list(uid_to_group.keys())
    labels = [uid_to_group[uid] for uid in uids]
    observed = _group_ratio(_build_group_values(uid_to_group, answers_map), all_questions)
    observed_ratio = observed["between_over_within"]

    rng = np.random.default_rng(seed)
    permuted: List[float] = []
    for _ in range(n_permutations):
        shuffled = labels.copy()
        rng.shuffle(shuffled)
        perm_map = {uid: shuffled[i] for i, uid in enumerate(uids)}
        perm_ratio = _group_ratio(_build_group_values(perm_map, answers_map), all_questions)
        permuted.append(perm_ratio["between_over_within"])

    permuted_arr = np.array(permuted, dtype=float)
    if not np.isfinite(observed_ratio):
        p_value = float("nan")
    else:
        p_value = float(np.mean(permuted_arr >= observed_ratio))

    return {
        "observed_ratio": float(observed_ratio),
        "permuted_mean": float(np.nanmean(permuted_arr)),
        "permuted_std": float(np.nanstd(permuted_arr)),
        "p_value": p_value,
        "n_permutations": int(n_permutations),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Group signal strength analysis")
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
    parser.add_argument("--n-permutations", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
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

    attributes_iter = tqdm(ATTRIBUTES, desc="Attributes", unit="attr") if not args.no_progress else ATTRIBUTES

    results: Dict[str, Dict] = {"human": {}, "models": {}}

    for attribute in attributes_iter:
        human_uid_map = _uid_group_map(vc, attribute, human_dist.keys())
        human_group_vals = _build_group_values(human_uid_map, human_dist)
        human_stats = _group_ratio(human_group_vals, vc.all_questions)
        human_total = _total_distance_to_global_centroid(human_dist, vc.all_questions, human_uid_map.keys())
        human_stats["residual_ratio"] = (
            human_stats["within_mean"] / human_total if np.isfinite(human_total) and human_total != 0 else float("nan")
        )
        human_stats["permutation_test"] = _permutation_p_value(
            human_uid_map,
            human_dist,
            vc.all_questions,
            n_permutations=args.n_permutations,
            seed=args.seed,
        )
        results["human"][attribute] = human_stats

        for model_name, model_results in models.items():
            model_dist = _model_distributions(vc, model_results)
            model_uid_map = _uid_group_map(vc, attribute, model_dist.keys())
            model_group_vals = _build_group_values(model_uid_map, model_dist)
            model_stats = _group_ratio(model_group_vals, vc.all_questions)
            model_total = _total_distance_to_global_centroid(model_dist, vc.all_questions, model_uid_map.keys())
            model_stats["residual_ratio"] = (
                model_stats["within_mean"] / model_total
                if np.isfinite(model_total) and model_total != 0
                else float("nan")
            )
            model_stats["vs_human_ratio"] = (
                model_stats["between_over_within"] / human_stats["between_over_within"]
                if np.isfinite(model_stats["between_over_within"])
                and np.isfinite(human_stats["between_over_within"])
                and human_stats["between_over_within"] != 0
                else float("nan")
            )
            model_stats["residual_vs_human"] = (
                model_stats["residual_ratio"] / human_stats["residual_ratio"]
                if np.isfinite(model_stats["residual_ratio"])
                and np.isfinite(human_stats["residual_ratio"])
                and human_stats["residual_ratio"] != 0
                else float("nan")
            )
            model_stats["permutation_test"] = _permutation_p_value(
                model_uid_map,
                model_dist,
                vc.all_questions,
                n_permutations=args.n_permutations,
                seed=args.seed,
            )
            results["models"].setdefault(model_name, {})[attribute] = model_stats

    output_path = os.path.abspath(args.output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    logger.info("Wrote results to %s", output_path)


if __name__ == "__main__":
    main()
