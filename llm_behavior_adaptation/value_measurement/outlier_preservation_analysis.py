import argparse
import json
import logging
import os
from typing import Dict, Iterable, List, Mapping, Set, Tuple

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


def _uid_group_map(vc: ValuesComparison, attribute: str, valid_uids: Iterable[str]) -> Dict[str, str]:
    df = vc.user_profile_dataset.copy()
    df["D_INTERVIEW"] = df["D_INTERVIEW"].astype(str)
    df = df.set_index("D_INTERVIEW")
    label_series = vc._group_label_series(df.reset_index(), attribute)
    label_series.index = df.index
    label_series = label_series.dropna()
    uid_set = set(valid_uids)
    return {uid: str(label_series.loc[uid]) for uid in label_series.index if uid in uid_set}


def _identify_outliers(
    uid_to_group: Dict[str, str],
    answers_map: Mapping[str, Mapping[str, int]],
    all_questions: Mapping[str, Mapping[str, int]],
    percentile: float = 90.0,
) -> Dict[str, Set[str]]:
    """
    Identify outliers in each group: individuals whose distance to group centroid
    is in the top percentile within their group.

    Returns: Dict mapping group_name -> set of outlier user IDs
    """
    groups = _build_group_values(uid_to_group, answers_map)
    centroids = _group_centroids_from_lists(groups, all_questions)

    outliers_by_group: Dict[str, Set[str]] = {}

    for group, uids in groups.items():
        if group not in centroids or len(uids) < 10:
            continue

        centroid = centroids[group]
        distances: List[Tuple[str, float]] = []

        for uid in [u for u in uid_to_group.keys() if uid_to_group[u] == group]:
            if uid not in answers_map:
                continue
            dist = emd_distance(answers_map[uid], centroid, all_questions)
            distances.append((uid, dist))

        if not distances:
            continue

        threshold = np.percentile([d for _, d in distances], percentile)
        outliers = {uid for uid, dist in distances if dist >= threshold}
        outliers_by_group[group] = outliers

    return outliers_by_group


def _outlier_preservation_analysis(
    human_outliers: Dict[str, Set[str]],
    uid_to_group: Dict[str, str],
    human_dist: Mapping[str, Mapping[str, int]],
    model_dist: Mapping[str, Mapping[str, int]],
    all_questions: Mapping[str, Mapping[str, int]],
    percentile: float = 90.0,
) -> Dict[str, object]:
    """
    Test if individuals who are outliers in human data remain outliers in model output.

    For each human outlier:
    - Check if they remain in top percentile in model output
    - Measure change in distance-to-centroid rank

    High retention rate = model preserves atypical individuals
    Low retention rate = model "corrects" outliers toward group stereotype
    """
    model_groups = _build_group_values(uid_to_group, model_dist)
    model_centroids = _group_centroids_from_lists(model_groups, all_questions)

    per_group_retention: Dict[str, Dict[str, object]] = {}
    all_retention: List[bool] = []
    rank_changes: List[float] = []

    for group, human_outlier_uids in human_outliers.items():
        if group not in model_centroids:
            continue

        model_centroid = model_centroids[group]
        group_uids = [uid for uid, g in uid_to_group.items() if g == group and uid in model_dist]

        if len(group_uids) < 10:
            continue

        # Compute model distances for all group members
        model_distances = [(uid, emd_distance(model_dist[uid], model_centroid, all_questions)) for uid in group_uids]
        model_distances.sort(key=lambda x: x[1], reverse=True)
        model_threshold = np.percentile([d for _, d in model_distances], percentile)

        # Create rank mapping
        model_ranks = {uid: rank for rank, (uid, _) in enumerate(model_distances)}

        # Check retention
        retained = 0
        total_outliers = 0
        group_rank_changes: List[float] = []

        for uid in human_outlier_uids:
            if uid not in model_dist:
                continue

            total_outliers += 1
            model_dist_val = emd_distance(model_dist[uid], model_centroid, all_questions)

            # Is still outlier in model?
            if model_dist_val >= model_threshold:
                retained += 1
                all_retention.append(True)
            else:
                all_retention.append(False)

            # Rank change (0 = highest distance/most atypical)
            model_rank = model_ranks.get(uid, len(group_uids))
            rank_pct_change = model_rank / len(group_uids)  # 0 = still most atypical, 1 = moved to typical
            group_rank_changes.append(rank_pct_change)
            rank_changes.append(rank_pct_change)

        if total_outliers > 0:
            per_group_retention[group] = {
                "n_outliers": total_outliers,
                "n_retained": retained,
                "retention_rate": float(retained / total_outliers),
                "mean_rank_percentile": float(np.mean(group_rank_changes)) if group_rank_changes else float("nan"),
            }

    overall = {
        "n_total_outliers": len(all_retention),
        "n_retained": int(sum(all_retention)),
        "retention_rate": float(np.mean(all_retention)) if all_retention else float("nan"),
        "mean_rank_percentile": float(np.mean(rank_changes)) if rank_changes else float("nan"),
        "interpretation": (
            (
                "strong_preservation"
                if np.mean(all_retention) >= 0.7
                else (
                    "moderate_preservation"
                    if np.mean(all_retention) >= 0.5
                    else "weak_preservation" if np.mean(all_retention) >= 0.3 else "homogenization"
                )
            )
            if all_retention
            else "insufficient_data"
        ),
    }

    return {
        "overall": overall,
        "per_group": per_group_retention,
    }


def _outlier_correction_test(
    human_outliers: Dict[str, Set[str]],
    uid_to_group: Dict[str, str],
    human_dist: Mapping[str, Mapping[str, int]],
    model_dist: Mapping[str, Mapping[str, int]],
    all_questions: Mapping[str, Mapping[str, int]],
) -> Dict[str, object]:
    """
    Specifically test if model "corrects" outliers toward group mean.

    Compare distance reduction for outliers vs non-outliers:
    - delta_outliers = (model_dist - human_dist) for outliers
    - delta_non_outliers = (model_dist - human_dist) for non-outliers

    If delta_outliers is more negative, model is pulling outliers toward centroid.
    """
    human_groups = _build_group_values(uid_to_group, human_dist)
    model_groups = _build_group_values(uid_to_group, model_dist)

    human_centroids = _group_centroids_from_lists(human_groups, all_questions)
    model_centroids = _group_centroids_from_lists(model_groups, all_questions)

    all_human_outliers = set()
    for outlier_set in human_outliers.values():
        all_human_outliers.update(outlier_set)

    outlier_deltas: List[float] = []
    non_outlier_deltas: List[float] = []

    for uid, group in uid_to_group.items():
        if uid not in human_dist or uid not in model_dist:
            continue
        if group not in human_centroids or group not in model_centroids:
            continue

        h_dist = emd_distance(human_dist[uid], human_centroids[group], all_questions)
        m_dist = emd_distance(model_dist[uid], model_centroids[group], all_questions)
        delta = m_dist - h_dist

        if uid in all_human_outliers:
            outlier_deltas.append(delta)
        else:
            non_outlier_deltas.append(delta)

    return {
        "outlier_mean_delta": float(np.mean(outlier_deltas)) if outlier_deltas else float("nan"),
        "non_outlier_mean_delta": float(np.mean(non_outlier_deltas)) if non_outlier_deltas else float("nan"),
        "delta_difference": (
            float(np.mean(outlier_deltas) - np.mean(non_outlier_deltas))
            if outlier_deltas and non_outlier_deltas
            else float("nan")
        ),
        "n_outliers": len(outlier_deltas),
        "n_non_outliers": len(non_outlier_deltas),
        "interpretation": (
            (
                "strong_correction"
                if (np.mean(outlier_deltas) - np.mean(non_outlier_deltas)) < -1.0
                else (
                    "moderate_correction"
                    if (np.mean(outlier_deltas) - np.mean(non_outlier_deltas)) < -0.3
                    else (
                        "neutral"
                        if -0.3 <= (np.mean(outlier_deltas) - np.mean(non_outlier_deltas)) <= 0.3
                        else "amplification"
                    )
                )
            )
            if outlier_deltas and non_outlier_deltas
            else "insufficient_data"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Outlier preservation analysis: tests if atypical individuals within demographic groups remain atypical in model outputs"
    )
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
        "--outlier-percentile",
        type=float,
        default=90.0,
        help="Percentile threshold for defining outliers (default: 90)",
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

    human_dist = _human_distributions(vc)
    results: Dict[str, Dict] = {"models": {}}

    attributes_iter = tqdm(ATTRIBUTES, desc="Attributes", unit="attr") if not args.no_progress else ATTRIBUTES

    for attribute in attributes_iter:
        human_uid_map = _uid_group_map(vc, attribute, human_dist.keys())

        # Identify human outliers
        human_outliers = _identify_outliers(
            human_uid_map, human_dist, vc.all_questions, percentile=args.outlier_percentile
        )

        for model_name, model_results in models.items():
            model_dist = _model_distributions(vc, model_results)
            _uid_group_map(vc, attribute, model_dist.keys())

            # Test outlier preservation
            preservation = _outlier_preservation_analysis(
                human_outliers,
                human_uid_map,
                human_dist,
                model_dist,
                vc.all_questions,
                percentile=args.outlier_percentile,
            )

            # Test outlier correction
            correction = _outlier_correction_test(
                human_outliers, human_uid_map, human_dist, model_dist, vc.all_questions
            )

            results["models"].setdefault(model_name, {})[attribute] = {
                "outlier_preservation": preservation,
                "outlier_correction": correction,
            }

    output_path = os.path.abspath(args.output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    logger.info("Wrote outlier preservation analysis to %s", output_path)


if __name__ == "__main__":
    main()
