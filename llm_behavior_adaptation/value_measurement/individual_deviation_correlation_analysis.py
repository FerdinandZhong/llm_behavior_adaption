import argparse
import json
import logging
import os
from typing import Dict, Iterable, List, Mapping, Tuple

import numpy as np
import pandas as pd
from scipy import stats
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


def _compute_deviations_and_ranks(
    uid_to_group: Dict[str, str],
    answers_map: Mapping[str, Mapping[str, int]],
    all_questions: Mapping[str, Mapping[str, int]],
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Compute each individual's deviation from their group centroid and their rank within group.

    Returns:
    - deviations: uid -> distance to group centroid
    - ranks: uid -> rank percentile within group (0 = most typical, 1 = most atypical)
    """
    groups = _build_group_values(uid_to_group, answers_map)
    centroids = _group_centroids_from_lists(groups, all_questions)

    deviations: Dict[str, float] = {}
    ranks: Dict[str, float] = {}

    for group, uids_in_group in groups.items():
        if group not in centroids or len(uids_in_group) < 2:
            continue

        centroid = centroids[group]
        group_deviations: List[Tuple[str, float]] = []

        for uid in [u for u in uid_to_group.keys() if uid_to_group[u] == group]:
            if uid not in answers_map:
                continue
            dist = emd_distance(answers_map[uid], centroid, all_questions)
            deviations[uid] = float(dist)
            group_deviations.append((uid, dist))

        # Compute ranks (sorted by distance, highest distance = rank 0)
        group_deviations.sort(key=lambda x: x[1], reverse=True)
        for rank, (uid, _) in enumerate(group_deviations):
            rank_percentile = rank / max(len(group_deviations) - 1, 1)
            ranks[uid] = float(rank_percentile)

    return deviations, ranks


def _deviation_correlation_analysis(
    uid_to_group: Dict[str, str],
    human_dist: Mapping[str, Mapping[str, int]],
    model_dist: Mapping[str, Mapping[str, int]],
    all_questions: Mapping[str, Mapping[str, int]],
) -> Dict[str, object]:
    """
    Test if model preserves individual's relative position within their group.

    Measures:
    1. Correlation between human and model deviations (distance to centroid)
    2. Correlation between human and model ranks within group
    3. Per-group correlations

    High correlation = model respects individual variance within groups
    Low correlation = model ignores individual differences
    """
    human_deviations, human_ranks = _compute_deviations_and_ranks(uid_to_group, human_dist, all_questions)
    model_deviations, model_ranks = _compute_deviations_and_ranks(uid_to_group, model_dist, all_questions)

    # Match common UIDs
    common_uids = sorted(set(human_deviations.keys()) & set(model_deviations.keys()))

    if len(common_uids) < 3:
        return {
            "overall": {
                "n": 0,
                "deviation_pearson": float("nan"),
                "deviation_spearman": float("nan"),
                "rank_pearson": float("nan"),
                "rank_spearman": float("nan"),
                "interpretation": "insufficient_data",
            },
            "per_group": {},
        }

    h_dev_arr = np.array([human_deviations[uid] for uid in common_uids])
    m_dev_arr = np.array([model_deviations[uid] for uid in common_uids])
    h_rank_arr = np.array([human_ranks[uid] for uid in common_uids])
    m_rank_arr = np.array([model_ranks[uid] for uid in common_uids])

    # Overall correlations
    dev_pearson = stats.pearsonr(h_dev_arr, m_dev_arr)[0] if len(common_uids) > 1 else float("nan")
    dev_spearman = stats.spearmanr(h_dev_arr, m_dev_arr)[0] if len(common_uids) > 1 else float("nan")
    rank_pearson = stats.pearsonr(h_rank_arr, m_rank_arr)[0] if len(common_uids) > 1 else float("nan")
    rank_spearman = stats.spearmanr(h_rank_arr, m_rank_arr)[0] if len(common_uids) > 1 else float("nan")

    overall = {
        "n": len(common_uids),
        "deviation_pearson": float(dev_pearson),
        "deviation_spearman": float(dev_spearman),
        "rank_pearson": float(rank_pearson),
        "rank_spearman": float(rank_spearman),
        "interpretation": (
            (
                "strong_preservation"
                if rank_spearman >= 0.7
                else (
                    "moderate_preservation"
                    if rank_spearman >= 0.5
                    else (
                        "weak_preservation"
                        if rank_spearman >= 0.3
                        else "poor_preservation" if rank_spearman >= 0 else "inverse_relationship"
                    )
                )
            )
            if np.isfinite(rank_spearman)
            else "insufficient_data"
        ),
    }

    # Per-group correlations
    per_group: Dict[str, Dict[str, object]] = {}
    groups = _build_group_values(uid_to_group, human_dist)

    for group in groups.keys():
        group_uids = [uid for uid in common_uids if uid_to_group.get(uid) == group]

        if len(group_uids) < 3:
            continue

        g_h_dev = np.array([human_deviations[uid] for uid in group_uids])
        g_m_dev = np.array([model_deviations[uid] for uid in group_uids])
        g_h_rank = np.array([human_ranks[uid] for uid in group_uids])
        g_m_rank = np.array([model_ranks[uid] for uid in group_uids])

        per_group[group] = {
            "n": len(group_uids),
            "deviation_pearson": float(stats.pearsonr(g_h_dev, g_m_dev)[0]) if len(group_uids) > 1 else float("nan"),
            "deviation_spearman": float(stats.spearmanr(g_h_dev, g_m_dev)[0]) if len(group_uids) > 1 else float("nan"),
            "rank_pearson": float(stats.pearsonr(g_h_rank, g_m_rank)[0]) if len(group_uids) > 1 else float("nan"),
            "rank_spearman": float(stats.spearmanr(g_h_rank, g_m_rank)[0]) if len(group_uids) > 1 else float("nan"),
        }

    return {
        "overall": overall,
        "per_group": per_group,
    }


def _relative_distance_preservation(
    uid_to_group: Dict[str, str],
    human_dist: Mapping[str, Mapping[str, int]],
    model_dist: Mapping[str, Mapping[str, int]],
    all_questions: Mapping[str, Mapping[str, int]],
) -> Dict[str, object]:
    """
    Test if relative distances between individuals within the same group are preserved.

    For each pair of individuals in the same group:
    - Compare human pairwise distance vs model pairwise distance
    - Correlation of these distances indicates preservation of internal group structure
    """
    groups = _build_group_values(uid_to_group, human_dist)

    human_pairwise: List[float] = []
    model_pairwise: List[float] = []

    for group in groups.keys():
        # Get UIDs for this group from uid_to_group, filtering to those in both distributions
        group_uid_list = [
            uid
            for uid in uid_to_group.keys()
            if uid_to_group[uid] == group and uid in human_dist and uid in model_dist
        ]

        if len(group_uid_list) < 2:
            continue

        # Sample pairs to avoid O(n^2) explosion on large groups
        sample_size = min(len(group_uid_list), 50)
        sampled_uids = np.random.choice(group_uid_list, size=sample_size, replace=False)

        for i in range(len(sampled_uids)):
            for j in range(i + 1, len(sampled_uids)):
                uid1, uid2 = sampled_uids[i], sampled_uids[j]

                h_dist = emd_distance(human_dist[uid1], human_dist[uid2], all_questions)
                m_dist = emd_distance(model_dist[uid1], model_dist[uid2], all_questions)

                human_pairwise.append(float(h_dist))
                model_pairwise.append(float(m_dist))

    if len(human_pairwise) < 3:
        return {
            "n_pairs": 0,
            "pearson": float("nan"),
            "spearman": float("nan"),
            "interpretation": "insufficient_data",
        }

    h_arr = np.array(human_pairwise)
    m_arr = np.array(model_pairwise)

    return {
        "n_pairs": len(human_pairwise),
        "pearson": float(stats.pearsonr(h_arr, m_arr)[0]),
        "spearman": float(stats.spearmanr(h_arr, m_arr)[0]),
        "interpretation": (
            "strong_preservation"
            if stats.spearmanr(h_arr, m_arr)[0] >= 0.7
            else "moderate_preservation" if stats.spearmanr(h_arr, m_arr)[0] >= 0.5 else "weak_preservation"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Individual deviation correlation analysis: tests if models preserve individual's relative position within their demographic group"
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
    parser.add_argument("--seed", type=int, default=42, help="Random seed for pairwise sampling")
    parser.add_argument(
        "--no-progress",
        action="store_true",
        default=False,
        help="Disable tqdm progress bars.",
    )
    args = parser.parse_args()

    np.random.seed(args.seed)

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

        for model_name, model_results in models.items():
            model_dist = _model_distributions(vc, model_results)
            _uid_group_map(vc, attribute, model_dist.keys())

            # Deviation correlation (rank preservation)
            deviation_corr = _deviation_correlation_analysis(human_uid_map, human_dist, model_dist, vc.all_questions)

            # Pairwise distance preservation
            pairwise_pres = _relative_distance_preservation(human_uid_map, human_dist, model_dist, vc.all_questions)

            results["models"].setdefault(model_name, {})[attribute] = {
                "deviation_correlation": deviation_corr,
                "pairwise_distance_preservation": pairwise_pres,
            }

    output_path = os.path.abspath(args.output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    logger.info("Wrote individual deviation correlation analysis to %s", output_path)


if __name__ == "__main__":
    main()
