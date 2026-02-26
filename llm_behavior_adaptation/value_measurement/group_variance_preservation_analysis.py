import argparse
import json
import logging
import os
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


def _within_group_variance_stats(
    group_values: Dict[str, List[Dict[str, int]]],
    all_questions: Mapping[str, Mapping[str, int]],
) -> Dict[str, object]:
    """
    Compute within-group variance statistics.
    Returns per-group std, mean, IQR of distances to group centroid.
    """
    per_group: Dict[str, Dict[str, float]] = {}
    all_distances: List[float] = []

    for group_name, answers_list in group_values.items():
        if len(answers_list) < 2:
            continue
        centroid = componentwise_centroid(answers_list, all_questions)
        distances = [emd_distance(ans, centroid, all_questions) for ans in answers_list]

        per_group[group_name] = {
            "mean": float(np.mean(distances)),
            "std": float(np.std(distances, ddof=1)),
            "median": float(np.median(distances)),
            "iqr": float(np.percentile(distances, 75) - np.percentile(distances, 25)),
            "n": len(distances),
        }
        all_distances.extend(distances)

    # Aggregate stats across all groups
    overall = {
        "mean_of_group_means": float(np.mean([g["mean"] for g in per_group.values()])) if per_group else float("nan"),
        "mean_of_group_stds": float(np.mean([g["std"] for g in per_group.values()])) if per_group else float("nan"),
        "mean_of_group_iqrs": float(np.mean([g["iqr"] for g in per_group.values()])) if per_group else float("nan"),
        "pooled_std": float(np.std(all_distances, ddof=1)) if all_distances else float("nan"),
        "n_groups": len(per_group),
        "n_total": len(all_distances),
    }

    return {"overall": overall, "per_group": per_group}


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


def _variance_preservation_ratio(
    human_stats: Dict[str, object],
    model_stats: Dict[str, object],
) -> Dict[str, float]:
    """
    Compute ratios: model_variance / human_variance.
    Ratio < 1 = model collapses individual differences.
    Ratio ≈ 1 = model preserves individual variance.
    Ratio > 1 = model amplifies individual differences.
    """
    h_overall = human_stats["overall"]
    m_overall = model_stats["overall"]

    std_ratio = (
        m_overall["mean_of_group_stds"] / h_overall["mean_of_group_stds"]
        if np.isfinite(m_overall["mean_of_group_stds"])
        and np.isfinite(h_overall["mean_of_group_stds"])
        and h_overall["mean_of_group_stds"] != 0
        else float("nan")
    )

    iqr_ratio = (
        m_overall["mean_of_group_iqrs"] / h_overall["mean_of_group_iqrs"]
        if np.isfinite(m_overall["mean_of_group_iqrs"])
        and np.isfinite(h_overall["mean_of_group_iqrs"])
        and h_overall["mean_of_group_iqrs"] != 0
        else float("nan")
    )

    pooled_std_ratio = (
        m_overall["pooled_std"] / h_overall["pooled_std"]
        if np.isfinite(m_overall["pooled_std"])
        and np.isfinite(h_overall["pooled_std"])
        and h_overall["pooled_std"] != 0
        else float("nan")
    )

    # Per-group ratios
    per_group_std_ratios: Dict[str, float] = {}
    per_group_iqr_ratios: Dict[str, float] = {}

    for group in human_stats["per_group"].keys():
        if group not in model_stats["per_group"]:
            continue
        h_group = human_stats["per_group"][group]
        m_group = model_stats["per_group"][group]

        if np.isfinite(h_group["std"]) and h_group["std"] != 0 and np.isfinite(m_group["std"]):
            per_group_std_ratios[group] = m_group["std"] / h_group["std"]

        if np.isfinite(h_group["iqr"]) and h_group["iqr"] != 0 and np.isfinite(m_group["iqr"]):
            per_group_iqr_ratios[group] = m_group["iqr"] / h_group["iqr"]

    return {
        "std_ratio": float(std_ratio),
        "iqr_ratio": float(iqr_ratio),
        "pooled_std_ratio": float(pooled_std_ratio),
        "per_group_std_ratios": per_group_std_ratios,
        "per_group_iqr_ratios": per_group_iqr_ratios,
        "interpretation": (
            "variance_collapse"
            if std_ratio < 0.8
            else (
                "variance_preserved"
                if 0.8 <= std_ratio <= 1.2
                else "variance_amplified" if std_ratio > 1.2 else "unknown"
            )
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Within-group variance preservation analysis: measures if models preserve individual differences within demographic groups"
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
        # Human variance
        human_uid_map = _uid_group_map(vc, attribute, human_dist.keys())
        human_group_vals = _build_group_values(human_uid_map, human_dist)
        human_variance = _within_group_variance_stats(human_group_vals, vc.all_questions)
        results["human"][attribute] = human_variance

        # Model variance and preservation ratios
        for model_name, model_results in models.items():
            model_dist = _model_distributions(vc, model_results)
            model_uid_map = _uid_group_map(vc, attribute, model_dist.keys())
            model_group_vals = _build_group_values(model_uid_map, model_dist)
            model_variance = _within_group_variance_stats(model_group_vals, vc.all_questions)

            preservation = _variance_preservation_ratio(human_variance, model_variance)

            results["models"].setdefault(model_name, {})[attribute] = {
                "variance_stats": model_variance,
                "preservation_ratio": preservation,
            }

    output_path = os.path.abspath(args.output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    logger.info("Wrote variance preservation analysis to %s", output_path)


if __name__ == "__main__":
    main()
