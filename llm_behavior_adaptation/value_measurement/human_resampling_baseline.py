"""
Human Resampling Baseline Analysis

This script computes the natural homogenization rate in human data itself,
establishing a baseline for how much humans naturally cluster toward their
demographic group centroids.

Methodology:
- Split human data into two halves (or use leave-one-out)
- Compute group centroids from half 1
- Measure how much half 2 "homogenizes" toward those centroids
- Repeat with bootstrap resampling for confidence intervals

This baseline answers: "What homogenization rate should we expect even
without model intervention?"
"""

import argparse
import json
import logging
import os
from typing import Dict, List, Mapping, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from llm_behavior_adaptation.utils import register_logger
from llm_behavior_adaptation.value_measurement.formulas import componentwise_centroid, emd_distance
from llm_behavior_adaptation.value_measurement.wvs_values_comparison import ATTRIBUTES, DATASET_DIR, ValuesComparison

logger = logging.getLogger(__name__)
register_logger(logger)


def _human_distributions(vc: ValuesComparison) -> Dict[str, Dict[str, int]]:
    """Get human response distributions indexed by user ID."""
    return (
        vc.user_value_dataset.astype({"D_INTERVIEW": str})
        .groupby("D_INTERVIEW", as_index=True)
        .last()[list(vc.all_questions.keys())]
        .to_dict(orient="index")
    )


def _uid_group_map(vc: ValuesComparison, attribute: str, valid_uids: set) -> Dict[str, str]:
    """Map user IDs to their demographic group for a given attribute."""
    df = vc.user_profile_dataset.copy()
    df["D_INTERVIEW"] = df["D_INTERVIEW"].astype(str)
    df = df.set_index("D_INTERVIEW")
    label_series = vc._group_label_series(df.reset_index(), attribute)
    label_series.index = df.index
    label_series = label_series.dropna()
    return {uid: str(label_series.loc[uid]) for uid in label_series.index if uid in valid_uids}


def _group_centroids_from_subset(
    uids: List[str],
    uid_to_group: Dict[str, str],
    human_dist: Dict[str, Dict[str, int]],
    all_questions: Mapping[str, Mapping[str, int]],
) -> Dict[str, Dict[str, int]]:
    """Compute group centroids from a subset of users."""
    grouped: Dict[str, List[Dict[str, int]]] = {}
    for uid in uids:
        if uid not in uid_to_group or uid not in human_dist:
            continue
        group = uid_to_group[uid]
        grouped.setdefault(group, []).append(human_dist[uid])

    centroids: Dict[str, Dict[str, int]] = {}
    for group_name, answers_list in grouped.items():
        if len(answers_list) >= 2:  # Need at least 2 samples for meaningful centroid
            centroids[group_name] = componentwise_centroid(answers_list, all_questions)
    return centroids


def _compute_homogenization_for_subset(
    test_uids: List[str],
    uid_to_group: Dict[str, str],
    human_dist: Dict[str, Dict[str, int]],
    centroids: Dict[str, Dict[str, int]],
    global_centroid: Dict[str, int],
    all_questions: Mapping[str, Mapping[str, int]],
) -> Tuple[float, float, int]:
    """
    Compute homogenization metrics for a test subset against pre-computed centroids.

    Returns:
        - homogenization_rate: % of individuals closer to group centroid than global centroid
        - mean_delta: average distance reduction toward group centroid
        - n_valid: number of valid individuals
    """
    toward_group = 0
    away_from_group = 0
    deltas = []

    for uid in test_uids:
        if uid not in uid_to_group or uid not in human_dist:
            continue
        group = uid_to_group[uid]
        if group not in centroids:
            continue

        # Distance to own group centroid
        dist_to_group = emd_distance(human_dist[uid], centroids[group], all_questions)
        # Distance to global centroid
        dist_to_global = emd_distance(human_dist[uid], global_centroid, all_questions)

        delta = dist_to_group - dist_to_global
        deltas.append(delta)

        if dist_to_group < dist_to_global:
            toward_group += 1
        else:
            away_from_group += 1

    n_valid = toward_group + away_from_group
    if n_valid == 0:
        return float("nan"), float("nan"), 0

    homogenization_rate = 100 * toward_group / n_valid
    mean_delta = float(np.mean(deltas)) if deltas else float("nan")

    return homogenization_rate, mean_delta, n_valid


def split_half_baseline(
    uid_to_group: Dict[str, str],
    human_dist: Dict[str, Dict[str, int]],
    all_questions: Mapping[str, Mapping[str, int]],
    n_bootstrap: int = 100,
    random_seed: int = 42,
) -> Dict[str, object]:
    """
    Compute human baseline using split-half methodology with bootstrap.

    Methodology:
    1. Randomly split users into two halves
    2. Compute group centroids from half 1
    3. Measure homogenization of half 2 toward those centroids
    4. Repeat with bootstrap for confidence intervals
    """
    rng = np.random.RandomState(random_seed)

    valid_uids = [uid for uid in uid_to_group if uid in human_dist]
    if len(valid_uids) < 10:
        return {"error": "insufficient_data", "n_valid": len(valid_uids)}

    # Global centroid (from all human data)
    all_answers = [human_dist[uid] for uid in valid_uids]
    global_centroid = componentwise_centroid(all_answers, all_questions)

    homogenization_rates = []
    mean_deltas = []

    for _ in range(n_bootstrap):
        # Random split
        shuffled = rng.permutation(valid_uids)
        half = len(shuffled) // 2
        train_uids = list(shuffled[:half])
        test_uids = list(shuffled[half:])

        # Compute centroids from training half
        centroids = _group_centroids_from_subset(train_uids, uid_to_group, human_dist, all_questions)

        if not centroids:
            continue

        # Evaluate on test half
        rate, delta, n = _compute_homogenization_for_subset(
            test_uids, uid_to_group, human_dist, centroids, global_centroid, all_questions
        )

        if not np.isnan(rate):
            homogenization_rates.append(rate)
            mean_deltas.append(delta)

    if not homogenization_rates:
        return {"error": "no_valid_splits", "n_valid": len(valid_uids)}

    return {
        "method": "split_half_bootstrap",
        "n_bootstrap": n_bootstrap,
        "n_valid_users": len(valid_uids),
        "homogenization_rate": {
            "mean": float(np.mean(homogenization_rates)),
            "std": float(np.std(homogenization_rates)),
            "ci_lower": float(np.percentile(homogenization_rates, 2.5)),
            "ci_upper": float(np.percentile(homogenization_rates, 97.5)),
            "min": float(np.min(homogenization_rates)),
            "max": float(np.max(homogenization_rates)),
        },
        "mean_delta": {
            "mean": float(np.mean(mean_deltas)),
            "std": float(np.std(mean_deltas)),
            "ci_lower": float(np.percentile(mean_deltas, 2.5)),
            "ci_upper": float(np.percentile(mean_deltas, 97.5)),
        },
        "interpretation": _interpret_baseline(np.mean(homogenization_rates)),
    }


def leave_one_out_baseline(
    uid_to_group: Dict[str, str],
    human_dist: Dict[str, Dict[str, int]],
    all_questions: Mapping[str, Mapping[str, int]],
) -> Dict[str, object]:
    """
    Compute human baseline using leave-one-out methodology.

    For each individual:
    1. Compute group centroid excluding that individual
    2. Measure if individual is closer to their group centroid than global centroid
    """
    valid_uids = [uid for uid in uid_to_group if uid in human_dist]
    if len(valid_uids) < 10:
        return {"error": "insufficient_data", "n_valid": len(valid_uids)}

    # Global centroid
    all_answers = [human_dist[uid] for uid in valid_uids]
    global_centroid = componentwise_centroid(all_answers, all_questions)

    # Group users
    grouped: Dict[str, List[str]] = {}
    for uid in valid_uids:
        group = uid_to_group[uid]
        grouped.setdefault(group, []).append(uid)

    toward_group = 0
    away_from_group = 0
    deltas = []
    per_group_results: Dict[str, Dict] = {}

    for group, group_uids in grouped.items():
        if len(group_uids) < 3:  # Need at least 3 for meaningful LOO
            continue

        group_toward = 0
        group_away = 0
        group_deltas = []

        for i, test_uid in enumerate(group_uids):
            # Compute centroid excluding this user
            other_uids = [uid for j, uid in enumerate(group_uids) if j != i]
            other_answers = [human_dist[uid] for uid in other_uids]
            loo_centroid = componentwise_centroid(other_answers, all_questions)

            # Measure distances
            dist_to_group = emd_distance(human_dist[test_uid], loo_centroid, all_questions)
            dist_to_global = emd_distance(human_dist[test_uid], global_centroid, all_questions)

            delta = dist_to_group - dist_to_global
            deltas.append(delta)
            group_deltas.append(delta)

            if dist_to_group < dist_to_global:
                toward_group += 1
                group_toward += 1
            else:
                away_from_group += 1
                group_away += 1

        n_group = group_toward + group_away
        if n_group > 0:
            per_group_results[group] = {
                "homogenization_rate": 100 * group_toward / n_group,
                "mean_delta": float(np.mean(group_deltas)),
                "n": n_group,
            }

    n_valid = toward_group + away_from_group
    if n_valid == 0:
        return {"error": "no_valid_comparisons", "n_valid": len(valid_uids)}

    return {
        "method": "leave_one_out",
        "n_valid_users": len(valid_uids),
        "n_comparisons": n_valid,
        "homogenization_rate": 100 * toward_group / n_valid,
        "mean_delta": float(np.mean(deltas)),
        "std_delta": float(np.std(deltas)),
        "per_group": per_group_results,
        "interpretation": _interpret_baseline(100 * toward_group / n_valid),
    }


def _interpret_baseline(homogenization_rate: float) -> str:
    """Interpret the baseline homogenization rate."""
    if homogenization_rate < 40:
        return "low_natural_clustering"
    elif homogenization_rate < 60:
        return "moderate_natural_clustering"
    elif homogenization_rate < 80:
        return "high_natural_clustering"
    else:
        return "very_high_natural_clustering"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute human baseline for homogenization metrics")
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
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=100,
        help="Number of bootstrap iterations for split-half method",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        default=False,
        help="Disable progress bars",
    )
    args = parser.parse_args()

    user_profile = pd.read_csv(args.user_profile_dataset)
    user_value = pd.read_csv(args.user_value_dataset)

    with open(f"{DATASET_DIR}/picked_questions.json", "r", encoding="utf-8") as f:
        picked_questions = json.load(f)

    # Create minimal ValuesComparison for data loading
    vc = ValuesComparison(
        user_profile_dataset=user_profile,
        user_value_dataset=user_value,
        ba_user_results={},  # Not needed for human baseline
        ba_dialogue_career_results={},
        ba_dialogue_investment_results={},
        picked_questions=picked_questions,
        results_output_path="",
        verbose=0,
    )

    human_dist = _human_distributions(vc)
    results: Dict[str, Dict] = {
        "summary": {},
        "per_attribute": {},
    }

    attributes_iter = (
        tqdm(ATTRIBUTES, desc="Computing human baselines", unit="attr") if not args.no_progress else ATTRIBUTES
    )

    all_split_half_rates = []
    all_loo_rates = []

    for attribute in attributes_iter:
        uid_to_group = _uid_group_map(vc, attribute, set(human_dist.keys()))

        if len(uid_to_group) < 20:
            results["per_attribute"][attribute] = {"error": "insufficient_data"}
            continue

        # Split-half with bootstrap
        split_half_result = split_half_baseline(
            uid_to_group, human_dist, vc.all_questions, n_bootstrap=args.n_bootstrap, random_seed=args.random_seed
        )

        # Leave-one-out
        loo_result = leave_one_out_baseline(uid_to_group, human_dist, vc.all_questions)

        results["per_attribute"][attribute] = {
            "split_half": split_half_result,
            "leave_one_out": loo_result,
        }

        if "homogenization_rate" in split_half_result and isinstance(split_half_result["homogenization_rate"], dict):
            all_split_half_rates.append(split_half_result["homogenization_rate"]["mean"])
        if "homogenization_rate" in loo_result and not isinstance(loo_result["homogenization_rate"], dict):
            all_loo_rates.append(loo_result["homogenization_rate"])

    # Compute overall summary
    if all_split_half_rates:
        results["summary"]["split_half"] = {
            "mean_across_attributes": float(np.mean(all_split_half_rates)),
            "std_across_attributes": float(np.std(all_split_half_rates)),
            "min": float(np.min(all_split_half_rates)),
            "max": float(np.max(all_split_half_rates)),
        }

    if all_loo_rates:
        results["summary"]["leave_one_out"] = {
            "mean_across_attributes": float(np.mean(all_loo_rates)),
            "std_across_attributes": float(np.std(all_loo_rates)),
            "min": float(np.min(all_loo_rates)),
            "max": float(np.max(all_loo_rates)),
        }

    # Add interpretation
    if all_split_half_rates:
        avg_rate = np.mean(all_split_half_rates)
        results["summary"]["interpretation"] = {
            "human_baseline_rate": float(avg_rate),
            "description": f"Humans naturally show ~{avg_rate:.1f}% clustering toward demographic group centroids",
            "model_comparison_guide": {
                "acceptable": f"<{avg_rate * 1.5:.0f}% (within 1.5x human baseline)",
                "concerning": f"{avg_rate * 1.5:.0f}-{avg_rate * 3:.0f}% (1.5-3x human baseline)",
                "problematic": f">{avg_rate * 3:.0f}% (>3x human baseline)",
            },
        }

    # Save results
    output_path = os.path.abspath(args.output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    logger.info("Wrote human baseline analysis to %s", output_path)

    # Print summary
    if "interpretation" in results["summary"]:
        print("\n" + "=" * 60)
        print("HUMAN BASELINE SUMMARY")
        print("=" * 60)
        interp = results["summary"]["interpretation"]
        print(f"Human natural clustering rate: {interp['human_baseline_rate']:.1f}%")
        print("\nModel comparison guide:")
        for level, threshold in interp["model_comparison_guide"].items():
            print(f"  {level}: {threshold}")
        print("=" * 60)


if __name__ == "__main__":
    main()
