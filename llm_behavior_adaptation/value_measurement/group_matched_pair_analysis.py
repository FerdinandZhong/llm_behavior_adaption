import argparse
import json
import logging
import os
from typing import Dict, List, Mapping, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from llm_behavior_adaptation.utils import register_logger
from llm_behavior_adaptation.value_measurement.formulas import emd_distance
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


def _match_pairs_cross_group(
    group_a: List[str],
    group_b: List[str],
    human_dist: Mapping[str, Mapping[str, int]],
    all_questions: Mapping[str, Mapping[str, int]],
    *,
    max_pairs: int,
    rng: np.random.Generator,
) -> List[Tuple[str, str, float]]:
    """Return matched cross-group pairs by nearest human distance (no reuse)."""
    a_ids = [uid for uid in group_a if uid in human_dist]
    b_ids = [uid for uid in group_b if uid in human_dist]
    if not a_ids or not b_ids:
        return []

    # Use smaller group as anchors
    if len(a_ids) > len(b_ids):
        a_ids, b_ids = b_ids, a_ids

    rng.shuffle(a_ids)
    rng.shuffle(b_ids)

    b_available = set(b_ids)
    pairs: List[Tuple[str, str, float]] = []

    for uid in a_ids:
        if len(pairs) >= max_pairs:
            break
        # find closest available in other group
        best = None
        best_d = None
        for vid in b_available:
            d = emd_distance(human_dist[uid], human_dist[vid], all_questions)
            if best_d is None or d < best_d:
                best_d = d
                best = vid
        if best is None:
            continue
        b_available.remove(best)
        pairs.append((uid, best, float(best_d)))

    return pairs


def _matched_pair_stats(
    pairs: List[Tuple[str, str, float]],
    human_dist: Mapping[str, Mapping[str, int]],
    model_dist: Mapping[str, Mapping[str, int]],
    all_questions: Mapping[str, Mapping[str, int]],
) -> Dict[str, float]:
    deltas: List[float] = []
    model_dists: List[float] = []
    human_dists: List[float] = []

    for uid, vid, h_dist in pairs:
        if uid not in model_dist or vid not in model_dist:
            continue
        m_dist = emd_distance(model_dist[uid], model_dist[vid], all_questions)
        model_dists.append(float(m_dist))
        human_dists.append(float(h_dist))
        deltas.append(float(m_dist - h_dist))

    return {
        "n_pairs": int(len(deltas)),
        "human_mean": float(np.mean(human_dists)) if human_dists else float("nan"),
        "model_mean": float(np.mean(model_dists)) if model_dists else float("nan"),
        "delta_mean": float(np.mean(deltas)) if deltas else float("nan"),
        "delta_median": float(np.median(deltas)) if deltas else float("nan"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Matched-pair cross-group analysis")
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
    parser.add_argument("--max-pairs-per-group", type=int, default=80)
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
    rng = np.random.default_rng(args.seed)

    results: Dict[str, Dict] = {"pairs": {}, "models": {}}

    attributes_iter = tqdm(ATTRIBUTES, desc="Attributes", unit="attr") if not args.no_progress else ATTRIBUTES
    for attribute in attributes_iter:
        group_dict = vc._get_user_id_list_for_groups(attribute)
        groups = list(group_dict.keys())
        group_pairs = [(groups[i], groups[j]) for i in range(len(groups)) for j in range(i + 1, len(groups))]

        attr_pairs: Dict[str, Dict] = {}
        for g, h in group_pairs:
            pairs = _match_pairs_cross_group(
                group_dict[g],
                group_dict[h],
                human_dist,
                vc.all_questions,
                max_pairs=args.max_pairs_per_group,
                rng=rng,
            )
            attr_pairs[f"{g}--{h}"] = {
                "n_pairs": len(pairs),
                "pairs": pairs,
            }
        results["pairs"][attribute] = attr_pairs

        for model_name, model_results in models.items():
            model_dist = _model_distributions(vc, model_results)
            per_pair_stats: Dict[str, Dict[str, float]] = {}
            all_pairs: List[Tuple[str, str, float]] = []

            for pair_key, payload in attr_pairs.items():
                pairs = payload["pairs"]
                stats = _matched_pair_stats(pairs, human_dist, model_dist, vc.all_questions)
                per_pair_stats[pair_key] = stats
                all_pairs.extend(pairs)

            overall = _matched_pair_stats(all_pairs, human_dist, model_dist, vc.all_questions)
            results["models"].setdefault(model_name, {})[attribute] = {
                "overall": overall,
                "per_pair": per_pair_stats,
            }

    output_path = os.path.abspath(args.output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    logger.info("Wrote results to %s", output_path)


if __name__ == "__main__":
    main()
