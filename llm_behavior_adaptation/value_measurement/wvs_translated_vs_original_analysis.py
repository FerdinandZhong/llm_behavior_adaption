r"""
Translated vs. Original Dialogue Analysis
==========================================
Compares model behaviour when dialogue history is in the user's native language
vs. the original English, using Qwen3-30B-A3B (or any model) career results.

Metrics computed
----------------
A. Cross-condition (new):
   1. original_vs_human       — Pearson/Spearman avg correlation + JSD vs human
   2. translated_vs_human     — same for translated dialogues
   3. cross_condition_agreement — % of (user, question) pairs with identical option_id
   4. per_language_agreement  — cross-condition agreement broken down by target_language
   5. per_language_correlation_delta — Δcorrelation (translated − original) per language

B. Preservation (reused analysis functions, grouped by target_language):
   6.  deviation_correlation  — rank preservation within language groups
   7.  outlier_preservation   — atypical individuals remain atypical after language switch
   8.  outlier_correction     — does translated dialogue pull outliers toward language mean?
   9.  variance_preservation  — model variance / human variance within language groups
   10. stereotype_amplification — does translated dialogue homogenise within language groups?

Usage
-----
python -m llm_behavior_adaptation.value_measurement.wvs_translated_vs_original_analysis \\
    --original-results   wvs_values_results/Qwen3-30B-A3B-Instruct/career/BA_dialogue_values_results/total_1000.jsonl \\
    --translated-results wvs_values_results/Qwen3-30B-A3B-Instruct/career/BA_translated_dialogue_values_results/total_1000.jsonl \\
    --translated-dialogues wvs_generated_dialogues/translated_dialogues/career/career_translated.jsonl \\
    --output-path wvs_values_results/Qwen3-30B-A3B-Instruct/career/translated_vs_original_analysis.json
"""

import argparse
import json
import logging
import os
from typing import Dict, Iterable, List, Mapping

import numpy as np
import pandas as pd
from scipy import stats

from llm_behavior_adaptation.utils import register_logger
from llm_behavior_adaptation.value_measurement.group_variance_preservation_analysis import (
    _variance_preservation_ratio,
    _within_group_variance_stats,
)

# Re-import the reusable analysis functions from their modules
from llm_behavior_adaptation.value_measurement.individual_deviation_correlation_analysis import (
    _build_group_values,
    _deviation_correlation_analysis,
    _relative_distance_preservation,
)
from llm_behavior_adaptation.value_measurement.outlier_preservation_analysis import (
    _identify_outliers,
    _outlier_correction_test,
    _outlier_preservation_analysis,
)
from llm_behavior_adaptation.value_measurement.stereotype_amplification_analysis import (
    _cross_group_stereotype_test,
    _stereotype_amplification_analysis,
)
from llm_behavior_adaptation.value_measurement.wvs_values_comparison import (
    DATASET_DIR,
    ValuesComparison,
    load_jsonl_file,
)

logger = logging.getLogger(__name__)
register_logger(logger)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _process_model_outputs(original_answers_list: List[Dict]) -> Dict[str, Dict[str, int]]:
    """Flatten JSONL results into {uid: {question_id: option_id}} dict."""
    processed: Dict[str, Dict] = {}
    for answer_details in original_answers_list:
        for user_id, answers in answer_details.items():
            per_user: Dict[str, Dict] = {}
            for cat_answers in answers.values():
                for answer in cat_answers:
                    per_user.update(answer)
            processed[user_id] = per_user
    return processed


def _human_distributions(vc: ValuesComparison) -> Dict[str, Dict[str, int]]:
    return (
        vc.user_value_dataset.astype({"D_INTERVIEW": str})
        .groupby("D_INTERVIEW", as_index=True)
        .last()[list(vc.all_questions.keys())]
        .to_dict(orient="index")
    )


def _model_dist(vc: ValuesComparison, model_results: Dict) -> Dict[str, Dict[str, int]]:
    return vc._pick_model_results_option_id(model_results)


def _uid_language_map(uid_to_language: Dict[str, str], valid_uids: Iterable[str]) -> Dict[str, str]:
    """Build {uid: target_language} restricted to valid_uids."""
    uid_set = set(valid_uids)
    return {uid: lang for uid, lang in uid_to_language.items() if uid in uid_set}


def _load_language_map(translated_dialogues_path: str) -> Dict[str, str]:
    """Parse the raw translated JSONL to extract {D_INTERVIEW: target_language}."""
    uid_to_lang: Dict[str, str] = {}
    with open(translated_dialogues_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            uid = str(entry.get("user_profile", {}).get("D_INTERVIEW", ""))
            lang = entry.get("target_language", "")
            if uid and lang:
                uid_to_lang[uid] = lang
    logger.info("Loaded language map: %d users, %d unique languages", len(uid_to_lang), len(set(uid_to_lang.values())))
    return uid_to_lang


# ---------------------------------------------------------------------------
# A. Cross-condition metrics
# ---------------------------------------------------------------------------


def _condition_vs_human(
    vc: ValuesComparison,
    human_dist: Dict[str, Dict[str, int]],
    model_results: Dict,
) -> Dict:
    """Pearson / Spearman correlation and JSD divergence for one condition vs human."""
    model_d = _model_dist(vc, model_results)

    pearson = vc._compute_id_matched_correlation(human_dist, model_d, method="pearson")
    spearman = vc._compute_id_matched_correlation(human_dist, model_d, method="spearman")
    core = vc._compute_id_matched_divergences(human_dist, model_d)
    baseline = vc._compute_baseline_two_random_matches(human_dist, model_d)

    return {
        "pearson": {
            "mean": float(np.nanmean(pearson["correlations"])),
            "median": float(np.nanmedian(pearson["correlations"])),
            "n_pairs": pearson["n_pairs"],
        },
        "spearman": {
            "mean": float(np.nanmean(spearman["correlations"])),
            "median": float(np.nanmedian(spearman["correlations"])),
            "n_pairs": spearman["n_pairs"],
        },
        "jsd": {
            "avg_divergence": float(np.mean(core["divergences"])),
            "std_divergence": float(np.std(core["divergences"])),
            "n_pairs": core["n_pairs"],
        },
        "jsd_ratio": (
            round(float(np.mean(core["divergences"])) / float(np.mean(baseline["per_user_means"])), 3)
            if baseline["per_user_means"]
            else float("nan")
        ),
    }


def _cross_condition_agreement(
    original_results: Dict[str, Dict],
    translated_results: Dict[str, Dict],
    all_questions: Mapping,
) -> Dict:
    """
    Per-(user, question) agreement rate between the two conditions.
    Also returns per-question and per-category breakdowns.
    """
    question_ids = list(all_questions.keys())
    common_uids = sorted(set(original_results.keys()) & set(translated_results.keys()))

    total = 0
    agreed = 0
    per_question_agree: Dict[str, int] = {qid: 0 for qid in question_ids}
    per_question_total: Dict[str, int] = {qid: 0 for qid in question_ids}

    for uid in common_uids:
        orig = original_results[uid]
        trans = translated_results[uid]
        for qid in question_ids:
            o = orig.get(qid)
            t = trans.get(qid)
            if o is None or t is None:
                continue
            o_id = o.get("option_id") if isinstance(o, dict) else o
            t_id = t.get("option_id") if isinstance(t, dict) else t
            if o_id is None or t_id is None:
                continue
            total += 1
            per_question_total[qid] += 1
            if o_id == t_id:
                agreed += 1
                per_question_agree[qid] += 1

    overall_rate = float(agreed / total) if total > 0 else float("nan")

    per_question_rates = {
        qid: (
            round(float(per_question_agree[qid] / per_question_total[qid]), 4)
            if per_question_total[qid] > 0
            else float("nan")
        )
        for qid in question_ids
    }

    return {
        "overall_agreement_rate": round(overall_rate, 4),
        "n_pairs": total,
        "n_agreed": agreed,
        "per_question": per_question_rates,
    }


def _per_language_agreement(
    uid_to_language: Dict[str, str],
    original_results: Dict[str, Dict],
    translated_results: Dict[str, Dict],
    all_questions: Mapping,
) -> Dict[str, Dict]:
    """Cross-condition agreement broken down by target_language."""
    question_ids = list(all_questions.keys())
    lang_stats: Dict[str, Dict[str, int]] = {}

    for uid, lang in uid_to_language.items():
        orig = original_results.get(uid)
        trans = translated_results.get(uid)
        if orig is None or trans is None:
            continue

        entry = lang_stats.setdefault(lang, {"total": 0, "agreed": 0})
        for qid in question_ids:
            o = orig.get(qid)
            t = trans.get(qid)
            if o is None or t is None:
                continue
            o_id = o.get("option_id") if isinstance(o, dict) else o
            t_id = t.get("option_id") if isinstance(t, dict) else t
            if o_id is None or t_id is None:
                continue
            entry["total"] += 1
            if o_id == t_id:
                entry["agreed"] += 1

    return {
        lang: {
            "agreement_rate": round(float(d["agreed"] / d["total"]), 4) if d["total"] > 0 else float("nan"),
            "n_pairs": d["total"],
            "n_agreed": d["agreed"],
        }
        for lang, d in sorted(lang_stats.items())
    }


def _per_language_correlation_delta(
    uid_to_language: Dict[str, str],
    human_dist: Dict[str, Dict[str, int]],
    original_dist: Dict[str, Dict[str, int]],
    translated_dist: Dict[str, Dict[str, int]],
    all_questions: Mapping,
) -> Dict[str, Dict]:
    """
    Per-user Pearson correlation with human, then average by language.
    Returns per-language: orig_mean_corr, trans_mean_corr, delta.
    """
    lang_orig: Dict[str, List[float]] = {}
    lang_trans: Dict[str, List[float]] = {}

    common_uids = set(human_dist.keys()) & set(original_dist.keys()) & set(translated_dist.keys())

    for uid in common_uids:
        lang = uid_to_language.get(uid)
        if lang is None:
            continue

        h_vec = human_dist[uid]
        o_vec = original_dist[uid]
        t_vec = translated_dist[uid]

        shared_qids = sorted(set(h_vec.keys()) & set(o_vec.keys()) & set(t_vec.keys()))
        if len(shared_qids) < 3:
            continue

        h_arr = np.array(
            [
                h_vec[q] if isinstance(h_vec[q], (int, float)) else h_vec[q].get("option_id", np.nan)
                for q in shared_qids
            ],
            dtype=float,
        )
        o_arr = np.array(
            [
                o_vec[q] if isinstance(o_vec[q], (int, float)) else o_vec[q].get("option_id", np.nan)
                for q in shared_qids
            ],
            dtype=float,
        )
        t_arr = np.array(
            [
                t_vec[q] if isinstance(t_vec[q], (int, float)) else t_vec[q].get("option_id", np.nan)
                for q in shared_qids
            ],
            dtype=float,
        )

        mask = ~(np.isnan(h_arr) | np.isnan(o_arr) | np.isnan(t_arr))
        if mask.sum() < 3:
            continue

        try:
            o_corr = stats.pearsonr(h_arr[mask], o_arr[mask])[0]
            t_corr = stats.pearsonr(h_arr[mask], t_arr[mask])[0]
        except Exception:
            continue

        lang_orig.setdefault(lang, []).append(float(o_corr))
        lang_trans.setdefault(lang, []).append(float(t_corr))

    results = {}
    all_langs = sorted(set(lang_orig.keys()) | set(lang_trans.keys()))
    for lang in all_langs:
        o_vals = lang_orig.get(lang, [])
        t_vals = lang_trans.get(lang, [])
        o_mean = float(np.nanmean(o_vals)) if o_vals else float("nan")
        t_mean = float(np.nanmean(t_vals)) if t_vals else float("nan")
        results[lang] = {
            "n_users": len(o_vals),
            "original_mean_corr": round(o_mean, 4),
            "translated_mean_corr": round(t_mean, 4),
            "delta": round(t_mean - o_mean, 4) if np.isfinite(o_mean) and np.isfinite(t_mean) else float("nan"),
        }

    return results


# ---------------------------------------------------------------------------
# B. Preservation metrics grouped by target_language
# ---------------------------------------------------------------------------


def _run_preservation_metrics(
    uid_to_language: Dict[str, str],
    human_dist: Dict[str, Dict[str, int]],
    original_dist: Dict[str, Dict[str, int]],
    translated_dist: Dict[str, Dict[str, int]],
    all_questions: Mapping,
    outlier_percentile: float = 90.0,
    seed: int = 42,
) -> Dict:
    """
    Run all reusable preservation metrics with target_language as the group attribute.
    Both original and translated conditions are evaluated.
    """
    np.random.seed(seed)
    common_uids = sorted(set(human_dist.keys()) & set(uid_to_language.keys()))
    uid_to_lang = _uid_language_map(uid_to_language, common_uids)

    results = {}

    for condition_name, model_dist in [("original", original_dist), ("translated", translated_dist)]:
        logger.info("Running preservation metrics for condition: %s", condition_name)

        model_d_filtered = {uid: model_dist[uid] for uid in uid_to_lang if uid in model_dist}

        # 1. Deviation correlation (rank preservation)
        dev_corr = _deviation_correlation_analysis(uid_to_lang, human_dist, model_d_filtered, all_questions)

        # 2. Pairwise distance preservation
        pairwise = _relative_distance_preservation(uid_to_lang, human_dist, model_d_filtered, all_questions)

        # 3. Outlier preservation
        human_outliers = _identify_outliers(uid_to_lang, human_dist, all_questions, percentile=outlier_percentile)
        outlier_pres = _outlier_preservation_analysis(
            human_outliers, uid_to_lang, human_dist, model_d_filtered, all_questions, percentile=outlier_percentile
        )
        outlier_corr = _outlier_correction_test(
            human_outliers, uid_to_lang, human_dist, model_d_filtered, all_questions
        )

        # 4. Variance preservation
        human_groups = _build_group_values(uid_to_lang, human_dist)
        model_groups = _build_group_values(uid_to_lang, model_d_filtered)
        human_var_stats = _within_group_variance_stats(human_groups, all_questions)
        model_var_stats = _within_group_variance_stats(model_groups, all_questions)
        var_ratio = _variance_preservation_ratio(human_var_stats, model_var_stats)

        # 5. Stereotype amplification
        stereo = _stereotype_amplification_analysis(uid_to_lang, human_dist, model_d_filtered, all_questions)
        cross_stereo = _cross_group_stereotype_test(uid_to_lang, human_dist, model_d_filtered, all_questions)

        results[condition_name] = {
            "deviation_correlation": dev_corr,
            "pairwise_distance_preservation": pairwise,
            "outlier_preservation": outlier_pres,
            "outlier_correction": outlier_corr,
            "variance_preservation_ratio": var_ratio,
            "stereotype_amplification": {
                "overall": stereo["overall"],
                "per_group": stereo["per_group"],
            },
            "cross_group_stereotype": cross_stereo,
        }

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare translated vs original dialogue results for a single model.")
    parser.add_argument(
        "--original-results",
        type=str,
        required=True,
        help="Path to BA_dialogue results with original English dialogues (JSONL)",
    )
    parser.add_argument(
        "--translated-results",
        type=str,
        required=True,
        help="Path to BA_dialogue results with translated dialogues (JSONL)",
    )
    parser.add_argument(
        "--translated-dialogues",
        type=str,
        required=True,
        help="Path to raw translated dialogues JSONL (for uid→language mapping)",
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
    parser.add_argument(
        "--picked-questions",
        type=str,
        default=f"{DATASET_DIR}/picked_questions.json",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Where to write the analysis JSON",
    )
    parser.add_argument("--outlier-percentile", type=float, default=90.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-progress", action="store_true", default=False)
    args = parser.parse_args()

    np.random.seed(args.seed)

    # --- Load data ---
    logger.info("Loading datasets...")
    user_profile = pd.read_csv(args.user_profile_dataset)
    user_value = pd.read_csv(args.user_value_dataset)
    with open(args.picked_questions, "r", encoding="utf-8") as f:
        picked_questions = json.load(f)

    original_results = _process_model_outputs(load_jsonl_file(args.original_results))
    translated_results = _process_model_outputs(load_jsonl_file(args.translated_results))
    uid_to_language = _load_language_map(args.translated_dialogues)

    logger.info("Original results: %d users", len(original_results))
    logger.info("Translated results: %d users", len(translated_results))

    # --- Build ValuesComparison (needed for _compute_id_matched_* methods) ---
    # Pass original as the career slot; investment slot is unused here.
    vc = ValuesComparison(
        user_profile_dataset=user_profile,
        user_value_dataset=user_value,
        ba_user_results={},
        ba_dialogue_career_results=original_results,
        ba_dialogue_investment_results={},
        picked_questions=picked_questions,
        results_output_path="",
        verbose=0,
    )

    human_dist = _human_distributions(vc)
    original_dist = _model_dist(vc, original_results)
    translated_dist = _model_dist(vc, translated_results)

    all_results: Dict = {}

    # --- A. Cross-condition metrics ---
    logger.info("Computing original vs human...")
    all_results["original_vs_human"] = _condition_vs_human(vc, human_dist, original_results)

    logger.info("Computing translated vs human...")
    # Swap career slot to translated for the translated pass
    vc_trans = ValuesComparison(
        user_profile_dataset=user_profile,
        user_value_dataset=user_value,
        ba_user_results={},
        ba_dialogue_career_results=translated_results,
        ba_dialogue_investment_results={},
        picked_questions=picked_questions,
        results_output_path="",
        verbose=0,
    )
    all_results["translated_vs_human"] = _condition_vs_human(vc_trans, human_dist, translated_results)

    logger.info("Computing cross-condition agreement...")
    all_results["cross_condition_agreement"] = _cross_condition_agreement(
        original_results, translated_results, vc.all_questions
    )

    logger.info("Computing per-language agreement...")
    all_results["per_language_agreement"] = _per_language_agreement(
        uid_to_language, original_results, translated_results, vc.all_questions
    )

    logger.info("Computing per-language correlation delta...")
    all_results["per_language_correlation_delta"] = _per_language_correlation_delta(
        uid_to_language, human_dist, original_dist, translated_dist, vc.all_questions
    )

    # --- B. Preservation metrics ---
    logger.info("Running preservation metrics by language group...")
    all_results["preservation_by_language"] = _run_preservation_metrics(
        uid_to_language=uid_to_language,
        human_dist=human_dist,
        original_dist=original_dist,
        translated_dist=translated_dist,
        all_questions=vc.all_questions,
        outlier_percentile=args.outlier_percentile,
        seed=args.seed,
    )

    # --- Write output ---
    output_path = os.path.abspath(args.output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    logger.info("Analysis written to %s", output_path)

    # Quick summary to stdout
    orig_p = all_results["original_vs_human"]["pearson"]["mean"]
    trans_p = all_results["translated_vs_human"]["pearson"]["mean"]
    agree = all_results["cross_condition_agreement"]["overall_agreement_rate"]
    print("\n=== Summary ===")
    print(f"Original vs human  — Pearson mean: {orig_p:.4f}")
    print(f"Translated vs human — Pearson mean: {trans_p:.4f}")
    print(f"Cross-condition agreement: {agree:.2%}")
    print(f"Languages covered: {len(all_results['per_language_agreement'])}")


if __name__ == "__main__":
    main()
