import argparse
import json
import os

import pandas as pd

from persona_understanding.value_measurement.values_comparison import (
    ValuesComparison,
    load_jsonl_file,
)


def _read_csv(csv_path):
    df = pd.read_csv(csv_path)
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    return df


def store_single_dataset_results(
    attribute, model_name, scenario, results, baselines, overall_baseline
):
    """store single dataset results"""
    os.makedirs(f"./values_results/{model_name}/vsm/{scenario}/", exist_ok=True)
    with open(
        f"./values_results/{model_name}/vsm/{scenario}/{attribute}.jsonl",
        "w",
        encoding="utf-8",
    ) as output_file:
        for result in results:
            output_file.write(json.dumps(result) + "\n")

    baselines["overall_baseline"] = overall_baseline

    with open(
        f"./values_results/{model_name}/vsm/{scenario}/{attribute}_baseline.json",
        "w",
        encoding="utf-8",
    ) as output_file:
        json.dump(baselines, output_file, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dataset",
        type=str,
        default="datasets/recruitment_data.csv",
    )
    parser.add_argument(
        "--generated_dialogues",
        type=str,
        default="datasets/generated_dialogues/first_1000_rows.jsonl",
    )
    parser.add_argument("--results_for_direct_questions", type=str, required=True)
    parser.add_argument("--results_for_dialogue_questions", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--ending_index", type=int, default=1000)

    args = parser.parse_args()

    user_profile_dataset = _read_csv(args.input_dataset)[: args.ending_index]

    direct_values_prediction = load_jsonl_file(
        args.results_for_direct_questions,
        "user_idx",
    )

    dialogue_values_prediction = load_jsonl_file(
        args.results_for_dialogue_questions,
        "user_idx",
    )

    generated_dialogues = load_jsonl_file(args.generated_dialogues)

    values_comparison_obj = ValuesComparison(
        user_profile_dataset=user_profile_dataset,
        direct_values_predictions=direct_values_prediction,
        dialogue_values_predictions=dialogue_values_prediction,
        generated_dialogues=generated_dialogues,
        verbose=1,
    )

    with open("./datasets/position_level_index.json", "r") as pl_file:
        position_levels_groups = json.load(pl_file)

    with open("./datasets/job_categories_index.json", "r") as pl_file:
        job_categories_groups = json.load(pl_file)

    single_dataset_scenarios = {
        "BA_user": "direct_values_predictions",
        "BA_dialogue": "dialogue_values_predictions",
    }
    attributes_list = [
        # ("Age", "age"),
        # ("Education Level", "education"),
        # ("Continent", "location"),
        # ("Development", "development_level"),
        # ("Position_Level", "position_level")
        ("Job Category", "job_category")
    ]

    for scenario, scenario_attr in single_dataset_scenarios.items():
        for groupby_attribute, storage_attribute in attributes_list:
            if storage_attribute == "position_level":
                results, groups_details, per_question_baselines, overall_baseline = (
                    values_comparison_obj.inner_dataset_groups_comparison(
                        groupby_attribute,
                        getattr(values_comparison_obj, scenario_attr),
                        groups=position_levels_groups,
                    )
                )
            elif storage_attribute == "job_category":
                results, groups_details, per_question_baselines, overall_baseline = (
                    values_comparison_obj.inner_dataset_groups_comparison(
                        groupby_attribute,
                        getattr(values_comparison_obj, scenario_attr),
                        groups=job_categories_groups,
                    )
                )
            else:
                results, groups_details, per_question_baselines, overall_baseline = (
                    values_comparison_obj.inner_dataset_groups_comparison(
                        groupby_attribute, getattr(values_comparison_obj, scenario_attr)
                    )
                )
            store_single_dataset_results(
                storage_attribute,
                args.model_name,
                scenario,
                results,
                per_question_baselines,
                overall_baseline,
            )

    # Consistency
    # cross_datasets_divergence = values_comparison_obj.cross_datasets_divergences()
    # cross_datasets_divergence = (
    #     values_comparison_obj.cross_datasets_divergences_id_based()
    # )
    # cross_datasets_divergence_baseline = (
    #     values_comparison_obj.cross_datasets_divergences_baseline()
    # )
    # cross_datasets_divergence_baseline = (
    #     values_comparison_obj.cross_datasets_divergences_baseline_id_based()
    # )

    # os.makedirs(f"./values_results/{args.model_name}/vsm/Consistency/", exist_ok=True)
    # with open(
    #     f"./values_results/{args.model_name}/vsm/Consistency/divergence_id_based.json",
    #     "w",
    # ) as json_file:
    #     json.dump(cross_datasets_divergence, json_file)

    # with open(
    #     f"./values_results/{args.model_name}/vsm/Consistency/baselines_id_based.json",
    #     "w",
    # ) as json_file:
    #     json.dump(cross_datasets_divergence_baseline, json_file)
