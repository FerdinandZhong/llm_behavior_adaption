import argparse
import ast
import json
import logging
import os
from typing import List, Union

import numpy as np
import pandas as pd

from understanding.constant import DATASETS_FOLDER, OUTPUTS_FOLDER
from understanding.utils import register_logger

# setup library logging
logger = logging.getLogger(__name__)
register_logger(logger)


JSON_FOLDER = os.path.join(os.getcwd(), "results_jsons")


def process_result_column(row, target_column):
    """
    Process the result column for a given row and target column.

    Args:
        row (dict): The current row being processed.
        target_column (str): The name of the target column to process.

    Returns:
        list: A list of integers representing the processed values.

    Raises:
        ValueError: If the input cannot be parsed into a list of integers.
    """
    target_item = row[target_column]

    if isinstance(target_item, list):
        return target_item
    else:
        try:
            if "[" in target_item and "]" in target_item:
                return [
                    int(single_item) for single_item in ast.literal_eval(target_item)
                ]
            elif "," in target_item:
                return [
                    int(single_item.strip()) for single_item in target_item.split(",")
                ]
            elif "." in target_item:
                return [
                    int(single_item.strip()) for single_item in target_item.split(".")
                ]
            else:
                return [
                    int(single_item.strip()) for single_item in target_item.split("\n")
                ]
        except Exception:
            logger.warning("Error processing %s with index: %s", target_item, row.name)
            return []


def process_df(target_df):
    """
    Process the DataFrame by adding columns for each user and few-shot number combination.

    Args:
        target_df (pandas.DataFrame): The input DataFrame to be processed.

    Returns:
        pandas.DataFrame: The processed DataFrame with additional columns.

    Notes:
        This function adds columns for each user ("user1" and "user2") and few-shot number combination (0, 1, 5, 10).
        It applies the `process_result_column` function to each row for these columns.
    """
    for user in ["user1", "user2"]:
        target_df[f"{user}_gt_index_list"] = target_df.apply(
            lambda row: process_result_column(row, f"{user}_gt_index_list"), axis=1
        )
        target_df[f"gpt_gt_{user}_new"] = target_df.apply(
            lambda row: process_result_column(row, f"gpt_gt_{user}_new"), axis=1
        )
        for few_shot_num in [0, 1, 5, 10]:
            target_df_col = f"{user}_{few_shot_num}_results"
            target_df[target_df_col] = target_df.apply(
                lambda row: process_result_column(row, target_df_col), axis=1
            )

    return target_df


def track_accuracy(row, output_col, gt_col, similarity_col, avg_similarity_col):
    """track accuracy"""
    if len(row[output_col]) == 0:
        return 0, np.nan, np.nan
    if row[output_col][0] in row[gt_col]:
        return 1, np.nan, np.nan
    else:
        selected_similarity = row[similarity_col][row[output_col][0] - 1]
        similarity_distance = (selected_similarity - row[avg_similarity_col]) / row[
            avg_similarity_col
        ]
        rank = get_rank(row[similarity_col], row[output_col][0] - 1)
        return 0, similarity_distance, rank


def track_accuracy_gpt_based(row, output_col, gpt_col):
    """track accuracy"""
    if len(row[gpt_col]) == 0:
        if len(row[output_col]) == 0:
            return 1
        else:
            return 0
    else:
        if len(row[output_col]) == 0:
            return 0
        if row[output_col][0] in row[gpt_col]:
            return 1
        else:
            return 0


def get_scores_df(target_df):
    for user in ["user1", "user2"]:
        gt_col = f"{user}_gt_index_list"
        similarity_col = f"{user}_candidates_similarities"
        avg_similarity_col = f"{user}_avg_similarity"
        for few_shot_num in [0, 1, 5, 10]:
            target_df_col = f"{user}_{few_shot_num}_results"
            accuracy_col = f"{user}_{few_shot_num}_accuracy"
            distance_col = f"{user}_{few_shot_num}_distance"
            rank_col = f"{user}_{few_shot_num}_rank"
            gpt_accuray_col = f"{user}_{few_shot_num}_gpt_accuracy"
            target_df[[accuracy_col, distance_col, rank_col]] = target_df.apply(
                lambda row: track_accuracy(
                    row, target_df_col, gt_col, similarity_col, avg_similarity_col
                ),
                axis=1,
            ).apply(pd.Series)
            target_df[gpt_accuray_col] = target_df.apply(
                lambda row: track_accuracy_gpt_based(
                    row, target_df_col, f"gpt_gt_{user}_new"
                ),
                axis=1,
            )

    return target_df


def get_average_scores(target_df):
    final_results_dict = {}
    for user in ["user1", "user2"]:
        for few_shot_num in [0, 1, 5, 10]:
            raw_output_list = f"{user}_{few_shot_num}_results"
            accuracy_col = f"{user}_{few_shot_num}_accuracy"
            distance_col = f"{user}_{few_shot_num}_distance"
            rank_col = f"{user}_{few_shot_num}_rank"
            gpt_accuray_col = f"{user}_{few_shot_num}_gpt_accuracy"
            final_results_dict[f"{user}_{few_shot_num}_accuracy_avg"] = round(
                target_df[accuracy_col].mean(), 3
            )
            final_results_dict[f"{user}_{few_shot_num}_distance_avg"] = round(
                target_df[distance_col].mean(), 3
            )
            final_results_dict[f"{user}_{few_shot_num}_similarity_rank"] = round(
                target_df[rank_col].mean(), 3
            )
            final_results_dict[f"{user}_{few_shot_num}_gpt_accuracy"] = round(
                target_df[gpt_accuray_col].mean(), 3
            )
            empty_list_percentage = round(
                (target_df[raw_output_list].apply(lambda x: len(x) == 0).mean()) * 100,
                3,
            )

            final_results_dict[f"{user}_{few_shot_num}_no_selection"] = (
                empty_list_percentage
            )

    return final_results_dict


def get_rank(data: List[Union[float, None]], index: int) -> Union[int, None]:
    """
    Get the rank of the value at the specified index in a list of floats and None values.
    Ranks are determined in descending order, ignoring None values.

    Args:
        data (List[Union[float, None]]): List containing floats and None.
        index (int): Index of the value whose rank is to be determined.

    Returns:
        int: Rank of the value at the specified index in descending order, or None if the index is invalid or value is None.
    """
    if index < 0 or index >= len(data) or data[index] is None:
        return None  # Invalid index or value is None

    # Filter out None values and sort the remaining floats in descending order
    sorted_values = sorted([x for x in data if x is not None], reverse=True)

    # Get the rank of the value at the specified index
    value = data[index]
    rank = sorted_values.index(value) + 1  # Rank is 1-based

    return rank


def concat_gt(results_df):
    similarity_df = pd.read_csv(
        f"{DATASETS_FOLDER}/personas_candidates_similarities.csv"
    )
    similarity_df = similarity_df.loc[
        :, ~similarity_df.columns.str.contains("^Unnamed")
    ]
    similarity_df = similarity_df.iloc[10:].reset_index(drop=True)
    columns_to_convert = [
        "user1_candidates_similarities",
        "user2_candidates_similarities",
    ]
    for col_name in columns_to_convert:
        similarity_df[col_name] = similarity_df[col_name].apply(ast.literal_eval)
    results_df = pd.concat(
        [
            similarity_df[
                [
                    "user1_gt_index_list",
                    "user2_gt_index_list",
                    "user1_candidates_similarities",
                    "user2_candidates_similarities",
                    "user1_avg_similarity",
                    "user2_avg_similarity",
                ]
            ],
            results_df,
        ],
        axis=1,
    )
    return results_df


def concat_gpt_results(results_df):
    gpt_df = pd.read_csv(f"{DATASETS_FOLDER}/personas_chatgpt.csv")
    gpt_df = gpt_df.loc[:, ~gpt_df.columns.str.contains("^Unnamed")]
    gpt_df = gpt_df.iloc[10:].reset_index(drop=True)
    columns_to_convert = [
        "gpt_gt_user1_new",
        "gpt_gt_user2_new",
    ]
    for col_name in columns_to_convert:
        gpt_df[col_name] = gpt_df[col_name].apply(ast.literal_eval)
    results_df = pd.concat(
        [
            gpt_df[
                [
                    "gpt_gt_user1_new",
                    "gpt_gt_user2_new",
                ]
            ],
            results_df,
        ],
        axis=1,
    )
    return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Add arguments
    parser.add_argument("--csv_name", type=str, help="Results csv", required=True)
    parser.add_argument(
        "--scores_json", type=str, help="Output score json", required=True
    )

    # Parse the arguments
    args = parser.parse_args()

    results_df = pd.read_csv(f"{OUTPUTS_FOLDER}/{args.csv_name}")
    results_df = results_df.loc[:, ~results_df.columns.str.contains("^Unnamed")]

    results_df = concat_gt(results_df)
    results_df = concat_gpt_results(results_df)
    results_df = process_df(results_df)
    print(results_df.head())
    results_df = get_scores_df(results_df)

    scores_dict = get_average_scores(results_df)

    with open(f"{JSON_FOLDER}/{args.scores_json}", "w") as output_file:
        json.dump(scores_dict, output_file, indent=2)
