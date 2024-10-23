import argparse
import ast
import json
import logging
import os
import pandas as pd

from understanding.utils import register_logger

# setup library logging
logger = logging.getLogger(__name__)
register_logger(logger)


DATASETS_FOLDER = os.path.join(os.getcwd(), "datasets")
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
        for few_shot_num in [0, 1, 5, 10]:
            target_df_col = f"{user}_{few_shot_num}_results"
            target_df[target_df_col] = target_df.apply(
                lambda row: process_result_column(row, target_df_col), axis=1
            )

    return target_df


def get_precision(row, target_col, gt_col):
    """
    Calculate the precision between the predicted output and ground truth.

    Args:
        row (pd.Series): The current row being evaluated.
        target_col (str): Name of the column containing predicted outputs.
        gt_col (str): Name of the column containing ground truth labels.

    Returns:
        float: Precision value as a percentage.

    Note:
        This function calculates the precision based on the intersection of sets.
    """
    try:
        output_set = set(row[target_col])
        gt_set = set(row[gt_col])

        intersection = output_set.intersection(gt_set)

        if row.name < 2:
            logger.info("output sample %s", output_set)
            logger.info("gt sample %s", gt_set)
            logger.info("intersection sample %s", intersection)

        if len(output_set) == 0:
            logger.info("empyt results row: %s", row.name)
            return 0

        precision = round((len(intersection) / len(output_set)), 2) * 100

        return precision
    except Exception as e:
        logger.warning(f"Error: {str(e)} for processing row: {row}")


def get_recall(row, target_col, gt_col):
    output_set = set(row[target_col])
    gt_set = set(row[gt_col])

    intersection = output_set.intersection(gt_set)

    if len(output_set) == 0:
        print(f"empyt results row: {row.name}")
        return 0

    precision = round((len(intersection) / len(gt_set)), 2) * 100

    return precision


def get_scores_df(target_df):
    for user in ["user1", "user2"]:
        for few_shot_num in [0, 1, 5, 10]:
            gt_col = f"{user}_gt_index_list"
            target_df_col = f"{user}_{few_shot_num}_results"
            precision_col = f"{user}_{few_shot_num}_precisions"
            recall_col = f"{user}_{few_shot_num}_recalls"
            target_df[precision_col] = target_df.apply(
                lambda row: get_precision(row, target_df_col, gt_col), axis=1
            )
            target_df[recall_col] = target_df.apply(
                lambda row: get_recall(row, target_df_col, gt_col), axis=1
            )

    return target_df


def get_average_scores(target_df):
    final_results_dict = {}
    for user in ["user1", "user2"]:
        for few_shot_num in [0, 1, 5, 10]:
            precision_col = f"{user}_{few_shot_num}_precisions"
            recall_col = f"{user}_{few_shot_num}_recalls"
            final_results_dict[f"{user}_{few_shot_num}_precision_avg"] = round(
                target_df[precision_col].mean(), 3
            )
            final_results_dict[f"{user}_{few_shot_num}_recall_avg"] = round(
                target_df[recall_col].mean(), 3
            )

    return final_results_dict


# def concat_gt(results_df):
#     raw_data_df = pd.read_csv(f"{DATASETS_FOLDER}/personas_candidates.csv")
#     raw_data_df = raw_data_df.loc[:, ~raw_data_df.columns.str.contains("^Unnamed")]
#     columns_to_convert = ["user1_gt_index_list", "user2_gt_index_list"]
#     for col_name in columns_to_convert:
#         raw_data_df[col_name] = raw_data_df[col_name].apply(ast.literal_eval)
#     raw_data_df = raw_data_df.iloc[10:].reset_index(drop=True)
#     results_df = pd.concat(
#         [raw_data_df[["user1_gt_index_list", "user2_gt_index_list"]], results_df],
#         axis=1,
#     )
#     return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Add arguments
    parser.add_argument("--csv_name", type=str, help="Results csv", required=True)
    parser.add_argument(
        "--scores_json", type=str, help="Output score json", required=True
    )

    # Parse the arguments
    args = parser.parse_args()

    results_df = pd.read_csv(f"{DATASETS_FOLDER}/{args.csv_name}")
    results_df = results_df.loc[:, ~results_df.columns.str.contains("^Unnamed")]

    results_df = process_df(results_df)
    results_df = get_scores_df(results_df)

    scores_dict = get_average_scores(results_df)

    with open(f"{JSON_FOLDER}/{args.scores_json}", "w") as output_file:
        json.dump(scores_dict, output_file, indent=2)
