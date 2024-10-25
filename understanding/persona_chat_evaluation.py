"""
Module description:
This module contains functions for evaluating chat personas using large language models.
It includes functionality for loading datasets, generating few-shot samples,
and querying LLMs to assess persona understanding.
"""

import argparse
import ast
import asyncio
import logging
import os
from typing import Dict, List

import pandas as pd

from understanding.llm_evaluation import (
    generate_dataset_fewshots,
    generate_fewshots_samples,
    generate_input_contents,
    query_server_in_chunk,
)
from understanding.utils import register_logger

# setup library logging
logger = logging.getLogger(__name__)
register_logger(logger)


DATASETS_FOLDER = os.path.join(os.getcwd(), "datasets")
OUTPUTS_FOLDER = os.path.join(os.getcwd(), os.environ["output_folder"])
DATASET_NAME = "personas_candidates.csv"


def load_dataset():
    """
    Load the dataset from the specified CSV file.

    Returns:
        pandas.DataFrame: Loaded DataFrame containing the dataset.
    """
    df = pd.read_csv(os.path.join(DATASETS_FOLDER, DATASET_NAME))
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    columns_to_convert = [
        "user1_personas_candidates",
        "user2_personas_candidates",
        "user1_gt_index_list",
        "user2_gt_index_list",
        "conversations",
    ]
    for col_name in columns_to_convert:
        df[col_name] = df[col_name].apply(ast.literal_eval)

    return df


async def model_query(
    few_shots_samples: List,
    few_shots_num: int,
    input_list: List,
    target_model: str,
    tested_user: str = "User 1",
    chunk_size: int = 50,
    selection_num: int = 1,
):
    """
    Prepare and execute queries using the model for few-shot samples.

    Args:
        few_shots_samples (List): List of sample dictionaries.
        few_shots_num (int): Number of samples to use.
        input_list (List): List of inputs for the query.
        target_model (str): Name of the target model
        tested_user (str): Name of the user being tested (default: "User 1").
        selection_num (int): Number of selection the model should give
    """
    logger.info("Prepare %s samples for %s", few_shots_num, tested_user)
    if few_shots_num == 0:
        samples = []
    else:
        samples = generate_fewshots_samples(
            samples_list=few_shots_samples[:few_shots_num],
            tested_user=tested_user,
            selection_num=selection_num,
        )

    requests_list = generate_dataset_fewshots(
        few_shots_examples=samples, input_list=input_list
    )

    if few_shots_num < 5:
        logger.info("Request sample: %s", requests_list[0])

    results = await query_server_in_chunk(
        requests_list, target_model, chunk_size=chunk_size
    )

    return results


async def main(
    few_shots_samples: List,
    inputs_list_dict: Dict,
    result_csv_name: str,
    target_model: str,
    num_samples: List[int],
    chunk_size: int = 50,
    selection_num: int = 1,
    existing_df: pd.DataFrame = None,
):
    """
    Main asynchronous function to run the evaluation process.

    Args:
        few_shots_samples (List): List of sample dictionaries.
        inputs_list_dict (Dict): Dictionary of input lists for each user.
        result_csv_name (str): Name of the output CSV file.
        target_model (str): Name of the target model to evaluate.
        num_samples (List[int]): List of sample numbers for testing.
        selection_num (int): Number of selection the model should give

    Returns:
        None: Saves the results to a CSV file.
    """
    results_dict = {}
    for tested_user in ["User 1", "User 2"]:
        for sample_num in num_samples:
            output_col = f"{tested_user.replace(' ', '').lower()}_{sample_num}_results"
            results_dict[output_col] = await model_query(
                few_shots_samples=few_shots_samples,
                few_shots_num=sample_num,
                input_list=inputs_list_dict[tested_user],
                target_model=target_model,
                tested_user=tested_user,
                chunk_size=chunk_size,
                selection_num=selection_num,
            )

    # Convert results_dict to DataFrame
    new_data_df = pd.DataFrame(results_dict)

    # Check if there's an existing DataFrame
    if existing_df is not None and not existing_df.empty:
        # Append the new columns to the existing DataFrame
        results_df = pd.concat([existing_df, new_data_df], axis=1)
    else:
        # If no existing DataFrame, use the new data as the DataFrame
        results_df = new_data_df

    # Save the updated DataFrame to a CSV file
    results_df.to_csv(os.path.join(OUTPUTS_FOLDER, result_csv_name), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Add arguments
    parser.add_argument("--model", type=str, help="Tested Model", required=True)
    parser.add_argument(
        "--results_csv", type=str, help="Name of results csv file", required=True
    )
    parser.add_argument("--chunk_size", type=int, help="query chunk size", default=50)
    parser.add_argument(
        "--selection_num", type=int, help="number of selections", default=1
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        nargs='+',
        default=[0, 1, 5, 10],
        help="List of sample numbers for testing (e.g., --num_samples 0 1 5 10)"
    )

    # Parse the arguments
    args = parser.parse_args()

    # Load an existing DataFrame if available
    result_csv_path = os.path.join(OUTPUTS_FOLDER, args.results_csv)
    existing_df = pd.read_csv(result_csv_path) if os.path.exists(result_csv_path) else None

    # Load dataset and set up data for testing
    persona_df = load_dataset()

    all_few_shots_samples = persona_df.iloc[:10].to_dict(orient="records")
    remaining_rows = persona_df.iloc[10:].reset_index(drop=True).to_dict(orient="records")

    user_inputs_list_dict = {
        "User 1": generate_input_contents(
            remaining_rows, tested_user="User 1", selection_num=args.selection_num
        ),
        "User 2": generate_input_contents(
            remaining_rows, tested_user="User 2", selection_num=args.selection_num
        ),
    }

    # Run the main function asynchronously
    asyncio.run(
        main(
            few_shots_samples=all_few_shots_samples,
            inputs_list_dict=user_inputs_list_dict,
            result_csv_name=args.results_csv,
            target_model=args.model,
            num_samples=args.num_samples,
            chunk_size=args.chunk_size,
            selection_num=args.selection_num,
            existing_df=existing_df
        )
    )
