import argparse
import ast
import json
import logging
import os
from copy import deepcopy
from typing import Dict, List

import pandas as pd
from openai import OpenAI
from tqdm import tqdm

from understanding.constant import (
    DATASETS_FOLDER,
    LLM_CHAT_MESSAGES,
    SAMPLE_USER_CONTENT_TEMPLATE_NO_FIXED,
)
from understanding.utils import register_logger

JSONL_FOLDER = os.path.join(os.getcwd(), os.environ["jsonl_folder"])


logger = logging.getLogger(__name__)
register_logger(logger)


def prepare_batch_json(input_list, model_name):
    """generate dataset with few shots

    Args:
        few_shots_examples (_type_): _description_
        input_list (_type_): _description_
    """
    all_requests = []
    chat_messages = deepcopy(LLM_CHAT_MESSAGES)

    fixed_dict = {
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {"model": model_name, "max_tokens": 64},
    }
    for input_index, input_content in enumerate(input_list):
        request_dict = deepcopy(fixed_dict)
        custom_id = f"persona_request-{input_index+1}"
        input_messages = chat_messages + [{"role": "user", "content": input_content}]
        request_dict["custom_id"] = custom_id
        request_dict["body"]["messages"] = input_messages
        all_requests.append(request_dict)

    return all_requests


def store_jsonl_file(jsonl_file_path, jsonl_inputs_list, starting_index, ending_index):
    with open(jsonl_file_path, "w") as jsonl_file:
        for entry in tqdm(
            jsonl_inputs_list[starting_index:ending_index], desc="Writing JSONL entries"
        ):
            jsonl_file.write(json.dumps(entry) + "\n")


def load_dataset():
    """
    Load the dataset from the specified CSV file.

    Returns:
        pandas.DataFrame: Loaded DataFrame containing the dataset.
    """
    df = pd.read_csv(os.path.join(DATASETS_FOLDER, "personas_chatgpt.csv"))
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    columns_to_convert = [
        "user1_personas",
        "user2_personas",
        "conversations",
    ]
    for col_name in columns_to_convert:
        df[col_name] = df[col_name].apply(ast.literal_eval)

    return df


def generate_input_contents(input_rows: List[Dict], tested_user: str = "User 1"):
    """generate input contents

    Args:
        input_rows (List[Dict]): List of samples in dictionary format
        tested_user (str): target of user
    """
    template = SAMPLE_USER_CONTENT_TEMPLATE_NO_FIXED
    user_column = f"{tested_user.replace(' ', '').lower()}_personas"
    return [
        template.format(
            dialogue_history="\n".join(input_row["conversations"]),
            attributes_candidates="\n".join(input_row[user_column]),
            user=tested_user,
        )
        for input_row in input_rows
    ]


def submit_batch(
    client,
    jsonl_file_path,
):
    batch_input_file = client.files.create(
        file=open(jsonl_file_path, "rb"), purpose="batch"
    )
    batch_input_file_id = batch_input_file.id
    batch_obj = client.batches.create(
        input_file_id=batch_input_file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"description": "persona attributes selection"},
    )
    print(batch_obj)
    with open(
        jsonl_file_path.replace(".jsonl", "_meta.json"), "w", encoding="utf-8"
    ) as meta_file:
        json.dump(batch_obj.to_dict(), meta_file, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Add arguments
    parser.add_argument("--model_name", type=str, default="gpt-4o-mini-2024-07-18")
    parser.add_argument(
        "--starting_index", type=int, help="starting index", required=True
    )
    parser.add_argument("--ending_index", type=int, help="ending index", required=True)

    # Parse the arguments
    args = parser.parse_args()
    user1_file_name = f"user1_inputs_{args.starting_index}_{args.ending_index}.jsonl"
    user2_file_name = f"user2_inputs_{args.starting_index}_{args.ending_index}.jsonl"
    user1_jsonl_file_path = os.path.join(JSONL_FOLDER, user1_file_name)
    user2_jsonl_file_path = os.path.join(JSONL_FOLDER, user2_file_name)

    # Load dataset and set up data for testing
    persona_df = load_dataset()

    user1_jsonl_inputs_list = prepare_batch_json(
        generate_input_contents(
            persona_df.to_dict(orient="records"), tested_user="User 1"
        ),
        args.model_name,
    )

    user2_jsonl_inputs_list = prepare_batch_json(
        generate_input_contents(
            persona_df.to_dict(orient="records"), tested_user="User 2"
        ),
        args.model_name,
    )

    store_jsonl_file(
        user1_jsonl_file_path,
        user1_jsonl_inputs_list,
        args.starting_index,
        args.ending_index,
    )
    store_jsonl_file(
        user2_jsonl_file_path,
        user2_jsonl_inputs_list,
        args.starting_index,
        args.ending_index,
    )

    client = OpenAI()

    submit_batch(client, user1_jsonl_file_path)
    submit_batch(client, user2_jsonl_file_path)
