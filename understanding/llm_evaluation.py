import asyncio
import os
from copy import deepcopy
from typing import Dict, List

import openai
from openai import AsyncOpenAI
from tqdm import tqdm

from .constant import (
    LLM_CHAT_MESSAGES,
    SAMPLE_USER_CONTENT_TEMPLATE_FIVE,
    SAMPLE_USER_CONTENT_TEMPLATE_SINGLE,
)

templates_dict = {
    1: SAMPLE_USER_CONTENT_TEMPLATE_SINGLE,
    5: SAMPLE_USER_CONTENT_TEMPLATE_FIVE,
}

openai = AsyncOpenAI(
    api_key=os.environ["api_key"],
    base_url=os.environ["base_url"],  # "http://localhost:8000/v1",
)


def generate_dataset_fewshots(few_shots_examples: List[Dict], input_list: List[str]):
    """generate dataset with few shots

    Args:
        few_shots_examples (_type_): _description_
        input_list (_type_): _description_
    """
    all_requests = []
    chat_messages = deepcopy(LLM_CHAT_MESSAGES)
    for example in few_shots_examples:
        chat_messages.append({"role": "user", "content": example["user"]})
        chat_messages.append({"role": "assistant", "content": example["assistant"]})

    for input_content in input_list:
        input_messages = chat_messages + [{"role": "user", "content": input_content}]
        all_requests.append(input_messages)

    return all_requests


async def query_server_in_chunk(
    chat_messages,
    model_name,
    chunk_size=10,
):
    async def _predict(messages, model_name):
        chat_completion = await openai.chat.completions.create(
            model=model_name, messages=messages, temperature=0.1, max_tokens=16
        )
        return chat_completion.choices[0].message.content

    generated_list = []
    chunk_original_sentences = []
    for content in tqdm((chat_messages), total=len(chat_messages)):

        chunk_original_sentences.append(content)
        if len(chunk_original_sentences) >= chunk_size:
            response_list = await asyncio.gather(
                *[
                    _predict(current_content, model_name=model_name)
                    for current_content in chunk_original_sentences
                ]
            )
            try:
                generated_list.extend(
                    [
                        [int(index) for index in response.split("\n")]
                        for response in response_list
                    ]
                )
            except ValueError:
                generated_list.extend(response_list)
            chunk_original_sentences = []

    if len(chunk_original_sentences) > 0:
        response_list = await asyncio.gather(
            *[
                _predict(current_content, model_name=model_name)
                for current_content in chunk_original_sentences
            ]
        )

        try:
            generated_list.extend(
                [
                    [int(index) for index in response.split("\n")]
                    for response in response_list
                ]
            )
        except ValueError:
            generated_list.extend(response_list)

    return generated_list


def generate_fewshots_samples(
    samples_list: List[Dict], tested_user: str = "User 1", selection_num: int = 1
):
    """generate samples for few shots

    Args:
        samples_list (List[Dict]): List of samples in dictionary format
        tested_user (str): target of user
    """
    template = templates_dict[selection_num]
    few_shots_samples = []
    user_column = f"{tested_user.replace(' ', '').lower()}_personas_candidates"
    for sample_dict in samples_list:
        few_shots_samples.append(
            {
                "user": template.format(
                    dialogue_history="\n".join(sample_dict["conversations"]),
                    attributes_candidates=sample_dict[user_column],
                    user=tested_user,
                ),
                "assistant": "\n".join(
                    [
                        str(index)
                        for index in sample_dict[
                            f"{tested_user.replace(' ', '').lower()}_gt_index_list"
                        ]
                    ][:selection_num]
                ),
            }
        )
    return few_shots_samples


def generate_input_contents(
    input_rows: List[Dict], tested_user: str = "User 1", selection_num: int = 1
):
    """generate input contents

    Args:
        input_rows (List[Dict]): List of samples in dictionary format
        tested_user (str): target of user
    """
    template = templates_dict[selection_num]
    user_column = f"{tested_user.replace(' ', '').lower()}_personas_candidates"
    return [
        template.format(
            dialogue_history="\n".join(input_row["conversations"]),
            attributes_candidates=input_row[user_column],
            user=tested_user,
        )
        for input_row in input_rows
    ]
