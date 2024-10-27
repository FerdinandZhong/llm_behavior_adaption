import ast
import logging
import os
import sys

import pandas as pd

from understanding.constant import DATASET_NAME, DATASETS_FOLDER


class ColorfulFormatter(logging.Formatter):
    grey = "\x1b[38;21m"
    yellow = "\x1b[33;21m"
    red = "\x1b[31;21m"
    green = "\x1b[32m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    formatter = (
        "%(asctime)s - {color}%(levelname)s{reset} - %(filename)s:%(lineno)d"
        " - %(module)s.%(funcName)s - %(process)d - %(message)s"
    )
    FORMATS = {
        logging.DEBUG: formatter.format(color=grey, reset=reset),
        logging.INFO: formatter.format(color=green, reset=reset),
        logging.WARNING: formatter.format(color=yellow, reset=reset),
        logging.ERROR: formatter.format(color=red, reset=reset),
        logging.CRITICAL: formatter.format(color=bold_red, reset=reset),
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def register_logger(logger=None):
    """register colorful debug log"""
    if not logger:
        logger = logging.getLogger()
    if not logger.hasHandlers():
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(ColorfulFormatter())
        logger.setLevel(logging.DEBUG)
        logger.addHandler(handler)


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
