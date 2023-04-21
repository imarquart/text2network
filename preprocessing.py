# %% Imports
import argparse
import configparser
import logging
import json
import os
import nltk

# Import components from our package
from text2network.preprocessing.nw_preprocessor import TextPreprocessor
from text2network.utils.file_helpers import check_create_folder, check_folder
from text2network.utils.logging_helpers import setup_logger, log

import yaml
from dataclasses import dataclass


@dataclass
class Config:
    maximum_sequence_length: int
    split_symbol: str
    logging_level: int
    other_loggers: int


def load_config(file_path):
    with open(file_path, "r") as f:
        config_dict = yaml.safe_load(f)
    return Config(**config_dict)


def main(args):
    if args.config:
        config = load_config(args.config)
    else:
        config = Config(
            maximum_sequence_length=args.maximum_sequence_length
            if args.maximum_sequence_length
            else 512,
            split_symbol=args.split_symbol if args.split_symbol else "_",
            logging_level=args.logging_level if args.logging_level else logging.INFO,
            other_loggers=args.other_loggers if args.other_loggers else logging.WARNING,
        )

    preprocessor = TextPreprocessor(
        config.maximum_sequence_length,
        config.split_symbol,
    )
    preprocessor.logging_level = config.logging_level
    preprocessor.other_loggers = config.other_loggers
    preprocessor.preprocess(
        args.folder if args.folder else "data/test",
        output_folder=args.output_folder if args.output_folder else "data/test_output",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Text Preprocessing")
    parser.add_argument("--folder", help="Path to the folder containing text files")
    parser.add_argument("--output_folder", help="Path to the output folder")
    parser.add_argument("--config", help="Path to the YAML configuration file")
    parser.add_argument(
        "--maximum_sequence_length", type=int, help="Maximum sequence length"
    )
    parser.add_argument(
        "--split_symbol", help="Symbol used to split parameters in the file name"
    )
    parser.add_argument(
        "--logging_level", type=int, help="Logging level for t2n logger"
    )
    parser.add_argument(
        "--other_loggers", type=int, help="Logging level for other loggers"
    )

    args = parser.parse_args()
    main(args)
