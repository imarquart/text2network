# %% Imports
import argparse
import configparser
import json
import logging
import os
from dataclasses import asdict, dataclass

import nltk
import yaml

from config.dataclasses import PreProcessConfig, TrainingConfig
# Import components from our package
from text2network.preprocessing.nw_preprocessor import TextPreprocessor
from text2network.training.bert_trainer import model_trainer
from text2network.utils.file_helpers import check_create_folder, check_folder
from text2network.utils.logging_helpers import log, setup_logger

nltk.download("stopwords")


def load_config(file_path):
    with open(file_path, "r") as f:
        config_dict = yaml.safe_load(f)
    return TrainingConfig(**config_dict)


def main(args):
    config = load_config(args.config)
    if args.logging_level:
        config.logging_level = args.logging_level
    if args.other_loggers:
        config.other_loggers = args.other_loggers
    setup_logger(logging_path = config.logging_folder ,logging_level=config.logging_level)
    trainer = model_trainer(**asdict(config))
    trainer.train_models()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Text Preprocessing")
    parser.add_argument("--config", help="Path to the YAML configuration file")
    parser.add_argument("--logging_level", type=int, help="Logging level for t2n logger")
    parser.add_argument("--other_loggers", type=int, help="Logging level for other loggers")

    args = parser.parse_args()
    main(args)
