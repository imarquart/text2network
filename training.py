# %% Imports
import argparse
import configparser
import logging
import json
from text2network.training.bert_trainer import bert_trainer
import os

from text2network.preprocessing.nw_preprocessor import nw_preprocessor
from text2network.functions.file_helpers import check_create_folder
from text2network.utils.logging_helpers import setup_logger


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess text files.')
    parser.add_argument('--config', metavar='path', required=True,
                        help='the path to the configuration file')
    args = parser.parse_args()
    # Set a configuration path
    configuration_path = args.config
    # Load Configuration file
    config = configparser.ConfigParser()
    logging.info("Loading config in {}".format(configuration_path))
    try:
        config.read(check_create_folder(configuration_path))
    except:
        logging.error("Could not read config.")
    # Setup logging
    logging.info("Setting up logger")
    logger = setup_logger(config['Paths']['log'], config['General']['logging_level'], "preprocessing.py")

    ##################### Training
    trainer=bert_trainer(config)
    trainer.train_berts()




