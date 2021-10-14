# %% Imports
import argparse
import configparser
import logging
import json
import os

from text2network.datasets.text_dataset import query_dataset
from text2network.preprocessing.nw_preprocessor import nw_preprocessor
from text2network.processing.nw_processor import nw_processor
from text2network.functions.file_helpers import check_create_folder
from text2network.utils.load_bert import get_bert_and_tokenizer
from text2network.utils.logging_helpers import setup_logger

import nltk
nltk.download('all')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess text files.')
    parser.add_argument('--config', metavar='path', required=True,
                        help='the path to the configuration file')
    args = parser.parse_args()
    # Set a configuration path
    configuration_path = args.config
    print("Loading config in {}".format(check_create_folder(configuration_path)))
    # Load Configuration file
    config = configparser.ConfigParser()
    try:
        config.read(check_create_folder(configuration_path))
    except:
        logging.error("Could not read config.")
        raise
    # Setup logging
    logger = setup_logger(config['Paths']['log'], int(config['General']['logging_level']), "processing.py")
    processor = nw_processor(config=config)

    #a=dataset[1]

    processor.run_all_queries(delete_incomplete=True, delete_all=True)





