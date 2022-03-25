# %% Imports
import argparse
import configparser
import logging
import nltk

from text2network.processing.nw_processor import nw_processor
from text2network.utils.file_helpers import check_create_folder
from text2network.utils.logging_helpers import setup_logger


configuration_path = "config/config.ini"

print("Loading config in {}".format(check_create_folder(configuration_path)))
# Load Configuration file
config = configparser.ConfigParser()
config.read(check_create_folder(configuration_path))
# Setup logging
logger = setup_logger(config['Paths']['log'], int(config['General']['logging_level']), "processing.py")


# Create Processor Instance
processor = nw_processor(config=config)

# Run all queries, DETELE DATABASE!
processor.run_all_queries(delete_incomplete_times=True, delete_all=False)





