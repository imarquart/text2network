# %% Imports
import argparse
import configparser
import logging

from text2network.processing.nw_processor import nw_processor
from text2network.utils.file_helpers import check_create_folder
from text2network.utils.logging_helpers import setup_logger

configuration_path = check_create_folder(
    "config/config.ini")

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

processor.run_all_queries(delete_incomplete_times=True, delete_all=False)





