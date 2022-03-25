# %% Imports
import tables
import argparse
import configparser
import logging
import json
import os
import nltk

# Import components from our package
from text2network.preprocessing.nw_preprocessor import nw_preprocessor
from text2network.utils.file_helpers import check_create_folder, check_folder
from text2network.utils.logging_helpers import setup_logger


# This needs to be done once
# nltk.download('all')

configuration_path = "config/config.ini"

# Load Configuration file
config = configparser.ConfigParser()

config.read(check_create_folder(configuration_path))
# Setup logging
logger = setup_logger(config['Paths']['log'], int(
    config['General']['logging_level']), "preprocessing.py")

# Initialize network preprocessor class
preprocessor = nw_preprocessor(config)
# Optional list of files to exclude
exclude_list = json.loads(config.get('Preprocessing', 'exclude_list'))

# Option 1: Without subfolders - year is the first parameter
preprocessor.preprocess_files(config['Paths']['import_folder'], overwrite=bool(
    config['Preprocessing']['overwrite_text_db']), excludelist=exclude_list)

preprocessor.preprocess_files()


# Option 2: WITH subfolders given by year
#preprocessor.preprocess_folders(config['Paths']['import_folder'], overwrite=bool(
#    config['Preprocessing']['overwrite_text_db']), excludelist=exclude_list)


