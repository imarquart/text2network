# %% Imports

# Import of namespaces important packages
import os
import argparse
import configparser
import logging
import sys
import traceback
import torch
import nltk
import gc
# Import the files we need from out package
# The trainer class
from text2network.training.bert_trainer import bert_trainer
# A tool to help with folders
from text2network.utils.file_helpers import check_create_folder
# Our logging tool
from text2network.utils.logging_helpers import setup_logger

# We will use nltk at times, so we just download all options (this needs to be done once)
# nltk.download('all')

# Set the configuration path
configuration_path = check_create_folder(
    "config/config.ini")  # Here: relative from main folder
# Load Configuration file
config = configparser.ConfigParser()
print("Loading config in {}".format(configuration_path))
try:
    config.read(check_create_folder(configuration_path))
except:
    logging.error("Could not read config.")
# Setup logging
logger = setup_logger(config['Paths']['log'],
                      config['General']['logging_level'], "training.py")

# We are now good to go!
# Create a trainer instance
trainer = bert_trainer(config)
# And run the trainer, given our configuration settings
result = trainer.train_berts()
