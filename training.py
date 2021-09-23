# %% Imports
import argparse
import configparser
import logging
import json
import sys
import traceback
import torch
from text2network.training.bert_trainer import bert_trainer
import os
import gc
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
    print("Loading config in {}".format(configuration_path))
    try:
        config.read(check_create_folder(configuration_path))
    except:
        logging.error("Could not read config.")
    # Setup logging
    logger = setup_logger(config['Paths']['log'], config['General']['logging_level'], "training.py")

    ##################### Training
    trainer=bert_trainer(config)
    success=False
    while success is not True:
        try:
            sucvar=trainer.train_berts()
        except:
            etype, value, _ = sys.exc_info()
            logging.error("Error in train_berts(): {}".format(value))
            logging.error("Traceback: {}".format(traceback.format_exc()))

            # Here we can try to release CUDA memory

            del bert_trainer
            gc.collect()
            print(torch.cuda.is_available())
            torch.cuda.empty_cache()

            trainer=bert_trainer(config)



            logging.info("trying to continue")
            continue
        if sucvar == 0:
            logging.info("Successfully trained BERTs, canceling")
            success=True

