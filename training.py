# %% Imports
import argparse
import configparser
import logging
import sys
import traceback
import torch
from text2network.training.bert_trainer import bert_trainer
import gc
from text2network.utils.file_helpers import check_create_folder
from text2network.utils.logging_helpers import setup_logger

import nltk
nltk.download('all')

def run_training(args):
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
    return trainer.train_berts()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess text files.')
    parser.add_argument('--config', metavar='path', required=True,
                        help='the path to the configuration file')
    args = parser.parse_args()
    
    success=False
    while success is not True:
        try:
            sucvar=run_training(args)
        except:
            etype, value, _ = sys.exc_info()
            logging.error("Error in train_berts(): {}".format(value))
            logging.error("Traceback: {}".format(traceback.format_exc()))

            # Here we can try to release CUDA memory

            gc.collect()
            print(torch.cuda.is_available())
            torch.cuda.empty_cache()
            logging.info("trying to continue")
            continue
        if sucvar == 0:
            logging.info("Successfully trained BERTs, canceling")
            success=True


