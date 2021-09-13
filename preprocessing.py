# %% Imports
import argparse
import configparser
import logging
import json
#from text2network.training.bert_trainer import bert_trainer
import os

from text2network.preprocessing.nw_preprocessor import nw_preprocessor
from text2network.functions.file_helpers import check_create_folder
from text2network.utils.logging_helpers import setup_logger

##################### Training
#trainer=bert_trainer(config)
#trainer.train_berts()

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

    preprocessor = nw_preprocessor(config)
    exclude_list = json.loads(config.get('Preprocessing', 'exclude_list'))
    from glob import glob
    paths = glob(os.getcwd()+os.path.normpath(config['Paths']['import_folder'])+"/*/")
    if paths == []:
        logging.info("Preprocessor in file mode: No subfolders found in {}".format(config['Paths']['import_folder']))
        preprocessor.preprocess_files(config['Paths']['import_folder'], overwrite=bool(config['Preprocessing']['overwrite_text_db']), excludelist=exclude_list)
    else:
        logging.info("Preprocessing subfolders in {}".format(config['Paths']['import_folder']))
        preprocessor.preprocess_folders(config['Paths']['import_folder'], overwrite=bool(config['Preprocessing']['overwrite_text_db']), excludelist=exclude_list)

    logging.info("Preprocessing of folder {} complete!".format(config['Paths']['import_folder']))



