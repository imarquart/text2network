# %% Imports
import argparse
import configparser
import logging
import json
import os

from text2network.preprocessing.nw_preprocessor import nw_preprocessor
from text2network.utils.file_helpers import check_create_folder, check_folder
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
    print("Loading config in {}".format(configuration_path))
    # Load Configuration file
    config = configparser.ConfigParser()
    try:
        config.read(check_create_folder(configuration_path))
    except:
        logging.error("Could not read config.")
        raise
    # Setup logging
    logger = setup_logger(config['Paths']['log'], int(config['General']['logging_level']), "preprocessing.py")

    preprocessor = nw_preprocessor(config)
    exclude_list = json.loads(config.get('Preprocessing', 'exclude_list'))

    globpath="".join([check_folder(config['Paths']['import_folder']), os.sep, '*', os.sep])
    from glob import glob
    logging.info("Checking files and folders in {}".format(globpath))
    paths = glob(globpath)
    if paths == []:
        logging.info("Preprocessor in file mode: No subfolders found in {}".format(config['Paths']['import_folder']))
        preprocessor.preprocess_files(config['Paths']['import_folder'], overwrite=bool(config['Preprocessing']['overwrite_text_db']), excludelist=exclude_list)
    else:
        logging.info("Preprocessing subfolders in {}".format(config['Paths']['import_folder']))
        preprocessor.preprocess_folders(config['Paths']['import_folder'], overwrite=bool(config['Preprocessing']['overwrite_text_db']), excludelist=exclude_list)

    logging.info("Preprocessing of folder {} complete!".format(config['Paths']['import_folder']))

    paths = check_create_folder(config['Paths']['database'], create_folder=False)
    logging.info("Confirmation, printing the first sentences in {}".format(paths))

    import tables

    table = tables.open_file(paths, mode="r")
    data = table.root.textdata.table
    for i, row in enumerate(data):
        logging.info("Seq {}, Run_id: {}, Year: {}, p1: {}".format(i, row['run_index'], row['year'], row['p1']))
        logging.info("Txt: {}".format(row['text']))
        if i > 5:
            break
    table.close()




