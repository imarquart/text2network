# %% Imports
import argparse
import configparser
import logging

from text2network.utils.file_helpers import check_create_folder
from text2network.utils.logging_helpers import setup_logger


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
    logging.info("Preprocessing of folder {} complete!".format(config['Paths']['import_folder']))
    paths = check_create_folder(config['Paths']['database'], create_folder=False)
    logging.info("Confirmation, printing the first sentences in {}".format(paths))

    import tables
    table = tables.open_file(paths, mode="r")
    data = table.root.textdata.table
    for i,row in enumerate(data):
        logging.info("Seq {}, Run_id: {}, Year: {}, p1: {}".format(i, row['run_index'], row['year'], row['p1']))
        logging.info("Txt: {}".format(row['text']))
        if i>5:
            break
    table.close()

    # items = data.read()[:]




