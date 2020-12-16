# %% Imports

import glob
import logging
import os
import time
import configparser
import json
from src.classes.neo4_preprocessor import neo4j_preprocessor
from src.classes.bert_trainer import bert_trainer
from src.utils.hash_file import hash_string, check_step, complete_step
from src.utils.load_bert import get_bert_and_tokenizer
from src.classes.neo4j_processor import neo4j_processor
from src.classes.neo4jnw import neo4j_network
from src.classes.neo4db import neo4j_database
from src.classes.neo4jnw_aggregator import neo4jnw_aggregator

# Load Configuration file
config = configparser.ConfigParser()
config.read('D:/NLP/HBR/config/config.ini')
logging_level = config['General'].getint('logging_level')

neo_creds = (config['NeoConfig']["db_uri"], (config['NeoConfig']["db_db"], config['NeoConfig']["db_pwd"]))

# Set up logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging_level)

rootLogger = logging.getLogger()
logFormatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s -   %(message)s')
fileHandler = logging.FileHandler("{0}/{1}.log".format(config['Paths']['log'], "preprocessing"))
fileHandler.setFormatter(logFormatter)
rootLogger.addHandler(fileHandler)
#consoleHandler = logging.StreamHandler()
#consoleHandler.setFormatter(logFormatter)
#rootLogger.addHandler(consoleHandler)


# Set up preprocessor
preprocessor = neo4j_preprocessor(config['Paths']['database'], config['Preprocessing'].getint('max_seq_length'),
                                  config['Preprocessing'].getint('char_mult'), config['Preprocessing']['split_symbol'],
                                  config['Preprocessing'].getint('number_params'), logging_level=logging_level)

# Preprocess file
#preprocessor.preprocess_folders(config['Paths']['import_folder'],overwrite=True,excludelist=['checked', 'Error'])

trainer=bert_trainer(config['Paths']['database'],config['Paths']['pretrained_bert'], config['Paths']['trained_berts'],config['BertTraining'],json.loads(config.get('General','split_hierarchy')),logging_level=logging_level)
trainer.train_berts()


#test = neo4j_database(neo_creds)
#neograph = neo4j_network(neo_creds)
#nagg = neo4jnw_aggregator(neograph)

#processor = neo4j_processor(config['Paths']['trained_berts'], neograph,
#                            config['Preprocessing'].getint('max_seq_length'), config['Processing'],
#                            text_db=config['Paths']['database'],
#                            split_hierarchy=json.loads(config.get('General', 'split_hierarchy')),
#                            processing_cache=config['Paths']['processing_cache'],
#                            logging_level=config['General'].getint('logging_level'))
#processor.run_all_queries(clean_database=True)

#neograph.condition(years=None, tokens=None, weight_cutoff=None, depth=None, context=None)
#neograph.export_gefx("E:/NLPInspeech/test4norpune.gexf")

