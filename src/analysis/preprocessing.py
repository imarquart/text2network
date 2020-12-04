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



# Load Configuration file
config = configparser.ConfigParser()
config.read('D:/NLP/InSpeech/BERTNLP/config/config.ini')
logging_level=config['General'].getint('logging_level')


# Set up logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging_level)

# Set up preprocessor
#preprocessor=neo4j_preprocessor(config['Paths']['database'], config['Preprocessing'].getint('max_seq_length'),config['Preprocessing'].getint('char_mult'),config['Preprocessing']['split_symbol'],config['Preprocessing'].getint('number_params'), logging_level=logging_level)

# Preprocess file
#preprocessor.preprocess_files(config['Paths']['import_folder'])


#trainer=bert_trainer(config['Paths']['database'],config['Paths']['pretrained_bert'], config['Paths']['trained_berts'],config['BertTraining'],json.loads(config.get('General','split_hierarchy')),logging_level=logging_level)
#trainer.train_berts()

from src.classes.neo4jnw import neo4j_network
from src.classes.neo4db import neo4j_database
from src.classes.neo4jnw_aggregator import neo4jnw_aggregator
db_uri = "http://localhost:7474"
db_pwd = ('neo4j', 'nlp')
neo_creds = (db_uri, db_pwd)

test=neo4j_database(neo_creds)
neograph = neo4j_network(neo_creds)
nagg=neo4jnw_aggregator(neograph)

#processor=neo4j_processor(config['Paths']['trained_berts'], neograph, config['Preprocessing'].getint('max_seq_length'), config['Processing'],text_db=config['Paths']['database'], split_hierarchy=json.loads(config.get('General','split_hierarchy')),logging_level=config['General'].getint('logging_level'))
#processor.run_all_queries()


neograph.condition(years=None, tokens=None, weight_cutoff=None, depth=None,  context=None)
neograph.export_gefx("E:/NLPInspeech/test.gefx")