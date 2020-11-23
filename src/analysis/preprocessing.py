# %% Imports

import glob
import logging
import os
import time
import configparser

from src.classes.neo4_preprocessor import neo4j_preprocessor
from src.classes.bert_trainer import bert_trainer
from src.utils.hash_file import hash_file, check_step, complete_step
from src.utils.load_bert import get_bert_and_tokenizer



# Load Configuration file
config = configparser.ConfigParser()
config.read('D:/NLP/InSpeech/BERTNLP/config/config.ini')

# Set up logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)

# Set up preprocessor
#preprocessor=neo4j_preprocessor(config['Paths']['database'], config['Preprocessing'].getint('max_seq_length'),config['Preprocessing'].getint('char_mult'),config['Preprocessing']['split_symbol'],config['Preprocessing'].getint('number_params'))

# Preprocess file
#preprocessor.preprocess_files(config['Paths']['import_folder'])


trainer=bert_trainer(config['Paths']['database'],config['Paths']['pretrained_bert'], config['Paths']['trained_berts'],config['BertTraining'])