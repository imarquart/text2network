from itertools import product

import pandas as pd
import tables
from text2network.functions.file_helpers import check_create_folder
from text2network.utils.load_bert import get_only_tokenizer
from text2network.utils.logging_helpers import setup_logger
from text2network.datasets.text_dataset import query_dataset, bert_dataset, bert_dataset_old

# Set a configuration path
configuration_path = 'config/2021/HBR40.ini'
# Settings
years = None
# Load Configuration file
import configparser

config = configparser.ConfigParser()
print(check_create_folder(configuration_path, False))
config.read(check_create_folder(configuration_path, False))
# Setup logging
setup_logger(config['Paths']['log'], config['General']['logging_level'], "dataset_debug.py")


path="data\\HBR40\\database\\db.h5"
bert="E:\\TrainedBerts\\HBR\St40\\2020"

path=check_create_folder(path, False)
bert=check_create_folder(bert, False)

tokenizer=get_only_tokenizer(bert)

#dataset=query_dataset(data_path=path, tokenizer=tokenizer, fixed_seq_length=40, maxn=None, query="year==2020")

b_dataset=bert_dataset(tokenizer=tokenizer,database=path,where_string="year==2020",block_size=40)
b_dataset_old=bert_dataset_old(tokenizer=tokenizer,database=path,where_string="year==2020",block_size=40)


for i in range (0,10):
    print(tokenizer.convert_ids_to_tokens(b_dataset[i].tolist()))
    print(tokenizer.convert_ids_to_tokens(b_dataset_old[i].tolist()))
    print("--------------------------------")