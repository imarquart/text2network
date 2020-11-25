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

import tables
import pandas as pd
import torch
import numbers
from torch.nn.utils.rnn import pad_sequence

from src.utils.get_uniques import get_uniques
split_hierarchy=json.loads(config.get('General','split_hierarchy'))
data_path=config['Paths']['database']
pretrained_folder=config['Paths']['pretrained_bert']
tokenizer, model = get_bert_and_tokenizer(pretrained_folder, True)
uniques=get_uniques(split_hierarchy,data_path)
queries=uniques['query']
query=queries[1]
db_conn = tables.open_file(data_path, mode="r")
data = db_conn.root.textdata.table

# Get data
items_text = data.read_where(query)['text']
items_year = data.read_where(query)['year']
items_seqid = data.read_where(query)['seq_id']  # ?
items_p1 = data.read_where(query)['p1']
items_p2 = data.read_where(query)['p2']
items_p3 = data.read_where(query)['p3']
items_p4 = data.read_where(query)['p4']

# Because of pyTables, we have to encode.
items_text = [x.decode("utf-8") for x in items_text]
items_p1 = [x.decode("utf-8") for x in items_p1]
items_p2 = [x.decode("utf-8") for x in items_p2]
items_p3 = [x.decode("utf-8") for x in items_p3]
items_p4 = [x.decode("utf-8") for x in items_p4]

data = pd.DataFrame({"year":items_year, "seq_id":items_seqid, "text": items_text, "p1":items_p1, "p2":items_p2, "p3":items_p3, "p4":items_p4})

fixed_seq_length=30
index=[10,30]


if torch.is_tensor(index):
    index = index.to_list(index)

if isinstance(index, numbers.Integral):
    index = [int(index)]

if type(index) is not list:
    index = [index]

item=data.iloc[index,:]

# Get numpy or torch vectors (numpy for the strings)
texts = item['text'].to_numpy()
year_vec= torch.tensor(item['year'].to_numpy(dtype="int32"), requires_grad=False)
p1_vec=item['p1'].to_numpy()
p2_vec=item['p2'].to_numpy()
p3_vec=item['p3'].to_numpy()
p4_vec=item['p4'].to_numpy()
seq_vec=item['seq_id'].to_numpy(dtype="int32")

token_input_vec = [] # tensor of padded inputs with special tokens
token_id_vec = [] # List of token ids for each sequence
for text in texts:
    indexed_tokens = tokenizer.encode(text, add_special_tokens=False)
    # Need fixed size, so we need to cut and pad
    indexed_tokens = indexed_tokens[:fixed_seq_length]
    indexed_tokens = tokenizer.build_inputs_with_special_tokens(indexed_tokens)
    inputs = torch.tensor([indexed_tokens], requires_grad=False)
    inputs = inputs.squeeze(dim=0).squeeze()
    token_input_vec.append(inputs)
    # Also save list of tokens
    token_id_vec.append(inputs[1:-1])

# Add padding sequence
token_input_vec.append(torch.zeros([fixed_seq_length + 2], dtype=torch.int))
token_input_vec = pad_sequence(token_input_vec, batch_first=True, padding_value=tokenizer.pad_token_id)
# Detele padding sequence
token_input_vec = token_input_vec[:-1, :]

# Prepare index vector and sequence vector by using sequence IDs found in database
lengths = ([x.shape[0] for x in token_id_vec])
index_vec = torch.repeat_interleave(torch.as_tensor(index), torch.as_tensor(lengths))
seq_vec = torch.repeat_interleave(torch.as_tensor(seq_vec), torch.as_tensor(lengths))

# Cat token_id_vec list into a single tensor for the whole batch
token_id_vec = torch.cat([x for x in token_id_vec])