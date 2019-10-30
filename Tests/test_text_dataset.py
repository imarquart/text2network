import torch
import numpy as np
import tables
from Experiments.text_dataset import text_dataset
from Experiments.text_dataset_simple import text_dataset_simple
from Experiments.text_dataset import text_dataset_collate
from Experiments.get_bert_tensor import get_bert_tensor
from torch.utils.data import Dataset, DataLoader

database='/home/ingo/PhD/BERT-NLP/data/texts.h5'
tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer','bert-base-uncased')  # Download vocabulary from S3 and cache.
bert = torch.hub.load('huggingface/pytorch-transformers', 'modelWithLMHead', 'bert-base-uncased')
text_file='/home/ingo/PhD/BERT-NLP/data/texts.h5'
MAX_SEQ_LENGTH = 40

batch_size = 500

# %% Initialize simple text dataset
dataset = text_dataset_simple(text_file, tokenizer, MAX_SEQ_LENGTH)

import time

start_time = time.time()
for i in range(0, dataset.nitems - batch_size, batch_size):
        batch = dataset[i, i + batch_size]
        predictions = get_bert_tensor(bert, batch, tokenizer.pad_token_id, tokenizer.mask_token_id, return_max=True)
        # print(tokenizer.convert_ids_to_tokens(predictions.numpy()))

dataset.close()
print("--- %s seconds ---" % (time.time() - start_time))

# Initialize simple text dataset
dataset = text_dataset_simple(text_file, tokenizer, MAX_SEQ_LENGTH)

import time

start_time = time.time()
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=False)

for i, batch in enumerate(dataloader):
        predictions = get_bert_tensor(bert, batch[0], tokenizer.pad_token_id, tokenizer.mask_token_id, return_max=True)
        # print(tokenizer.convert_ids_to_tokens(predictions.numpy()))

dataset.close()
print("--- %s seconds ---" % (time.time() - start_time))

# %% Initialize text dataset
dataset = text_dataset(text_file, tokenizer, MAX_SEQ_LENGTH)

import time

start_time = time.time()
for i in range(0, dataset.nitems - batch_size, batch_size):
        batch = dataset[i, i + batch_size]
        predictions = get_bert_tensor(bert, batch[0], tokenizer.pad_token_id, tokenizer.mask_token_id, return_max=True)
        # print(tokenizer.convert_ids_to_tokens(predictions.numpy()))

dataset.close()
print("--- %s seconds ---" % (time.time() - start_time))

# %% Initialize text dataset
dataset = text_dataset(text_file, tokenizer, MAX_SEQ_LENGTH)

import time

start_time = time.time()
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=False,
                        collate_fn=text_dataset_collate)

for i, batch in enumerate(dataloader):
        predictions = get_bert_tensor(bert, batch[0], tokenizer.pad_token_id, tokenizer.mask_token_id, return_max=True)
        # print(tokenizer.convert_ids_to_tokens(predictions.numpy()))

dataset.close()
print("--- %s seconds ---" % (time.time() - start_time))