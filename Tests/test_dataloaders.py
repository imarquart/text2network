import torch
import numpy as np
import tables
from Experiments.text_dataset import text_dataset
from Experiments.text_dataset_simple import text_dataset_simple
from Experiments.text_dataset import text_dataset_collate_batchsample
from Experiments.get_bert_tensor import get_bert_tensor
from torch.utils.data import Dataset, DataLoader
from pytorch_memlab import MemReporter
from pytorch_memlab import profile

from torch.utils.data import BatchSampler
from torch.utils.data import SequentialSampler

database = '/home/ingo/PhD/BERT-NLP/data/texts.h5'
tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer',
                           'bert-base-uncased')  # Download vocabulary from S3 and cache.
bert = torch.hub.load('huggingface/pytorch-transformers', 'modelWithLMHead', 'bert-base-uncased')
text_file = '/home/ingo/PhD/BERT-NLP/data/texts.h5'
MAX_SEQ_LENGTH = 40

batches = [1, 2, 5, 25, 50, 100, 200, 500,1000,10000]

dataset = text_dataset_simple(text_file, tokenizer, MAX_SEQ_LENGTH)
import time

for batch_size in batches:
    print("#######################################")
    print("Batch size is %i" % batch_size)
    print("#######################################")

    start_time = time.time()
    bt_sampler = BatchSampler(SequentialSampler(range(dataset.nitems)), batch_size=batch_size, drop_last=True)
    bt_shapes = []
    sdbt_shapes = []
    dataloader = DataLoader(dataset, batch_sampler=bt_sampler, num_workers=16, pin_memory=False)
    for i, batch in enumerate(dataloader):
        batch = batch.squeeze(1)
        send_batch = batch[0].unsqueeze(0)
        bt_shapes.append(batch.shape)
        sdbt_shapes.append(send_batch.shape)
        # predictions = get_bert_tensor(bert, send_batch, tokenizer.pad_token_id, tokenizer.mask_token_id, return_max=False)
    dataset.close()
    print("DataLoader Batch_Sampler bt_sampler:")
    print("-------------------------------------")
    print("--- %s seconds ---" % (time.time() - start_time))
    print("Nr of iterations %i" % i)
    print("Batch shape {0}".format(bt_shapes[0]))
    print("send_batch shape {0}".format(sdbt_shapes[0]))
    print("-------------------------------------")

    start_time = time.time()
    bt_sampler = BatchSampler(SequentialSampler(range(dataset.nitems)), batch_size=batch_size, drop_last=True)
    dataloader = DataLoader(dataset, sampler=bt_sampler, batch_size=None, num_workers=16, pin_memory=False)
    bt_shapes = []
    sdbt_shapes = []
    for i, batch in enumerate(dataloader):
        send_batch = batch[0].unsqueeze(0)
        bt_shapes.append(batch.shape)
        sdbt_shapes.append(send_batch.shape)
    # predictions = get_bert_tensor(bert, send_batch, tokenizer.pad_token_id, tokenizer.mask_token_id, return_max=False)
    dataset.close()
    print("DataLoader Sampler bt_sampler:")
    print("-------------------------------------")
    print("--- %s seconds ---" % (time.time() - start_time))
    print("Nr of iterations %i" % i)
    print("Batch shape {0}".format(bt_shapes[0]))
    print("send_batch shape {0}".format(sdbt_shapes[0]))
    print("-------------------------------------")

    start_time = time.time()
    dataloader = DataLoader(dataset, shuffle=False, num_workers=16, batch_size=batch_size, pin_memory=False,drop_last=True)
    bt_shapes = []
    sdbt_shapes = []
    for i, batch in enumerate(dataloader):
        batch = batch.squeeze(1)
        send_batch = batch[0].unsqueeze(0)
        bt_shapes.append(batch.shape)
        sdbt_shapes.append(send_batch.shape)
    # predictions = get_bert_tensor(bert, send_batch, tokenizer.pad_token_id, tokenizer.mask_token_id, return_max=False)
    dataset.close()
    print("DataLoader Random Sampler:")
    print("-------------------------------------")
    print("--- %s seconds ---" % (time.time() - start_time))
    print("Nr of iterations %i" % i)
    print("Batch shape {0}".format(bt_shapes[0]))
    print("send_batch shape {0}".format(sdbt_shapes[0]))
    print("-------------------------------------")

    start_time = time.time()
    bt_shapes = []
    sdbt_shapes = []
    for idx, i in enumerate(range(0, dataset.nitems - batch_size, batch_size)):
        batch = dataset[i:i+batch_size]
        send_batch=batch[0]
        bt_shapes.append(batch.shape)
        sdbt_shapes.append(send_batch.shape)
        # predictions = get_bert_tensor(bert, batch, tokenizer.pad_token_id, tokenizer.mask_token_id, return_max=False)
    dataset.close()
    print("For Loop:")
    print("-------------------------------------")
    print("--- %s seconds ---" % (time.time() - start_time))
    print("Nr of iterations %i" % idx)
    print("Batch shape {0}".format(bt_shapes[0]))
    print("send_batch shape {0}".format(sdbt_shapes[0]))
    print("-------------------------------------")
