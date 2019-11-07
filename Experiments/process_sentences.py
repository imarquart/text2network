import torch
import numpy as np
import tables
from NLP.Experiments.text_dataset import text_dataset
from NLP.Experiments.text_dataset_simple import text_dataset_simple
from NLP.Experiments.text_dataset import text_dataset_collate
from NLP.Experiments.get_bert_tensor import get_bert_tensor
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import BatchSampler, SequentialSampler


def process_sentences(tokenizer, bert, text_file, filepath, MAX_SEQ_LENGTH, DICT_SIZE, batch_size):
    """
    Extracts probability distributions from texts and saves them in pyTables database
    in three formats:

    Sequence Data: Tensor for each sequence including distribution for each word.

    Token-Sequence Data: One entry for each token in each sequence, including all distributions of
    the sequence (duplication). Allows indexing by token.

    Token Data: One entry for each token in each sequence, but only includes fields for
    distribution of focal token, and and aggregation (average) of all contextual tokens.

    Parameters
        tokenizer : BERT tokenizer (pyTorch)

        bert : BERT model

        texts : List of sentences (n sequences of length k_i, where k_i<= MAX_SEQ_LENGTH)

        seq_ids : List of IDs to pass for each sequence

        filepath : Path to HDF5 PyTables file

        MAX_SEQ_LENGTH : maximal length of sequences

        batch_size : Batch size to use for sending texts to BERT (e.g. via GPU)

    Returns

    """

    class Seq_Particle(tables.IsDescription):
        seq_id = tables.UInt32Col()
        token_ids = tables.UInt32Col(shape=[1, MAX_SEQ_LENGTH])
        token_dist = tables.Float32Col(shape=(MAX_SEQ_LENGTH, DICT_SIZE))
        seq_size = tables.UInt32Col()

    class Token_Seq_Particle(tables.IsDescription):
        token_id = tables.UInt32Col()
        seq_id = tables.UInt32Col()
        seq_size = tables.UInt32Col()
        pos_id = tables.UInt32Col()
        token_ids = tables.UInt32Col(shape=[1, MAX_SEQ_LENGTH])
        token_dist = tables.Float32Col(shape=(MAX_SEQ_LENGTH, DICT_SIZE))

    class Token_Particle(tables.IsDescription):
        token_id = tables.UInt32Col()
        seq_id = tables.UInt32Col()
        pos_id = tables.UInt32Col()
        seq_size = tables.UInt32Col()
        own_dist = tables.Float32Col(shape=(1, DICT_SIZE))
        context_dist = tables.Float32Col(shape=(1, DICT_SIZE))

    try:
        data_file = tables.open_file(filepath, mode="a", title="Data File")
    except:
        data_file = tables.open_file(filepath, mode="w", title="Data File")

    try:
        seq_table = data_file.root.seq_data.table
    except:
        group = data_file.create_group("/", 'seq_data', 'Sequence Data')
        seq_table = data_file.create_table(group, 'table', Seq_Particle, "Sequence Table")

    try:
        token_seq_table = data_file.root.token_seq_data.table
    except:
        group = data_file.create_group("/", 'token_seq_data', 'Token Sequence Data')
        token_seq_table = data_file.create_table(group, 'table', Token_Seq_Particle, "Token Sequence Table")

    try:
        token_table = data_file.root.token_data.table
    except:
        group = data_file.create_group("/", 'token_data', 'Token Data')
        token_table = data_file.create_table(group, 'table', Token_Particle, "Token Table")

    # TODO: Batching
    # TODO: Parallelize
    # for batch etc

    batch_size = 20

    # %% Initialize text dataset
    dataset = text_dataset(text_file, tokenizer, MAX_SEQ_LENGTH)
    batch_sampler = BatchSampler(SequentialSampler(range(0, dataset.nitems)), batch_size=batch_size, drop_last=False)
    dataloader=DataLoader(dataset=dataset,batch_size=None, sampler=batch_sampler,num_workers=16)
    for batch in dataloader:
        #predictions = get_bert_tensor(bert, batch[0], tokenizer.pad_token_id, tokenizer.mask_token_id, return_max=True)
        # print(tokenizer.convert_ids_to_tokens(predictions.numpy()))
        print((batch[0].shape))

    dataset.close()
    data_file.flush()
    data_file.close()
    return(batch)
