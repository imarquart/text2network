# TODO: Redo Comments

import torch
import numpy as np
import tables
from NLP.Experiments.text_dataset import text_dataset
from NLP.Experiments.text_dataset import text_dataset_collate_batchsample
from NLP.Experiments.get_bert_tensor import get_bert_tensor
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import BatchSampler, SequentialSampler
from scipy import sparse
import tqdm

def process_sentences_sparse(tokenizer, bert, text_db, tensor_db, MAX_SEQ_LENGTH, DICT_SIZE, batch_size):
    """
    Extracts probability distributions from texts and saves them in pyTables database
    in three formats:

    Sequence Data: Tensor for each sequence including distribution for each word.

    Token-Sequence Data: One entry for each token in each sequence, including all distributions of
    the sequence (duplication). Allows indexing by token.

    Token Data: One entry for each token in each sequence, but only includes fields for
    distribution of focal token, and and aggregation (average weighted by attention) of all contextual tokens.

    :param tokenizer: BERT tokenizer (pyTorch)
    :param bert: BERT model
    :param text_db: HDF5 File of processes sentences, string of tokens, ending with punctuation
    :param tensor_db: HDF5 File to save processed tensors
    :param MAX_SEQ_LENGTH:  maximal length of sequences
    :param DICT_SIZE: tokenizer dict size
    :param batch_size: batch size to send to BERT
    :return: None
    """

    filters = tables.Filters(complevel=9, complib='blosc', fletcher32=False)

    class Token_Particle(tables.IsDescription):
        token_id = tables.UInt32Col()
        peer_id = tables.UInt32Col()
        value = tables.Float16Col()
        seq_id = tables.UInt32Col()
        pos_id = tables.UInt32Col()

    try:
        data_file = tables.open_file(tensor_db, mode="a", title="Data File")
    except:
        data_file = tables.open_file(tensor_db, mode="w", title="Data File", filters=filters)

    try:
        own_table = data_file.root.token_data.table
    except:
        group = data_file.create_group("/", 'token_data', 'Token Data')
        own_table = data_file.create_table(group, 'owndist', Token_Particle, "Own Distributions", filters=filters)
        own_table.cols.token_id.create_index()
        own_table.cols.peer_id.create_index()

    try:
        con_table = data_file.root.token_data.table
    except:
        group = data_file.create_group("/", 'token_data', 'Token Data')
        con_table = data_file.create_table(group, 'contdist', Token_Particle, "Context distributions", filters=filters)
        con_table.cols.token_id.create_csindex()
        con_table.cols.peer_id.create_csindex()


    # Push BERT to GPU
    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bert.to(device)
    bert.eval()

    # %% Initialize text dataset
    dataset = text_dataset(text_db, tokenizer, MAX_SEQ_LENGTH)
    batch_sampler = BatchSampler(SequentialSampler(range(0, dataset.nitems)), batch_size=batch_size, drop_last=False)
    dataloader=DataLoader(dataset=dataset,batch_size=None, sampler=batch_sampler,num_workers=16,collate_fn=text_dataset_collate_batchsample, pin_memory=False)

    for batch, seq_ids, token_ids in tqdm.tqdm(dataloader, desc="Iteration"):
        # This seems to allow slightly higher batch sizes on my GPU
        #torch.cuda.empty_cache()
        # Run BERT and get predictions
        predictions, attn = get_bert_tensor(0, bert, batch, tokenizer.pad_token_id, tokenizer.mask_token_id, device,return_max=False)
        # %% Sequence Table
        # TODO: Needs to use sparse matrix bc. file-size too large otherwise
        # Iterate over sequences
        for sequence_id in np.unique(seq_ids):

            sequence_mask = seq_ids == sequence_id
            sequence_size = sum(sequence_mask)

            # Pad and add sequence of IDs
            idx = torch.zeros([1, MAX_SEQ_LENGTH], requires_grad=False, dtype=torch.int32)
            idx[0, :sequence_size] = token_ids[sequence_mask]

            # Pad and add distributions per token, we need to save to maximum sequence size
            dists = torch.zeros([MAX_SEQ_LENGTH, DICT_SIZE], requires_grad=False)
            dists[:sequence_size, :] = predictions[sequence_mask, :]

            # %% Token Table
            # Extract attention
            seq_attn=attn[sequence_mask,:].cpu()
            # Curtail to tokens in sequence
            seq_attn=seq_attn[:,1:sequence_size+1]
            # Delete diagonal attention
            seq_attn[torch.eye(sequence_size).bool()]=0
            # Normalize
            seq_attn=torch.div(seq_attn.transpose(-1, 0), torch.sum(seq_attn, dim=1)).transpose(-1, 0)

            for pos, token in enumerate(token_ids[sequence_mask]):
                # Particles
                own_p = own_table.row
                con_p = con_table.row

                # Base Entries
                own_p['token_id'] = token
                own_p['pos_id'] = pos
                own_p['seq_id'] = sequence_id
                con_p['token_id'] = token
                con_p['pos_id'] = pos
                con_p['seq_id'] = sequence_id


                ## Context distribution with attention weights
                context_index = np.zeros([MAX_SEQ_LENGTH], dtype=np.bool)
                context_index[:sequence_size] = True
                context_dist = dists[context_index, :]
                context_dist =(torch.sum((seq_attn[pos] * context_dist.transpose(-1, 0)).transpose(-1, 0)
                           , dim=0).unsqueeze(0))

                # Simple average
                #context_index = np.arange(sequence_size) != pos
                #context_index = np.concatenate(
                #    [context_index, np.zeros([MAX_SEQ_LENGTH - sequence_size], dtype=np.bool)])
                # particle['context_dist'] = (torch.sum(context_dist, dim=0).unsqueeze(0)) / sequence_size
                particle.append()






    data_file.flush()


    dataset.close()
    data_file.close()
