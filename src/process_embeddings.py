# TODO: Redo Comments

import torch
import numpy as np
import tables
import time
import logging
import networkx as nx
from NLP.utils.rowvec_tools import simple_norm, get_weighted_edgelist, calculate_cutoffs, add_to_networks
from NLP.src.datasets.text_dataset import text_dataset_subset, text_dataset_collate_batchsample
from NLP.src.datasets.dataloaderX import DataLoaderX
from NLP.src.text_processing.get_bert_embeddings import get_bert_embeddings
from torch.utils.data import BatchSampler, SequentialSampler
from NLP.utils.delwords import create_stopword_list
from NLP.utils.load_bert import get_bert_and_tokenizer
import tqdm


def process_embeddings(bert_folder, text_db, interest_set, MAX_SEQ_LENGTH, batch_size):
    """
    Extracts pre-processed sentences, gets predictions by BERT and creates a network

    Network is created for both context distribution and for replacement distribution
    Each sentence is added via parallel ties in a multigraph

    :param tokenizer: BERT tokenizer (pyTorch)
    :param bert: BERT model
    :param text_db: HDF5 File of processes sentences, string of tokens, ending with punctuation
    :param MAX_SEQ_LENGTH:  maximal length of sequences
    :param DICT_SIZE: tokenizer dict size
    :param batch_size: batch size to send to BERT
    :param nr_workers: Nr workers for dataloader. Probably should be set to 0 on windows
    :param method: "attention": Weigh by BERT attention; "context_element": Sum probabilities unweighted
    :param cutoff_percent: Amount of probability mass to use to create links. Smaller values, less ties.
    :return: graph, context_graph (networkx DiMultiGraphs)
    """
    tables.set_blosc_max_threads(15)

    tokenizer, bert = get_bert_and_tokenizer(bert_folder, True, True, True)

    # %% Initialize text dataset
    dataset = text_dataset_subset(text_db, interest_set, tokenizer, MAX_SEQ_LENGTH)
    logging.info("Number of sentences found: %i"%dataset.nitems)
    batch_sampler = BatchSampler(SequentialSampler(range(0, dataset.nitems)), batch_size=batch_size, drop_last=False)
    dataloader = DataLoaderX(dataset=dataset, batch_size=None, sampler=batch_sampler, num_workers=0,
                             collate_fn=text_dataset_collate_batchsample, pin_memory=False)

    # Push BERT to GPU
    torch.cuda.empty_cache()
    if torch.cuda.is_available(): logging.info("Using CUDA.")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    bert.to(device)
    bert.eval()

    # Get token id's for interest set
    interest_tokens= tokenizer.convert_tokens_to_ids(interest_set)
    DICT_SIZE = tokenizer.vocab_size

    # Create Stopwords
    delwords = create_stopword_list(tokenizer)

    # Create dict to hold data
    pickle_list=[]

    # Counter for timing
    model_timings = []
    process_timings = []
    load_timings = []
    start_time = time.time()
    for batch, seq_ids, token_ids in tqdm.tqdm(dataloader, desc="Iteration"):

        # Data spent on loading batch
        load_time = time.time() - start_time
        load_timings.append(load_time)
        # This seems to allow slightly higher batch sizes on my GPU
        # torch.cuda.empty_cache()
        # Run BERT and get predictions
        predictions, attn, embeddings = get_bert_embeddings(0, bert, batch, pad_token_id=tokenizer.pad_token_id, mask_token_id=tokenizer.mask_token_id, device=device)

        # compute model timings time
        prepare_time = time.time() - start_time - load_time
        model_timings.append(prepare_time)

        # %% Sequence Table
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


            #%% Extract attention for sequence
            seq_attn = attn[sequence_mask, :].cpu()
            seq_embeddings = embeddings[sequence_mask,:].cpu()

            # Curtail to tokens in sequence
            # attention row vectors for each token are of
            # size sequence_size+2, where position 0 is <CLS>
            # and position n+1 is <SEP>, these we ignore

            # DISABLED ATTENTION
            #seq_attn = seq_attn[:, 1:sequence_size + 1]

            # Delete diagonal attention
            # DISABLED ATTENTION
            #seq_attn[torch.eye(sequence_size).bool()] = 0

            #%% Context element distribution: we sum over all probabilities in a sequence
            seq_ce = torch.ones([sequence_size, sequence_size])
            seq_ce[torch.eye(sequence_size).bool()] = 0

            # TODO: Re-Enable Lines of attention
            for pos, token in enumerate(token_ids[sequence_mask]):
                # Should all be np
                token=token.item()
                if token in interest_tokens:
                    replacement = dists[pos, :]
                    token_attention = seq_attn[pos]
                    token_embeddings = seq_embeddings[pos,:]
                    seq_tokens=tokenizer.convert_ids_to_tokens(token_ids[sequence_mask].numpy())

                    pickle_list.append({"Token": token, "Position": pos, "Sequence": sequence_id, "Tokens": seq_tokens,
                                        "Replacement": replacement, "Attention": token_attention, "Embeddings": token_embeddings})

                    ## Attention
                    # DISABLED ATTENTION
                    #context_att = (
                    #    torch.sum((seq_attn[pos] * context_dist.transpose(-1, 0)).transpose(-1, 0), dim=0).unsqueeze(0))

            del dists

        del predictions, attn

        # compute processing time
        process_timings.append(time.time() - start_time - prepare_time - load_time)
        # New start time
        start_time = time.time()

    dataset.close()

    logging.info("Average Load Time: %s seconds" % (np.mean(load_timings)))
    logging.info("Average Model Time: %s seconds" % (np.mean(model_timings)))
    logging.info("Average Processing Time: %s seconds" % (np.mean(process_timings)))
    logging.info("Ratio Load/Operations: %s seconds" % (np.mean(load_timings) / np.mean(process_timings + model_timings)))

    return pickle_list
