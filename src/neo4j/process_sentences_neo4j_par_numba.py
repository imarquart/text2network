# TODO: Redo Comments

import torch
import numpy as np
import tables
import time
import logging
from NLP.src.neo4j.neo4j_network import neo4j_network
import asyncio
from joblib import Parallel, delayed
import multiprocessing
from numba import jit

from NLP.utils.rowvec_tools import simple_norm
from NLP.src.neo4j.neo_row_tools import  get_weighted_edgelist, calculate_cutoffs
from NLP.src.datasets.text_dataset import text_dataset, text_dataset_collate_batchsample
from NLP.src.datasets.dataloaderX import DataLoaderX
from NLP.src.text_processing.get_bert_tensor import get_bert_tensor
from torch.utils.data import BatchSampler, SequentialSampler
from NLP.utils.delwords import create_stopword_list
import tqdm


async def wait_for_event_loop():
    tasks = []
    logging.info('Total tasks: %i' % len(asyncio.Task.all_tasks()))
    nr_task =len([task for task in asyncio.Task.all_tasks() if not task.done()])
    while nr_task > 1:
        logging.info('Active tasks: %i' % nr_task)
        nr_task = len([task for task in asyncio.Task.all_tasks() if not task.done()])
        await asyncio.sleep(0.2)

def process_sentences_neo4j_par(tokenizer, bert, text_db, neograph, year, MAX_SEQ_LENGTH, DICT_SIZE, batch_size, maxn=None, nr_workers=0,
                              cutoff_percent=99,max_degree=50, par=True):
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

    # %% Initialize text dataset
    # !!! TODO REMOVE MAXN !!!
    dataset = text_dataset(text_db, tokenizer, MAX_SEQ_LENGTH,maxn=maxn)
    logging.info("Number of sentences found: %i"%dataset.nitems)
    batch_sampler = BatchSampler(SequentialSampler(range(0, dataset.nitems)), batch_size=batch_size, drop_last=False)
    dataloader = DataLoaderX(dataset=dataset, batch_size=None, sampler=batch_sampler, num_workers=nr_workers,
                             collate_fn=text_dataset_collate_batchsample, pin_memory=False)

    # Push BERT to GPU
    torch.cuda.empty_cache()
    if torch.cuda.is_available(): logging.info("Using CUDA.")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    bert.to(device)
    bert.eval()

    # Create Stopwords
    delwords = create_stopword_list(tokenizer)

    # Initgraph



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
        predictions, attn = get_bert_tensor(0, bert, batch, tokenizer.pad_token_id, tokenizer.mask_token_id, device,
                                            return_max=False)

        # compute model timings time
        prepare_time = time.time() - start_time - load_time
        model_timings.append(prepare_time)

        # %% Sequence Table
        # Iterate over sequences
        sequence_id_container=[]
        dists_container=[]
        token_id_container=[]
        for sequence_id in np.unique(seq_ids):
            sequence_mask = seq_ids == sequence_id
            sequence_size = sum(sequence_mask)

            # Pad and add distributions per token, we need to save to maximum sequence size
            dists = np.zeros([MAX_SEQ_LENGTH, DICT_SIZE])
            dists[:sequence_size, :] = predictions[sequence_mask, :]
            dists_container.append(dists.copy())
            token_id_container.append(token_ids[sequence_mask])
            sequence_id_container.append(sequence_id)
        del sequence_size,sequence_mask,dists,sequence_id

        inputs=list(zip(sequence_id_container,dists_container,token_id_container))

        @jit(nopython=True)
        def calculate_ties(input_vec):
            sequence_id=input_vec[0]
            dists=input_vec[1]
            idx=input_vec[2]
            results=[]
            # TODO: This should probably not be a loop
            # but done smartly with matrices and so forth
            for pos, token in enumerate(idx):
                # Should all be np
                token=token.item()
                if token not in delwords:
                    replacement = dists[pos, :]

                    # Flatten, since it is one row each
                    replacement = replacement.flatten()

                    # Sparsify
                    # TODO: try without setting own-link to zero!
                    replacement[token]=0
                    replacement[replacement==np.min(replacement)]=0

                    # Get rid of delnorm links
                    replacement[delwords]=0

                    # We norm the distributions here
                    replacement = simple_norm(replacement)

                    # Add values to network
                    # TODO implement sequence id and pos properties on network
                    cutoff_number, cutoff_probability = calculate_cutoffs(replacement, method="percent",
                                                                          percent=cutoff_percent, max_degree=max_degree)
                    ties=get_weighted_edgelist(token, replacement, year, cutoff_number, cutoff_probability,sequence_id,pos)
                    #neograph.insert_edges_multiple(ties)
                    results.append(ties)
            return results

        num_cores = 8
        if par==True:
            results = Parallel(n_jobs=num_cores)(delayed(calculate_ties)(i) for i in inputs)
        elif par=="Threading":
            results = Parallel(n_jobs=num_cores,prefer="threads")(delayed(calculate_ties)(i) for i in inputs)
        else:
            results=[]
            for i in inputs:
                results.append(calculate_ties(i))

        for tie_vec in results:
            for ties in tie_vec:
                neograph.insert_edges_multiple(ties)

        del predictions #, attn, token_ids, seq_ids, batch
        #neograph.write_queue()
        # compute processing time
        process_timings.append(time.time() - start_time - prepare_time - load_time)
        # New start time
        start_time = time.time()


    # Write remaining
    neograph.write_queue()
    torch.cuda.empty_cache()

    dataset.close()
    del dataloader, dataset, batch_sampler
    logging.info("Average Load Time: %s seconds" % (np.mean(load_timings)))
    logging.info("Average Model Time: %s seconds" % (np.mean(model_timings)))
    logging.info("Average Processing Time: %s seconds" % (np.mean(process_timings)))
    logging.info("Ratio Load/Operations: %s seconds" % (np.mean(load_timings) / np.mean(process_timings + model_timings)))

