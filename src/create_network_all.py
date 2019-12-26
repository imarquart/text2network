# TODO: Commment
# TODO: Add Logger

from sklearn.cluster import SpectralClustering
import torch
from NLP.src.datasets.dataloaderX import DataLoaderX
import numpy as np
import tables
import networkx as nx
import os
from tqdm import tqdm
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
from numpy import inf
from sklearn.cluster import KMeans
import hdbscan
import itertools
from NLP.utils.delwords import create_stopword_list
from torch.utils.data import BatchSampler, SequentialSampler
import time
from sklearn.cluster import MeanShift, estimate_bandwidth
from NLP.src.datasets.tensor_dataset import tensor_dataset, tensor_dataset_collate_batchsample
from NLP.utils.rowvec_tools import simple_norm, get_weighted_edgelist, calculate_cutoffs, add_to_networks


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    x = np.exp(x - np.max(x))
    return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)



def create_network_all(database, tokenizer, batch_size=0, dset_method=0, cutoff_percent=80):
    dataset = tensor_dataset(database, method=dset_method)

    # Delete from this stopwords and the like
    # print("Total nodes: %i" % dataset.nodes.shape[0])
    delwords = create_stopword_list(tokenizer)
    dataset.nodes = np.setdiff1d(dataset.nodes, delwords)
    nr_nodes = dataset.nodes.shape[0]
    # print("Nodes after deletion: %i" % dataset.nodes.shape[0])

    graph = nx.MultiDiGraph()
    context_graph = nx.MultiDiGraph()
    graph.add_nodes_from(range(0, tokenizer.vocab_size))
    context_graph.add_nodes_from(range(0, tokenizer.vocab_size))

    btsampler = BatchSampler(SequentialSampler(dataset.nodes), batch_size=batch_size, drop_last=False)
    dataloader = DataLoaderX(dataset=dataset, batch_size=None, sampler=btsampler, num_workers=0,
                             collate_fn=tensor_dataset_collate_batchsample, pin_memory=False)

    # Counter for timing
    prep_timings = []
    process_timings = []
    load_timings = []
    start_time = time.time()

    for chunk, token_idx, own_dists, context_dists in tqdm(dataloader):
        # chunk, token_idx, own_dists, context_dists = batch
        # Data spent on loading batch
        load_time = time.time() - start_time
        load_timings.append(load_time)

        nr_rows = len(token_idx)
        if nr_rows == 0:
            raise AssertionError("Database error: Token information missing")

        context_dists[:, delwords] = 0  # np.min(context_dists)
        own_dists[:, delwords] = 0

        # Find a better way to do this
        # thsi is terrible
        for token in chunk:
            context_dists[token_idx == token, token] = 0
            own_dists[token_idx == token, token] = 0

        # compute model timings time
        prepare_time = time.time() - start_time - load_time
        prep_timings.append(prepare_time)

        for token in chunk:
            mask = (token_idx == token)
            if len(token_idx[mask]) > 0:

                replacement = own_dists[mask, :]
                context = context_dists[mask, :]


                for i in range(0, len(token_idx[mask])):

                    replacement[i, :] = simple_norm(replacement[i, :])
                    context[i, :] = simple_norm(context[i, :])


                    graph,context_graph=add_to_networks(graph,context_graph,replacement[i,:],context[i, :],token,cutoff_percent,0,i)


        process_timings.append(time.time() - start_time - prepare_time - load_time)
    print(" ")
    print("Average Load Time: %s seconds" % (np.mean(load_timings)))
    print("Average Prep Time: %s seconds" % (np.mean(prep_timings)))
    print("Average Processing Time: %s seconds" % (np.mean(process_timings)))
    print("Ratio Load/Operations: %s seconds" % (np.mean(load_timings) / np.mean(process_timings + prep_timings)))
    return graph, context_graph
