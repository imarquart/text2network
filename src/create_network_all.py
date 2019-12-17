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


def calculate_cutoffs(x, method="mean"):
    """
    Different methods to calculate cutoff probability and number.

    :param x: Contextual vector
    :param method: To implement. Currently: mean
    :return: cutoff_number and probability
    """
    if method == "mean":
        cutoff_probability = max(np.mean(x), 0.01)
        cutoff_number = max(np.int(len(x) / 100), 100)
    elif method == "80":
        sortx = np.sort(x)[::-1]
        cum_sum = np.cumsum(sortx)
        cutoff = cum_sum[-1] * 0.8
        cutoff_number = np.where(cum_sum >= cutoff)[0][0]
        cutoff_probability = 0.01
    else:
        cutoff_probability = 0
        cutoff_number = 0

    return cutoff_number, cutoff_probability


def get_weighted_edgelist(token, x, cutoff_number=100, cutoff_probability=0):
    """
    Sort probability distribution to get the most likely neighbor nodes.
    Return a networksx weighted edge list for a given focal token as node.

    :param token: Numerical, token which to add
    :param x: Probability distribution
    :param cutoff_number: Number of neighbor token to consider. Not used if 0.
    :param cutoff_probability: Lowest probability to consider. Not used if 0.
    :return: List of tuples compatible with networkx
    """
    # Get the most pertinent words
    if cutoff_number > 0:
        neighbors = np.argsort(-x)[:cutoff_number]
    else:
        neighbors = np.argsort(-x)[:]

    # Cutoff probability (zeros)
    if len(neighbors > 0):
        neighbors = neighbors[x[neighbors] > cutoff_probability]
        weights = x[neighbors]
        # edgelist = [(token, x) for x in neighbors]
    return [(token, x[0], x[1]) for x in list(zip(neighbors, weights))]


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    x = np.exp(x - np.max(x))
    return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)


def simple_norm(x):
    """Just want to start at zero and sum to 1, without norming anything else"""
    x = x - np.min(x, axis=-1)
    if np.sum(x, axis=-1) > 0:
        return x / np.sum(x, axis=-1)
    else:
        return x


def create_network_all(database, tokenizer, start_token, nr_clusters, batch_size=0, dset_method=0):
    dataset = tensor_dataset(database, method=0)

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
                    cutoff_number, cutoff_probability = calculate_cutoffs(replacement[i, :], method="80")
                    graph.add_weighted_edges_from(
                        get_weighted_edgelist(token, replacement[i, :], cutoff_number, cutoff_probability), 'weight',
                        seq=i)

                    context[i, :] = simple_norm(context[i, :])
                    cutoff_number, cutoff_probability = calculate_cutoffs(context[i, :], method="80")
                    context_graph.add_weighted_edges_from(
                        get_weighted_edgelist(token, context[i, :], cutoff_number, cutoff_probability), 'weight',
                        seq=i)

        process_timings.append(time.time() - start_time - prepare_time - load_time)
    print(" ")
    print("Average Load Time: %s seconds" % (np.mean(load_timings)))
    print("Average Prep Time: %s seconds" % (np.mean(prep_timings)))
    print("Average Processing Time: %s seconds" % (np.mean(process_timings)))
    print("Ratio Load/Operations: %s seconds" % (np.mean(load_timings) / np.mean(process_timings + prep_timings)))
    return graph, context_graph
