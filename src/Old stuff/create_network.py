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
from NLP.src.utils.delwords import create_stopword_list
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


def create_network(database, tokenizer, start_token, nr_clusters, batch_size=0, dset_method=0):
    dataset = tensor_dataset(database, method=dset_method)

    # Delete from this stopwords and the like
    #print("Total nodes: %i" % dataset.nodes.shape[0])
    delwords = create_stopword_list(tokenizer)
    dataset.nodes = np.setdiff1d(dataset.nodes, delwords)
    nr_nodes = dataset.nodes.shape[0]
    #print("Nodes after deletion: %i" % dataset.nodes.shape[0])

    # Now we will situate the contexts of the start token
    start_token = tokenizer.convert_tokens_to_ids(start_token)

    chunk, token_idx, own_dists, context_dists = dataset[dataset.tokenid_to_index(start_token)]
    nr_rows = len(token_idx)
    if nr_rows == 0:
        raise AssertionError("Start token not in dataset")

    context_dists[:, start_token] = np.min(context_dists)
    context_dists[:, delwords] = np.min(context_dists)
    own_dists[:, start_token] = np.min(own_dists)
    own_dists[:, delwords] = np.min(own_dists)

    # context_dists = softmax(context_dists)
    # own_dists = softmax(own_dists)

    # Cluster according the context distributions
    nr_clusters = min(nr_clusters, nr_rows)
    # Clusterer
    if nr_clusters > 1:
        clusterer = KMeans(n_clusters=nr_clusters).fit(context_dists)
        # clusterer = hdbscan.HDBSCAN(min_cluster_size=4, prediction_data=True).fit(context_dists)
        # bandwidth = estimate_bandwidth(context_dists, quantile=0.2)
        # clusterer = MeanShift(bandwidth=0.3, bin_seeding=True).fit(context_dists)
        # spcluster=SpectralClustering(n_clusters=nr_clusters,assign_labels = "discretize",random_state = 0).fit(context_dists)
        # In case some cluster is empty
        # nr_clusters = len(np.unique(spcluster.labels_))
        nr_clusters = len(np.unique(clusterer.labels_))

    #print("".join(["Number of clusters: ", str(nr_clusters)]))

    # Create di-Graphs to store network
    # Each cluster is stored separately
    graphs = [nx.DiGraph() for i in range(nr_clusters)]
    context_graphs = [nx.DiGraph() for i in range(nr_clusters)]

    # Add all potential tokens
    for g in graphs: g.add_nodes_from(range(0, tokenizer.vocab_size))
    for g in context_graphs: g.add_nodes_from(range(0, tokenizer.vocab_size))

    # for i in range(nr_clusters):
    #    replacement = np.sum(own_dists[clusterer.labels_==i-1,:],axis=0)
    #    context=np.sum(own_dists[clusterer.labels_==i-1,:],axis=0)
    #    replacement=simple_norm(replacement)
    #    context=simple_norm(context)
    #    graphs[i].add_weighted_edges_from(get_weighted_edgelist(token,replacement))

    # Now parallel loop over all nodes
    # TODO: Parallel
    # TODO: Batch the DB request (at least)

    btsampler = BatchSampler(SequentialSampler(dataset.nodes), batch_size=batch_size, drop_last=False)
    dataloader=DataLoaderX(dataset=dataset,batch_size=None, sampler=btsampler,num_workers=0,collate_fn=tensor_dataset_collate_batchsample, pin_memory=False)

    # Counter for timing
    prep_timings = []
    process_timings = []
    load_timings = []
    start_time = time.time()

    for chunk, token_idx, own_dists, context_dists in tqdm(dataloader):
        #chunk, token_idx, own_dists, context_dists = batch
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

        # context_dists = softmax(context_dists)
        # own_dists = softmax(own_dists)

        # Predict to label
        # labels, strengths = hdbscan.approximate_predict(clusterer, context_dists)
        if nr_clusters > 1:
            labels = clusterer.predict(context_dists)
        else:
            labels = np.zeros(nr_rows, dtype=int)

        # compute model timings time
        prepare_time = time.time() - start_time - load_time
        prep_timings.append(prepare_time)

        for i in range(nr_clusters):
            for token in chunk:
                mask = (token_idx == token) & (labels == i)
                if len(labels[mask]) > 0:
                    replacement = np.sum(own_dists[mask, :], axis=0)
                    context = np.sum(context_dists[mask, :], axis=0)
                    replacement = simple_norm(replacement)
                    context = simple_norm(context)
                    cutoff_number, cutoff_probability = calculate_cutoffs(replacement, method="mean")
                    graphs[i].add_weighted_edges_from(
                        get_weighted_edgelist(token, replacement, cutoff_number, cutoff_probability))

                    cutoff_number, cutoff_probability = calculate_cutoffs(context, method="mean")
                    context_graphs[i].add_weighted_edges_from(
                        get_weighted_edgelist(token, context, cutoff_number, cutoff_probability))
        process_timings.append(time.time() - start_time - prepare_time - load_time)
    print(" ")
    print("Average Load Time: %s seconds" % (np.mean(load_timings)))
    print("Average Prep Time: %s seconds" % (np.mean(prep_timings)))
    print("Average Processing Time: %s seconds" % (np.mean(process_timings)))
    print("Ratio Load/Operations: %s seconds" % (np.mean(load_timings) / np.mean(process_timings + prep_timings)))
    return graphs, context_graphs
