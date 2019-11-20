# TODO: Commment

from sklearn.cluster import SpectralClustering
import torch
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
from sklearn.cluster import MeanShift, estimate_bandwidth

def get_weighted_edgelist(token, x, cutoff_number=1000, cutoff_probability=0):
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
    if len(neighbors>0):
        neighbors=neighbors[x[neighbors]>cutoff_probability]
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


def create_network(database, tokenizer, start_token, nr_clusters, batch_size=0):
    # Open database connection, and table
    try:
        data_file = tables.open_file(database, mode="r", title="Data File")
        token_table = data_file.root.token_data.table
    except:
        raise FileNotFoundError("Could not read token table from database.")

    # Creat index?

    # Get all unique tokens in database
    nodes = np.unique(token_table.col('token_id'))
    # Delete from this stopwords and the like
    delwords = create_stopword_list(tokenizer)
    nodes = np.setdiff1d(nodes, delwords)
    nr_nodes = nodes.shape[0]

    # Now we will situate the contexts of the start token
    start_token = tokenizer.convert_tokens_to_ids(start_token)
    token = start_token
    query = "".join(['token_id==', str(token)])
    rows = token_table.read_where(query)
    nr_rows = len(rows)
    if nr_rows == 0:
        raise AssertionError("Start token not in dataset")

    # Create context distributions
    context_dists = np.stack([x[0] for x in rows], axis=0).squeeze()
    own_dists = np.stack([x[1] for x in rows], axis=0).squeeze()

    if nr_rows == 1:
        context_dists = np.reshape(context_dists, (-1, context_dists.shape[0]))
        own_dists = np.reshape(own_dists, (-1, own_dists.shape[0]))

    context_dists[:, token] = np.min(context_dists)
    context_dists[:, delwords] = np.min(context_dists)
    own_dists[:, token] = np.min(own_dists)
    own_dists[:, delwords] = np.min(own_dists)

    context_dists = softmax(context_dists)
    own_dists = softmax(own_dists)

    # Cluster according the context distributions
    nr_clusters = min(nr_clusters, nr_rows)
    # spcluster=SpectralClustering(n_clusters=nr_clusters,assign_labels = "discretize",random_state = 0).fit(context_dists)
    # In case some cluster is empty
    # nr_clusters = len(np.unique(spcluster.labels_))

    # Clusterer
    #clusterer = KMeans(n_clusters=nr_clusters,n_jobs=-1).fit(context_dists)
    #clusterer = hdbscan.HDBSCAN(min_cluster_size=4, prediction_data=True).fit(context_dists)
    bandwidth = estimate_bandwidth(context_dists, quantile=0.2)
    clusterer = MeanShift(bandwidth=0.3, bin_seeding=True).fit(context_dists)
    nr_clusters = len(np.unique(clusterer.labels_))

    # Create di-Graphs to store network
    # Each cluster is stored separately
    graphs = [nx.DiGraph() for i in range(nr_clusters)]
    # Add all potential tokens
    for g in graphs: g.add_nodes_from(range(0, tokenizer.vocab_size))

    # for i in range(nr_clusters):
    #    replacement = np.sum(own_dists[clusterer.labels_==i-1,:],axis=0)
    #    context=np.sum(own_dists[clusterer.labels_==i-1,:],axis=0)
    #    replacement=simple_norm(replacement)
    #    context=simple_norm(context)
    #    graphs[i].add_weighted_edges_from(get_weighted_edgelist(token,replacement))

    # Now parallel loop over all nodes
    # TODO: Parallel
    # TODO: Batch the DB request (at least)

    nodes = np.sort(nodes)

    if batch_size>1:
        print("Batch size > 1, using new algo")
        btsampler=BatchSampler(SequentialSampler(nodes), batch_size=batch_size, drop_last=False)
        for chunk in tqdm(btsampler):
            chunk=nodes[chunk]
            limits = [chunk[0], chunk[-1]]
            query = "".join(['(token_id>=', str(limits[0]), ') & (token_id<=', str(limits[1]), ')'])
            rows = token_table.read_where(query)
            nr_rows = len(rows)
            if nr_rows == 0:
                raise AssertionError("Database error: Token information missing")

            # Create context distributions
            context_dists = np.stack([x[0] for x in rows], axis=0).squeeze()
            own_dists = np.stack([x[1] for x in rows], axis=0).squeeze()
            token_idx = np.stack([x['token_id'] for x in rows], axis=0).squeeze()

            if nr_rows == 1:
                context_dists = np.reshape(context_dists, (-1, context_dists.shape[0]))
                own_dists = np.reshape(own_dists, (-1, own_dists.shape[0]))

            context_dists[:, token] = np.min(context_dists)
            context_dists[:, delwords] = np.min(context_dists)
            own_dists[:, token] = np.min(own_dists)
            own_dists[:, delwords] = np.min(own_dists)

            context_dists = softmax(context_dists)
            own_dists = softmax(own_dists)

            # Predict to label
            #labels, strengths = hdbscan.approximate_predict(clusterer, context_dists)
            labels=clusterer.predict(context_dists)

            for i in range(nr_clusters):
                for token in chunk:
                    mask=(token_idx==token) & (labels == i - 1)
                    if len(labels[mask]) > 0:
                        replacement = np.sum(own_dists[mask,:], axis=0)
                        context = np.sum(context_dists[mask, :], axis=0)
                        replacement = simple_norm(replacement)
                        context = simple_norm(context)
                        graphs[i].add_weighted_edges_from(get_weighted_edgelist(token, replacement))
    else:
        print("Batch size <= 1, using old algo")
        for idx, token in tqdm(enumerate(nodes)):
            query = "".join(['token_id==', str(token)])
            rows = token_table.read_where(query)
            nr_rows = len(rows)
            if nr_rows == 0:
                raise AssertionError("Database error: Token information missing")

            # Create context distributions
            context_dists = np.stack([x[0] for x in rows], axis=0).squeeze()
            own_dists = np.stack([x[1] for x in rows], axis=0).squeeze()

            if nr_rows == 1:
                context_dists = np.reshape(context_dists, (-1, context_dists.shape[0]))
                own_dists = np.reshape(own_dists, (-1, own_dists.shape[0]))

            context_dists[:, token] = np.min(context_dists)
            context_dists[:, delwords] = np.min(context_dists)
            own_dists[:, token] = np.min(own_dists)
            own_dists[:, delwords] = np.min(own_dists)

            context_dists = softmax(context_dists)
            own_dists = softmax(own_dists)

            # Predict to label
            #labels, strengths = hdbscan.approximate_predict(clusterer, context_dists)
            labels=clusterer.predict(context_dists)

            for i in range(nr_clusters):
                if len(labels[labels == i - 1]) > 0:
                    replacement = np.sum(own_dists[labels == i - 1, :], axis=0)
                    context = np.sum(own_dists[labels == i - 1, :], axis=0)
                    replacement = simple_norm(replacement)
                    context = simple_norm(context)
                    graphs[i].add_weighted_edges_from(get_weighted_edgelist(token, replacement))

    return graphs
