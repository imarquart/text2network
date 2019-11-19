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

def batches(iterable, size):
    source = iter(iterable)
    while True:
        chunk = [val for _, val in zip(range(size), source)]
        if not chunk:
            raise StopIteration
        yield chunk




def get_weighted_edgelist(token,x,cutoff_number=5,cutoff_probability=0):
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

    if cutoff_probability >0:
        # TODO
        cutoff_probability=0
    
    weights = x[neighbors]
    #edgelist = [(token, x) for x in neighbors]
    return [(token, x[0], x[1]) for x in list(zip(neighbors, weights))]

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    x = np.exp(x - np.max(x))
    return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)

def simple_norm(x):
    """Just want to start at zero and sum to 1, without norming anything else"""
    x=x-np.min(x,axis=-1)
    return x/np.sum(x,axis=-1)

def create_network(database,tokenizer,start_token,nr_clusters,batch_size):

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
    delwords=create_stopword_list(tokenizer)
    nodes = np.setdiff1d(nodes, delwords)
    nr_nodes = nodes.shape[0]


    # Now we will situate the contexts of the start token
    start_token=tokenizer.convert_tokens_to_ids(start_token)
    token=start_token
    query = "".join(['token_id==', str(token)])
    rows = token_table.read_where(query)
    nr_rows = len(rows)
    if nr_rows == 0:
        raise AssertionError("Start token not in dataset")

    # Create context distributions
    context_dists=np.stack([x[0] for x in rows],axis=0).squeeze()
    own_dists=np.stack([x[1] for x in rows],axis=0).squeeze()

    if nr_rows == 1:
        context_dists=np.reshape(context_dists, (-1, context_dists.shape[0]))
        own_dists=np.reshape(own_dists, (-1, own_dists.shape[0]))

    context_dists[:, token] = np.min(context_dists)
    context_dists[:, delwords] = np.min(context_dists)
    own_dists[:, token] = np.min(own_dists)
    own_dists[:, delwords] = np.min(own_dists)

    context_dists = softmax(context_dists)
    own_dists = softmax(own_dists)

    # Cluster according the context distributions
    nr_clusters=min(nr_clusters,nr_rows)
    #spcluster=SpectralClustering(n_clusters=nr_clusters,assign_labels = "discretize",random_state = 0).fit(context_dists)
    # In case some cluster is empty
    #nr_clusters = len(np.unique(spcluster.labels_))


    # HDBSCAN Clusterer
    clusterer = hdbscan.HDBSCAN(min_cluster_size=2, prediction_data=True).fit(context_dists)
    nr_clusters = len(np.unique(clusterer.labels_))




    # Create di-Graphs to store network
    # Each cluster is stored separately
    graphs=[nx.DiGraph() for i in range(nr_clusters)]
    # Add all potential tokens
    for g in graphs: g.add_nodes_from(range(0, tokenizer.vocab_size))

    #for i in range(nr_clusters):
    #    replacement = np.sum(own_dists[clusterer.labels_==i-1,:],axis=0)
    #    context=np.sum(own_dists[clusterer.labels_==i-1,:],axis=0)
    #    replacement=simple_norm(replacement)
    #    context=simple_norm(context)
    #    graphs[i].add_weighted_edges_from(get_weighted_edgelist(token,replacement))

    # Now parallel loop over all nodes
    # TODO: Parallel
    # TODO: Batch the DB request (at least)


    nodes=np.sort(nodes)
    node_batch=batches(nodes,batch_size)

    for chunk in tqdm(node_batch):
        limits=[chunk[0],chunk[-1]]
        query = "".join(['(token_id>=', str(limits[0]),') & (token_id<=', str(limits[1]),')' ])
        rows = token_table.where(query)
        own_dists = {}
        context_dists = {}
        ar=np.zeros((1,30522))
        for token in chunk:
            own_dists[token]=ar
            context_dists[token] = ar

        def token_id_sel(row):
            return row['token_id']

        for token_id, rows_grouped_by_token_id in itertools.groupby(rows, token_id_sel):
            if token_id not in delwords:
                own_dists[token_id] = np.stack([own_dists[token_id]],np.stack(r['own_dist'] for r in rows_grouped_by_token_id))
                context_dists[token_id] = np.stack([context_dists[token_id]],np.stack(r['own_dist'] for r in rows_grouped_by_token_id))


        nr_rows = len(rows)
        if nr_rows == 0:
            raise AssertionError("Database error: Token information missing")

        # Create context distributions
        context_dists=np.stack([x[0] for x in rows],axis=0).squeeze()
        own_dists=np.stack([x[1] for x in rows],axis=0).squeeze()
       
        if nr_rows == 1:
            context_dists=np.reshape(context_dists, (-1, context_dists.shape[0]))
            own_dists=np.reshape(own_dists, (-1, own_dists.shape[0]))

        context_dists[:, token] = np.min(context_dists)
        context_dists[:, delwords] = np.min(context_dists)
        own_dists[:, token] = np.min(own_dists)
        own_dists[:, delwords] = np.min(own_dists)

        context_dists = softmax(context_dists)
        own_dists = softmax(own_dists)

        # Predict to label
        labels, strengths = hdbscan.approximate_predict(clusterer, context_dists)

        for i in range(nr_clusters):
            if len(labels[labels==i-1]) >0:
                replacement = np.sum(own_dists[labels==i-1,:],axis=0)
                context=np.sum(own_dists[labels==i-1,:],axis=0)
                replacement=simple_norm(replacement)
                context=simple_norm(context)
                graphs[i].add_weighted_edges_from(get_weighted_edgelist(token,replacement))

        #if idx>10:
        #    break





    return graphs