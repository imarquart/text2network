# TODO: Commment

from sklearn.cluster import SpectralClustering
import torch
import numpy as np
import tables
import networkx as nx
import os
from utils.load_bert import get_bert_and_tokenizer
from tqdm import tqdm
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
from numpy import inf
from sklearn.cluster import KMeans

def create_stopword_list(tokenizer):
    """
    Return a list of tokenized tokens for words which should be removed from the data set

    :param tokenizer: BERT tokenizer
    :return: Numerical list of tokens to take out of sample 
    """
    alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
                'v', 'w', 'x', 'y', 'z']
    alphabet_ids = tokenizer.convert_tokens_to_ids(alphabet)
    numbers = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'one', 'two', 'three', 'four']
    numbers_ids = tokenizer.convert_tokens_to_ids(numbers)
    pronouns = ['we', 'us', 'my', 'yourself', 'you', 'me', 'he', 'her', 'his', 'him', 'she', 'they', 'their', 'them',
                'me', 'myself', 'himself', 'herself', 'themselves']
    pronouns_ids = tokenizer.convert_tokens_to_ids(pronouns)
    stopwords = ['&', ',', '.', 'and', '-', 'the', '##d', '...', 'that', 'to', 'as', 'for', '"', 'in', "'", 'a', 'of',
                 'only', ':', 'so', 'all', 'one', 'it', 'then', 'also', 'with', 'but', 'by', 'on', 'just', 'like',
                 'again', ';', 'more', 'this', 'not', 'is', 'there', 'was', 'even', 'still', 'after', 'here', 'later',
                 '!', 'over', 'from', 'i', 'or', '?', 'at', 'first', '##s', 'while', ')', 'before', 'when', 'once',
                 'too', 'out', 'yet', 'because', 'some', 'though', 'had', 'instead', 'always', '(', 'well', 'back',
                 'tonight', 'since', 'about', 'through', 'will', 'them', 'left', 'often', 'what', 'ever', 'until',
                 'sometimes', 'if', 'however', 'finally', 'another', 'somehow', 'everything', 'further', 'really',
                 'last', 'an', '/', 'rather', 's', 'may', 'be', 'each', 'thus', 'almost', 'where', 'anyway', 'their',
                 'has', 'something', 'already', 'within', 'any', 'indeed', '##a', '[UNK]', '~', 'every', 'meanwhile',
                 'would', '##e', 'have', 'nevertheless', 'which', 'how', '1', 'are', 'either', 'along', 'thereafter',
                 'otherwise', 'did', 'quite', 'these', 'can', '2', 'its', 'merely', 'actually', 'certainly', '3',
                 'else', 'upon', 'except', 'those', 'especially', 'therefore', 'beside', 'apparently', 'besides',
                 'third', 'whilst', '*', 'although', 'were', 'likewise', 'mainly', 'four', 'seven', 'into', 'm', ']',
                 'than', 't', 'surely', '|', '#', 'till', '##ly', '_', 'al']
    stopwords_ids = tokenizer.convert_tokens_to_ids(stopwords)

    delwords = np.union1d(stopwords_ids, pronouns_ids)
    delwords = np.union1d(delwords, alphabet_ids)
    delwords = np.union1d(delwords, numbers_ids)
    return delwords

def get_weighted_edgelist(token,x,cutoff_number=25,cutoff_probability=0):
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
        neighbors = np.argsort(-x)[:25]
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

def create_network(database,tokenizer,start_token,nr_clusters):


    # Open database connection, and table
    try:
        data_file = tables.open_file(database, mode="r", title="Data File")
    except:
        print("ERROR")
    try:
        token_table = data_file.root.token_data.table
    except:
        print("ERROR")

    # Get all unique tokens in database
    nodes = np.unique(token_table.col('token_id'))
    # Delete from this stopwords and the like
    delwords=create_stopword_list(tokenizer)
    nodes = np.setdiff1d(nodes, delwords)
    nr_nodes = nodes.shape[0]


    # Now we will situate the contexts of the start token
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
    spcluster=SpectralClustering(n_clusters=nr_clusters,assign_labels = "discretize",random_state = 0).fit(context_dists)

    # In case some cluster is empty
    nr_clusters = len(np.unique(spcluster.labels_))

    # Create di-Graphs to store network
    # Each cluster is stored separately
    graphs=[nx.DiGraph() for i in range(nr_clusters)]
    # Add all potential tokens
    for g in graphs: g.add_nodes_from(range(1, tokenizer.vocab_size))

    for i in range(nr_clusters):
        replacement = np.sum(own_dists[spcluster.labels_==i,:],axis=0)
        context=np.sum(own_dists[spcluster.labels_==i,:],axis=0)
        replacement=simple_norm(replacement)
        context=simple_norm(context)
        graphs[i].add_weighted_edges_from(get_weighted_edgelist(token,replacement))

    # Now parallel loop over all nodes
    # TODO: Parallel

    for idx, token in enumerate(nodes):
        query = "".join(['token_id==', str(token)])
        rows = token_table.read_where(query)
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
        results=spcluster.predict(context_dists)
        for i in range(nr_clusters):
            if len(results.label_) >0:
                replacement = np.sum(own_dists[spcluster.labels_==i,:],axis=0)
                context=np.sum(own_dists[spcluster.labels_==i,:],axis=0)
                replacement=simple_norm(replacement)
                context=simple_norm(context)
                graphs[i].add_weighted_edges_from(get_weighted_edgelist(token,replacement))





