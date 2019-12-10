
from sklearn.cluster import SpectralClustering
import torch
from NLP.src.datasets.dataloaderX import DataLoaderX
from sklearn.decomposition import PCA
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


def create_clusters(database, tokenizer, start_token, nr_clusters, cutoff=10, dset_method=0):
    dataset = tensor_dataset(database, method=0)

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
    print("".join(["Rows found:",str(nr_rows)]))
    if nr_rows == 0:
        raise AssertionError("Start token not in dataset")

    context_dists[:, start_token] = np.min(context_dists)
    context_dists[:, delwords] = np.min(context_dists)
    own_dists[:, start_token] = np.min(own_dists)
    own_dists[:, delwords] = np.min(own_dists)

    pca = PCA(n_components=cutoff)
    pca.fit(context_dists)
    trunc_dist=pca.transform(context_dists)
    print("".join(['Number of PCA components to explain ', str(cutoff*100),'% var: ',str(pca.n_components_)]))

    # Cluster according the context distributions
    nr_clusters = min(nr_clusters, nr_rows)

    # Get variances of dimension
    #context_vars=np.var(context_dists, axis=0)
    #replacement_vars=np.var(own_dists, axis=0)
    #word_idx = np.argsort(-context_vars)[0:cutoff]
    #trunc_dist=context_dists[:,word_idx]

    clusterer = KMeans(n_clusters=nr_clusters).fit(trunc_dist)
    # #cluster_selection_method="leaf",
    clusterer = hdbscan.HDBSCAN(min_cluster_size=nr_clusters, min_samples=1, metric='l1',prediction_data=True).fit(trunc_dist)
    print("Context clustering:")
    for cluster in np.unique(clusterer.labels_):
        mask=clusterer.labels_==cluster
        context = np.sum(context_dists[mask, :], axis=0)
        rep = np.sum(own_dists[mask, :], axis=0)
        idx2 = np.argsort(-rep)[0:15]
        idx = np.argsort(-context)[0:25]
        print("".join(['--- Cluster: ',str(cluster+1),' ---']))
        print("".join(['Context: ',str(tokenizer.convert_ids_to_tokens(idx))]))
        print("".join(['Replacement: ',str(tokenizer.convert_ids_to_tokens(idx2))]))
        print("--- ---")
