import numpy as np
import scipy as sp
import pandas as pd
from sklearn.cluster import KMeans
import hdbscan
from scipy.spatial.distance import cosine, pdist,squareform
import pandas as pd
import glob
import xlsxwriter
import os, time, sys
import logging
from shutil import copyfile
from sys import exit
from NLP.src.text_processing.preprocess_files_HBR import preprocess_files_HBR
from NLP.config.config import configuration
from NLP.src.process_sentences_network import process_sentences_network
from NLP.utils.load_bert import get_bert_and_tokenizer
from NLP.utils.network_tools import load_graph, make_symmetric, graph_merge,load_graph_overall
from NLP.src.run_bert import bert_args, run_bert
from NLP.src.dynamic_clustering import dynamic_clustering, louvain_cluster, overall_clustering, \
    overall_onelevel_clustering
from NLP.src.centrality_measures import dynamic_centralities, raw_ego_network
import networkx as nx
from NLP.src.draw_networks import draw_ego_network_mem, draw_ego_network
import itertools
import community

def louvain_cluster(graph,resolution=1):
    clustering = community.best_partition(graph,weight="weight",resolution=resolution)

    return [[k for k, v in clustering.items() if v == val] for val in list(set(clustering.values()))]

cfg = configuration()

csv=pd.read_csv("E:/NLP/cluster_xls/centralities/Book1.csv",delimiter=";")
tokens=csv.iloc[:150,0].to_numpy()
interest_tokens=[np.str(x) for x in tokens]
#graph=load_graph(1991,cfg,cfg.ma_folder,"Rgraph-Sum-Rev_order3")
graph=load_graph_overall(cfg,cfg.merged_folder,"Rgraph-Sum-Rev_avg_r")
sym_graph=make_symmetric(graph,technique="min-sym-avg")

sym_subgraph=nx.subgraph(sym_graph,interest_tokens)
subgraph=nx.subgraph(graph,interest_tokens)
egograph=nx.ego_graph(graph,'leader',2)
ego_sym=make_symmetric(egograph,technique="min-sym-avg")
tokens=np.array(subgraph.nodes)

egograph['leader']

network=sym_subgraph

cluster_graphs=[]
res_param=[0.1,0.2,0.3,0.4]
for res in res_param:
    cluster_it=louvain_cluster(network, resolution =res)
    cluster_graph = nx.Graph()
    cluster_graph.add_nodes_from(network)
    for cluster in cluster_it:
        pairs = itertools.combinations(cluster, r=2)
        cluster_graph.add_edges_from(list(pairs))
    cluster_graphs.append(cluster_graph.copy())

consensus_graph = graph_merge(cluster_graphs, average_links=True, method="majority", merge_mode="safe")
cluster_it=louvain_cluster(consensus_graph, resolution =1)

for cluster in cluster_it:
    if not len(np.intersect1d(cluster,interest_tokens))==0: print(np.intersect1d(cluster,interest_tokens))

network=ego_sym
cluster_graphs=[]
res_param=0.2
cluster_it=louvain_cluster(network, resolution =res_param)
for cluster in cluster_it:
    if not len(np.intersect1d(cluster, interest_tokens)) == 0: print(np.intersect1d(cluster, interest_tokens))

#################



adjacency_matrix = nx.to_scipy_sparse_matrix(subgraph)
sums=adjacency_matrix.sum(axis=1)[:,None]
sums[np.min(sums,axis=1)==0]=0.00001
adjacency_matrix_normed=adjacency_matrix/sums


Y=1-pdist(adjacency_matrix_normed, 'cosine')
sim_matrix=squareform(Y)


sim_matrix=np.zeros([len(adjacency_matrix),len(adjacency_matrix)])
for u,v in itertools.combinations(range(0,len(adjacency_matrix)),r=2):
    row_u=adjacency_matrix_normed[u,:]
    row_v=adjacency_matrix_normed[v,:]
    if not np.all(row_u==0) and not np.all(row_v==0):
        distance=cosine(adjacency_matrix_normed[u,:],adjacency_matrix_normed[v,:])
    else:
        distance=1
    sim_matrix[u,v]=1-distance
    sim_matrix[v,u] = 1-distance

a=nx.from_numpy_matrix(sim_matrix)
mapping = dict(zip(range(0, len(tokens)), tokens))
a = nx.relabel_nodes(a, mapping)

k = 90
comp = nx.algorithms.community.centrality.girvan_newman(ego_sym)
limited = itertools.takewhile(lambda c: len(c) <= k, comp)
for communities in limited:
    print(tuple(sorted(c) for c in communities))


clusterer = hdbscan.HDBSCAN(min_cluster_size=2, min_samples=1, metric='l1', prediction_data=True).fit(
    sim_matrix)
labels=clusterer.labels_
print(labels)
for cluster in (np.unique(labels)):
    if not cluster==-1: print(tokens[labels==cluster])


# %%
csv=pd.read_csv("E:/NLP/cluster_xls/centralities/Book4.csv",delimiter=";")
tokens=csv.iloc[:-1,0].to_numpy()
tokens=[np.str(x) for x in tokens]
years=pd.to_numeric(csv.columns[1:]).to_numpy()
data=csv.iloc[0:-1,1:]
long_data=data.transpose()
long_data.apply(pd.to_numeric)
np_data=long_data.to_numpy()


#clusterer = KMeans(n_clusters=nr_clusters).fit(trunc_dist)
# #cluster_selection_method="leaf",
clusterer = hdbscan.HDBSCAN(min_cluster_size=3, min_samples=2, metric='l2', prediction_data=True).fit(
    np_data)
labels=clusterer.labels_
print(labels)
means=[]
for cluster in np.unique(clusterer.labels_):
    vec=np_data[labels==cluster,:]
    avg_vec=np.mean(vec,axis=0)
    years_vec=years[labels==cluster]
    means.append(pd.Series(avg_vec,index=tokens))

mean_frame=pd.DataFrame(means)
year_frame=pd.DataFrame(labels,index=years)

mean_frame.to_csv("E:/NLP/cluster_xls/centralities/mean_frame.csv")
year_frame.to_csv("E:/NLP/cluster_xls/centralities/year_frame.csv")





