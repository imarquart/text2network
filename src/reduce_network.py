import networkx as nx
import scipy as sp
from networkx.readwrite.gexf import read_gexf,write_gexf
from networkx.readwrite.gml import write_gml
from NLP.utils.network_tools import graph_merge, load_graph, plural_elimination
from NLP.utils.rowvec_tools import prune_network_edges
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

def moving_avg_networks(years,cfg,ma_order=3,network_type="Rgraph-Sum",average_links=True,load_all=False):
    graphs={}
    graph_nodes = []


    if load_all==True:
        logging.info("Loading all graphs in memory")
        for year in years:
           # data_folder = ''.join([cfg.data_folder, '/', str(year)])
            #network_file = ''.join([data_folder, cfg.sums_folder,'/', network_type, '.gexf'])
            graph = load_graph(year, cfg,cfg.sums_folder,network_type)
            graphs.update({year: graph})

    for year in tqdm(years, desc="Year MA calculation", leave=False, position=0):
        lb=max(years[0],year-ma_order+1)
        graph_range=range(lb,year+1)
        merge_list=[]
        for i in graph_range:
            if load_all==False:
                merge_list.append(load_graph(i, cfg,cfg.sums_folder,network_type))
            else:
                merge_list.append(graphs[i])
        # Merge
        merged_graph=graph_merge(merge_list,average_links=average_links, merge_mode="safe")
        # Save yearly graph
        data_folder = ''.join([cfg.data_folder, '/', str(year)])
        network_file = ''.join([data_folder, cfg.ma_folder, '/', network_type,'_order',str(ma_order), '.gexf'])
        # Prune only nodes and zero edges
        merged_graph = prune_network_edges(merged_graph, 0)
        nx.write_gexf(merged_graph,network_file)


def reduce_network(network_file,cfg,reverse=True,method="sum",save_folder=None, plural_elim=False):

    network=read_gexf(network_file)
    # Prune only nodes and zero edges
    network = prune_network_edges(network,0)
    if reverse==True:
        network=network.reverse()

    if method=="sum":
        nodes=list(network.nodes)
        A = nx.to_scipy_sparse_matrix(network)
        del network
        graph=nx.convert_matrix.from_scipy_sparse_matrix(A, create_using=nx.DiGraph)
        token_map = {v: nodes[v] for v in graph.nodes}
        graph = nx.relabel_nodes(graph, token_map)

    if plural_elim==True:
        logging.info("Eliminating plural links.")
        graph=plural_elimination(graph,method=cfg.plural_method)

    if save_folder is not None:
        logging.info("Saving graph to %s" % save_folder)
        network = write_gexf(graph, save_folder)

    return graph

