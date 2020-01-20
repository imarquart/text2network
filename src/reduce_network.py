import logging

import networkx as nx
from networkx.readwrite.gexf import read_gexf, write_gexf
from tqdm import tqdm

from NLP.utils.network_tools import graph_merge, load_graph, plural_elimination, make_symmetric, save_graph_year, save_graph
from NLP.utils.rowvec_tools import prune_network_edges


def save_merged_ego_graph(years, focal_token, cfg, save_folder, sum_folder, ego_radius=2, links="avg",
                          network_type="Rgraph-Sum-Rev"):
    # Assorted Lists and dicts
    ego_graphs = []  # Dict for year:ego graphs
    logging.info("Loading graph data.")
    # First we load and prepare the networks.
    for year in years:
        graph = load_graph(year, cfg, sum_folder, network_type)
        # Do some pruning here but keep weights intact
        graph = prune_network_edges(graph, edge_weight=0)
        ego_graph = nx.generators.ego.ego_graph(graph, focal_token, radius=ego_radius, center=True,
                                                undirected=False)
        ego_graphs.append(ego_graph.copy())
        del graph

    logging.info("Merging %i graphs" % len(ego_graphs))
    if links=="avg" or links=="both":
        ego_graph = graph_merge(ego_graphs, average_links=True, method=None, merge_mode="safe")
        ego_graph = prune_network_edges(ego_graph, 0)
        save_graph(ego_graph, cfg, save_folder, ''.join([network_type, '_', "avg", '_r', str(ego_radius)]))


    if links == "sum" or links == "both":
        ego_graph = graph_merge(ego_graphs, average_links=False, method=None, merge_mode="safe")
        ego_graph = prune_network_edges(ego_graph, 0)
        save_graph(ego_graph, cfg, save_folder, ''.join([network_type, '_', "sum", '_r', str(ego_radius)]))
    logging.info("Merging %i graphs complete" % len(ego_graphs))

    del ego_graphs


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
        network_file = ''.join([data_folder, cfg.ma_folder, '/2', network_type,'_order',str(ma_order), '.gexf'])
        # Prune only nodes and zero edges
        merged_graph = prune_network_edges(merged_graph, 0)
        nx.write_gexf(merged_graph,network_file)


def min_symmetric_network(year,cfg,folder,save_folder,network_type="Rgraph-Sum-Rev",method="min-sym-avg"):
    graph = load_graph(year, cfg, folder, network_type)
    graph=make_symmetric(graph,method)
    name=''.join([network_type,'_',method])
    save_graph_year(graph,year,cfg,save_folder,name)



def reduce_network(network_file,cfg,reverse=True,method="sum",save_folder=None, plural_elim=False):

    network=read_gexf(network_file)
    logging.info("File loaded. Pruning.")
    # Prune only nodes and zero edges
    network = prune_network_edges(network,0)
    if reverse==True:
        logging.info("Reversing graph.")
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

