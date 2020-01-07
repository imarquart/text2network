import community
import networkx as nx
import numpy as np
import scipy as scp
import itertools
import logging
from tqdm import tqdm
from NLP.config.config import configuration
from networkx.algorithms.community.centrality import girvan_newman
import hdbscan
from NLP.utils.rowvec_tools import make_symmetric, prune_network_edges, inverse_edge_weight


def cut_list(cluster, node_info, nr_retain):
    proximities = [node_info[v][0] for v in cluster]
    sortkey = np.argsort(-np.array(proximities))
    cluster = np.array(cluster)
    cluster = cluster[sortkey]
    return cluster[:nr_retain]


def louvain_cluster(graph):
    graph = make_symmetric(graph)
    clustering = community.best_partition(graph)

    return [[k for k, v in clustering.items() if v == val] for val in list(set(clustering.values()))]


# TODO:
def hdbclustering(graph):
    A = nx.to_scipy_sparse_matrix(graph)
    nodes_list = list(graph.nodes)


def load_graph(year, cfg, sum_folder, network_type):
    data_folder = ''.join([cfg.data_folder, '/', str(year)])
    network_file = ''.join([data_folder, sum_folder, '/', network_type, '.gexf'])
    return nx.read_gexf(network_file)


def dynamic_clustering(years, focal_token, cfg, cluster_function, sum_folder, window=3, num_retain=15, ego_radius=2,
                       network_type="Rgraph-Sum", pruning_method="majority", cluster_levels=3):
    """
    This runs a dynamic cluster across years and smoothes the clusters within each year
    by running a consensus algorithm across a timed window.

    Limitations: Graphs need to the same nodes, thus graphs are intersected toward common nodes

    :param years: years range
    :param focal_token: string of focal token
    :param cfg: config class
    :param cluster_function: cluster function to use
    :param window: total window, odd number
    :param ego_radius: radius of ego network to consider
    :param network_type: filename of gexf file
    :param pruning_method: Pruning method for consensus clustering
    :return:
    """
    # Assorted Lists and dicts
    ego_graphs = {}  # Dict for year:ego graphs
    cluster_graphs = {}  # Dict for year:cluster graphs
    ego_nodes = []
    graph_nodes = []
    graphs = {}  # Original graphs
    logging.info("Loading graph data.")
    logging.info("Pruning links smaller than %f" % cfg.prune_min)
    # First we load and prepare the networks. Because these are small compared to RAM, we load them all
    for year in years:
        graph = load_graph(year, cfg, sum_folder, network_type)
        # Do some pruning here.
        graph = prune_network_edges(graph, edge_weight=cfg.prune_min)
        # Keep only one graph in memory
        # graphs.update({year: graph})
        # Create a set that is the intersection of all nodes. We need to keep the adjacency matrix consistent
        # so nodes should all be the same!
        if year == years[0]:
            graph_nodes = list(graph.nodes)
        else:
            graph_nodes = np.intersect1d(graph_nodes, list(graph.nodes))
    del graph

    logging.info("Intersecting graphs, creating interest list.")
    # We now take all subgraphs of the intersection, and we also create the ego networks
    for year in years:
        graph = load_graph(year, cfg, sum_folder, network_type)
        graph = nx.subgraph(graph, graph_nodes)
        # Create ego network for given year
        # This will not be the final ego network, we only use it to get a
        # union of nodes we are interested in
        # Reason: Want same network for all years
        ego_graph = nx.generators.ego.ego_graph(graph, focal_token, radius=ego_radius, center=True, undirected=False)
        ego_nodes = np.union1d(ego_nodes, list(ego_graph.nodes))
    del graph, ego_graph

    # The nodes we will track throughout all years
    ego_nodes = ego_nodes.flatten()

    logging.info("Creating ego graphs and cluster graphs.")
    for year in years:
        # Create actual ego network from nodes of interest
        # Use copy so it is not just a view of the graph object
        graph = load_graph(year, cfg, sum_folder, network_type)
        ego_graph = nx.subgraph(graph, ego_nodes).copy()
        ego_graphs.update({year: ego_graph})
        del graph

        # The cluster function should return a list of lists, where each top-level list is a cluster
        cluster_it = cluster_function(ego_graph)
        # Create the cluster graph
        # Nodes have a link, if they are in the same cluster
        cluster_graph = nx.Graph()
        cluster_graph.add_nodes_from(ego_nodes)
        for cluster in cluster_it:
            pairs = itertools.combinations(cluster, r=2)
            cluster_graph.add_edges_from(list(pairs))

        cluster_graphs.update({year: cluster_graph})

    # These variables will hold our return values for the function
    year_info = {}
    central_graphs = {}

    # This is the actual loop to create the clustering via consensus method
    for t, year in enumerate(years):
        logging.info("Consensus clustering for year %i." % year)
        # Overflowing time windows will just be cut off
        lb = year - years[0]
        ub = years[-1] - year
        lb = int(max(years[0], year - (window - 1) / 2))
        ub = int(min(years[-1], year + (window - 1) / 2))
        # Range +1 because python counts at 0
        floating_range = range(lb, ub + 1)

        # When using adjacency matrices, we lose the node labels
        # Preserve this in a list
        # (NetworkX otherwise messes up the order!)
        nodes_list = list(cluster_graphs[year].nodes)

        # For all years in the floating range, add together the cluster graphs
        for i in floating_range:
            if i == floating_range[0]:
                A = nx.to_scipy_sparse_matrix(cluster_graphs[i])
            else:
                A = A + nx.to_scipy_sparse_matrix(cluster_graphs[i])
        # Average
        A = A / len(floating_range)
        # TODO: Add majority etc.
        # "Majority" voting. Note that this is bad for a sparse matrix.
        # It also creates zeros.
        A[A < 0.5] = 0
        A.eliminate_zeros()
        # Reconvert to graphs, and apply labels again
        consensus_graph = nx.convert_matrix.from_scipy_sparse_matrix(A)
        mapping = dict(zip(range(0, len(nodes_list)), nodes_list))
        consensus_graph = nx.relabel_nodes(consensus_graph, mapping)
        # Now cluster this consensus graph to derive the final clusters
        cluster_it = cluster_function(consensus_graph)

        node_infos = []
        c_data = []
        # Given final clusters, we compute centralities via page_rank
        egocentralities = nx.pagerank_numpy(ego_graphs[year], weight='weight')

        for cluster in tqdm(cluster_it):
            # Single node clusters don't work for any measure, skip
            if len(cluster) >= 2:
                node_info = {}
                cluster_info = {}
                # %% Get Node Info
                # We first want to create a dictionary of nodes with their
                # correponding centralities and proximities to the focal node
                f_cluster = cluster.append(focal_token)
                cluster_subgraph = nx.subgraph(ego_graphs[year], f_cluster)
                centralities = nx.pagerank_numpy(cluster_subgraph, weight='weight')
                for v in cluster:
                    try:
                        distance = nx.dijkstra_path_length(cluster_subgraph, focal_token, v,
                                                           weight=inverse_edge_weight)
                    except:
                        distance = 0

                    if cluster_subgraph.has_edge(focal_token, v):
                        # Proximity
                        try:
                            proximity = cluster_subgraph[focal_token][v]['weight']
                        except:
                            proximity = 0

                        # Constraint
                        try:
                            coTo = nx.local_constraint(cluster_subgraph, v, focal_token, weight='weight')
                            coFrom = nx.local_constraint(cluster_subgraph, focal_token, v, weight='weight')

                        except:
                            coTo = 0
                            coFrom = 0

                    else:
                        proximity = 0
                        coTo = 0
                        coFrom = 0

                    centrality = centralities[v]
                    egocentrality = egocentralities[v]
                    node_info.update({v: (proximity, distance, centrality, coTo, coFrom, egocentrality)})
                node_infos.append(node_info)

                # %% Clustering
                cluster_nest = []
                cluster_nest_cut = []
                cluster_nest.append([cluster])
                cluster_nest_cut.append([cut_list(cluster, node_info, num_retain)])

                for i in range(1, cluster_levels):
                    n_list = []
                    n_list_cut = []
                    for n_cluster in cluster_nest[i - 1]:
                        cluster_subgraph = nx.subgraph(ego_graphs[year], n_cluster)
                        n_cluster_it = cluster_function(cluster_subgraph)
                        for c in n_cluster_it:
                            n_list.append(c)
                            n_list_cut.append(cut_list(c, node_info, num_retain))
                    cluster_nest.append(n_list)
                    cluster_nest_cut.append(n_list_cut)

                cluster_info.update({'nest': cluster_nest_cut})
                cluster_info.update({'info': node_info})
                c_data.append(cluster_info)

        # Save clusters etc
        year_info.update({year: c_data})

    return year_info
