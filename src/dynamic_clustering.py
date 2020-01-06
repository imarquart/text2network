import community
import networkx as nx
import numpy as np
import scipy as scp
import itertools
import logging
from NLP.config.config import configuration
from networkx.algorithms.community.centrality import girvan_newman
import hdbscan
from NLP.utils.rowvec_tools import make_symmetric, prune_network_edges, inverse_edge_weight



def louvain_cluster(graph):
    graph = make_symmetric(graph)
    clustering = community.best_partition(graph)

    return [[k for k, v in clustering.items() if v == val] for val in list(set(clustering.values()))]

# TODO:
def hdbclustering(graph):
    A = nx.to_scipy_sparse_matrix(graph)
    nodes_list = list(graph.nodes)


def dynamic_clustering(years, focal_token, cfg, cluster_function, window=3, num_retain=15, ego_radius=2,
                       network_type="Rgraph-Sum", pruning_method="majority"):
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
    ego_graphs = {} # Dict for year:ego graphs
    cluster_graphs = {} # Dict for year:cluster graphs
    ego_nodes = []
    graph_nodes = []
    graphs = {} # Original graphs
    logging.info("Loading graph data.")
    logging.info("Pruning links smaller than %f" % cfg.prune_min)
    # First we load and prepare the networks. Because these are small compared to RAM, we load them all
    for year in years:
        data_folder = ''.join([cfg.data_folder, '/', str(year)])
        network_file = ''.join([data_folder, '/networks/sums/', network_type, '.gexf'])
        graph = nx.read_gexf(network_file)
        # Do some pruning here.
        graph = prune_network_edges(graph, edge_weight=cfg.prune_min)
        graphs.update({year: graph})
        # Create a set that is the intersection of all nodes. We need to keep the adjacency matrix consistent
        # so nodes should all be the same!
        if year == years[0]:
            graph_nodes = list(graph.nodes)
        else:
            graph_nodes = np.intersect1d(graph_nodes, list(graph.nodes))

    logging.info("Intersecting graphs.")
    # We now take all subgraphs of the intersection, and we also create the ego networks
    for year in years:
        graph = nx.subgraph(graphs[year], graph_nodes)
        graphs.update({year: graph})
        # Create ego network for given year
        # This will not be the final ego network, we only use it to get a
        # union of nodes we are interested in
        # Reason: Want same network for all years
        ego_graph = nx.generators.ego.ego_graph(graph, focal_token, radius=ego_radius, center=True, undirected=False)
        ego_nodes = np.union1d(ego_nodes, list(ego_graph.nodes))

    # The nodes we will track throughout all years
    ego_nodes = ego_nodes.flatten()

    logging.info("Creating ego graphs and cluster graphs.")
    for year in years:
        # Create actual ego network from nodes of interest
        # Use copy so it is not just a view of the graph object
        ego_graph = nx.subgraph(graphs[year], ego_nodes).copy()
        ego_graphs.update({year: ego_graph})

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
        cluster_info = []
        # Given final clusters, we compute centralities via page_rank
        # Add other values to characterize nodes in cluster here
        for cluster in cluster_it:
            # Single node clusters don't work for any measure, skip
            if len(cluster) >= 2:
                # The following simply builds a dictionary of centralities
                # of nodes within their cluster subnetwork
                cluster_subgraph = nx.subgraph(ego_graphs[year], cluster).copy()
                # centralities=nx.eigenvector_centrality(cluster_subgraph)
                centralities = nx.pagerank_numpy(cluster_subgraph, weight='weight')
                # centralities_weighted=nx.eigenvector_centrality(cluster_subgraph,weight='weight')
                centralities_k = [k for k, v in sorted(centralities.items(), key=lambda item: item[1])]
                centralities_v = [v for k, v in sorted(centralities.items(), key=lambda item: item[1])]
                centralities_k = centralities_k[-num_retain:]
                centralities_v = centralities_v[-num_retain:]
                cent_dict = dict(zip(centralities_k, centralities_v))
                cluster_info.append(cent_dict)

        # %% Graph clusters (shitty version)
        # Get the most central node for each cluster, and add the focal token
        central_nodes = [list(c.keys())[-1] for c in cluster_info]
        central_nodes.append(focal_token)
        # Get all directed pairs of these focal tokens
        pairs = list(itertools.permutations(central_nodes, r=2))
        # We will create a new graph which only includes the cluster "labels"
        central_graph = nx.DiGraph()
        central_graph.add_nodes_from(central_nodes)
        # The connection between clusters is for now simply the distance between most central nodes, if present
        for pair in pairs:
            try:
                edge_weight = nx.dijkstra_path_length(ego_graphs[year], pair[0], pair[1], weight=inverse_edge_weight)
                central_graph.add_edge(pair[0], pair[1], weight=edge_weight)
            except:
                continue

        # Save clusters etc
        central_graphs.update({year: central_graph})
        year_info.update({year: cluster_info})

    return central_graphs, year_info
