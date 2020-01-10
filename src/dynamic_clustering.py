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
from NLP.utils.network_tools import load_graph, graph_merge


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


def overall_clustering(years, focal_token, cfg, cluster_function, sum_folder, num_retain=15, ego_radius=2,
                       network_type="Rgraph-Sum", cluster_levels=3):
    # Assorted Lists and dicts
    ego_graphs = []  # Dict for year:ego graphs
    logging.info("Loading graph data.")
    logging.info("Pruning links smaller than %f" % cfg.prune_min)
    # First we load and prepare the networks.
    for year in years:
        graph = load_graph(year, cfg, sum_folder, network_type)
        # Do some pruning here.
        graph = prune_network_edges(graph, edge_weight=cfg.prune_min)
        ego_graph = nx.generators.ego.ego_graph(graph, focal_token, radius=ego_radius, center=True, undirected=False)
        ego_graphs.append(ego_graph.copy())

    logging.info("Merging %i graphs" % len(ego_graphs))
    ego_graph = graph_merge(ego_graphs, average_links=True, method=None, merge_mode="safe")
    logging.info("Merging %i graphs complete" % len(ego_graphs))
    ego_graph = prune_network_edges(ego_graph, cfg.prune_min)
    del ego_graphs

    # %% Clustering
    cluster_it = cluster_function(ego_graph)

    # These lists will store the data over the years
    node_infos = []
    c_data = []

    for cluster in tqdm(cluster_it, desc="Top Clusters", leave=False, position=0):
        # Single node clusters don't work for any measure, skip
        if len(cluster) >= 2:
            # This works in two parts. First, we derive the measures for each
            # node within the first cluster, saving them as node info
            # Later, we will use hierarchical clustering
            node_info = {}
            cluster_info = {}
            # %% Step 1: Get Node Info
            # We first want to create a dictionary of nodes with their
            # correponding centralities and proximities to the focal node
            # Need to add the focal node for this
            f_cluster = cluster.append(focal_token)
            cluster_subgraph = nx.subgraph(ego_graph, f_cluster).copy()
            centralities = nx.pagerank_numpy(cluster_subgraph, weight='weight')
            indeg = nx.in_degree_centrality(cluster_subgraph)
            outdeg = nx.out_degree_centrality(cluster_subgraph)
            try:
                egocentralities = nx.betweenness_centrality(cluster_subgraph,
                                                            k=int(np.log(len(list(ego_graphs[year].nodes)))))
                # centralities = {0: 0}
            except:
                logging.info("No success with betweeness centrality")
                egocentralities = {0: 0}

            for v in tqdm(cluster, desc="Single Node Measures", leave=False, position=0):
                try:
                    distance = max(1000, nx.dijkstra_path_length(cluster_subgraph, focal_token, v,
                                                                 weight=inverse_edge_weight))
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
                indegree = indeg[v]
                outdegree = outdeg[v]
                if v in egocentralities:
                    egocentrality = egocentralities[v]
                else:
                    egocentrality = 0
                node_info.update(
                    {v: (proximity, distance, centrality, coTo, coFrom, egocentrality, indegree, outdegree)})

            # Each year and each cluster has associated one of these dicts with values for all
            # Nodes of interest in the ego network
            node_infos.append(node_info)

            # %% Step 2 Clustering
            # We use two lists. The first keeps all nodes, the second only retains the top n
            # Nodes in terms of proximity to the focal node
            cluster_nest = []
            cluster_nest_cut = []
            # The first cluster is already given by consensus
            cluster_nest.append([cluster])
            # We use the function cut_list and the node_info dict generated earlier
            cluster_nest_cut.append([cut_list(cluster, node_info, num_retain)])

            # We can cluster hierarchically with a loop, because each level is a list of lists of tokens
            # Structure is thus:
            # Level -> Cluster -> Token
            for i in tqdm(range(1, cluster_levels), desc="Sub Clustering", leave=False, position=0):
                n_list = []
                n_list_cut = []
                # Take all clusters from previous level, do clustering on these
                for n_cluster in cluster_nest[i - 1]:
                    cluster_subgraph = nx.subgraph(ego_graph, n_cluster)
                    n_cluster_it = cluster_function(cluster_subgraph)

                    # For each new cluster, add the elements (lists) to the current level list
                    for c in n_cluster_it:
                        n_list.append(c)
                        n_list_cut.append(cut_list(c, node_info, num_retain))

                # Append current level list to the list of all levels
                cluster_nest.append(n_list)
                cluster_nest_cut.append(n_list_cut)

            # We create a dict of nested lists and node info for this year
            cluster_info.update({'nest': cluster_nest_cut})
            cluster_info.update({'info': node_info})
            # Add this to the overall (yearly) data list
            c_data.append(cluster_info)

    return c_data


def dynamic_clustering(years, focal_token, cfg, cluster_function, sum_folder, window=3, num_retain=15, ego_radius=2,
                       network_type="Rgraph-Sum", pruning_method="majority", cluster_levels=3):
    """
    This runs a dynamic cluster across years and smoothes the clusters within each year
    by running a consensus algorithm across a timed window.

    Then, each cluster is clustered again hierarchically.
    Measures are computed for each node based on the first cluster assignment!

    Limitations: Graphs need to the same nodes, thus graphs are intersected toward common nodes

    :param years: years range
    :param focal_token: string of focal token
    :param cfg: config class
    :param cluster_function: cluster function to use
    :param window: total window, odd number
    :param ego_radius: radius of ego network to consider
    :param network_type: filename of gexf file
    :param pruning_method: Pruning method for consensus clustering
    :param cluster_levels: How deep to cluster hierarchically
    :return:
    """
    # Assorted Lists and dicts
    ego_graphs = {}  # Dict for year:ego graphs
    cluster_graphs = {}  # Dict for year:cluster graphs
    ego_nodes = []
    graph_nodes = []
    logging.info("Loading graph data.")
    logging.info("Pruning links smaller than %f" % cfg.prune_min)
    # First we load and prepare the networks.
    for year in years:
        graph = load_graph(year, cfg, sum_folder, network_type)
        # Do some pruning here.
        graph = prune_network_edges(graph, edge_weight=cfg.prune_min)
        # Create a set that is the intersection of all nodes. We need to keep the adjacency matrix consistent
        # so nodes should all be the same!
        if year == years[0]:
            graph_nodes = list(graph.nodes)
        else:
            graph_nodes = np.intersect1d(graph_nodes, list(graph.nodes))
    # Due to memory sizes, try to be vigilant here
    del graph

    logging.info("Intersecting graphs, creating interest list.")
    # We now take all subgraphs of the intersection, and we also create the ego networks
    for year in years:
        graph = load_graph(year, cfg, sum_folder, network_type)
        graph = nx.subgraph(graph, graph_nodes).copy()
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
    for year in tqdm(years, desc="Cluster Graph Creation", leave=False, position=0):
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
    for t, year in enumerate(tqdm(years, desc="Year Consensus Clustering", leave=False, position=0)):
        # Overflowing time windows will just be cut off
        lb = int(max(years[0], year - (window - 1) / 2))
        ub = int(min(years[-1], year + (window - 1) / 2))
        # Range +1 because python counts at 0
        floating_range = range(lb, ub + 1)
        graph_list = [cluster_graphs[i] for i in floating_range]
        consensus_graph = graph_merge(graph_list, average_links=True, method="majority")

        # Now cluster this consensus graph to derive the final clusters
        cluster_it = cluster_function(consensus_graph)

        # These lists will store the data over the years
        node_infos = []
        c_data = []
        # Compute once all centralities in the ego network
        # egocentralities = nx.pagerank_numpy(ego_graphs[year], weight='weight')

        for cluster in tqdm(cluster_it, desc="Top Clusters", leave=False, position=0):
            # Single node clusters don't work for any measure, skip
            if len(cluster) >= 2:
                # This works in two parts. First, we derive the measures for each
                # node within the first cluster, saving them as node info
                # Later, we will use hierarchical clustering
                node_info = {}
                cluster_info = {}
                # %% Step 1: Get Node Info
                # We first want to create a dictionary of nodes with their
                # correponding centralities and proximities to the focal node
                # Need to add the focal node for this
                f_cluster = cluster.append(focal_token)
                cluster_subgraph = nx.subgraph(ego_graphs[year], f_cluster).copy()
                centralities = nx.pagerank_numpy(cluster_subgraph, weight='weight')
                indeg = nx.in_degree_centrality(cluster_subgraph)
                outdeg = nx.out_degree_centrality(cluster_subgraph)
                try:
                    egocentralities = nx.betweenness_centrality(cluster_subgraph,
                                                                k=int(np.log(len(list(ego_graphs[year].nodes)))))
                    # centralities = {0: 0}
                except:
                    logging.info("No success with betweeness centrality")
                    egocentralities = {0: 0}

                for v in tqdm(cluster, desc="Single Node Measures", leave=False, position=0):
                    try:
                        distance = max(1000, nx.dijkstra_path_length(cluster_subgraph, focal_token, v,
                                                                     weight=inverse_edge_weight))
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
                    indegree = indeg[v]
                    outdegree = outdeg[v]
                    if v in egocentralities:
                        egocentrality = egocentralities[v]
                    else:
                        egocentrality = 0
                    node_info.update(
                        {v: (proximity, distance, centrality, coTo, coFrom, egocentrality, indegree, outdegree)})

                # Each year and each cluster has associated one of these dicts with values for all
                # Nodes of interest in the ego network
                node_infos.append(node_info)

                # %% Step 2 Clustering
                # We use two lists. The first keeps all nodes, the second only retains the top n
                # Nodes in terms of proximity to the focal node
                cluster_nest = []
                cluster_nest_cut = []
                # The first cluster is already given by consensus
                cluster_nest.append([cluster])
                # We use the function cut_list and the node_info dict generated earlier
                cluster_nest_cut.append([cut_list(cluster, node_info, num_retain)])

                # We can cluster hierarchically with a loop, because each level is a list of lists of tokens
                # Structure is thus:
                # Level -> Cluster -> Token
                for i in tqdm(range(1, cluster_levels), desc="Sub Clustering", leave=False, position=0):
                    n_list = []
                    n_list_cut = []
                    # Take all clusters from previous level, do clustering on these
                    for n_cluster in cluster_nest[i - 1]:
                        cluster_subgraph = nx.subgraph(ego_graphs[year], n_cluster)
                        n_cluster_it = cluster_function(cluster_subgraph)

                        # For each new cluster, add the elements (lists) to the current level list
                        for c in n_cluster_it:
                            n_list.append(c)
                            n_list_cut.append(cut_list(c, node_info, num_retain))

                    # Append current level list to the list of all levels
                    cluster_nest.append(n_list)
                    cluster_nest_cut.append(n_list_cut)

                # We create a dict of nested lists and node info for this year
                cluster_info.update({'nest': cluster_nest_cut})
                cluster_info.update({'info': node_info})
                # Add this to the overall (yearly) data list
                c_data.append(cluster_info)

        # Now we want to add this in form of a dict
        year_info.update({year: c_data})

    return year_info
