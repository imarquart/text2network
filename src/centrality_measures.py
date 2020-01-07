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
import time


def dynamic_centralities(years, focal_token, cfg, num_retain=15,
                       network_type="Rgraph-Sum"):
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

    ego_graphs = {}
    cluster_graphs = {}
    ego_nodes = []
    graph_nodes = []
    graphs = {}
    neighbor_set=[]
    weight_set=[]
    logging.info("Loading graph data.")
    logging.info("Pruning links smaller than %f" % cfg.prune_min)
    for year in years:
        data_folder = ''.join([cfg.data_folder, '/', str(year)])
        network_file = ''.join([data_folder, '/networks/sums/', network_type, '.gexf'])
        graph = nx.read_gexf(network_file)
        graph = prune_network_edges(graph, edge_weight=cfg.prune_min)
        # To save memory, I have changed this such that only one graph is kept in memory
        #graphs.update({year: graph})
        logging.info("Creating list of most connected terms.")
        neighbors = graph[focal_token]
        edge_weights = [x['weight'] for x in neighbors.values()]
        edge_sort = np.argsort(-np.array(edge_weights))
        neighbors = [x for x in neighbors]
        neighbor_set.extend(neighbors)
        weight_set.extend(edge_weights)


    logging.info("Creating interest set")
    weight_set=np.array(weight_set)
    neighbor_set=np.array(neighbor_set)
    edge_sort = np.argsort(-weight_set)
    neighbor_set=neighbor_set[edge_sort]
    weight_set=weight_set[edge_sort]
    interest_set={}
    counter=0
    for i in range(0,len(neighbor_set)):
        if neighbor_set[i] not in interest_set.keys():
            interest_set.update({neighbor_set[i]:weight_set[i]})
            counter=counter+1
        if counter >= num_retain:
            break

    interest_nodes=list(interest_set.keys())
    if focal_token not in interest_nodes: interest_nodes.append(focal_token)
    if "man" not in interest_nodes: interest_nodes.append("man")
    if "men" not in interest_nodes: interest_nodes.append("men")
    if "male" not in interest_nodes: interest_nodes.append("male")
    if "female" not in interest_nodes: interest_nodes.append("female")
    if "women" not in interest_nodes: interest_nodes.append("women")
    if "woman" not in interest_nodes: interest_nodes.append("woman")

    year_info = {}
    central_graphs = {}

    for t, year in enumerate(years):
        measures={}
        logging.info("Centrality calculation for year %i." % year)
        data_folder = ''.join([cfg.data_folder, '/', str(year)])
        network_file = ''.join([data_folder, '/networks/sums/', network_type, '.gexf'])
        graph = nx.read_gexf(network_file)
        graph = prune_network_edges(graph, edge_weight=cfg.prune_min)
        ego_graph = nx.ego_graph(graph, focal_token, cfg.ego_radius)


        #%% Closeness to leader
        start_time = time.time()
        proximity={}
        try:
            centralities=ego_graph[focal_token]
        except:
            centralities = {0: 0}
        for node in interest_nodes:
            if node in centralities.keys():
                proximity.update({node: centralities[node]['weight']})
            else:
                proximity.update({node: 0})

        measures.update({'EgoProximityFocal': proximity})
        logging.info("Proximity time in %s seconds" % (time.time() - start_time))

        # %% Katz
        #start_time = time.time()
        #katz = {}
        #alpha = 1 / (max(nx.adjacency_spectrum(ego_graph))+0.01)
        #try:
        #    centralities = nx.katz_centrality_numpy(ego_graph, alpha, weight='weight')
        #except:
        #    centralities = {0: 0}
        #    logging.info("No success with Katz centrality")  #
        #for node in interest_nodes:
        #    if node in centralities.keys():
        #        katz.update({node: centralities[node]})
         #   else:
         #       katz.update({node: 0})
        #measures.update({'Katz': katz})
        #logging.info("Katz Centrality time in %s seconds" % (time.time() - start_time))

        # %% Constraint
        start_time = time.time()

        constraint = {}
        for node in interest_nodes:
            if node in list(ego_graph.nodes):
                try:
                    co=nx.local_constraint(ego_graph, focal_token, node, weight='weight')
                except:
                    co=0
                    logging.info("No success with Constraint centrality, node %s" % node)
                constraint.update({node: co})
            else:
                constraint.update({node: 0})

        measures.update({'EgoFocalConstrBy': constraint})
        logging.info("Constraint time in %s seconds" % (time.time() - start_time))
        # %% Constraint

        start_time = time.time()

        constraint_to = {}
        for node in interest_nodes:
            if node in list(ego_graph.nodes):
                try:
                    co = nx.local_constraint(ego_graph, node , focal_token, weight='weight')
                except:
                    co=0
                    logging.info("No success with Constraint-Towards centrality, node %s" % node)
                constraint.update({node: co})
            else:
                constraint.update({node: 0})

        measures.update({'EgoFocalConstraining': constraint})
        logging.info("Constraint time in %s seconds" % (time.time() - start_time))

        #%% Page Rank Weighted
        start_time = time.time()
        pageranks={}
        try:
            centralities = nx.pagerank_scipy(ego_graph, weight='weight')
        except:
            centralities = {0:0}

        for node in interest_nodes:
            if node in centralities.keys():
                pageranks.update({node: centralities[node]})
            else:
                pageranks.update({node: 0})

        measures.update({'EgoPageRank-Weighted': pageranks})
        logging.info("PageRank-Weighted time in %s seconds" % (time.time() - start_time))

        # %% Page Rank UnWeighted
        start_time = time.time()
        pageranks = {}
        try:
            centralities = nx.pagerank_scipy(ego_graph,weight=None)
        except:
            centralities = {0: 0}

        for node in interest_nodes:
            if node in centralities.keys():
                pageranks.update({node: centralities[node]})
            else:
                pageranks.update({node: 0})

        measures.update({'EgoPageRank-unweighted': pageranks})
        logging.info("PageRank-unweighted time in %s seconds" % (time.time() - start_time))

        # %% In degree
        start_time = time.time()

        indegree = {}
        try:
            centralities = nx.in_degree_centrality(ego_graph)
        except:
            centralities = {0: 0}

        for node in interest_nodes:
            if node in centralities.keys():
                indegree.update({node: centralities[node]})
            else:
                indegree.update({node: 0})

        measures.update({'EgoInDegree': indegree})
        logging.info("In Degree time in %s seconds" % (time.time() - start_time))

        # %% Out degree
        start_time = time.time()

        outdegree = {}
        try:
            centralities = nx.out_degree_centrality(ego_graph)
        except:
            centralities = {0: 0}

        for node in interest_nodes:
            if node in centralities.keys():
                outdegree.update({node: centralities[node]})
            else:
                outdegree.update({node: 0})

        measures.update({'EgoOutDegree': outdegree})
        logging.info("Out Degree time in %s seconds" % (time.time() - start_time))

        # %% Betweenness
        start_time = time.time()
        sym_between = {}
        sym_graph=make_symmetric(ego_graph, technique="min-sym")
        logging.info("Symmetrized graph in %s seconds" % (time.time() - start_time))
        try:
            centralities = nx.betweenness_centrality(sym_graph,k=int(np.log(len(list(ego_graph.nodes)))))
            #centralities = {0: 0}
        except:
            logging.info("No success with symmetric betweeness centrality")
            centralities = {0: 0}

        for node in interest_nodes:
            if node in centralities.keys():
                sym_between.update({node: centralities[node]})
            else:
                sym_between.update({node: 0})

        measures.update({'EgoSymmetricEstBetweeness': sym_between})
        logging.info("Symmetric Betweenness time in %s seconds" % (time.time() - start_time))

        # %% Betweenness
        start_time = time.time()

        between = {}
        try:
            centralities = nx.betweenness_centrality(ego_graph,k=int(np.log(len(list(ego_graph.nodes)))))
            #centralities = {0: 0}
        except:
            logging.info("No success with symmetric betweeness centrality")
            centralities = {0: 0}

        for node in interest_nodes:
            if node in centralities.keys():
                between.update({node: centralities[node]})
            else:
                between.update({node: 0})

        measures.update({'EgoEstBetweeness': between})
        logging.info("Betweenness time in %s seconds" % (time.time() - start_time))


        # %% Eigenvector
        start_time = time.time()

        eigen = {}
        try:
            centralities = nx.eigenvector_centrality_numpy(ego_graph)
            #centralities = {0: 0}
        except:
            centralities = {0: 0}
            logging.info("No success with eigenvector centrality")

        for node in interest_nodes:
            if node in centralities.keys():
                eigen.update({node: centralities[node]})
            else:
                eigen.update({node: 0})

        measures.update({'EgoEigenvector': eigen})
        logging.info("Eigenvector time in %s seconds" % (time.time() - start_time))

        # %% Closeness
        start_time = time.time()

        close = {}

        for node in interest_nodes:
            if node in list(ego_graph.nodes):
                try:
                    co = nx.closeness_centrality(ego_graph, node)
                except:
                    co = 0
                    logging.info("No success with Closeness centrality, node %s" % node)
                close.update({node: co})
            else:
                close.update({node: 0})

        measures.update({'EgoCloseness': close})
        logging.info("closeness time in %s seconds" % (time.time() - start_time))



        #%% Update year info
        year_info.update({year: measures})

    return year_info
