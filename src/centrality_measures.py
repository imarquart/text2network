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
    logging.info("Loading graph data.")
    logging.info("Pruning links smaller than %f" % cfg.prune_min)
    for year in years:
        data_folder = ''.join([cfg.data_folder, '/', str(year)])
        network_file = ''.join([data_folder, '/networks/sums/', network_type, '.gexf'])
        graph = nx.read_gexf(network_file)
        graph = prune_network_edges(graph, edge_weight=cfg.prune_min)
        graphs.update({year: graph})

    logging.info("Creating list of most connected terms.")
    neighbor_set=[]
    weight_set=[]
    for year in years:
        neighbors = graphs[year][focal_token]
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
    if "male" not in interest_nodes: interest_nodes.append("female")
    if "female" not in interest_nodes: interest_nodes.append("female")
    if "women" not in interest_nodes: interest_nodes.append("woman")
    if "woman" not in interest_nodes: interest_nodes.append("woman")

    year_info = {}
    central_graphs = {}

    for t, year in enumerate(years):
        measures={}
        logging.info("Centrality calculation for year %i." % year)
        graph=graphs[year]

        #%% Closeness to leader
        proximity={}
        try:
            centralities=graph[focal_token]
        except:
            centralities = {0: 0}
        for node in interest_nodes:
            if node in centralities.keys():
                proximity.update({node: centralities[node]})
            else:
                proximity.update({node: 0})

        measures.update({'ProximityFocal': proximity})

        # %% Constraint
        start_time = time.time()

        constraint = {}
        for node in interest_nodes:
            if node in list(graph.nodes):
                try:
                    co=nx.local_constraint(graph, focal_token, node, weight='weight')
                except:
                    co=0
                    logging.info("No success with Constraint centrality, node %s" % node)
                constraint.update({node: co})
            else:
                constraint.update({node: 0})

        measures.update({'FocalConstrBy': constraint})
        logging.info("Constraint time in %s seconds" % (time.time() - start_time))
        # %% Constraint

        start_time = time.time()

        constraint_to = {}
        for node in interest_nodes:
            if node in list(graph.nodes):
                try:
                    co = nx.local_constraint(graph, node , focal_token, weight='weight')
                except:
                    co=0
                    logging.info("No success with Constraint-Towards centrality, node %s" % node)
                constraint.update({node: co})
            else:
                constraint.update({node: 0})

        measures.update({'FocalConstraining': constraint})
        logging.info("Constraint time in %s seconds" % (time.time() - start_time))

        #%% Page Rank Weighted
        start_time = time.time()
        pageranks={}
        try:
            centralities = nx.pagerank_scipy(graph, weight='weight')
        except:
            centralities = {0:0}

        for node in interest_nodes:
            if node in centralities.keys():
                pageranks.update({node: centralities[node]})
            else:
                pageranks.update({node: 0})

        measures.update({'PageRank-Weighted': pageranks})
        logging.info("PageRank-Weighted time in %s seconds" % (time.time() - start_time))

        # %% Page Rank UnWeighted
        start_time = time.time()
        pageranks = {}
        try:
            centralities = nx.pagerank_scipy(graph,weight=None)
        except:
            centralities = {0: 0}

        for node in interest_nodes:
            if node in centralities.keys():
                pageranks.update({node: centralities[node]})
            else:
                pageranks.update({node: 0})

        measures.update({'PageRank-unweighted': pageranks})
        logging.info("PageRank-unweighted time in %s seconds" % (time.time() - start_time))

        # %% In degree
        start_time = time.time()

        indegree = {}
        try:
            centralities = nx.in_degree_centrality(graph)
        except:
            centralities = {0: 0}

        for node in interest_nodes:
            if node in centralities.keys():
                indegree.update({node: centralities[node]})
            else:
                indegree.update({node: 0})

        measures.update({'InDegree': indegree})
        logging.info("In Degree time in %s seconds" % (time.time() - start_time))

        # %% In degree
        start_time = time.time()

        outdegree = {}
        try:
            centralities = nx.out_degree_centrality(graph)
        except:
            centralities = {0: 0}

        for node in interest_nodes:
            if node in centralities.keys():
                outdegree.update({node: centralities[node]})
            else:
                outdegree.update({node: 0})

        measures.update({'OutDegree': outdegree})
        logging.info("Out Degree time in %s seconds" % (time.time() - start_time))

        # %% Betweenness
        start_time = time.time()

        between = {}
        try:
            centralities = nx.betweenness_centrality(graph)
            #centralities = {0: 0}
        except:
            logging.info("No success with betweeness centrality")
            centralities = {0: 0}

        for node in interest_nodes:
            if node in centralities.keys():
                between.update({node: centralities[node]})
            else:
                between.update({node: 0})

        measures.update({'Betweeness': between})
        logging.info("Betweenness time in %s seconds" % (time.time() - start_time))

        # %% Betweenness
        start_time = time.time()

        sym_between = {}
        sym_graph=make_symmetric(graph, technique="min-sym")
        try:
            centralities = nx.betweenness_centrality(sym_graph)
            #centralities = {0: 0}
        except:
            logging.info("No success with symmetric betweeness centrality")
            centralities = {0: 0}

        for node in interest_nodes:
            if node in centralities.keys():
                between.update({node: centralities[node]})
            else:
                between.update({node: 0})

        measures.update({'SymBetweeness': between})
        logging.info("Symmetric Betweenness time in %s seconds" % (time.time() - start_time))

        # %% Eigenvector
        start_time = time.time()

        eigen = {}
        try:
            centralities = nx.eigenvector_centrality_numpy(graph)
            #centralities = {0: 0}
        except:
            centralities = {0: 0}
            logging.info("No success with eigenvector centrality")

        for node in interest_nodes:
            if node in centralities.keys():
                eigen.update({node: centralities[node]})
            else:
                eigen.update({node: 0})

        measures.update({'Eigenvector': eigen})
        logging.info("Eigenvector time in %s seconds" % (time.time() - start_time))

        # %% Closeness
        start_time = time.time()

        close = {}
        try:
            centralities = nx.closeness_centrality(graph)
            #centralities = {0: 0}
        except:
            centralities = {0: 0}
            logging.info("No success with Closeness centrality")

        for node in interest_nodes:
            if node in centralities.keys():
                close.update({node: centralities[node]})
            else:
                close.update({node: 0})

        measures.update({'Closeness': close})
        logging.info("closeness time in %s seconds" % (time.time() - start_time))


        # %% Katz
        #start_time = time.time()
        #katz = {}
        #try:
        #    centralities = nx.katz_centrality_numpy(graph)
        #except:
        #    centralities = {0: 0}
        #    logging.info("No success with Katz centrality")#
        #for node in interest_nodes:
        #    if node in centralities.keys():
        #        katz.update({node: centralities[node]})
        #    else:
        #        katz.update({node: 0})
        #measures.update({'Katz': katz})
        #logging.info("Katz Centrality time in %s seconds" % (time.time() - start_time))

        #%% Update year info
        year_info.update({year: measures})

    return year_info
