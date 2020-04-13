import community
import networkx as nx
import numpy as np
import scipy as scp
import itertools
import logging
from NLP.config.config import configuration
from networkx.algorithms.community.centrality import girvan_newman
import hdbscan
import xlsxwriter
from NLP.utils.rowvec_tools import prune_network_edges, inverse_edge_weight
from NLP.utils.network_tools import load_graph, make_symmetric
import time
from tqdm import tqdm
import pandas as pd

def return_dataframe(year_info,years,cluster_dict=None):

    if cluster_dict==None:
        token_id = []
        cents = []
        measure_id = []
        c_year = []
        # Pandas
        for year in years:
            for measure in year_info[year]:
                mdict = year_info[year][measure]
                for token, cent in mdict.items():
                    token_id.append(token)
                    cents.append(cent)
                    measure_id.append(measure)
                    c_year.append(year)
        data = {'Year': c_year, 'Token': token_id, 'Values': cents, 'Measure': measure_id}
        dframe = pd.DataFrame(data)
    else:
        # Averaged Clusters
        clusters=cluster_dict.values()
        unique_clusters=np.unique(list(cluster_dict.values()))
        c_cluster = []
        c_wavg = []
        c_savg = []
        c_measure = []
        c_missing = []
        c_year = []
        # Pandas
        for year in years:
            # Get proximities
            prox_dict = {}
            logging.info("Starting year %i" %year)
            logging.info("Starting proximity dict")
            for token in cluster_dict:
                mdict = year_info[year]['Proximity']
                if token in mdict:
                    prox_dict.update({token: mdict[token]})
                else:
                    prox_dict.update({token: 0})
            logging.info("Starting measure calcs")
            # Go through measures
            for measure in year_info[year]:
                mdict = year_info[year][measure]
                for cur_cluster in unique_clusters:
                    # Get a list of tokens corresponding to cluster
                    indexes=np.array(list(cluster_dict.values()))==cur_cluster
                    t_list = np.array(list(cluster_dict.keys()))[indexes]
                    # Make a list for the measures
                    measure_list = []
                    # Make a list for missing tokens
                    missing = np.zeros(len(t_list), dtype=np.bool)
                    for i, token in enumerate(t_list):
                        if token in mdict:
                            measure_list.append(mdict[token])
                        else:
                            measure_list.append(0)
                            missing[i] = True
                    # How many tokens had measures?
                    logging.info("Finished measure dict for measure %s" %measure)
                    nr_not_missing = len(np.where(missing == False)[0])
                    nr_missing = len(np.where(missing == True)[0])
                    logging.info("Missing: %i, Not Missing %i" %(nr_missing,nr_not_missing))

                    # Extract proximities for the tokens
                    p_list=np.array(list(prox_dict.values()))
                    p_list_keys = np.array(list(prox_dict.keys()))
                    indexes=[i for i, e in enumerate(p_list_keys) if e in set(t_list)]
                    p_list=p_list[indexes]
                    # Weigh proximities for missing tokens by zero
                    p_list[missing] = 0
                    prox_sum = np.sum(p_list)
                    if prox_sum==0:
                        weights = p_list * 0
                    else:
                        weights = p_list / prox_sum
                    # Calculate weights
                    w_avg = np.sum(weights * np.array(measure_list))
                    s_avg = np.sum((1 / max(1, nr_not_missing)) * np.array(measure_list))
                    # Append to dataframe
                    c_cluster.append(cur_cluster)
                    c_wavg.append(w_avg)
                    c_savg.append(s_avg)
                    c_missing.append(nr_missing)
                    c_measure.append(measure)
                    c_year.append(year)

        data = {'Year': c_year, 'Cluster': c_cluster, 'WAvg': c_wavg, 'SAvg': c_savg, 'Measure': c_measure,
                'Missing': c_missing}
        dframe = pd.DataFrame(data)

    return dframe

def pair_analysis(years, u,v, cfg, save_folder, cutoff, network_type="2Rgraph-Sum-Rev_order3",
                    sums_folder="/networksNoCut/sums"):
    logging.info("Pruning links smaller than %f" % cutoff)
    year_info={}
    interest_nodes=[u,v]
    for year in tqdm(years, leave=False, position=0):
        measure_info={}
        graph = load_graph(year, cfg, sums_folder, network_type)
        graph = prune_network_edges(graph, edge_weight=cutoff)
        if not u in graph.nodes or not v in graph.nodes:
            AssertionError("Both nodes are not in network of year %i" % year)

        # Calc Centralities
        pagerank={}
        try:
            centralities = nx.pagerank_scipy(graph, weight='weight')
        except:
            centralities = {0: 0}
        for node in interest_nodes:
            if node in centralities.keys():
                pagerank.update({node: centralities[node]})
            else:
                pagerank.update({node: 0})

        measure_info.update({"PageRank": pagerank})
        between={}
        try:
            #centralities = nx.betweenness_centrality(graph, k=int(10 * np.log(len(list(graph.nodes)))))
            centralities = nx.betweenness_centrality(graph)

        except:
            centralities = {0: 0}
        for node in interest_nodes:
            if node in centralities.keys():
                between.update({node: centralities[node]})
            else:
                between.update({node: 0})

        measure_info.update({"Betweeness": between})
        # Calc Constraint
        constraint_to = {}
        try:
            #co = 0
            co = nx.local_constraint(graph, u, v, weight='weight')
        except:
            co = 0
            logging.info("No success with constraint from %s to %s" % (u,v))
        try:
            #co = 0
            co2 = nx.local_constraint(graph, v, u, weight='weight')
        except:
            co2 = 0
            logging.info("No success with constraint from %s to %s" % (v,u))
        # Update Measures
        constraint_to.update({u: co})
        constraint_to.update({v: co2})
        measure_info.update({"Constraint": constraint_to})

        year_info.update({year: measure_info})





    filename = ''.join([save_folder, '/unweight_', u,'_',v, '.xlsx'])
    workbook = xlsxwriter.Workbook(filename)
    worksheet = workbook.add_worksheet()

    col=1
    for year in years:
        row = 1
        col=col+1
        logging.info("Writing year %i" %year)
        worksheet.write_number(0, col, year)
        for measure in year_info[year]:
            worksheet.write_string(row,1,measure)
            worksheet.write_string(row+1, 1, measure)
            mdict=year_info[year][measure]
            for token in mdict:
                worksheet.write_string(row,0,token)
                worksheet.write_number(row, col, mdict[token])
                row=row+1

    workbook.close()



def raw_ego_network(years, focal_token, cfg, max_retain, save_folder, network_type="Rgraph-Sum-Rev",
                    sums_folder="/networksNoCut/sums"):
    logging.info("Loading graph data.")
    logging.info("Pruning links smaller than %f" % cfg.prune_min)
    logging.info("Creating list of most connected terms.")
    neighbor_set = {}
    weight_set = {}
    for year in tqdm(years, leave=False, position=0):
        graph = load_graph(year, cfg, sums_folder, network_type)
        graph = prune_network_edges(graph, edge_weight=0)
        if focal_token in graph.nodes:
            neighbors = graph[focal_token]
            edge_weights = [x['weight'] for x in neighbors.values()]
            edge_sort = np.argsort(-np.array(edge_weights))
            edge_weights=np.array(edge_weights)
            edge_weights=edge_weights[edge_sort]
            neighbors = np.array([x for x in neighbors])
            neighbors=neighbors[edge_sort]
        else:
            neighbors = []
            edge_weights = []

        neighbor_set.update({year: neighbors})
        weight_set.update({year: edge_weights})
        del graph

    filename = ''.join([save_folder, '/ego_links_', focal_token, '.xlsx'])
    workbook = xlsxwriter.Workbook(filename)
    worksheet = workbook.add_worksheet()
    worksheet.write_string(0, 0, focal_token)
    col = 0
    for year in years:
        row = 1
        logging.info("Writing year %i" %year)
        worksheet.write_number(row, col, year)
        worksheet.write_string(row, col+1, "Proximity")
        logging.info("Neighbor set %s" % str(neighbor_set[year]))
        row = 2
        for i, token in enumerate(neighbor_set[year]):
            worksheet.write_string(row, col, token)
            worksheet.write_number(row, col + 1, weight_set[year][i])
            row = row + 1
        col = col + 3
    workbook.close()


def dynamic_centralities(years, focal_token, cfg, num_retain=15, cutoff=0.025,
                         network_type="Rgraph-Sum", sums_folder="/networksNoCut/sums", external_list=[]):
    """

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
    neighbor_set = []
    weight_set = []
    logging.info("Loading graph data.")
    logging.info("Creating list of most connected terms.")

    for year in years:
        graph = load_graph(year, cfg, sums_folder, network_type)
        graph = prune_network_edges(graph, edge_weight=cutoff)
        # To save memory, I have changed this such that only one graph is kept in memory
        # graphs.update({year: graph})
        neighbors = graph[focal_token]
        edge_weights = [x['weight'] for x in neighbors.values()]
        edge_sort = np.argsort(-np.array(edge_weights))
        neighbors = [x for x in neighbors]
        neighbor_set.extend(neighbors)
        weight_set.extend(edge_weights)

    logging.info("Creating interest set")
    weight_set = np.array(weight_set)
    neighbor_set = np.array(neighbor_set)
    edge_sort = np.argsort(-weight_set)
    neighbor_set = neighbor_set[edge_sort]
    weight_set = weight_set[edge_sort]
    interest_set = {}
    counter = 0
    for i in range(0, len(neighbor_set)):
        if neighbor_set[i] not in interest_set.keys():
            interest_set.update({neighbor_set[i]: weight_set[i]})
            counter = counter + 1
        if counter >= num_retain:
            break

    interest_nodes = list(interest_set.keys())

    external_list2 = [
        'alpha', 'champion', 'competitor', 'force', 'pioneer', 'player', 'quarterback', 'winner',
        'boss', 'captain', 'commander', 'controller', 'director', 'general', 'head', 'king', 'master', 'monarch',
        'owner', 'president', 'superior',
        'believer', 'giant', 'hero', 'magnet', 'speaker', 'star', 'superstar', 'visionary', 'activist',
        'challenger', 'razor', 'renegade', 'revolutionary', 'builder', 'coach', 'conductor', 'consultant',
        'coordinator', 'mentor', 'parent', 'partner', 'teacher',
        'designer', 'expert', 'scientist', 'solution', 'man', 'men', 'male', 'woman', 'female', 'women']
    [interest_nodes.append(x) for x in external_list if x not in interest_nodes]
    [interest_nodes.append(x) for x in external_list2 if x not in interest_nodes]

    if focal_token not in interest_nodes: interest_nodes.append(focal_token)

    year_info = {}
    central_graphs = {}
    logging.info("Identified %i terms of interest" % len(interest_nodes))
    for t, year in enumerate(tqdm(years, desc="Centralities per year", leave=False, position=0)):
        measures = {}
        logging.info("Centrality calculation for year %i." % year)
        ego_graph = load_graph(year, cfg, sums_folder, network_type)
        ego_graph = prune_network_edges(ego_graph, edge_weight=cutoff)
        ego_graph = nx.ego_graph(ego_graph, focal_token, cfg.ego_radius).copy()

        # %% Closeness to leader
        start_time = time.time()
        proximity = {}
        try:
            centralities = ego_graph[focal_token]
        except:
            centralities = {0: 0}
        for node in interest_nodes:
            if node in centralities.keys():
                proximity.update({node: centralities[node]['weight']})
            else:
                proximity.update({node: 0})

        measures.update({'Proximity': proximity})
        logging.info("Proximity time in %s seconds" % (time.time() - start_time))

        # %% Constraint
        start_time = time.time()

        constraint = {}
        try:
            #co1 = nx.constraint(ego_graph, interest_nodes, weight='weight')
            co1 = {0: 0}
        except:
            co1 = {0:0}
            logging.info("No success with Constraint centrality, node %s" % node)

        constraint.update({node: co1})

        measures.update({'Constraint': constraint})
        logging.info("Constraint time in %s seconds" % (time.time() - start_time))
        # %% Constraint
        start_time = time.time()

        constraint = {}
        for node in interest_nodes:
            if node in list(ego_graph.nodes):
                try:
                    co1 = 0
                    #co1 = nx.local_constraint(ego_graph, focal_token, node, weight='weight')
                except:
                    co1 = 0
                    logging.info("No success with Constraint centrality, node %s" % node)
                constraint.update({node: co1})
            else:
                constraint.update({node: 0})

        measures.update({'EgoFocalConstrBy': constraint})
        logging.info("Dyadic Constraint time in %s seconds" % (time.time() - start_time))
        # %% Constraint

        start_time = time.time()

        constraint_to = {}
        for node in interest_nodes:
            if node in list(ego_graph.nodes):
                try:
                    co = 0
                    #co = nx.local_constraint(ego_graph, node, focal_token, weight='weight')
                except:
                    co = 0
                    logging.info("No success with Constraint-Towards centrality, node %s" % node)
                constraint_to.update({node: co})
            else:
                constraint_to.update({node: 0})

        measures.update({'EgoFocalConstraining': constraint_to})
        logging.info("Dyadic Constraint time in %s seconds" % (time.time() - start_time))

        # %% Page Rank Weighted
        start_time = time.time()
        pageranks = {}
        try:
            centralities = nx.pagerank_scipy(ego_graph, weight='weight')
        except:
            centralities = {0: 0}

        for node in interest_nodes:
            if node in centralities.keys():
                pageranks.update({node: centralities[node]})
            else:
                pageranks.update({node: 0})

        measures.update({'EgoPageRank-Weighted': pageranks})
        logging.info("PageRank-Weighted time in %s seconds" % (time.time() - start_time))


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

        between = {}
        try:
            centralities = nx.betweenness_centrality(ego_graph, k=int(10 * np.log(len(list(ego_graph.nodes)))))
            # centralities = {0: 0}
        except:
            logging.info("No success with  betweeness centrality")
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
            # centralities = {0: 0}
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


        # %% Update year info
        year_info.update({year: measures})

    return year_info
