# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 16:47:51 2020

@author: marquart
"""

import networkx as nx
import itertools
import logging
import numpy as np


def proximity(nw_graph, focal_tokens=None, alter_subset=None):
    """
    Calculate proximities for given tokens.

    Parameters
    ----------
    nw_graph : networkx graph
        semantic network to use.
    focal_tokens : list, str, optional
        List of tokens of interest. If not provided, centralities for all tokens will be returned.
    alter_subset : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    proximity_dict : dict
        Dictionary of form {token_id:{alter_id: proximity}}.

    """

    proximity_dict = {}
    if focal_tokens is None:
        focal_tokens=list(nw_graph.nodes)
    for token in focal_tokens:        
        # Get list of alter token ids, either those found in network, or those specified by user
        if alter_subset == None:
            alter_subset = list(nw_graph.nodes)
        logging.debug("alter_subset token_ids {}".format(alter_subset))
        if isinstance(alter_subset, int):
            alter_subset = [alter_subset]

        # Extract
        neighbors = nw_graph[token]
        n_keys = list(neighbors.keys())
        # Choose only relevant alters
        n_keys = tuple(np.intersect1d(alter_subset, n_keys))
        neighbors = {k: v for k, v in neighbors.items() if k in n_keys}

        # Extract edge weights and sort by weight
        edge_weights = [x['weight'] for x in neighbors.values()]
        edge_sort = np.argsort(-np.array(edge_weights))
        neighbors = [x for x in neighbors]
        edge_weights = np.array(edge_weights)
        neighbors = np.array(neighbors)
        edge_weights = edge_weights[edge_sort]
        neighbors = neighbors[edge_sort]

        tie_dict = dict(zip(neighbors, edge_weights))
        proximity_dict.update({token: tie_dict})

    return {"proximity":proximity_dict}


def yearly_centrality(nw_graph, year_list, focal_tokens=None,  types=["PageRank", "normedPageRank"]):
    """
    Compute directly year-by-year centralities for provided list.

    Parameters
    ----------
    semantic_network : semantic network class
        semantic network to use.
    year_list : list
        List of years for which to calculate centrality.
    focal_tokens : list, str
        List of tokens of interest. If not provided, centralities for all tokens will be returned.
    types : list, optional
        ypes of centrality to calculate. The default is ["PageRank"].
    Returns
    -------
    dict
        Dict of years with dict of centralities for focal tokens.

    """
    cent_year = {}
    assert isinstance(year_list, list), "Please provide list of years."
    for year in year_list:
        cent_measures = centrality(nw_graph,
                                   focal_tokens=focal_tokens, types=types)
        cent_year.update({year: cent_measures})

    return {'yearly_centralities':cent_year}


def centrality(nw_graph, focal_tokens=None,  types=["PageRank", "normedPageRank"]):
    """
    Calculate centralities for given tokens over an aggregate of given years.
    If no graph is supplied via nw, the semantic network will be conditioned according to the parameters given.

    Parameters
    ----------
    semantic_network : semantic network class
        semantic network to use.
    focal_tokens : list, str, optional
        List of tokens of interest. If not provided, centralities for all tokens will be returned.
    types : list, optional
        Types of centrality to calculate. The default is ["PageRank", "normedPageRank"].

    Returns
    -------
    dict
        Dict of centralities for focal tokens.

    """
    # Input checks
    if isinstance(types, str):
        types = [types]
    elif not isinstance(types, list):
        logging.error("Centrality types must be list")
        raise ValueError("Centrality types must be list")

    # Get list of token ids
    logging.debug("Tokens {}".format(focal_tokens))
    if focal_tokens is None:
        focal_tokens=list(nw_graph.nodes)
    logging.debug("token_ids {}".format(focal_tokens))
    if isinstance(focal_tokens, int):
        focal_tokens = [focal_tokens]
    measures = {}
    for measure in types:
        # PageRank centrality
        centralities = compute_centrality(nw_graph, measure)
        cent_nodes = np.array(list(centralities.keys()))
        cent_vals = np.array(list(centralities.values()))
        edge_sort = np.argsort(-cent_vals)
        cent_nodes = cent_nodes[edge_sort]
        cent_vals = cent_vals[edge_sort]
        subset = np.in1d(cent_nodes, focal_tokens)
        cent_nodes = cent_nodes[subset]
        cent_vals = cent_vals[subset]
        centralities = dict(zip(cent_nodes, cent_vals))

        measures.update({measure: centralities})

    return {"centralities":measures}


def compute_centrality(nw_graph, measure):
    if measure == "PageRank":
        # PageRank centrality
        try:
            centralities = nx.pagerank_scipy(
                nw_graph, weight='weight')
            logging.debug(
                "Calculated {} PageRank centralities".format(
                    len(centralities)))

        except:
            logging.error("Could not calculate Page Rank centralities")
            raise Exception("Could not calculate Page Rank centralities")
    elif measure == "normedPageRank":
        # PageRank centrality
        try:
            centralities = nx.pagerank_scipy(
                nw_graph, weight='weight')
            logging.debug(
                "Calculated {} normalized PageRank centralities".format(
                    len(centralities)))
            centvec = np.array(list(centralities.values()))
            normconst = np.sum(centvec)
            nr_n = len(centvec)

            centvec = (centvec/normconst)*nr_n
            logging.debug("For N={}, sum of changed vector is {}".format(
                nr_n, np.sum(centvec)))
            centralities = dict(zip(centralities.keys(), centvec))

        except:
            logging.error(
                "Could not calculate normed Page Rank centralities")
            raise Exception(
                "Could not calculate normed Page Rank centralities")
    else:
        raise AttributeError(
            "Centrality measure {} not found in list".format(measure))

    return centralities
