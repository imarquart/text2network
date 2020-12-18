# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 16:47:51 2020

@author: marquart
"""
from src.classes.neo4jnw import neo4j_network
from src.functions.backout_measure import backout_measure
from src.utils.input_check import input_check
import networkx as nx
import itertools
import logging
import numpy as np


def proximities(semantic_network, focal_tokens=None, years=None, context=None, alter_subset=None, nw=None, weight_cutoff=None, norm_ties=True):
    """
    Calculate proximities for given tokens.

    Parameters
    ----------
    semantic_network : semantic network class
        semantic network to use.
    focal_tokens : list, str, optional
        List of tokens of interest. If not provided, centralities for all tokens will be returned.
    years : dict, int, optional
        Given year, or an interval dict {"start":YYYYMMDD,"end":YYYYMMDD}. The default is None.
    context : list, optional
        List of tokens that need to appear in the context distribution of a tie. The default is None.
    alter_subset : TYPE, optional
        DESCRIPTION. The default is None.
    nw : TYPE, optional
        DESCRIPTION. The default is None.
    weight_cutoff : TYPE, optional
        DESCRIPTION. The default is None.
    norm_ties : TYPE, optional
        DESCRIPTION. The default is True.

    Returns
    -------
    proximity_dict : dict
        Dictionary of form {token_id:{alter_id: proximity}}.

    """
    # Input checks
    input_check(years, focal_tokens)
    input_check(tokens=alter_subset)

    proximity_dict = {}
    if focal_tokens is None:
        if nw == None:
            focal_tokens=list(semantic_network.graph.nodes)
        else:
            focal_tokens=list(nw.nodes)
    for token in focal_tokens:
        if nw == None:
            logging.debug(
                "Conditioning year(s) {} with focus on token {}".format(years, token))
            semantic_network.condition(years, tokens=[
                token], weight_cutoff=weight_cutoff, depth=1, context=context, norm=norm_ties)
            nw_graph = semantic_network.graph
        else:
            nw_graph = nw
            
        # Get list of alter token ids, either those found in network, or those specified by user
        if alter_subset == None:
            token_ids = list(nw_graph.nodes)
        else:
            token_ids = semantic_network.ensure_ids(alter_subset)
        logging.debug("token_ids {}".format(token_ids))
        if isinstance(token_ids, int):
            token_ids = [token_ids]

        # Extract
        neighbors = nw_graph[token]
        n_keys = list(neighbors.keys())
        # Choose only relevant alters
        n_keys = tuple(np.intersect1d(token_ids, n_keys))
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

        if nw is not None:
            # Decondition
            semantic_network.decondition()
    return {"proximity":proximity_dict}


def yearly_centralities(semantic_network, year_list, focal_tokens=None, ego_nw_tokens=None, depth=1, types=["PageRank", "normedPageRank"], nw=None, weight_cutoff=None, norm_ties=True):
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
    ego_nw_tokens : list, optional
        List of tokens on which to condition an ego network. The default is None.
    depth : int, optional
        Maximal path length for ego network. The default is 1.
    types : list, optional
        ypes of centrality to calculate. The default is ["PageRank"].
    nw : networkx graph, optional
        Existing networkx graph. The default is None.
    weight_cutoff : float, optional
        Only links of higher weight are considered in conditioning. The default is None.
    norm_ties : bool, optional
        Please see semantic network class. The default is True.

    Returns
    -------
    None.

    """
    cent_year = {}
    assert isinstance(year_list, list), "Please provide list of years."
    for year in year_list:
        cent_measures = centrality(semantic_network,
                                   focal_tokens=focal_tokens, years=year, ego_nw_tokens=ego_nw_tokens, depth=depth, types=types, nw=nw, weight_cutoff=weight_cutoff, norm_ties=norm_ties)
        cent_year.update({year: cent_measures})

    return {'yearly_centralities':cent_year}


def centrality(semantic_network, focal_tokens=None, years=None, ego_nw_tokens=None, depth=1, types=["PageRank", "normedPageRank"], nw=None, weight_cutoff=None, norm_ties=True):
    """
    Calculate centralities for given tokens over an aggregate of given years.
    If no graph is supplied via nw, the semantic network will be conditioned according to the parameters given.

    Parameters
    ----------
    semantic_network : semantic network class
        semantic network to use.
    focal_tokens : list, str, optional
        List of tokens of interest. If not provided, centralities for all tokens will be returned.
    years : dict, int, optional
        Given year, or an interval dict {"start":YYYYMMDD,"end":YYYYMMDD}. The default is None.
    ego_nw_tokens : list, optional
         List of tokens for an ego-network if desired. Only used if no graph is supplied. The default is None.
    depth : TYPE, optional
        Maximal path length for ego network. Only used if no graph is supplied. The default is 1.
    types : list, optional
        Types of centrality to calculate. The default is ["PageRank", "normedPageRank"].
    nw : networkx graph, optional
        Modified graph to use instead of conditioning semantic_network. The default is None.
    weight_cutoff : float, optional
        Only links of higher weight are considered in conditioning.. The default is None.
    norm_ties : bool, optional
        Please see semantic network class.. The default is True.


    Returns
    -------
    dict
        Dict of centralities for focal tokens.

    """
    # Input checks
    input_check(years, focal_tokens)

    if isinstance(types, str):
        types = [types]
    elif not isinstance(types, list):
        logging.error("Centrality types must be list")
        raise ValueError("Centrality types must be list")
    # Condition either overall, or via ego network
    if nw is None:
        if ego_nw_tokens == None:
            logging.debug("Conditioning year(s) {} with focus on tokens {}".format(
                years, focal_tokens))
            semantic_network.condition(years, tokens=None, weight_cutoff=weight_cutoff,
                                       depth=None, context=None, norm=norm_ties)
            logging.debug("Finished conditioning, {} nodes and {} edges in graph".format(
                len(semantic_network.graph.nodes), len(semantic_network.graph.edges)))
            nw_graph=semantic_network.graph
        else:
            logging.debug("Conditioning ego-network for {} tokens with depth {}, for year(s) {} with focus on tokens {}".format(
                len(ego_nw_tokens), depth, years, focal_tokens))
            semantic_network.condition(years, tokens=ego_nw_tokens, weight_cutoff=weight_cutoff,
                                       depth=depth, context=None, norm=norm_ties)
            logging.debug("Finished ego conditioning, {} nodes and {} edges in graph".format(
                len(semantic_network.graph.nodes), len(semantic_network.graph.edges)))
            nw_graph=semantic_network.graph
    else:
        nw_graph=nw
    # Get list of token ids
    logging.debug("Tokens {}".format(focal_tokens))
    if focal_tokens == None:
        token_ids = semantic_network.ids
    else:
        token_ids = semantic_network.ensure_ids(focal_tokens)
    logging.debug("token_ids {}".format(token_ids))
    if isinstance(token_ids, int):
        token_ids = [token_ids]
    measures = {}
    for measure in types:
        # PageRank centrality
        centralities = compute_centrality(nw_graph, measure)
        cent_nodes = np.array(list(centralities.keys()))
        cent_vals = np.array(list(centralities.values()))
        edge_sort = np.argsort(-cent_vals)
        cent_nodes = cent_nodes[edge_sort]
        cent_vals = cent_vals[edge_sort]
        subset = np.in1d(cent_nodes, token_ids)
        cent_nodes = cent_nodes[subset]
        cent_vals = cent_vals[subset]
        centralities = dict(zip(cent_nodes, cent_vals))

        measures.update({measure: centralities})

    # Decondition
    semantic_network.decondition()
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
