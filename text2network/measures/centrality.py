import logging
from typing import Optional

from text2network.functions.node_measures import centrality
# from src.classes import neo4jnw
from text2network.utils.input_check import input_check


def centralities(snw, focal_tokens=None, types=["PageRank", "normedPageRank"], reverse_ties: Optional[bool] = False):
    """
    Calculate centralities for given tokens over an aggregate of given years.
    If not conditioned, error will be thrown!

    Parameters
    ----------
    snw : semantic network
        semantic network class
    focal_tokens : list, str, optional
        List of tokens of interest. If not provided, centralities for all tokens will be returned.
    types : list, optional
        Types of centrality to calculate. The default is ["PageRank", "normedPageRank"].
    reverse_ties : bool, optional
        Reverse all ties. The default is False.

    Returns
    -------
    dict
        Dict of centralities for focal tokens.

    """

    input_check(tokens=focal_tokens)
    if focal_tokens is not None:
        focal_tokens = snw.ensure_ids(focal_tokens)

    if not snw.conditioned:
        snw.condition_error()

    # Reverse ties if requested
    if reverse_ties:
        snw.to_reverse()

    cent_dict = centrality(
        snw.graph, focal_tokens=focal_tokens, types=types)

    # Reverse ties if requested
    if reverse_ties:
        snw.to_reverse()

    return cent_dict


def yearly_centralities(snw, year_list, focal_tokens=None, types=["PageRank", "normedPageRank"],
                        depth=None, context=None, weight_cutoff=None, compositional=None,
                        symmetric: Optional[bool] = False,
                        reverse_ties: Optional[bool] = False, backout: Optional[bool] = False, max_degree=100):
    """
    Compute directly year-by-year centralities for provided list.

    This will decondition and re-condition the network across years

    Parameters
    ----------
    snw : semantic network
    year_list : list
        List of years for which to calculate centrality.
    focal_tokens : list, str
        List of tokens of interest. If not provided, centralities for all tokens will be returned.
    types : list, optional
        types of centrality to calculate. The default is ["PageRank"].
    ego_nw_tokens : list, optional - used when conditioning
         List of tokens for an ego-network if desired. Only used if no graph is supplied. The default is None.
    depth : TYPE, optional - used when conditioning
        Maximal path length for ego network. Only used if no graph is supplied. The default is None.
    context : list, optional - used when conditioning
        List of tokens that need to appear in the context distribution of a tie. The default is None.
    weight_cutoff : float, optional - used when conditioning
        Only links of higher weight are considered in conditioning.. The default is None.
    norm_ties : bool, optional - used when conditioning
        Please see semantic network class. The default is True.
    reverse_ties : bool, optional
        Reverse all ties. The default is False.

    Returns
    -------
    dict
        Dict of years with dict of centralities for focal tokens.

    """

    cent_year = {}
    assert isinstance(year_list, list), "Please provide list of years."

    for year in year_list:
        snw.decondition()
        logging.debug(
            "Conditioning network on year {} with {} focal tokens and depth {}".format(year, len(focal_tokens), depth))
        snw.condition(tokens=focal_tokens, times=[
            year], depth=depth, context=context, weight_cutoff=weight_cutoff,
                      compositional=compositional, max_degree=max_degree)
        if backout:
            snw.to_backout()
        if symmetric:
            snw.to_symmetric()
        logging.debug("Computing centralities for year {}".format(year))
        cent_measures = snw.centralities(focal_tokens=focal_tokens, types=types, reverse_ties=reverse_ties)
        cent_year.update({year: cent_measures})

    return {'yearly_centrality': cent_year}
