import logging
from typing import Optional, Dict, Union

from text2network.functions.node_measures import centrality
from text2network.utils.input_check import input_check


def centralities(snw, focal_tokens=None, types=["PageRank", "normedPageRank"]) -> Dict:
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


    Returns
    -------
    dict
        Dict of centralities for focal tokens.

    """
    if types is None:
        types = ["PageRank", "normedPageRank"]

    input_check(tokens=focal_tokens)
    if focal_tokens is not None:
        focal_tokens = snw.ensure_ids(focal_tokens)

    if not snw.conditioned:
        snw.condition_error()

    cent_dict = centrality(
        snw.graph, focal_tokens=focal_tokens, types=types)

    return cent_dict


def yearly_centralities(snw, year_list: list, focal_tokens: Optional[Union[list, str]] = None,
                        types: Optional[list] = ("PageRank", "normedPageRank"),
                        depth: Optional[int] = None, context: Optional[list] = None,
                        weight_cutoff: Optional[float] = None,
                        max_degree: Optional[int] = 100,
                        symmetric: Optional[bool] = False,
                        compositional: Optional[bool] = False,
                        reverse: Optional[bool] = False, normalization: Optional[str] = None) -> Dict:
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
        types of centrality to calculate. The default is ("PageRank", "normedPageRank").
    depth : TYPE, optional - used when conditioning
        Maximal path length for ego network starting from focal token. The default is None.
    context : list, optional - used when conditioning
        List of tokens that need to appear in the context distribution of a tie. The default is None.
    weight_cutoff : float, optional - used when conditioning
        Only links of higher weight are considered in conditioning.. The default is None.
    max_degree: int, optional
        The top max_degree ties in terms of tie weight are considered for any token.
   normalization: optional, str
        Given that each point in time has differently many sequences, we can norm either:
        -> "sequences" - divide each tie weight by #sequences/1000 in the given year
        -> "occurrences" - divide each tie weight by the total #occurrences/1000 in the given year
        Note that this differs from compositional mode, where each norm is individual to each token/year
    symmetric: bool, optional
        Transform directed network to undirected network
        use symmetric_method to specify how. See semantic_network.to_symmetric
    compositional: bool, optional
        Use compositional ties. See semantic_network.to_compositional
    reverse: bool, optional
        Reverse ties. See semantic_network.to_reverse()

    Returns
    -------
    dict
        Dict of years with dict of centralities for focal tokens.

    """
    if types is None:
        types = ["PageRank", "normedPageRank"]

    cent_year = {}
    if not isinstance(year_list, list):
        raise AssertionError("Please provide list of years.")

    for year in year_list:
        snw.decondition()
        logging.debug(
            "Conditioning network on year {} with {} focal tokens and depth {}".format(year, len(focal_tokens), depth))
        snw.condition(tokens=focal_tokens, times=[
            year], depth=depth, context=context, weight_cutoff=weight_cutoff, max_degree=max_degree)
        if normalization == "sequences":
            snw.norm_by_total_nr_sequences(times=ma_years)
        elif normalization == "occurrences":
            snw.norm_by_total_nr_occurrences(times=ma_years)
        elif normalization is not None:
            msg = "For yearly normalization, please either specify 'sequences' or 'occcurrences' or None"
            logging.error(msg)
            raise AttributeError(msg)
        if reverse:
            snw.to_reverse()
        if compositional:
            snw.to_compositional()
        if symmetric:
            snw.to_symmetric()
        logging.debug("Computing centralities for year {}".format(year))
        cent_measures = snw.centralities(focal_tokens=focal_tokens, types=types)
        cent_year.update({year: cent_measures})

    return {'yearly_centrality': cent_year}
