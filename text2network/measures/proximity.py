import logging
from typing import Optional, List, Dict

import numpy as np

# from src.classes import neo4jnw
# from src.classes.neo4jnw import neo4j_network
from text2network.functions.node_measures import proximity
# from src.classes import neo4jnw
from text2network.utils.input_check import input_check


def proximities(snw, focal_tokens: Optional[List] = None, alter_subset: Optional[List] = None,
                reverse_ties: Optional[bool] = False, to_backout: Optional[bool] = False) -> Dict:
    """
    Calculate proximities for given tokens.

    Throwns error if network is not conditioned!

    Parameters
    ----------
    snw : semantic network
        semantic network class
    focal_tokens : list, str, optional
        List of tokens of interest. If not provided, centralities for all tokens will be returned.
    alter_subset : list, str optional
        List of alters to show. Others are hidden. The default is None.
    reverse_ties : bool, optional
        Reverse all ties. The default is False.

    Returns
    -------
    proximity_dict : dict
        Dictionary of form {token_id:{alter_id: proximity}}.

    """

    input_check(tokens=focal_tokens)
    input_check(tokens=alter_subset)

    if alter_subset is not None:
        alter_subset = snw.ensure_ids(alter_subset)
    if focal_tokens is not None:
        focal_tokens = snw.ensure_ids(focal_tokens)
        if not isinstance(focal_tokens, list):
            focal_tokens = [focal_tokens]
    else:
        focal_tokens = snw.ids

    if not snw.conditioned:
        snw.condition_error()

    proximity_dict = {}

    # Reverse ties if requested
    if reverse_ties:
        snw.to_reverse()
    if to_backout:
        snw.to_backout()
    # Get proximities from conditioned network
    for token in focal_tokens:
        if token in snw.graph.nodes:
            tie_dict = proximity(snw.graph, focal_tokens=[token], alter_subset=alter_subset)[
                'proximity'][token]
            proximity_dict.update({token: tie_dict})

    # Reverse ties if requested
    if reverse_ties:
        snw.to_reverse()

    return {"proximity": proximity_dict}



def yearly_proximities(snw, year_list, focal_tokens=None, alter_subset: Optional[List] = None,symmetric=False,
                        context=None, weight_cutoff=None, compositional=None, moving_average=None,
                        reverse_ties: Optional[bool] = False, backout: Optional[bool]=False):
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
    alter_subset: Optional[List] = None
        List of alters to return proximities for. Others are discarded.
    symmetric : bool, optional
        Symmetrize Network?
    context : list, optional - used when conditioning
        List of tokens that need to appear in the context distribution of a tie. The default is None.
    weight_cutoff : float, optional - used when conditioning
        Only links of higher weight are considered in conditioning.. The default is None.
    compositional : bool, optional - used when conditioning
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

    if not isinstance(focal_tokens,list):
        focal_tokens=[focal_tokens]

    for year in year_list:
        snw.decondition()
        logging.info("Conditioning network on year {} with {} focal tokens".format(year,len(focal_tokens)))

        if moving_average is not None:
            start_year = max(year_list[0], year - moving_average[0])
            end_year = min(year_list[-1], year + moving_average[1])
            ma_years = np.arange(start_year, end_year + 1)
            logging.info(
                "Calculating proximities for fixed relevant clusters for year {} with moving average -{} to {} over {}".format(
                    year,
                    moving_average[
                        0],
                    moving_average[
                        1], ma_years))
        else:
            ma_years = year

        if reverse_ties or symmetric:
            snw.condition(tokens=focal_tokens, times=ma_years, depth=1, context=context, weight_cutoff=weight_cutoff,
                          compositional=compositional)
        else:
            snw.condition(tokens=focal_tokens, times=ma_years, depth=0, context=context, weight_cutoff=weight_cutoff,
                          compositional=compositional)


        if not compositional:
            snw.norm_by_time(ma_years)
        if reverse_ties:
            snw.to_reverse()
        if symmetric:
            snw.to_symmetric()
        if backout:
            snw.to_backout()
        logging.debug("Computing proximities for year {}".format(year))
        proximity_dict={}
        # Get proximities from conditioned network
        tie_dict = snw.proximities( focal_tokens=focal_tokens, alter_subset=alter_subset)
        logging.info("Identified {} proximate tokens for year {}".format(len(list(tie_dict['proximity'].values())[0]),year))
        cent_year.update({year: tie_dict})

    return {'yearly_proximity': cent_year}
