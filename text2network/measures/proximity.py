import logging
from typing import Optional, List, Dict, Union

import numpy as np

from text2network.functions.node_measures import proximity
from text2network.utils.input_check import input_check


def proximities(snw, focal_tokens: Optional[List] = None, alter_subset: Optional[List] = None) -> Dict:
    """
    Calculate proximities for given tokens.

    Throws error if network is not conditioned!

    Parameters
    ----------
    snw : semantic network
        semantic network class
    focal_tokens : list, str, optional
        List of tokens of interest. If not provided, centralities for all tokens will be returned.
    alter_subset : list, str optional
        List of alters to show. Others are hidden. The default is None.

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

    # Get proximities from conditioned network
    for token in focal_tokens:
        if token in snw.graph.nodes:
            tie_dict = proximity(snw.graph, focal_tokens=[token], alter_subset=alter_subset)[
                'proximity'][token]
            proximity_dict.update({token: tie_dict})

    return {"proximity": proximity_dict}


def yearly_proximities(snw, year_list: Union[list, int], focal_tokens: Optional[Union[list, str]] = None,
                       alter_subset: Optional[list] = None, max_degree: Optional[int] = None,
                       context: Optional[Union[list, str]] = None, weight_cutoff: Optional[float] = None,
                       moving_average: Optional[tuple] = None,symmetric:Optional[bool]=False,
                        compositional:Optional[bool]=False,
                        reverse:Optional[bool]=False, normalization: Optional[str] = None,
                       symmetric_method:Optional[str]=None, prune_min_frequency: Optional[int] = None):
    """
    Compute directly year-by-year centralities for provided list.

    This will decondition and re-condition the network across years

    Parameters
    ----------

    snw : semantic network

    year_list : list
        List of years for which to calculate centrality.

    focal_tokens : list, str
        List of tokens of interest. If not provided, proximities for all tokens will be returned.

    alter_subset: list, optional
        List of alters to return proximities for. Others are discarded.

    context : list, optional - used when conditioning
        List of tokens that need to appear in the context distribution of a tie. The default is None.

    weight_cutoff : float, optional - used when conditioning
        Only links of higher weight are considered in conditioning.. The default is None.

    moving_average: tuple
        Pass as (a,b), where for a focal year x the conditioning window will be
        [x-a,x+b]

    normalization: optional, str
        Given that each point in time has differently many sequences, we can norm either:
        -> "sequences" - divide each tie weight by #sequences/1000 in the given year
        -> "occurrences" - divide each tie weight by the total #occurrences/1000 in the given year
        Note that this differs from compositional mode, where each norm is individual to each token/year

    max_degree: int
        When conditioning, extract at most the top max_degree ties for any token in terms of weight

    prune_min_frequency : int, optional
    Will remove nodes entirely which occur less than  prune_min_frequency+1 times

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
        Dict of years with dict of proxmities for focal tokens.

    """

    if symmetric or reverse:
        depth=1
    else:
        depth=0

    cent_year = {}
    if not isinstance(year_list, list):
        raise AssertionError("Please provide list of years.")

    if not isinstance(focal_tokens, list):
        focal_tokens = [focal_tokens]

    orig_focal_tokens = focal_tokens.copy()
    if alter_subset is not None:
        # Add alter subset to focal tokens
        focal_tokens = focal_tokens+alter_subset
        # Set depth=0 to only get this network
        depth=0

    for year in year_list:
        snw.decondition()
        logging.info("Conditioning network on year {} with {} focal tokens".format(year, len(focal_tokens)))

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

        snw.condition(tokens=focal_tokens, times=ma_years, depth=depth, context=context, weight_cutoff=weight_cutoff,
                      max_degree=max_degree,prune_min_frequency=prune_min_frequency)
        if normalization=="sequences":
            snw.norm_by_total_nr_sequences(times=ma_years)
        elif normalization=="occurrences":
            snw.norm_by_total_nr_occurrences(times=ma_years)
        elif normalization is not None:
            msg="For yearly normalization, please either specify 'sequences' or 'occcurrences' or None"
            logging.error(msg)
            raise AttributeError(msg)
        if reverse:
            snw.to_reverse()
        if compositional:
            snw.to_compositional()
        if symmetric:
            snw.to_symmetric(technique=symmetric_method)
        logging.debug("Computing proximities for year {}".format(year))
        proximity_dict = {}
        # Get proximities from conditioned network
        tie_dict = snw.proximities(focal_tokens=orig_focal_tokens, alter_subset=alter_subset)
        logging.info(
            "Identified {} proximate tokens for year {}".format(len(list(tie_dict['proximity'].values())[0]), year))
        cent_year.update({year: tie_dict})

    return {'yearly_proximity': cent_year}
