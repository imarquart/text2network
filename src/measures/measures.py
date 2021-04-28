import configparser
import logging
from typing import Optional, Callable, Union, List, Dict

import numpy as np
import pandas as pd

# from src.classes import neo4jnw
# from src.classes.neo4jnw import neo4j_network
from src.functions.file_helpers import check_create_folder
from src.functions.graph_clustering import consensus_louvain
from src.functions.node_measures import proximity, centrality
# from src.classes import neo4jnw
from src.utils.input_check import input_check


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


def proximities(snw, focal_tokens: Optional[List] = None, alter_subset: Optional[List] = None,
                reverse_ties: Optional[bool] = False) -> Dict:
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

    # Get proximities from conditioned network
    for token in focal_tokens:
        tie_dict = proximity(snw.graph, focal_tokens=[token], alter_subset=alter_subset)[
            'proximity'][token]
        proximity_dict.update({token: tie_dict})

    # Reverse ties if requested
    if reverse_ties:
        snw.to_reverse()

    return {"proximity": proximity_dict}


def yearly_centralities(snw, year_list, focal_tokens=None, types=["PageRank", "normedPageRank"], ego_nw_tokens=None,
                        depth=None, context=None, weight_cutoff=None, norm_ties=None,
                        reverse_ties: Optional[bool] = False):
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

    # Get default normation behavior
    if norm_ties is None:
        norm_ties = snw.norm_ties

    cent_year = {}
    assert isinstance(year_list, list), "Please provide list of years."

    for year in year_list:
        snw.decondition()
        snw.condition(focal_tokens=focal_tokens, years=[
            year], ego_nw_tokens=ego_nw_tokens, depth=depth, context=context, weight_cutoff=weight_cutoff,
                      norm_ties=norm_ties)
        cent_measures = snw.centralities(focal_tokens=focal_tokens, types=types, reverse_ties=reverse_ties)
        cent_year.update({year: cent_measures})

    return {'yearly_centrality': cent_year}


def average_fixed_cluster_proximities(focal_token: str, interest_list: list, nw, levels: int,
                                      times: Optional[Union[list,int]] = None,
                                      depth: Optional[int] = 1, context: Optional[list] = None,
                                      weight_cutoff: Optional[float] = None,
                                      cluster_cutoff: Optional[float] = 0, do_reverse: Optional[bool] = False,
                                      moving_average: Optional[tuple] = False,
                                      filename: Optional[str] = None,
                                      compositional: Optional[bool] = False,
                                      reverse_ties: Optional[bool] = False,
                                      seed: Optional[int] = None) -> pd.DataFrame:
    """
    First, derives clusters from overall network (across all years), then creates year-by-year average proximities for these clusters

    Parameters
    ----------
    focal_token: str
        Token to which proximities are calculated
    interest_list: list
        List of tokens of relevance. Only clusters that contain these tokens are considered.
    nw: neo4jnw
        Semantic Network
    levels: int
        How many levels to cluster?
    depth: int
        Depth of the ego network of focal_token to consider. If 0, consider whole semantic network.
    context: list
        Contextual tokens
    weight_cutoff: float
        Cutoff when querying the network
    do_reverse: bool
        Calculate reverse proximities, note this needs to query the entire graph for each year
    moving_average: tuple
        Consider a moving_average window for years, given as tuple where (nr_prior,nr_past)
        So for example, a length 3 window around the focal year would be given by (1,1).
        A length 3 window behind the focal year would be (2,0)
    cluster_cutoff: float
        After clusters are derived, throw away proximities of less than this value
    filename: str
        If given, will save the resulting dataframe to this file as xlsx
    compositional : bool
        Whether to use compositional ties
    reverse_ties : bool
        Whether to reverse ties after conditioning

    Returns
    -------
    pd.DataFrame
        Resulting average proximities of cluster year-by-year. year=-100 is overall cluster across all years.
    """
    nw.decondition()

    if seed is not None:
        nw.set_random_seed(seed)

    if times is None:
        logging.info("Getting years.")
        times = np.array(nw.get_times_list())
        times = np.sort(times)

    # First, derive clusters
    if depth > 0:
        nw.condition(tokens=focal_token, times=times, context=context,depth=depth,weight_cutoff=weight_cutoff,compositional=compositional,reverse_ties=reverse_ties)

    else:
        nw.condition(times=times,weight_cutoff=weight_cutoff, context=context,compositional=compositional,
                     reverse_ties=reverse_ties)
    # Get clusters
    clusters = nw.cluster(interest_list=interest_list, levels=levels, to_measure=[proximity],
                          algorithm=consensus_louvain)
    cluster_dict = {}
    cluster_dataframe = []
    logging.info("Extracting relevant clusters at level {} across all years {}".format(levels,times))
    for cl in clusters:
        if cl['level'] == 0:  # Use zero cluster, where all tokens are present, to get proximities
            rev_proxim = nw.pd_format(cl['measures'])[0].loc[:, focal_token]
            proxim = nw.pd_format(cl['measures'])[0].loc[focal_token, :]
        if len(cl['graph'].nodes) > 0 and cl['level'] == levels:  # Consider only the last level
            # Get List of tokens
            nodes = nw.ensure_tokens(list(cl['graph'].nodes))
            # Check if this is a cluster of interest
            if len(np.intersect1d(nodes, interest_list)) > 0:
                proximate_nodes = proxim.reindex(nodes, fill_value=0)
                rev_proximate_nodes = rev_proxim.reindex(nodes, fill_value=0)
                # We only care about proximate nodes
                proximate_nodes = proximate_nodes[proximate_nodes > cluster_cutoff]
                if len(proximate_nodes) > 0:
                    cluster_measures = return_measure_dict(proximate_nodes)
                    mean_cluster_prox = np.mean(proximate_nodes)
                    mean_cluster_rev_prox = np.mean(rev_proximate_nodes)
                    top_node = proximate_nodes.idxmax()
                    # Default cluster entry
                    year = -100
                    name = "-".join(list(proximate_nodes.nlargest(5).index))
                    df_dict = {'Year': year, 'Level': cl['level'], 'Clustername': name, 'Prom_Node': top_node,
                               'Parent': cl['parent'], 'Cluster_Avg_Prox': mean_cluster_prox,
                               'Cluster_rev_proximity': mean_cluster_rev_prox, 'Nr_ProxNodes': len(proximate_nodes),
                               'NrNodes': len(nodes), 'Ma': 0}
                    cluster_dict.update({name: cl})
                    df_dict.update(cluster_measures)
                    cluster_dataframe.append(df_dict.copy())



    for year in times:
        nw.decondition()
        nw.condition(times=year,weight_cutoff=weight_cutoff, context=context, compositional=compositional,
                     reverse_ties=reverse_ties)

        if moving_average is not None:
            start_year = max(times[0], year - moving_average[0])
            end_year = min(times[-1], year + moving_average[1])
            ma_years = np.arange(start_year, end_year + 1)
        else:
            ma_years = year
        logging.info(
            "Calculating proximities for fixed relevant clusters for year {} with moving average -{} to {} over {}".format(year,
                                                                                                                   moving_average[
                                                                                                                       0],
                                                                                                                   moving_average[
                                                                                                                       1], ma_years))
        if do_reverse is True:
            year_proxim = nw.proximities()
            year_proxim = nw.pd_format(year_proxim)[0]
            rev_proxim = year_proxim.loc[:, focal_token]
            proxim = year_proxim.loc[focal_token, :]
        else:
            year_proxim = nw.proximities(focal_tokens=[focal_token])
            year_proxim = nw.pd_format(year_proxim)[0]
            proxim = year_proxim.loc[focal_token, :]
            rev_proxim = 0
        for cl_name in cluster_dict:
            cl = cluster_dict[cl_name]
            nodes = nw.ensure_tokens(list(cl['graph'].nodes))
            proximate_nodes = proxim.reindex(nodes, fill_value=0)
            if do_reverse is True:
                rev_proximate_nodes = rev_proxim.reindex(nodes, fill_value=0)
            else:
                rev_proximate_nodes = [0]
            proximate_nodes = proximate_nodes[proximate_nodes > cluster_cutoff]

            mean_cluster_prox = np.mean(proximate_nodes)
            mean_cluster_rev_prox = np.mean(rev_proxim)
            cluster_measures = return_measure_dict(proximate_nodes)
            if len(proximate_nodes) > 0:
                top_node = proximate_nodes.idxmax()
            else:
                top_node = "empty"
            df_dict = {'Year': year, 'Level': cl['level'], 'Clustername': cl_name, 'Prom_Node': top_node,
                       'Parent': cl['parent'], 'Cluster_Avg_Prox': mean_cluster_prox,
                       'Cluster_rev_proximity': mean_cluster_rev_prox, 'Nr_ProxNodes': len(proximate_nodes),
                       'NrNodes': len(nodes), 'Ma': len(ma_years)}
            df_dict.update(cluster_measures)
            cluster_dataframe.append(df_dict.copy())

    df = pd.DataFrame(cluster_dataframe)

    if filename is not None:
        filename = check_create_folder(filename)
        df.to_excel(filename + ".xlsx")
        # nw.export_gefx(filename=filename)
    return df


def extract_all_clusters(level: int, cutoff: float, focal_token: str,
                         snw, depth: Optional[int] = 0, context:Optional[list] = None,
                         interest_list: Optional[list] = None, algorithm: Optional[Callable] = None,
                         times: Optional[Union[list, int]] = None,
                         filename: Optional[str] = None, compositional: Optional[bool] = False,
                         reverse_ties: Optional[bool] = False, add_focal_to_clusters: Optional[bool] = False,seed: Optional[int] = None) -> pd.DataFrame:
    """
    Create and extract all clusters relative to a focal token until a given level.

    Parameters
    ----------
    level : int
        How many levels to cluster?
    cutoff : float
        Cutoff weight for querying network
    depth : int
        If nonzero, query an ego network of given depth instead of full network, default is 0
    focal_token : str
        The focal token to consider for proximities and ego network
    semantic_network : neo4jnw
        Initiated semantic network
    interest_list : list
        List of tokens that are of interest. Clusters that do not contain these tokens are discarded and not further clustered for deeper levels
    algorithm : callable
        Clustering algorithm to use
    times : list, int
        Times to use when conditioning
    filename : str
        If not None, dataframe is saved to filename
    compositional : bool
        Whether to use compositional ties
    reverse_ties : bool
        Whether to reverse ties after conditioning
    add_focal_to_clusters: bool
        If true, add focal token to each cluster before clustering
    Returns
    -------
    pd.DataFrame:
        DataFrame with all cluster and all tokens by cluster levels.
    """

    if algorithm is None:
        algorithm = consensus_louvain

    if add_focal_to_clusters:
        add_focal_to_clusters=focal_token
    else:
        add_focal_to_clusters=None

    snw.decondition()

    if seed is not None:
        snw.set_random_seed(seed)

    dataframe_list = []

    # First, derive clusters
    if depth > 0:
        snw.condition(tokens=focal_token, times=times, context=context,depth=depth,weight_cutoff=cutoff,compositional=compositional,reverse_ties=reverse_ties)

    else:
        snw.condition(times=times,weight_cutoff=cutoff, context=context,compositional=compositional,
                     reverse_ties=reverse_ties)
    # Get clusters
    clusters = snw.cluster(interest_list=interest_list, levels=level, to_measure=[proximity],
                          algorithm=algorithm, add_ego_tokens=add_focal_to_clusters)
    for cl in clusters:
        if cl['level'] == 0:
            rev_proxim = snw.pd_format(cl['measures'])[0].loc[:, focal_token]
            proxim = snw.pd_format(cl['measures'])[0].loc[focal_token, :]
        if len(cl['graph'].nodes) >= 1 and cl['level'] > 0:
            nodes = snw.ensure_tokens(list(cl['graph'].nodes))
            proximate_nodes = proxim.reindex(nodes, fill_value=0)
            proximate_nodes = proximate_nodes[proximate_nodes > 0]
            if len(proximate_nodes) > 0:
                cluster_measures = return_measure_dict(proximate_nodes)

                top_node = proximate_nodes.idxmax()
                # Default cluster entry
                node_prox = 0
                node_rev_prox = 0
                delta_prox = -100
                name = "-".join(list(proximate_nodes.nlargest(5).index))
                df_dict = {'Token': name, 'Level': cl['level'], 'Clustername': cl['name'], 'Prom_Node': top_node,
                           'Parent': cl['parent'], 'Proximity': node_prox,
                           'Rev_Proximity': node_rev_prox, 'Delta_Proximity': delta_prox,
                           'Nr_ProxNodes': len(proximate_nodes), 'NrNodes': len(nodes)}
                df_dict.update(cluster_measures)
                dataframe_list.append(df_dict.copy())
                # if len(proximate_nodes) > 0:
                # logging.info("Name: {}, Level: {}, Parent: {}".format(cl['name'], cl['level'], cl['parent']))
                # logging.info("Nodes: {}".format(nodes))
                # Add focal node
                if focal_token in nodes:
                    proximate_nodes[focal_token] = -999
                for node in list(proximate_nodes.index):
                    if proxim.reindex([node], fill_value=0)[0] > 0 or node==focal_token:
                        node_prox = proxim.reindex([node], fill_value=0)[0]
                        node_rev_prox = rev_proxim.reindex([node], fill_value=0)[0]
                        delta_prox = node_prox - node_rev_prox
                        df_dict = {'Token': node, 'Level': cl['level'], 'Clustername': cl['name'],
                                   'Prom_Node': top_node,
                                   'Parent': cl['parent'],
                                   'Proximity': node_prox,
                                   'Rev_Proximity': node_rev_prox, 'Delta_Proximity': delta_prox,
                                   'Nr_ProxNodes': len(proximate_nodes), 'NrNodes': len(nodes)}
                        df_dict.update(cluster_measures)
                        dataframe_list.append(df_dict.copy())

    df = pd.DataFrame(dataframe_list)

    if filename is not None:
        df.to_excel(check_create_folder(filename + ".xlsx"))
        snw.export_gefx(filename=check_create_folder(filename + ".gexf"))

    return df


def return_measure_dict(vec: Union[list, np.array]):
    """
    Helper function to create cluster measures

    Parameters
    ----------
    vec:
        vector of proximities or centralities

    Returns
    -------
        dict of measures
    """
    if isinstance(vec, list):
        vec = np.array(vec)
    if len(vec) == 0:
        return {}

    avg = np.mean(vec)
    weights = vec / np.sum(vec)
    w_avg = np.average(vec, weights=weights)
    sum_vec = np.sum(vec)
    sd = np.std(vec)

    return {"Avg": avg, "w_Avg": w_avg, "Std": sd, "Sum": sum_vec}
