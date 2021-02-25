from typing import Optional

from src.classes import neo4jnw
from src.functions.file_helpers import check_create_folder
from src.functions.node_measures import proximity, centrality
from src.functions.graph_clustering import consensus_louvain
import numpy as np
import pandas as pd
import logging
import configparser


def yearly_centralities(nw, year_list, focal_tokens=None, types=["PageRank", "normedPageRank"], ego_nw_tokens=None,
                        depth=1, context=None, weight_cutoff=None, norm_ties=None):
    """
    Compute directly year-by-year centralities for provided list.

    Parameters
    ----------
    year_list : list
        List of years for which to calculate centrality.
    focal_tokens : list, str
        List of tokens of interest. If not provided, centralities for all tokens will be returned.
    types : list, optional
        types of centrality to calculate. The default is ["PageRank"].
    ego_nw_tokens : list, optional - used when conditioning
         List of tokens for an ego-network if desired. Only used if no graph is supplied. The default is None.
    depth : TYPE, optional - used when conditioning
        Maximal path length for ego network. Only used if no graph is supplied. The default is 1.
    context : list, optional - used when conditioning
        List of tokens that need to appear in the context distribution of a tie. The default is None.
    weight_cutoff : float, optional - used when conditioning
        Only links of higher weight are considered in conditioning.. The default is None.
    norm_ties : bool, optional - used when conditioning
        Please see semantic network class. The default is True.

    Returns
    -------
    dict
        Dict of years with dict of centralities for focal tokens.

    """

    # Get default normation behavior
    if norm_ties == None:
        norm_ties = nw.norm_ties

    cent_year = {}
    assert isinstance(year_list, list), "Please provide list of years."
    for year in year_list:
        cent_measures = nw.centralities(focal_tokens=focal_tokens, types=types, years=[
            year], ego_nw_tokens=ego_nw_tokens, depth=depth, context=context, weight_cutoff=weight_cutoff,
                                          norm_ties=norm_ties)
        cent_year.update({year: cent_measures})

    return {'yearly_centrality': cent_year}


def average_fixed_cluster_proximities(focal_token:str, interest_list:list, nw:neo4jnw, levels:int, depth:Optional[int]=1, context:Optional[list]=None, weight_cutoff:Optional[float]=None, norm_ties:Optional[bool]=None, cluster_cutoff:Optional[float]=0,filename:Optional[str]=None)->pd.DataFrame:
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
    norm_ties: bool
        Whether to norm ties (compositional mode)
    cluster_cutoff: float
        After clusters are derived, throw away proximities of less than this value
    filename: str
        If given, will save the resulting dataframe to this file as xlsx

    Returns
    -------
    pd.DataFrame
        Resulting average proximities of cluster year-by-year. year=-100 is overall cluster across all years.
    """
    # First, derive clusters
    if depth > 0:
        clusters = nw.cluster(ego_nw_tokens=focal_token, interest_list=interest_list, depth=depth, levels=levels, weight_cutoff=weight_cutoff,
                                            to_measure=[proximity], algorithm=consensus_louvain)
    else:
        clusters = nw.cluster(levels=levels, interest_list=interest_list, weight_cutoff=weight_cutoff, to_measure=[proximity],
                                            algorithm=consensus_louvain)

    cluster_dict={}
    cluster_dataframe=[]
    logging.info("Extracting relevant clusters at level across all years {}".format(levels))
    for cl in clusters:
        if cl['level'] == 0: # Use zero cluster, where all tokens are present, to get proximities
            rev_proxim = nw.pd_format(cl['measures'])[0].loc[:, focal_token]
            proxim = nw.pd_format(cl['measures'])[0].loc[focal_token, :]
        if len(cl['graph'].nodes) > 2 and cl['level'] == levels: # Consider only the last level
            # Get List of tokens
            nodes = nw.ensure_tokens(list(cl['graph'].nodes))
            # Check if this is a cluster of interest
            if len(np.intersect1d(nodes, interest_list))>0:
                proximate_nodes = proxim.reindex(nodes, fill_value=0)
                rev_proximate_nodes = rev_proxim.reindex(nodes, fill_value=0)
                # We only care about proximate nodes
                proximate_nodes = proximate_nodes[proximate_nodes > cluster_cutoff]
                if len(proximate_nodes)>0:
                    mean_cluster_prox = np.mean(proximate_nodes)
                    mean_cluster_rev_prox = np.mean(rev_proxim)
                    top_node = proximate_nodes.idxmax()
                    # Default cluster entry
                    year = -100
                    name="-".join(list(proximate_nodes.nlargest(5).index))
                    df_dict = {'Year':year,'Level': cl['level'], 'Clustername': name, 'Prom_Node': top_node,
                               'Parent': cl['parent'], 'Cluster_Avg_Prox': mean_cluster_prox,  'Cluster_rev_proximity': mean_cluster_rev_prox, 'Nr_ProxNodes': len(proximate_nodes), 'NrNodes': len(nodes)}
                    cluster_dict.update({name: cl})
                    cluster_dataframe.append(df_dict.copy())

    logging.info("Getting years.")
    years = np.array(nw.get_times_list())


    for year in years:
        nw.decondition()
        logging.info("Calculating proximities for fixed relevant clusters for year {}".format(year))
        year_proxim=nw.proximities(focal_tokens=focal_token, years=year, context=context, weight_cutoff=weight_cutoff, norm_ties=norm_ties)
        rev_proxim = nw.pd_format(year_proxim)[0].loc[:, focal_token]
        proxim = nw.pd_format(year_proxim)[0].loc[focal_token, :]
        for cl_name in cluster_dict:
            cl=cluster_dict[cl_name]
            nodes = nw.ensure_tokens(list(cl['graph'].nodes))
            proximate_nodes = proxim.reindex(nodes, fill_value=0)
            rev_proximate_nodes = rev_proxim.reindex(nodes, fill_value=0)
            proximate_nodes = proximate_nodes[proximate_nodes > cluster_cutoff]
            mean_cluster_prox = np.mean(proximate_nodes)
            mean_cluster_rev_prox = np.mean(rev_proxim)
            top_node = proximate_nodes.idxmax()
            df_dict = {'Year':year,'Level': cl['level'], 'Clustername': cl_name, 'Prom_Node': top_node,
                       'Parent': cl['parent'], 'Cluster_Avg_Prox': mean_cluster_prox,
                       'Cluster_rev_proximity': mean_cluster_rev_prox, 'Nr_ProxNodes': len(proximate_nodes),
                       'NrNodes': len(nodes)}
            cluster_dataframe.append(df_dict.copy())


    df = pd.DataFrame(cluster_dataframe)

    if filename is not None:
        df.to_excel(filename)

    return df



def extract_all_clusters(config:configparser.ConfigParser, level:int, cutoff:float, focal_token:str, semantic_network:neo4jnw, depth:Optional[int]=0, interest_list:Optional[list]=None)->pd.DataFrame:
    """
    Create and extract all clusters relative to a focal token until a given level.

    Parameters
    ----------
    config : ConfigParser
        Config file
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

    Returns
    -------
    pd.DataFrame:
        DataFrame with all cluster and all tokens by cluster levels.
    """


    filename = "".join(
        [config['Paths']['csv_outputs'], "/", str(focal_token), "_con_egocluster_lev", str(level), "_cut",
         str(cutoff), "_depth", str(depth),
         ".xlsx"])
    filename_csv = "".join(
        [config['Paths']['csv_outputs'], "/", str(focal_token), "_con_egocluster_lev", str(level), "_cut",
         str(cutoff), "_depth", str(depth),
         ".csv"])
    logging.info("Network clustering: {}".format(filename))
    filename = check_create_folder(filename)
    filename_csv = check_create_folder(filename_csv)
    semantic_network.deconditition()
    dataframe_list = []
    if depth > 0:
        clusters = semantic_network.cluster(ego_nw_tokens=focal_token, interest_list=interest_list,depth=depth, levels=level,
                                            weight_cutoff=cutoff,
                                            to_measure=[proximity], algorithm=consensus_louvain)
    else:
        clusters = semantic_network.cluster(levels=level, interest_list=interest_list, weight_cutoff=cutoff, to_measure=[proximity],
                                            algorithm=consensus_louvain)
        # clusters = semantic_network.cluster(levels=level, weight_cutoff=cutoff, to_measure=[proximity])
    for cl in clusters:
        if cl['level'] == 0:
            rev_proxim = semantic_network.pd_format(cl['measures'])[0].loc[:, focal_token]
            proxim = semantic_network.pd_format(cl['measures'])[0].loc[focal_token, :]
        if len(cl['graph'].nodes) > 2 and cl['level'] > 0:
            nodes = semantic_network.ensure_tokens(list(cl['graph'].nodes))
            proximate_nodes = proxim.reindex(nodes, fill_value=0)
            proximate_nodes = proximate_nodes[proximate_nodes > 0]
            if len(proximate_nodes) > 0:
                mean_cluster_prox = np.mean(proximate_nodes)
                top_node = proximate_nodes.idxmax()
                # Default cluster entry
                node_prox = 0
                node_rev_prox = 0
                delta_prox = -100
                name = "-".join(list(proximate_nodes.nlargest(5).index))
                df_dict = {'Token': name, 'Level': cl['level'], 'Clustername': cl['name'], 'Prom_Node': top_node,
                           'Parent': cl['parent'], 'Cluster_Avg_Prox': mean_cluster_prox, 'Proximity': node_prox,
                           'Rev_Proximity': node_rev_prox, 'Delta_Proximity': delta_prox,
                           'Nr_ProxNodes': len(proximate_nodes), 'NrNodes': len(nodes)}
                dataframe_list.append(df_dict.copy())
                if len(proximate_nodes) > 0:
                    logging.info("Name: {}, Level: {}, Parent: {}".format(cl['name'], cl['level'], cl['parent']))
                    logging.info("Nodes: {}".format(nodes))
                    logging.info(proximate_nodes)
                for node in list(proximate_nodes.index):
                    if proxim.reindex([node], fill_value=0)[0] > 0:
                        node_prox = proxim.reindex([node], fill_value=0)[0]
                        node_rev_prox = rev_proxim.reindex([node], fill_value=0)[0]
                        delta_prox = node_prox - node_rev_prox
                        df_dict = {'Token': node, 'Level': cl['level'], 'Clustername': cl['name'],
                                   'Prom_Node': top_node,
                                   'Parent': cl['parent'], 'Cluster_Avg_Prox': mean_cluster_prox,
                                   'Proximity': node_prox,
                                   'Rev_Proximity': node_rev_prox, 'Delta_Proximity': delta_prox,
                                   'Nr_ProxNodes': len(proximate_nodes), 'NrNodes': len(nodes)}
                        dataframe_list.append(df_dict.copy())

    df = pd.DataFrame(dataframe_list)
    df.to_excel(filename)
    df.to_csv(filename_csv)
    return df