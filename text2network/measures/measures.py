import logging
from typing import Optional, Callable, Union, List, Dict

import numpy as np
import pandas as pd

# from src.classes import neo4jnw
# from src.classes.neo4jnw import neo4j_network
from text2network.utils.file_helpers import check_create_folder
from text2network.functions.graph_clustering import consensus_louvain
from text2network.functions.node_measures import proximity, centrality
# from src.classes import neo4jnw
from text2network.utils.input_check import input_check



def average_cluster_proximities(focal_token: str,  nw, levels: int,
                                interest_list: Optional[list]=None,
                                times: Optional[Union[list, int]] = None,
                                depth: Optional[int] = 1, context: Optional[list] = None,
                                weight_cutoff: Optional[float] = None,
                                cluster_cutoff: Optional[Union[float,int]] = 0, do_reverse: Optional[bool] = False,
                                add_individual_nodes: Optional[bool] = False, year_by_year: Optional[bool] = False,
                                include_all_levels: Optional[bool] = False,
                                moving_average: Optional[tuple] = None,
                                to_back_out:Optional[bool]=False,
                                filename: Optional[str] = None,
                                algorithm: Optional[Callable] = None,
                                compositional: Optional[bool] = False,
                                reverse_ties: Optional[bool] = False,
                                symmetric: Optional[bool] = False, max_degree:Optional[int]=None,
                                export_network: Optional[bool] = False,
                                add_focal_to_clusters: Optional[bool] = False,
                                mode: Optional[str] = "replacement", occurrence: Optional[bool] = False,
                                seed: Optional[int] = None) -> pd.DataFrame:
    """
    First, derives clusters from overall network (across all years), then creates year-by-year average proximities for these clusters

    Parameters
    ----------
    focal_token: str
        Token to which proximities are calculated
    interest_list: list
        List of tokens of relevance. Only clusters that contain these tokens are retained.
    nw: neo4jnw
        Semantic Network
    levels: int
        How many levels to cluster?
    times: list, int
        Overall timeframe to consider
    depth: int
        Depth of the ego network of focal_token to consider. If 0, consider whole semantic network.
    context: list
        Contextual tokens
    weight_cutoff: float
        Cutoff when querying the network
    do_reverse: bool
        Calculate reverse proximities, note this needs to query the entire graph for each year.
    seed : int
        numpy random seed (e.g. for clustering)
    algorithm: callable
        Clustering algorithm (consensus_louvain by default)
    include_all_levels: bool
        If True, dataframe will include all cluster levels. If False, only the last one. Default is False.
    year_by_year: bool
        After getting the overall clustering across times, take the clusters and calculate measures
        for each element in times. Default is False.
    add_individual_nodes: bool
        If True, add all tokens to the DataFrame with their respective measures and cluster associations. Default is False.
    moving_average: tuple
        Consider a moving_average window for years, given as tuple where (nr_prior,nr_past)
        So for example, a length 3 window around the focal year would be given by (1,1).
        A length 3 window behind the focal year would be (2,0)
    to_back_out: bool
        Whether to back out network where i,j is the discounted and weighted similarity between i and j across
        all paths in the network (cf. centrality measures). Default is False.
    cluster_cutoff: float,int
        If int:
            sparsify with cluster_cutoff % cutoff
        if float:
            After clusters are derived, throw away proximities of less than this value
    filename: str
        If given, will save the resulting dataframe to this file as xlsx
    compositional : bool
        Whether to use compositional ties
    reverse_ties : bool
        Whether to reverse ties after conditioning
    symmetric : bool
        Whether to make ties symmetric after conditioning
    export_network : bool
        If True, will try to export the network as gexf File
    add_focal_to_clusters: bool
        If true, add focal token to each cluster before clustering
    mode: str
        Whether to derive replacement or contextual clusters
            "replacement" (Default)
            "context"
    occurrence: bool
        When mode is set to "context", query either
            False (Default): Context are those words that are plausible replacement in the sentence
            True: Contexts are the words that actually occur in the sentence
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
        query_times=None
    else:
        query_times=times

    if algorithm is None:
        algorithm = consensus_louvain

    if isinstance(cluster_cutoff,int):
        percentage=cluster_cutoff
        cluster_cutoff=0
    else:
        percentage=0

    if cluster_cutoff==None:
        cluster_cutoff=0

    if symmetric or reverse_ties:
        if depth <= 1:
            depth = 1

    # First, derive clusters
    if mode == "context":
        nw.context_condition(tokens=focal_token, times=query_times, depth=depth, weight_cutoff=weight_cutoff,
                             occurrence=occurrence, batchsize=None, max_degree=max_degree)

    else:
        nw.condition(tokens=focal_token, times=query_times, context=context, depth=depth, weight_cutoff=weight_cutoff,
                     compositional=compositional,max_degree=max_degree)

    if percentage>0:
        nw.sparsify(percentage)
    if to_back_out:
        nw.to_backout()
    if reverse_ties:
        nw.to_reverse()
    if symmetric:
        nw.to_symmetric()
    # Populate alter tokens if not given
    if interest_list is None:
        interest_list=nw.ensure_tokens(list(nw.graph.nodes))

    # Get clusters
    if add_focal_to_clusters:
        add_ego_tokens = focal_token
    else:
        add_ego_tokens = None
    clusters = nw.cluster(interest_list=interest_list, levels=levels, to_measure=[proximity, centrality],
                          algorithm=algorithm, add_ego_tokens=add_ego_tokens)

    cluster_dict = {}
    cluster_dataframe = []
    logging.info("Extracting relevant clusters at level {} across all years {}".format(levels, times))
    for cl in clusters:
        if cl['level'] == 0:  # Use zero cluster, where all tokens are present, to get proximities
            rev_proxim = nw.pd_format(cl['measures'])[0].loc[:, focal_token]
            proxim = nw.pd_format(cl['measures'])[0].loc[focal_token, :]
            overall_pagerank = nw.pd_format(cl['measures'])[1]['normedPageRank']
        if len(cl['graph'].nodes) > 0 and (cl['level'] == levels or include_all_levels):  # Consider only the last level
            # Get List of tokens
            nodes = nw.ensure_tokens(list(cl['graph'].nodes))
            # Check if this is a cluster of interest
            if len(np.intersect1d(nodes, interest_list)) > 0:
                proximate_nodes = proxim.reindex(nodes, fill_value=0)
                rev_proximate_nodes = rev_proxim.reindex(nodes, fill_value=0)
                overall_cluster_pagerank = overall_pagerank.reindex(nodes, fill_value=0)
                cluster_pagerank=nw.pd_format(cl['measures'])[1]['normedPageRank'].reindex(nodes, fill_value=0)
                # We only care about proximate nodes
                proximate_nodes = proximate_nodes[proximate_nodes > cluster_cutoff]
                rev_proximate_nodes = rev_proximate_nodes[rev_proximate_nodes > cluster_cutoff]
                if len(proximate_nodes) > 0:
                    cluster_measures = return_measure_dict(proximate_nodes)
                    rev_cluster_measures=return_measure_dict(rev_proximate_nodes, prefix="rev_")
                    overall_pagerank_measures=return_measure_dict(overall_cluster_pagerank, prefix="opr_")
                    pagerank_measures = return_measure_dict(cluster_pagerank, prefix="pr_")
                    # Default cluster entry
                    year = -100
                    name = "-".join(list(proximate_nodes.nlargest(5).index))

                    topelements_pr = np.array(cluster_pagerank.nlargest(6).index)
                    topelements_pr =list(topelements_pr[topelements_pr != focal_token])

                    if topelements_pr==[]:
                        topelements_pr=["empty"]
                    name_pr = "-".join(topelements_pr[0:5])
                    top_node_pr = topelements_pr[0]
                    top_node = proximate_nodes.idxmax()
                    df_dict = {'Year': year, 'Token': name, 'Token_pr': name_pr, 'Prom_Node_pr': top_node_pr,'Prom_Node': top_node, 'Level': cl['level'],
                               'Cluster_Id': cl['name'],
                               'Parent': cl['parent'], 'Nr_ProxNodes': len(proximate_nodes),
                               'NrNodes': len(nodes), 'Ma': 0, 'Node_Proximity': 0,
                               'Node_Rev_Proximity': 0, 'Node_Delta_Proximity': -100, 'Node_Centrality':0}
                    cluster_dict.update({name: cl})
                    df_dict.update(cluster_measures)
                    df_dict.update(rev_cluster_measures)
                    df_dict.update(pagerank_measures)
                    df_dict.update(overall_pagerank_measures)
                    cluster_dataframe.append(df_dict.copy())
                    # Add each node
                    if add_individual_nodes:
                        if focal_token in nodes:
                            proximate_nodes[focal_token] = -999
                        for node in list(proximate_nodes.index):
                            if proxim.reindex([node], fill_value=0)[0] > 0 or node == focal_token:
                                node_prox = proxim.reindex([node], fill_value=0)[0]
                                node_cent = cluster_pagerank.reindex([node], fill_value=0)[0]
                                node_rev_prox = rev_proxim.reindex([node], fill_value=0)[0]
                                delta_prox = node_prox - node_rev_prox
                                df_dict = {'Year': year, 'Token': node, 'Token_pr': node, 'Prom_Node_pr': top_node_pr,'Prom_Node': top_node, 'Level': cl['level'],
                                           'Cluster_Id': cl['name'],
                                           'Parent': cl['parent'],
                                           'Nr_ProxNodes': len(proximate_nodes),
                                           'NrNodes': len(nodes), 'Ma': 0, 'Node_Proximity': node_prox,
                                           'Node_Rev_Proximity': node_rev_prox, 'Node_Delta_Proximity': delta_prox,'Node_Centrality':node_cent}
                                df_dict.update(cluster_measures)
                                df_dict.update(rev_cluster_measures)
                                df_dict.update(pagerank_measures)
                                df_dict.update(overall_pagerank_measures)
                                cluster_dataframe.append(df_dict.copy())

    if year_by_year:
        for year in times:
            nw.decondition()

            if moving_average is not None:
                start_year = max(times[0], year - moving_average[0])
                end_year = min(times[-1], year + moving_average[1])
                ma_years = list(np.arange(start_year, end_year + 1))
                logging.info(
                    "Calculating proximities for fixed relevant clusters for year {} with moving average -{} to {} over {}".format(
                        year,
                        moving_average[
                            0],
                        moving_average[
                            1], ma_years))
            else:
                ma_years = [year]



            if mode == "context":
                nw.context_condition(tokens=focal_token, times=ma_years, depth=depth, weight_cutoff=weight_cutoff,
                                     occurrence=occurrence, max_degree=max_degree)

            else:
                nw.condition(tokens=focal_token, depth=depth, times=ma_years, weight_cutoff=weight_cutoff, context=context, compositional=compositional, max_degree=max_degree)


            if to_back_out:
                nw.to_backout()
            if symmetric:
                nw.to_symmetric()
            if reverse_ties:
                nw.to_reverse()

            if do_reverse is True:
                year_proxim = nw.proximities()
                year_proxim = nw.pd_format(year_proxim)[0]
                rev_proxim = year_proxim.loc[:, focal_token]
                proxim = year_proxim.loc[focal_token, :]
                year_pagerank= nw.centralities()
                year_pagerank = nw.pd_format(year_pagerank)[0]['normedPageRank']
            else:
                year_proxim = nw.proximities(focal_tokens=[focal_token])
                year_proxim = nw.pd_format(year_proxim)[0]
                proxim = year_proxim.loc[focal_token, :]
                rev_proxim = 0
                year_pagerank= nw.centralities()
                year_pagerank = nw.pd_format(year_pagerank)[0]['normedPageRank']
            for cl_name in cluster_dict:
                cl = cluster_dict[cl_name]
                nodes = nw.ensure_tokens(list(cl['graph'].nodes))
                proximate_nodes = proxim.reindex(nodes, fill_value=0)
                proximate_nodes = proximate_nodes[proximate_nodes > cluster_cutoff]
                overall_cluster_year_pagerank = year_pagerank.reindex(nodes, fill_value=0)
                if do_reverse is True:
                    rev_proximate_nodes = rev_proxim.reindex(nodes, fill_value=0)
                    rev_proximate_nodes = rev_proximate_nodes[rev_proximate_nodes > cluster_cutoff]
                else:
                    rev_proximate_nodes = [0]

                cluster_measures = return_measure_dict(proximate_nodes)
                rev_cluster_measures = return_measure_dict(rev_proximate_nodes, prefix="rev_")
                overall_pagerank_measures = return_measure_dict(overall_cluster_year_pagerank, prefix="opr_")
                pagerank_measures = return_measure_dict(overall_cluster_year_pagerank, prefix="pr_")

                if len(proximate_nodes) > 0:
                    top_node = proximate_nodes.idxmax()

                else:
                    top_node = "empty"

                topelements_pr = np.array(cluster_pagerank.nlargest(6).index)
                topelements_pr =list(topelements_pr[topelements_pr != focal_token])
                if topelements_pr == []:
                    topelements_pr = ["empty"]
                name_pr = "-".join(topelements_pr[0:5])
                top_node_pr = topelements_pr[0]

                df_dict = {'Year': year, 'Token': cl_name, 'Token_pr': name_pr, 'Prom_Node_pr': top_node_pr,'Prom_Node': top_node, 'Level': cl['level'],
                           'Cluster_Id': cl['name'],
                           'Parent': cl['parent'], 'Nr_ProxNodes': len(proximate_nodes),
                           'NrNodes': len(nodes), 'Ma': len(ma_years), 'Node_Proximity': 0,
                           'Node_Rev_Proximity': 0, 'Node_Delta_Proximity': -100}

                df_dict.update(cluster_measures)
                df_dict.update(rev_cluster_measures)
                df_dict.update(pagerank_measures)
                df_dict.update(overall_pagerank_measures)
                cluster_dataframe.append(df_dict.copy())

    df = pd.DataFrame(cluster_dataframe)

    if filename is not None:
        if export_network:
            nw.export_gefx(filename=check_create_folder(filename + ".gexf"))
        filename = check_create_folder(filename + ".xlsx")
        df.to_excel(filename)

    return df


def extract_all_clusters(level: int, cutoff: float, focal_token: str,
                         snw, depth: Optional[int] = None, context: Optional[list] = None,
                         cluster_cutoff: Optional[float] = 0,
                         interest_list: Optional[list] = None, algorithm: Optional[Callable] = None,
                         times: Optional[Union[list, int]] = None,
                         filename: Optional[str] = None, compositional: Optional[bool] = False,
                         reverse_ties: Optional[bool] = False, add_focal_to_clusters: Optional[bool] = False,
                         to_back_out: Optional[bool] = False,
                         mode: Optional[str] = "replacement", occurrence: Optional[bool] = False,
                         seed: Optional[int] = None) -> pd.DataFrame:
    return average_cluster_proximities(focal_token=focal_token, interest_list=interest_list, nw=snw, levels=level, times=times, depth=depth, context=context,
                                       weight_cutoff=cutoff, cluster_cutoff=cluster_cutoff, do_reverse=True,
                                       add_individual_nodes=True, year_by_year=False, moving_average=None,
                                       filename=filename, include_all_levels=True, to_back_out=to_back_out,
                                       compositional=compositional, reverse_ties=reverse_ties,
                                       add_focal_to_clusters=add_focal_to_clusters, mode=mode, occurrence=occurrence,
                                       algorithm=algorithm, seed=seed)



def return_measure_dict(vec: Union[list, np.array], prefix=""):
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


    avg = np.mean(vec)
    sum_vec = np.sum(vec)
    if sum_vec <= 0:
        w_avg = 0
    else:
        weights = vec / sum_vec
        w_avg = np.average(vec, weights=weights)
    if sum(vec.shape) >= 2:
        sd = np.std(vec)
    else:
        sd = 0

    return {prefix+"Avg": avg, prefix+"w_Avg": w_avg, prefix+"Std": sd, prefix+"Sum": sum_vec}
