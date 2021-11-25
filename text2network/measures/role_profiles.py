# TODO COMMENTS AND DOCSTRINGS


import logging
from typing import Optional, Union, Callable

import numpy as np
import pandas as pd
from tqdm import tqdm

from text2network.classes.neo4jnw import neo4j_network
# from src.classes import neo4jnw
# from src.classes.neo4jnw import neo4j_network
#
from text2network.functions.graph_clustering import consensus_louvain
from text2network.measures.measures import return_measure_dict
from text2network.utils.file_helpers import check_create_folder

def get_pos_profile(snw: neo4j_network, focal_token: Union[str, int], role_cluster: Union[str, int, list],
                    times: Union[list, int], pos: str, context_mode: Optional[str] = "bidirectional",
                    return_sentiment: Optional[bool] = True, weight_cutoff: Optional[float] = 0) -> pd.DataFrame:
    # Want to have a list here
    if not isinstance(role_cluster, (list, np.ndarray)):
        role_cluster = [role_cluster]
    if not isinstance(times, (list, np.ndarray)):
        times = [times]

    pd_list = []
    for alter in role_cluster:
        pd_list.append(
            snw.get_dyad_context(occurrence=alter, replacement=focal_token, times=times, weight_cutoff=weight_cutoff,
                                 part_of_speech=pos, context_mode=context_mode, return_sentiment=return_sentiment)[
                'dyad_context'])

    df = pd.concat([pd.DataFrame(x) for x in pd_list])
    if len(df) > 0:
        # TODO Think about Sum vs. Mean here
        df = df.groupby(["idx", "pos"]).agg(weight=('weight', 'sum'), sentiment=('sentiment', 'mean'),
                                            subjectivity=('subjectivity', 'mean'))
        # df = df.groupby(["idx", "pos"]).sum()
        df = df.reset_index(drop=False).sort_values(by="weight", ascending=True)
        df["context_token"] = snw.ensure_tokens(df.idx)
        if return_sentiment:
            df = df[["idx", "context_token", "pos", "weight", "sentiment", "subjectivity"]]
        else:
            df = df[["idx", "context_token", "pos", "weight"]]
    else:
        df = None
    return df


def create_YOY_role_profile(snw: neo4j_network, focal_token: Union[str, int], role_cluster: Union[str, int, list],
                            times: Union[list, int], pos_list: Optional[list] = None, keep_top_k: Optional[int] = None,
                            filename: Optional[str] = None, moving_average: Optional[tuple] = None,
                            context_mode: Optional[str] = "bidirectional", return_sentiment: Optional[bool] = True,
                            weight_cutoff: Optional[float] = 0, seed: Optional[int] = None) -> pd.DataFrame:
    snw.decondition()

    if seed is not None:
        snw.set_random_seed(seed)

    if not isinstance(role_cluster, (list, tuple, np.ndarray)):
        role_cluster = [role_cluster]

    if times is None:
        logging.info("Getting years.")
        times = np.array(snw.get_times_list())
        times = np.sort(times)
        query_times = None
    else:
        query_times = times

    cluster_name = "-".join(list(role_cluster[:5]))
    logging.info("YOY Role cluster for {}".format(cluster_name))

    if pos_list is None:
        logging.info("Getting POS in Database")
        res = snw.db.receive_query("MATCH (n:part_of_speech) RETURN DISTINCT n.part_of_speech as pos")
        pos_list = [x['pos'] for x in res if x['pos'] != '.']

    df_list = []
    for year in tqdm(times, desc="Getting yearly profiles for {}".format(cluster_name)):
        if moving_average is not None:
            start_year = max(times[0], year - moving_average[0])
            end_year = min(times[-1], year + moving_average[1])
            ma_years = list(np.arange(start_year, end_year + 1))
            logging.debug(
                "Calculating proximities for fixed relevant clusters for year {} with moving average -{} to {} over {}".format(
                    year,
                    moving_average[
                        0],
                    moving_average[
                        1], ma_years))
        else:
            ma_years = [year]

        year_df_list = []
        for pos in pos_list:
            temp_df = get_pos_profile(snw=snw, focal_token=focal_token, role_cluster=role_cluster, times=ma_years,
                                      pos=pos, context_mode=context_mode, return_sentiment=return_sentiment,
                                      weight_cutoff=weight_cutoff)
            if temp_df is not None:
                if keep_top_k is not None:
                    # temp_df.sort_values(by="weight", ascending=False)
                    # temp_df=temp_df.iloc[0:keep_top_k, :]
                    temp_df = temp_df.nlargest(keep_top_k, columns=["weight"])
                temp_df["time"] = year
                temp_df["ma_time"] = str(ma_years)
                year_df_list.append(temp_df)
        if len(year_df_list) > 0:
            df_list.append(pd.concat(year_df_list))

    if len(df_list) > 0:
        df = pd.concat(df_list)

        if filename is not None:
            df.to_excel(filename + ".xlsx")

    else:
        df = None

    return df


def create_overall_role_profile(snw: neo4j_network, focal_token: Union[str, int], role_cluster: Union[str, int, list],
                                times: Union[list, int], pos_list: Optional[list] = None,
                                keep_top_k: Optional[int] = None,
                                filename: Optional[str] = None, moving_average: Optional[tuple] = None,
                                context_mode: Optional[str] = "bidirectional", return_sentiment: Optional[bool] = True,
                                weight_cutoff: Optional[float] = 0, seed: Optional[int] = None) -> pd.DataFrame:
    snw.decondition()

    if seed is not None:
        snw.set_random_seed(seed)

    if not isinstance(role_cluster, (list, tuple, np.ndarray)):
        role_cluster = [role_cluster]

    if times is None:
        logging.info("Getting years.")
        times = np.array(snw.get_times_list())
        times = np.sort(times)
        query_times = None
    else:
        query_times = times

    cluster_name = "-".join(list(role_cluster[:5]))
    logging.info("YOY Role cluster for {}".format(cluster_name))

    if pos_list is None:
        logging.info("Getting POS in Database")
        res = snw.db.receive_query("MATCH (n:part_of_speech) RETURN DISTINCT n.part_of_speech as pos")
        pos_list = [x['pos'] for x in res if x['pos'] != '.']

    df_list = []
    for pos in pos_list:
        logging.info("Now checking {}".format(pos))
        temp_df = get_pos_profile(snw=snw, focal_token=focal_token, role_cluster=role_cluster, times=times,
                                  pos=pos, context_mode=context_mode, return_sentiment=return_sentiment,
                                  weight_cutoff=weight_cutoff)
        if temp_df is not None:
            if keep_top_k is not None:
                # temp_df.sort_values(by="weight", ascending=False)
                # temp_df=temp_df.iloc[0:keep_top_k, :]
                temp_df = temp_df.nlargest(keep_top_k, columns=["weight"])
            temp_df["time"] = str(times)
            temp_df["ma_time"] = str(times)
            df_list.append(temp_df)

    if len(df_list) > 0:
        df = pd.concat(df_list)

        if filename is not None:
            df.to_excel(filename + ".xlsx")

    else:
        df = None

    return df


def extract_yoy_role_profiles(semantic_network, df_cluster_ids, df_clusters, df_nodes, times, focal_token, keep_top_k,
                              cutoff,
                              depth, context_mode, filename=None, moving_average=None):
    # YOY
    cluster_df_list = []
    for cluster_id in df_cluster_ids:
        logging.info("Considering cluster with id {}".format(cluster_id))
        nodes = df_nodes[df_nodes.Cluster_Id == cluster_id].sort_values(by="Node_Proximity",
                                                                        ascending=False).Token.to_list()
        cluster_name = df_clusters[df_clusters.Cluster_Id == cluster_id].Token.to_list()[0]
        cluster_df = create_YOY_role_profile(snw=semantic_network, focal_token=focal_token, role_cluster=nodes,
                                             weight_cutoff=cutoff, times=times, keep_top_k=keep_top_k,
                                             context_mode=context_mode, moving_average=moving_average)
        if cluster_df is not None:
            print(cluster_df)
            all_years = cluster_df.groupby(by=["idx", "context_token", "pos"], as_index=False).mean()
            all_years.time = 0
            all_years["ma_time"] = 0
            cluster_df = pd.concat([cluster_df, all_years])
            cluster_df["cluster_name"] = cluster_name
            cluster_df_list.append(cluster_df)

    df = pd.concat(cluster_df_list)

    if filename is not None:
        logging.info("Network clustering: {}".format(filename))
        df.to_excel(filename + ".xlsx", merge_cells=False)

    return df


def get_clustered_role_profile(snw: neo4j_network, focal_token: Union[str, int], cluster_nodes: Union[list, str, int],
                               level: int, cluster_cutoff: Optional[float] = 0.0,
                               pos_list: Optional[list] = None, times: Optional[Union[list, int]] = None,
                               keep_top_k: Optional[int] = None, weight_cutoff: Optional[float] = None,
                               context_mode: Optional[str] = "bidirectional",
                               add_individual_nodes: Optional[bool] = True,
                               max_degree: Optional[int] = None, include_all_levels: Optional[bool] = True,
                               sym: Optional[bool] = False, depth: Optional[int] = 1, algorithm: Optional[Callable] = None,
                               export_network: Optional[bool] = True, filename: Optional[str] = None):

    if algorithm is None:
        algorithm = consensus_louvain
    # Get role profile
    df = create_overall_role_profile(snw=snw, focal_token=focal_token, role_cluster=cluster_nodes, times=times,
                                     weight_cutoff=weight_cutoff,
                                     keep_top_k=keep_top_k, context_mode=context_mode, pos_list=pos_list)

    interest_list = np.unique(df.context_token).tolist()

    snw.condition_given_dyad(dyad_substitute=focal_token, dyad_occurring=cluster_nodes, times=times,
                             focal_tokens=interest_list, weight_cutoff=weight_cutoff, depth=depth,
                             max_degree=max_degree)
    logging.info("Finished conditioning")
    clusters = snw.cluster(levels=level, interest_list=interest_list, algorithm=algorithm)
    logging.info("Finished clustering, found {} clusters".format(len(clusters)))

    pos_list = np.unique(df.pos).tolist()
    cluster_dict = {}
    cluster_dataframe = []
    for pos in pos_list:
        logging.info("Extracting clusters for pos: {}".format(pos))
        df_subset = df[df.pos == pos]
        proxim = df_subset[["idx", "context_token", "pos", "weight"]]
        proxim = proxim.set_index(proxim.context_token)
        sentiment = df_subset[["idx", "context_token", "pos", "sentiment"]]
        sentiment = sentiment.set_index(sentiment.context_token)
        subjectivity = df_subset[["idx", "context_token", "pos", "subjectivity"]]
        subjectivity = subjectivity.set_index(subjectivity.context_token)
        logging.info("Extracting relevant clusters at level {} across all years {}".format(level, times))
        for cl in tqdm(clusters, desc="Extracting all clusters for POS {}".format(pos)):
            if len(cl['graph'].nodes) > 0 and (
                    cl['level'] == level or include_all_levels):  # Consider only the last level
                # Get List of tokens
                nodes = snw.ensure_tokens(list(cl['graph'].nodes))
                # Check if this is a cluster of interest
                if len(np.intersect1d(nodes, interest_list)) > 0:
                    proxim_in_cluster = proxim.reindex(nodes, fill_value=0)
                    proxim_in_cluster = proxim_in_cluster[proxim_in_cluster.weight > cluster_cutoff]
                    sentiment_in_cluster = sentiment.reindex(proxim_in_cluster.context_token, fill_value=0)
                    subjectivity_in_cluster = subjectivity.reindex(proxim_in_cluster.context_token, fill_value=0)
                    if len(proxim_in_cluster) > 0:
                        cluster_measures = return_measure_dict(proxim_in_cluster.weight)
                        sent_measures = return_measure_dict(sentiment_in_cluster.sentiment, prefix="sentiment_")
                        sub_measures = return_measure_dict(subjectivity_in_cluster.subjectivity, prefix="subjectivity_")
                        # Default cluster entry
                        year = -100
                        name = "-".join(list(proxim_in_cluster.weight.nlargest(5).index))
                        top_node = proxim_in_cluster.weight.idxmax()
                        df_dict = {'Year': year, 'POS': pos, 'Token': name,
                                   'Prom_Node': top_node, 'Level': cl['level'],
                                   'Cluster_Id': cl['name'],
                                   'Parent': cl['parent'], 'Nr_ProxNodes': len(proxim_in_cluster),
                                   'NrNodes': len(nodes), 'Ma': 0, 'Node_Weight': 0,
                                   'Node_Sentiment': 0, 'Node_Subjectivity': 0, 'Type': "Cluster"}
                        cluster_dict.update({name: cl})
                        df_dict.update(cluster_measures)
                        df_dict.update(sent_measures)
                        df_dict.update(sub_measures)
                        cluster_dataframe.append(df_dict.copy())
                        # Add each node
                        if add_individual_nodes:
                            for node in list(proxim_in_cluster.index):
                                if proxim.reindex([node], fill_value=0).weight[0] > 0:
                                    node_prox = proxim.reindex([node], fill_value=0).weight[0]
                                    node_sent = sentiment.reindex([node], fill_value=0).sentiment[0]
                                    node_sub = subjectivity.reindex([node], fill_value=0).subjectivity[0]
                                    df_dict = {'Year': year, 'POS': pos, 'Token': node,
                                               'Prom_Node': top_node, 'Level': cl['level'],
                                               'Cluster_Id': cl['name'],
                                               'Parent': cl['parent'],
                                               'Nr_ProxNodes': len(proxim_in_cluster),
                                               'NrNodes': len(nodes), 'Ma': 0, 'Node_Weight': node_prox,
                                               'Node_Sentiment': node_sent, 'Node_Subjectivity': node_sub,
                                               'Type': "Cluster"}
                                    df_dict.update(cluster_measures)
                                    df_dict.update(sent_measures)
                                    df_dict.update(sub_measures)
                                    cluster_dataframe.append(df_dict.copy())

    ### For all POS
    pos = "ALL"
    df_subset = df[["idx", "context_token", "pos", "weight", "sentiment", "subjectivity"]]
    df_subset = df_subset.groupby(by=["idx", "context_token"]).agg(weight=('weight', 'sum'),
                                                                   sentiment=('sentiment', 'mean'),
                                                                   subjectivity=('subjectivity', 'mean'))
    df_subset = df_subset.reset_index(drop=False).sort_values(by="weight", ascending=True)
    proxim = df_subset[["idx", "context_token", "weight"]]
    proxim = proxim.set_index(proxim.context_token)
    sentiment = df_subset[["idx", "context_token","sentiment"]]
    sentiment = sentiment.set_index(sentiment.context_token)
    subjectivity = df_subset[["idx", "context_token", "subjectivity"]]
    subjectivity = subjectivity.set_index(subjectivity.context_token)
    logging.info("Extracting relevant clusters at level {} across all years {}".format(level, times))
    for cl in tqdm(clusters, desc="Extracting all clusters for POS {}".format(pos)):
        if len(cl['graph'].nodes) > 0 and (cl['level'] == level or include_all_levels):  # Consider only the last level
            # Get List of tokens
            nodes = snw.ensure_tokens(list(cl['graph'].nodes))
            # Check if this is a cluster of interest
            if len(np.intersect1d(nodes, interest_list)) > 0:
                proxim_in_cluster = proxim.reindex(nodes, fill_value=0)
                proxim_in_cluster = proxim_in_cluster[proxim_in_cluster.weight > cluster_cutoff]
                sentiment_in_cluster = sentiment.reindex(proxim_in_cluster.context_token, fill_value=0)
                subjectivity_in_cluster = subjectivity.reindex(proxim_in_cluster.context_token, fill_value=0)
                if len(proxim_in_cluster) > 0:
                    cluster_measures = return_measure_dict(proxim_in_cluster.weight)
                    sent_measures = return_measure_dict(sentiment_in_cluster.sentiment, prefix="sentiment_")
                    sub_measures = return_measure_dict(subjectivity_in_cluster.subjectivity, prefix="subjectivity_")
                    # Default cluster entry
                    year = -100
                    name = "-".join(list(proxim_in_cluster.weight.nlargest(5).index))
                    top_node = proxim_in_cluster.weight.idxmax()
                    df_dict = {'Year': year, 'POS': pos, 'Token': name,
                               'Prom_Node': top_node, 'Level': cl['level'],
                               'Cluster_Id': cl['name'],
                               'Parent': cl['parent'], 'Nr_ProxNodes': len(proxim_in_cluster),
                               'NrNodes': len(nodes), 'Ma': 0, 'Node_Weight': 0,
                               'Node_Sentiment': 0, 'Node_Subjectivity': 0, 'Type': "Cluster"}
                    cluster_dict.update({name: cl})
                    df_dict.update(cluster_measures)
                    df_dict.update(sent_measures)
                    df_dict.update(sub_measures)
                    cluster_dataframe.append(df_dict.copy())
                    # Add each node
                    if add_individual_nodes:
                        for node in list(proxim_in_cluster.index):
                            if proxim.reindex([node], fill_value=0).weight[0] > 0:
                                node_prox = proxim.reindex([node], fill_value=0).weight[0]
                                node_sent = sentiment.reindex([node], fill_value=0).sentiment[0]
                                node_sub = subjectivity.reindex([node], fill_value=0).subjectivity[0]
                                df_dict = {'Year': year, 'POS': pos, 'Token': node,
                                           'Prom_Node': top_node, 'Level': cl['level'],
                                           'Cluster_Id': cl['name'],
                                           'Parent': cl['parent'],
                                           'Nr_ProxNodes': len(proxim_in_cluster),
                                           'NrNodes': len(nodes), 'Ma': 0, 'Node_Weight': node_prox,
                                           'Node_Sentiment': node_sent, 'Node_Subjectivity': node_sub,
                                           'Type': "Cluster"}
                                df_dict.update(cluster_measures)
                                df_dict.update(sent_measures)
                                df_dict.update(sub_measures)
                                cluster_dataframe.append(df_dict.copy())

    df = pd.DataFrame(cluster_dataframe)

    if filename is not None:
        if export_network:
            snw.export_gefx(filename=check_create_folder(filename + ".gexf"))
        filename = check_create_folder(filename + ".xlsx")
        df.to_excel(filename)

    return df
