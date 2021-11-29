import logging
from typing import Optional, Union, Callable

from tqdm import tqdm

from text2network.classes.neo4jnw import neo4j_network
from typing import Optional, Union

import numpy as np
import pandas as pd

from text2network.classes.neo4jnw import neo4j_network
from text2network.functions.graph_clustering import consensus_louvain
from text2network.measures.measures import return_measure_dict
from text2network.utils.file_helpers import check_create_folder


def contextual_tokens_for_pos(snw: neo4j_network,pos: str,  focal_occurrences: Optional[Union[list,str, int]]=None, focal_substitutes: Optional[Union[list,str, int]]=None, times: Optional[Union[list, int]]=None, context_mode: Optional[str] = "bidirectional",
                    return_sentiment: Optional[bool] = True, weight_cutoff: Optional[float] = 0) -> pd.DataFrame:
    """

    This function returns a dataframe with a list of contextual tokens that appear in the context of another dyad.
    The focal dyad can be specified by occurrence tokens, and substitute tokens, given as lists. The algorithm
    will consider the combination of each element dyad=(occurrence, substitute).

    For each such dyad, contextual tokens are returned, also from a dyad that occurs in the same sequence.
    Which token gets returned, and with which weight, depends on the parameter context mode
    If context_mode="occuring", give the likelihood that the token appears as written in the context of a substitution
    If context_mode="bidirectional", give the likelihood that the token appears, or according to BERT could appear
    If context_mode="substitution", give the likelihood that the token could appear when it does not

    Specify pos to give the part of speech in the focal sequence that the contextual token should be associated with!

    Parameters
    ----------

    snw : neo4j_network
        Semantic Network

    focal_substitutes: list, str, int, Optional
        Terms that substitute for an occurring term in the focal dyad

    focal_occurrences:  list, str, int, Optional
        Terms that occur in the focal dyad

    pos: str, Optional
        Only consider context terms, where the occurring word is classified as the given Part of Speech

    times: list, Optional
        Aggregate across these times

    context_mode: str, Optional, Default "bidirectional"
        If context_mode="occuring", give the likelihood that the token appears as written in the context of a substitution
        If context_mode="bidirectional", give the likelihood that the token appears, or according to BERT could appear
        If context_mode="substitution", give the likelihood that the token could appear when it does not

    return_sentiment: bool, Optional, Default True
        Return sentiment and subjectivity (Averaged) for the focal tie

    weight_cutoff: float, Optional, Default None
        Ignore any network ties that are less than this value in weight

    Returns
    -------
        Pandas DataFrame with all tokens and group associations

    """
    # Want to have a list here
    if not isinstance(focal_occurrences, (list, np.ndarray)):
        role_cluster = [focal_occurrences]
    if not isinstance(focal_substitutes, (list, np.ndarray)):
        role_cluster = [focal_substitutes]
    if not isinstance(times, (list, np.ndarray)):
        times = [times]

    df=pd.DataFrame(snw.get_dyad_context(focal_occurrences=focal_occurrences, focal_substitutes=focal_substitutes, times=times, weight_cutoff=weight_cutoff,context_pos=pos, context_mode=context_mode, return_sentiment=return_sentiment)['dyad_context'])
    df["context_token"] = snw.ensure_tokens(df.idx)
    return df


def context_per_pos(snw: neo4j_network,  focal_occurrences: Optional[Union[list,str, int]], focal_substitutes: Optional[Union[list,str, int]],
                                times: Optional[Union[list, int]] = None, pos_list: Optional[list] = None,
                                keep_top_k: Optional[int] = None,
                                filename: Optional[str] = None, moving_average: Optional[tuple] = None,
                                context_mode: Optional[str] = "bidirectional", return_sentiment: Optional[bool] = True,
                                weight_cutoff: Optional[float] = 0, seed: Optional[int] = None) -> pd.DataFrame:
    snw.decondition()

    if seed is not None:
        snw.set_random_seed(seed)

    if not isinstance(focal_occurrences, (list, np.ndarray)):
        role_cluster = [focal_occurrences]
    if not isinstance(focal_substitutes, (list, np.ndarray)):
        role_cluster = [focal_substitutes]
    if not isinstance(times, (list, np.ndarray)):
        times = [times]


    if times is None:
        logging.info("Getting years.")
        times = np.array(snw.get_times_list())
        times = np.sort(times)
        query_times = None
    else:
        query_times = times

    if pos_list is None:
        logging.info("Getting POS in Database")
        res = snw.db.receive_query("MATCH (n:part_of_speech) RETURN DISTINCT n.part_of_speech as pos")
        pos_list = [x['pos'] for x in res if x['pos'] != '.']

    df_list = []
    for pos in pos_list:
        logging.info("Now checking {}".format(pos))
        temp_df = contextual_tokens_for_pos(snw=snw, pos=pos, focal_substitutes=focal_substitutes, focal_occurrences=focal_occurrences, times=times,
                                  context_mode=context_mode, return_sentiment=return_sentiment,
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



def xx(snw: neo4j_network,  focal_occurrences: Optional[Union[list,str, int]], focal_substitutes: Optional[Union[list,str, int]],
                                times: Optional[Union[list, int]] = None, pos_list: Optional[list] = None,
                                keep_top_k: Optional[int] = None,
                                filename: Optional[str] = None, algorithm: Optional[Callable]=None,
                                context_mode: Optional[str] = "bidirectional", return_sentiment: Optional[bool] = True,
                                weight_cutoff: Optional[float] = 0, seed: Optional[int] = None) -> pd.DataFrame:
    snw.decondition()

    if seed is not None:
        snw.set_random_seed(seed)

    if not isinstance(focal_occurrences, (list, np.ndarray)):
        role_cluster = [focal_occurrences]
    if not isinstance(focal_substitutes, (list, np.ndarray)):
        role_cluster = [focal_substitutes]
    if not isinstance(times, (list, np.ndarray)):
        times = [times]


    if times is None:
        logging.info("Getting years.")
        times = np.array(snw.get_times_list())
        times = np.sort(times)
        query_times = None
    else:
        query_times = times

    if pos_list is None:
        logging.info("Getting POS in Database")
        res = snw.db.receive_query("MATCH (n:part_of_speech) RETURN DISTINCT n.part_of_speech as pos")
        pos_list = [x['pos'] for x in res if x['pos'] != '.']

    df_list = []
    for pos in pos_list:
        logging.info("Now checking {}".format(pos))
        temp_df = contextual_tokens_for_pos(snw=snw, pos=pos, focal_substitutes=focal_substitutes, focal_occurrences=focal_occurrences, times=times,
                                  context_mode=context_mode, return_sentiment=return_sentiment,
                                  weight_cutoff=weight_cutoff)
        if temp_df is not None:
            if keep_top_k is not None:
                # temp_df.sort_values(by="weight", ascending=False)
                # temp_df=temp_df.iloc[0:keep_top_k, :]
                temp_df = temp_df.nlargest(keep_top_k, columns=["weight"])
            temp_df["time"] = str(times)
            temp_df["ma_time"] = str(times)
            df_list.append(temp_df)

        interest_list = np.unique(temp_df.context_token).tolist()

        logging.info("Now checking {} via substitution".format(pos))
        cl_filename=filename + "_sub_" + str(pos)
        snw.condition_given_dyad(dyad_substitute=focal_substitutes, dyad_occurring=focal_occurrences, times=times, focal_tokens=interest_list, contextual_relations=False, context_mode=context_mode)



    if len(df_list) > 0:
        df = pd.concat(df_list)

        if filename is not None:
            df.to_excel(filename + ".xlsx")

    else:
        df = None

    return df


def context_cluster_per_pos(snw: neo4j_network, focal_substitutes: Union[list, str, int]=None, focal_occurrences: Union[list, str, int]=None,
                               level: int=10, cluster_cutoff: Optional[float] = 0.0,
                               pos_list: Optional[list] = None, times: Optional[Union[list, int]] = None,
                               keep_top_k: Optional[int] = None, weight_cutoff: Optional[float] = None,
                               context_mode: Optional[str] = "bidirectional",
                               add_individual_nodes: Optional[bool] = True,
                                contextual_relations: Optional[bool] = False,
                               max_degree: Optional[int] = None, include_all_levels: Optional[bool] = True,
                               sym: Optional[bool] = False, depth: Optional[int] = 1, algorithm: Optional[Callable] = None,
                               export_network: Optional[bool] = True, filename: Optional[str] = None):

    if algorithm is None:
        algorithm = consensus_louvain
    # Get role profile
    df = context_per_pos(snw=snw, focal_substitutes=focal_substitutes, focal_occurrences=focal_occurrences, times=times,
                                     keep_top_k=keep_top_k, context_mode=context_mode, pos_list=pos_list)

    pos_list = np.unique(df.pos).tolist()
    cluster_dict = {}
    cluster_dataframe = []
    for pos in pos_list:
        logging.info("Extracting clusters information for pos: {}".format(pos))
        df_subset = df[df.pos == pos]
        interest_list = np.unique(df_subset.context_token).tolist()
        proxim = df_subset[["idx", "context_token", "pos", "weight"]]
        proxim = proxim.set_index(proxim.context_token)
        sentiment = df_subset[["idx", "context_token", "pos", "sentiment"]]
        sentiment = sentiment.set_index(sentiment.context_token)
        subjectivity = df_subset[["idx", "context_token", "pos", "subjectivity"]]
        subjectivity = subjectivity.set_index(subjectivity.context_token)
        if depth==0:
            keep_only_tokens = True
            logging.info("Keeping only contextual tokens in conditioning network!")
        else:
            keep_only_tokens= False
        snw.condition_given_dyad(dyad_substitute=focal_substitutes, dyad_occurring=focal_occurrences, times=times,
                                 focal_tokens=interest_list, weight_cutoff=weight_cutoff, depth=depth, keep_only_tokens=keep_only_tokens,
                                 contextual_relations=contextual_relations,
                                 max_degree=max_degree)
        logging.info("Finished conditioning")
        clusters = snw.cluster(levels=level, interest_list=interest_list, algorithm=algorithm)
        logging.info("Finished clustering, found {} clusters".format(len(clusters)))
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


    df = pd.DataFrame(cluster_dataframe)

    if filename is not None:
        if export_network:
            snw.export_gefx(filename=check_create_folder(filename + ".gexf"))
        filename = check_create_folder(filename + ".xlsx")
        df.to_excel(filename)

    return df



def context_cluster_all_pos(snw: neo4j_network, focal_substitutes: Union[list, str, int]=None, focal_occurrences: Union[list, str, int]=None,
                               level: int=10, cluster_cutoff: Optional[float] = 0.0,
                               pos_list: Optional[list] = None, times: Optional[Union[list, int]] = None,
                               keep_top_k: Optional[int] = None, weight_cutoff: Optional[float] = None,
                               context_mode: Optional[str] = "bidirectional",
                               add_individual_nodes: Optional[bool] = True,
                                contextual_relations: Optional[bool] = False,
                               max_degree: Optional[int] = None, include_all_levels: Optional[bool] = True,
                               sym: Optional[bool] = False, depth: Optional[int] = 1, algorithm: Optional[Callable] = None,
                               export_network: Optional[bool] = True, filename: Optional[str] = None):

    if algorithm is None:
        algorithm = consensus_louvain
    if pos_list is not None:
        logging.warning("Extracting context clusters for part of speech {}".format(pos_list))
    # Get role profiles

    df = context_per_pos(snw=snw, focal_substitutes=focal_substitutes, focal_occurrences=focal_occurrences, times=times,
                                     keep_top_k=keep_top_k, context_mode=context_mode, pos_list=pos_list)
    cluster_dict = {}
    cluster_dataframe = []

    interest_list = np.unique(df.context_token).tolist()

    # If a word appears in several distinct POS, change context token description for one instance
    df=df.set_index(df.context_token)
    df.loc[df.index.duplicated(), "context_token"] = df.context_token.loc[df.index.duplicated()] + df.pos.loc[
        df.index.duplicated()]
    df = df.set_index(df.context_token)
    assert len(df.index[df.index.duplicated()]) ==0

    proxim = df[["idx", "context_token", "pos", "weight"]]
    proxim = proxim.set_index(proxim.context_token)
    sentiment = df[["idx", "context_token", "pos", "sentiment"]]
    sentiment = sentiment.set_index(sentiment.context_token)
    subjectivity = df[["idx", "context_token", "pos", "subjectivity"]]
    subjectivity = subjectivity.set_index(subjectivity.context_token)
    if depth==0:
        keep_only_tokens = True
        logging.info("Keeping only contextual tokens in conditioning network!")
    else:
        keep_only_tokens= False
    snw.condition_given_dyad(dyad_substitute=focal_substitutes, dyad_occurring=focal_occurrences, times=times,
                             focal_tokens=interest_list, weight_cutoff=weight_cutoff, depth=depth, keep_only_tokens=keep_only_tokens,
                             contextual_relations=contextual_relations,
                             max_degree=max_degree)
    logging.info("Finished conditioning")
    clusters = snw.cluster(levels=level, interest_list=interest_list, algorithm=algorithm)
    logging.info("Finished clustering, found {} clusters".format(len(clusters)))
    logging.info("Extracting relevant clusters at level {} across all years {}".format(level, times))
    for cl in tqdm(clusters, desc="Extracting all clusters"):
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
                    df_dict = {'Year': year, 'POS': "ALL", 'Token': name,
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
                                df_dict = {'Year': year, 'POS': "ALL", 'Token': node,
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

    return df, clusters

def grouped_dyadic_context(snw: neo4j_network, focal_substitutes: Optional[Union[list, str, int]] = None,
                           focal_occurrences: Optional[Union[list, str, int]] = None, groups: Optional[list] = None,
                           context_pos: Optional[str] = None, times: Union[list, int] = None,
                           context_mode: Optional[str] = "bidirectional", return_sentiment: Optional[bool] = True,
                           weight_cutoff: Optional[float] = None) -> (pd.DataFrame,dict):
    """
    This function returns a dataframe with a list of contextual tokens that appear in the context of another dyad.
    The focal dyad can be specified by occurrence tokens, and substitute tokens, given as lists. The algorithm
    will consider the combination of each element dyad=(occurrence, substitute).

    For each such dyad, contextual tokens are returned, also from a dyad that occurs in the same sequence.
    Which token gets returned, and with which weight, depends on the parameter context mode
    If context_mode="occuring", give the likelihood that the token appears as written in the context of a substitution
    If context_mode="bidirectional", give the likelihood that the token appears, or according to BERT could appear
    If context_mode="substitution", give the likelihood that the token could appear when it does not

    Values are aggregated across sequences with the substitution weight of the original dyad.


    Parameters
    ----------
    snw : neo4j_network
        Semantic Network

    focal_substitutes: list, str, int, Optional
        Terms that substitute for an occurring term in the focal dyad

    focal_occurrences:  list, str, int, Optional
        Terms that occur in the focal dyad

    groups: iterable over lists of tokens, Optional
        Provide some collection of lists of tokens, for example, clusters.
        Output dataframe will be grouped by the tokens.

    context_pos: str, Optional
        Only consider context terms, where the occurring word is classified as the given Part of Speech

    times: list, Optional
        Aggregate across these times

    context_mode: str, Optional, Default "bidirectional"
        If context_mode="occuring", give the likelihood that the token appears as written in the context of a substitution
        If context_mode="bidirectional", give the likelihood that the token appears, or according to BERT could appear
        If context_mode="substitution", give the likelihood that the token could appear when it does not

    return_sentiment: bool, Optional, Default True
        Return sentiment and subjectivity (Averaged) for the focal tie

    weight_cutoff: float, Optional, Default None
        Ignore any network ties that are less than this value in weight

    Returns
    -------
        Pandas DataFrame with all tokens and group associations

    """

    # Format inputs
    if focal_substitutes is not None:
        if not isinstance(focal_substitutes, (list, np.ndarray)):
            focal_substitutes = [focal_substitutes]
    if focal_occurrences is not None:
        if not isinstance(focal_occurrences, (list, np.ndarray)):
            focal_occurrences = [focal_occurrences]
    if times is None:
        times = snw.get_times_list()
    if not isinstance(times, (list, np.ndarray)):
        times = [times]

    df=pd.DataFrame(snw.get_dyad_context(focal_occurrences=focal_occurrences, focal_substitutes=focal_substitutes, times=times, weight_cutoff=weight_cutoff,
                                 context_pos=context_pos, context_mode=context_mode, return_sentiment=return_sentiment)[
                'dyad_context'])

    df["context_token"] = snw.ensure_tokens(df.idx)
    df_list=[]
    df_dict={}
    for group in groups:
        group_idx = snw.ensure_ids(group)
        group_tk = snw.ensure_tokens(group_idx)
        group_df = df[df.idx.isin(group_idx)]
        if len(group_df)>0:
            group_df = group_df.groupby(["idx","context_token", "pos"]).agg(weight=('weight', 'sum'), sentiment=('sentiment', 'mean'),
                                                subjectivity=('subjectivity', 'mean'))


            group_df = group_df.reset_index(drop=False).sort_values(by="weight", ascending=False)
            if return_sentiment:
                group_df = group_df[["idx", "context_token", "pos", "weight", "sentiment", "subjectivity"]]
            else:
                group_df = group_df[["idx", "context_token", "pos", "weight"]]
        else:
            group_df = None
        group_name=group_df.iloc[0:6].context_token
        group_name = "-".join(list(group_name.nlargest(5).index))
        group_df.group_name = group_name
        df_list.append(group_df)
        df_dict[group_name]=group_df

    df = pd.concat(df_list)

    return df, df_dict
