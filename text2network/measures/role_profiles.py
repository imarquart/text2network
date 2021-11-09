import logging
from typing import Optional, Callable, Union

import numpy as np
import pandas as pd
from text2network.classes.neo4jnw import neo4j_network
from text2network.functions.graph_clustering import consensus_louvain
from text2network.functions.node_measures import proximity, centrality
# from src.classes import neo4jnw
# from src.classes.neo4jnw import neo4j_network
from text2network.utils.file_helpers import check_create_folder


def get_pos_profile(snw:neo4j_network, focal_token:Union[str,int], role_cluster:Union[str,int,list], times:Union[list, int],  pos:str, context_mode:Optional[str]="bidirectional", return_sentiment:Optional[bool]=True, weight_cutoff:Optional[float]=0)->pd.DataFrame:

    # Want to have a list here
    if not isinstance(role_cluster, (list, np.ndarray)):
        role_cluster=[role_cluster]
    if not isinstance(times, (list, np.ndarray)):
        times=[times]

    pd_list=[]
    for alter in role_cluster:
        pd_list.append(snw.get_dyad_context(occurrence=alter, replacement=focal_token, times=times, weight_cutoff=weight_cutoff, part_of_speech=pos, context_mode=context_mode, return_sentiment=return_sentiment)['dyad_context'])

    df=pd.concat([pd.DataFrame(x) for x in pd_list])
    df=df.groupby(["idx", "pos"]).mean().reset_index(drop=False).sort_values(by="weight", ascending=True)
    df["context_token"]=snw.ensure_tokens(df.idx)
    if return_sentiment:
        df=df[["idx", "context_token", "pos", "weight", "sentiment", "subjectivity"]]
    else:
        df = df[["idx", "context_token", "pos", "weight"]]

    return df


def create_YOY_role_profile(snw:neo4j_network, focal_token:Union[str,int], role_cluster:Union[str,int,list], times:Union[list, int],  pos_list:list, context_mode:Optional[str]="bidirectional", return_sentiment:Optional[bool]=True, weight_cutoff:Optional[float]=0)->pd.DataFrame:

    for pos in pos_list:
        temp_df = get_pos_profile(snw=snw, focal_token=focal_token, role_cluster=role_cluster, times=times, pos=pos, context_mode=context_mode, return_sentiment=return_sentiment, weight_cutoff=weight_cutoff)


