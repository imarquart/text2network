import logging
from typing import Optional, Callable, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

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
    if len(df) > 0:
        df=df.groupby(["idx", "pos"]).mean().reset_index(drop=False).sort_values(by="weight", ascending=True)
        df["context_token"]=snw.ensure_tokens(df.idx)
        if return_sentiment:
            df=df[["idx", "context_token", "pos", "weight", "sentiment", "subjectivity"]]
        else:
            df = df[["idx", "context_token", "pos", "weight"]]
    else:
        df= None
    return df


def create_YOY_role_profile(snw:neo4j_network, focal_token:Union[str,int], role_cluster:Union[str,int,list], times:Union[list, int],  pos_list:Optional[list] = None, keep_top_k:Optional[int]=None,filename: Optional[str] = None, moving_average: Optional[tuple] = None, context_mode:Optional[str]="bidirectional", return_sentiment:Optional[bool]=True, weight_cutoff:Optional[float]=0, seed: Optional[int] = None)->pd.DataFrame:
    snw.decondition()

    if seed is not None:
        snw.set_random_seed(seed)

    if not isinstance(role_cluster, (list,tuple,np.ndarray)):
        role_cluster=[role_cluster]

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
        res=snw.db.receive_query("MATCH (n:part_of_speech) RETURN DISTINCT n.part_of_speech as pos")
        pos_list=[x['pos'] for x in res if x['pos'] != '.']

    df_list=[]
    for year in tqdm(times, desc="Getting yearly profiles for {}".format(cluster_name)):
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

        year_df_list=[]
        for pos in pos_list:
            temp_df = get_pos_profile(snw=snw, focal_token=focal_token, role_cluster=role_cluster, times=ma_years, pos=pos, context_mode=context_mode, return_sentiment=return_sentiment, weight_cutoff=weight_cutoff)
            if temp_df is not None:
                if keep_top_k is not None:
                    #temp_df.sort_values(by="weight", ascending=False)
                    #temp_df=temp_df.iloc[0:keep_top_k, :]
                    temp_df=temp_df.nlargest(keep_top_k, columns=["weight"])
                temp_df["time"]=year
                temp_df["ma_time"]=str(ma_years)
                year_df_list.append(temp_df)
        if len(year_df_list)>0:
            df_list.append(pd.concat(year_df_list))

    if len(df_list) > 0:
        df=pd.concat(df_list)

        if filename is not None:
            df.to_excel(filename+".xlsx")

    else:
        df = None

    return df
