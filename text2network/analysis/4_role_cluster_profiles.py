
from itertools import product

import pandas as pd

from text2network.utils.file_helpers import check_create_folder, check_folder
from text2network.measures.measures import average_cluster_proximities
from text2network.utils.logging_helpers import setup_logger
import logging
import numpy as np
from text2network.functions.graph_clustering import consensus_louvain
from text2network.classes.neo4jnw import neo4j_network
from text2network.measures.role_profiles import get_pos_profile, create_YOY_role_profile

# Set a configuration path
configuration_path = 'config/analyses/LeaderSenBert40.ini'
# Settings
import os
os.environ['NUMEXPR_MAX_THREADS'] = '16'
alter_subset=None
# Load Configuration file
import configparser

config = configparser.ConfigParser()
print(check_create_folder(configuration_path))
config.read(check_create_folder(configuration_path))
# Setup logging
setup_logger(config['Paths']['log'], config['General']['logging_level'], "3_YOY_clusters.py")

semantic_network = neo4j_network(config)

def get_semantic_clusters_from_file(file, level):
    df = pd.read_excel(file)
    df=df[df.Level==level]
    df_clusters = df[df.Node_Delta_Proximity == -100]
    df_cluster_ids = df_clusters.Cluster_Id.to_list()
    df_nodes = df[df.Node_Delta_Proximity != -100]
    return df_cluster_ids, df_clusters, df_nodes

def get_semantic_clusters(semantic_network,  times, focal_token,  max_degree, sym, cutoff, level,
                              depth,  rs):
    df = average_cluster_proximities(focal_token=focal_token, nw=semantic_network, levels=level, times=times,
                                     do_reverse=False,
                                     depth=depth, weight_cutoff=cutoff, year_by_year=False,
                                     add_focal_to_clusters=False,
                                     include_all_levels=False, add_individual_nodes=True,
                                     reverse_ties=False, symmetric=sym, seed=rs, export_network=False,
                                     max_degree=max_degree)
    df_clusters = df[df.Node_Delta_Proximity == -100]
    df_cluster_ids = df_clusters.Cluster_Id.to_list()
    df_nodes = df[df.Node_Delta_Proximity != -100]


    return df_cluster_ids, df_clusters, df_nodes

def extract_yoy_role_profiles(semantic_network, df_cluster_ids, df_clusters, df_nodes, config, times, focal_token, keep_top_k, max_degree, sym, cutoff, level,
                              depth, context_mode, rs):

    # YOY
    cluster_df_list = []
    for cluster_id in df_cluster_ids:
        logging.info("Considering cluster with id {}".format(cluster_id))
        nodes = df_nodes[df_nodes.Cluster_Id == cluster_id].sort_values(by="Node_Proximity",
                                                                        ascending=False).Token.to_list()
        cluster_name = df_clusters[df_clusters.Cluster_Id == cluster_id].Token.to_list()[0]
        cluster_df = create_YOY_role_profile(snw=semantic_network, focal_token=focal_token, role_cluster=nodes,
                                             weight_cutoff=cutoff, times=times, keep_top_k=keep_top_k,
                                             context_mode=context_mode)
        if cluster_df is not None:
            print(cluster_df)
            all_years = cluster_df.groupby(by=["idx", "context_token", "pos"], as_index=False).mean()
            all_years.time = 0
            all_years["ma_time"] = 0
            cluster_df = pd.concat([cluster_df, all_years])
            cluster_df["cluster_name"] = cluster_name
            cluster_df_list.append(cluster_df)

    df = pd.concat(cluster_df_list)

    filename = "".join(
        [config['Paths']['csv_outputs'], "/RoleCluster", str(focal_token), "_max_degree", str(max_degree), "_sym",
         str(sym),
         "_lev", str(level), "_cut", "_keeptopk", str(keep_top_k), "_cm", str(context_mode),
         str(cutoff), "_depth", str(depth), "_rs", str(rs)])
    logging.info("Network clustering: {}".format(filename))
    df.to_excel(filename + ".xlsx", merge_cells=False)



times = list(range(1980, 2021))
focal_token="leader"
sym=False
keep_top_k = 20
max_degree=200
rs=100
cutoff=0.0025
level=10
depth=1

file="/EgoCluster_leader_max_degree200_symTrue_revFalse_faddFalse_altersFalse_normFalse_lev15_cut0.0025_clcut0_algoconsensus_louvain_depth1_rs100.xlsx"
filename = "".join(
        [config['Paths']['csv_outputs'],file])
df_cluster_ids, df_clusters, df_nodes = get_semantic_clusters_from_file(filename, level)

# Extract Clusters
#df_cluster_ids, df_clusters, df_nodes = get_semantic_clusters(semantic_network,  times, focal_token,  max_degree, sym, cutoff, level,
#                              depth,  rs)

context_mode="bidirectional"
extract_yoy_role_profiles(semantic_network=semantic_network,df_cluster_ids=df_cluster_ids, df_clusters=df_clusters, df_nodes=df_nodes,config=config, times=times, focal_token=focal_token, keep_top_k=keep_top_k, max_degree=max_degree, sym=sym, cutoff=cutoff, level=level,
                              depth=depth, context_mode=context_mode, rs=rs)
context_mode="occurrence"
extract_yoy_role_profiles(semantic_network=semantic_network,df_cluster_ids=df_cluster_ids, df_clusters=df_clusters, df_nodes=df_nodes,config=config, times=times, focal_token=focal_token, keep_top_k=keep_top_k, max_degree=max_degree, sym=sym, cutoff=cutoff, level=level,
                              depth=depth, context_mode=context_mode, rs=rs)

context_mode="substitution"
extract_yoy_role_profiles(semantic_network=semantic_network,df_cluster_ids=df_cluster_ids, df_clusters=df_clusters, df_nodes=df_nodes,config=config, times=times, focal_token=focal_token, keep_top_k=keep_top_k, max_degree=max_degree, sym=sym, cutoff=cutoff, level=level,
                              depth=depth, context_mode=context_mode, rs=rs)


cutoff=0
file="/EgoCluster_leader_max_degree200_symTrue_revFalse_faddTrue_altersFalse_normFalse_lev15_cut0_clcut0_algoconsensus_louvain_depth1_rs100.xlsx"
filename = "".join(
        [config['Paths']['csv_outputs'],file])
df_cluster_ids, df_clusters, df_nodes = get_semantic_clusters_from_file(filename, level)
# Extract Clusters
#df_cluster_ids, df_clusters, df_nodes = get_semantic_clusters(semantic_network,  times, focal_token,  max_degree, sym, cutoff, level,
#                              depth,  rs)


context_mode="bidirectional"
extract_yoy_role_profiles(semantic_network=semantic_network,df_cluster_ids=df_cluster_ids, df_clusters=df_clusters, df_nodes=df_nodes,config=config, times=times, focal_token=focal_token, keep_top_k=keep_top_k, max_degree=max_degree, sym=sym, cutoff=cutoff, level=level,
                              depth=depth, context_mode=context_mode, rs=rs)
context_mode="occurrence"
extract_yoy_role_profiles(semantic_network=semantic_network,df_cluster_ids=df_cluster_ids, df_clusters=df_clusters, df_nodes=df_nodes,config=config, times=times, focal_token=focal_token, keep_top_k=keep_top_k, max_degree=max_degree, sym=sym, cutoff=cutoff, level=level,
                              depth=depth, context_mode=context_mode, rs=rs)

context_mode="substitution"
extract_yoy_role_profiles(semantic_network=semantic_network,df_cluster_ids=df_cluster_ids, df_clusters=df_clusters, df_nodes=df_nodes,config=config, times=times, focal_token=focal_token, keep_top_k=keep_top_k, max_degree=max_degree, sym=sym, cutoff=cutoff, level=level,
                              depth=depth, context_mode=context_mode, rs=rs)



level=6
file="/EgoCluster_leader_max_degree200_symTrue_revFalse_faddTrue_altersFalse_normFalse_lev15_cut0_clcut0_algoconsensus_louvain_depth1_rs100.xlsx"
filename = "".join(
        [config['Paths']['csv_outputs'],file])
df_cluster_ids, df_clusters, df_nodes = get_semantic_clusters_from_file(filename, level)
# Extract Clusters
#df_cluster_ids, df_clusters, df_nodes = get_semantic_clusters(semantic_network,  times, focal_token,  max_degree, sym, cutoff, level,
#                              depth,  rs)


context_mode="bidirectional"
extract_yoy_role_profiles(semantic_network=semantic_network,df_cluster_ids=df_cluster_ids, df_clusters=df_clusters, df_nodes=df_nodes,config=config, times=times, focal_token=focal_token, keep_top_k=keep_top_k, max_degree=max_degree, sym=sym, cutoff=cutoff, level=level,
                              depth=depth, context_mode=context_mode, rs=rs)
context_mode="occurrence"
extract_yoy_role_profiles(semantic_network=semantic_network,df_cluster_ids=df_cluster_ids, df_clusters=df_clusters, df_nodes=df_nodes,config=config, times=times, focal_token=focal_token, keep_top_k=keep_top_k, max_degree=max_degree, sym=sym, cutoff=cutoff, level=level,
                              depth=depth, context_mode=context_mode, rs=rs)

context_mode="substitution"
extract_yoy_role_profiles(semantic_network=semantic_network,df_cluster_ids=df_cluster_ids, df_clusters=df_clusters, df_nodes=df_nodes,config=config, times=times, focal_token=focal_token, keep_top_k=keep_top_k, max_degree=max_degree, sym=sym, cutoff=cutoff, level=level,
                              depth=depth, context_mode=context_mode, rs=rs)
