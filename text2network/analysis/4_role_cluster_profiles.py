
from itertools import product

import pandas as pd

from text2network.utils.file_helpers import check_create_folder, check_folder
from text2network.measures.measures import average_cluster_proximities
from text2network.utils.logging_helpers import setup_logger
import logging
import numpy as np
from text2network.functions.graph_clustering import consensus_louvain
from text2network.classes.neo4jnw import neo4j_network
from text2network.measures.role_profiles import get_pos_profile, create_YOY_role_profile, extract_yoy_role_profiles, \
    get_clustered_role_profile

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

def get_semantic_clusters_from_file(file, level, keep_tok_k_clusters=20):
    df = pd.read_excel(file)
    df=df[df.Level==level]
    df_clusters = df[df.Node_Delta_Proximity == -100]
    df_clusters = df_clusters.nlargest(keep_tok_k_clusters, columns=["w_Avg"])
    df_cluster_ids = df_clusters.Cluster_Id.to_list()
    df_nodes = df[df.Node_Delta_Proximity != -100]
    df_nodes = df_nodes[df_nodes.Cluster_Id.isin(df_cluster_ids)]
    return df_cluster_ids, df_clusters, df_nodes

def get_semantic_clusters(semantic_network,  times, focal_token,  max_degree, sym, cutoff, level,
                              depth,  rs, keep_tok_k_clusters=20):
    df = average_cluster_proximities(focal_token=focal_token, nw=semantic_network, levels=level, times=times,
                                     do_reverse=False,
                                     depth=depth, weight_cutoff=cutoff, year_by_year=False,
                                     add_focal_to_clusters=False,
                                     include_all_levels=False, add_individual_nodes=True,
                                     reverse_ties=False, symmetric=sym, seed=rs, export_network=False,
                                     max_degree=max_degree)
    df_clusters = df[df.Node_Delta_Proximity == -100]
    df_clusters = df_clusters.nlargest(keep_tok_k_clusters, columns=["w_Avg"])
    df_cluster_ids = df_clusters.Cluster_Id.to_list()
    df_nodes = df[df.Node_Delta_Proximity != -100]
    df_nodes = df_nodes[df_nodes.Cluster_Id.isin(df_cluster_ids)]


    return df_cluster_ids, df_clusters, df_nodes



times = list(range(1980, 2021))
focal_token="leader"
sym=False
keep_top_k = 40
max_degree=200
rs=100
cutoff=0.0025
level=10
depth=1

ma=(2,2)

file="/EgoCluster_leader_max_degree200_symTrue_revFalse_faddFalse_altersFalse_normFalse_lev15_cut0.0025_clcut0_algoconsensus_louvain_depth1_rs100.xlsx"
filename = "".join(
        [config['Paths']['csv_outputs'],file])
df_cluster_ids, df_clusters, df_nodes = get_semantic_clusters_from_file(filename, level)

# Extract Clusters
#df_cluster_ids, df_clusters, df_nodes = get_semantic_clusters(semantic_network,  times, focal_token,  max_degree, sym, cutoff, level,
#                              depth,  rs)


context_mode="bidirectional"
filename = "".join(
    [config['Paths']['csv_outputs'], "/RoleCluster", str(focal_token), "_max_degree", str(max_degree), "_sym",
     str(sym),
     "_lev", str(level), "_ma", str(ma), "_cut", "_keeptopk", str(keep_top_k), "_cm", str(context_mode),
     str(cutoff), "_depth", str(depth), "_rs", str(rs)])
logging.info("Network clustering: {}".format(filename))
#extract_yoy_role_profiles(semantic_network=semantic_network,df_cluster_ids=df_cluster_ids, df_clusters=df_clusters, df_nodes=df_nodes, times=times, focal_token=focal_token, keep_top_k=keep_top_k, max_degree=max_degree, sym=sym, cutoff=cutoff, level=level,
#                              depth=depth, context_mode=context_mode, moving_average=ma, filename=filename)

cluster_nodes=["boss","subordinate","superior","supervisor"]


df=get_clustered_role_profile(semantic_network,focal_token=focal_token, cluster_nodes=cluster_nodes, times=times, keep_top_k=keep_top_k, max_degree=max_degree, sym=sym, weight_cutoff=cutoff, level=level,
                              depth=depth, context_mode=context_mode, filename=filename)