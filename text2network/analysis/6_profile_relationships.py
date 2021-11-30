import pickle
from itertools import product

import pandas as pd
import networkx as nx
from text2network.measures.context_profiles import context_per_pos, context_cluster_per_pos, context_cluster_all_pos
from text2network.measures.proximity import get_top_100
from text2network.utils.file_helpers import check_create_folder, check_folder
from text2network.measures.measures import average_cluster_proximities
from text2network.utils.logging_helpers import setup_logger
import logging
import numpy as np
from text2network.functions.graph_clustering import consensus_louvain, infomap_cluster, cluster_distances, \
    cluster_distances_from_clusterlist, get_cluster_dict
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
setup_logger(config['Paths']['log'], config['General']['logging_level'], "6_profile_relationships.py")

semantic_network = neo4j_network(config)


times = list(range(1980, 1982))
focal_token="leader"
sym_list=[False]
keep_top_k_list = [30,50,100]
max_degree_list=[50,100,200]
rs_list=[100]
cutoff_list=[0.2,0.1,0.01]
level_list=[10,15]
depth_list=[0,1]
context_mode_list=["bidirectional","substitution"]
rev_list=[False, True]
algo_list=[consensus_louvain, infomap_cluster]
ma_list=[(2,2)]
keep_only_tokens_list=[True]
contextual_relations_list=[True,False]
pos_list_list=[["NOUN","VERB"]]
param_list=product(rs_list,cutoff_list, level_list, depth_list, context_mode_list, max_degree_list, sym_list, keep_top_k_list, ma_list, algo_list, contextual_relations_list,keep_only_tokens_list,pos_list_list)

times = list(range(1982))
focal_token="leader"
sym_list=[False]
keep_top_k_list = [30]
max_degree_list=[50]
rs_list=[100]
cutoff_list=[0.2]
level_list=[10]
depth_list=[0]
context_mode_list=["bidirectional"]
rev_list=[False]
algo_list=[consensus_louvain]
ma_list=[(2,2)]
keep_only_tokens_list=[True]
contextual_relations_list=[True]
pos_list_list=[["NOUN","VERB"]]
tfidf_list=["pmi","nweight", "rel_weight", "pmi_weight", None]
param_list=product(rs_list,cutoff_list, level_list, depth_list, context_mode_list, max_degree_list, sym_list, keep_top_k_list, ma_list, algo_list, contextual_relations_list,keep_only_tokens_list,pos_list_list,tfidf_list)



for rs, cutoff, level, depth, context_mode, max_degree, sym, keep_top_k, ma, algo, contextual_relations, keep_only_tokens, pos_list,tfidf in param_list:

    filename = "".join(
        [config['Paths']['csv_outputs'], "/OvrlCluster", str(focal_token), "_", "-".join(pos_list),"_max_degree", str(max_degree),  "_algo", str(algo.__name__), "_depth", str(depth), "_conRel", str(contextual_relations),
          "_depth", str(depth), "_keeptopk", str(keep_top_k), "_cm", str(context_mode),"_keeponlyt_", str(depth==0),
         str(cutoff),"_lev", str(level), "_tfidf", str(tfidf),"_rs", str(rs)])
    logging.info("Overall Profiles  ALL POS: {}".format(filename))
    df, clusters = context_cluster_all_pos(semantic_network,focal_substitutes=focal_token, times=times, keep_top_k=keep_top_k, max_degree=max_degree, sym=sym, weight_cutoff=cutoff, level=level,
                                      depth=depth, context_mode=context_mode, algorithm=algo,contextual_relations=contextual_relations, pos_list=pos_list,tfidf=tfidf,  filename=filename)

    logging.info("Extracted Clusters, now getting relationships")

    clusterdict, all_nodes=get_cluster_dict(clusters, level=level)
    rlgraph = cluster_distances_from_clusterlist(clusters, level=level)
    cl_df = nx.convert_matrix.to_pandas_edgelist(rlgraph)
    cl_filename = filename+"_CLREL"
    cl_df.to_excel(cl_filename+".xlsx", merge_cells=False)
    # Dump cluster dict
    pickle.dump(clusterdict, open(filename+"_CLDICT.p", "wb"))

    logging.info("Extracting cluster relationships across years")
    df_list=[]
    for year in times:
        semantic_network.decondition()
        semantic_network.condition_given_dyad(dyad_substitute=focal_token, dyad_occurring=None, times=[year],
                                 focal_tokens=all_nodes, weight_cutoff=cutoff, depth=0,
                                 keep_only_tokens=True,
                                 contextual_relations=contextual_relations,
                                 max_degree=max_degree)
        rlgraph = cluster_distances(semantic_network.graph, clusterdict)
        cl_df = nx.convert_matrix.to_pandas_edgelist(rlgraph)
        cl_df["year"] = year
        df_list.append(cl_df)

    df = pd.concat(df_list)
    cl_filename = filename + "_CLREL_YOY"
    df.to_excel(cl_filename + ".xlsx", merge_cells=False)



