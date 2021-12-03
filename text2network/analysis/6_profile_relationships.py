import logging
import pickle

import networkx as nx
import pandas as pd

from text2network.classes.neo4jnw import neo4j_network
from text2network.functions.graph_clustering import consensus_louvain, cluster_distances, \
    cluster_distances_from_clusterlist, get_cluster_dict
from text2network.measures.context_profiles import context_cluster_all_pos
from text2network.utils.file_helpers import check_create_folder
from text2network.utils.logging_helpers import setup_logger

# Set a configuration path
configuration_path = 'config/analyses/LeaderSenBert40.ini'
# Settings
import os

os.environ['NUMEXPR_MAX_THREADS'] = '16'
alter_subset = None
# Load Configuration file
import configparser

config = configparser.ConfigParser()
print(check_create_folder(configuration_path))
config.read(check_create_folder(configuration_path))
# Setup logging
setup_logger(config['Paths']['log'], config['General']['logging_level'], "6_profile_relationships.py")
output_path = check_create_folder(config['Paths']['csv_outputs'])
output_path = check_create_folder(config['Paths']['csv_outputs']+"/profile_relationships")
semantic_network = neo4j_network(config)

times = list(range(1980, 1982))
focal_token = "leader"
sym = False
keep_top_k = 2
max_degree = 50
rs = 100
cutoff = 0.2
level = 10
depth = 0
context_mode = "bidirectional"
rev = False
algo = consensus_louvain
ma = (2, 2)
keep_only_tokens = True
contextual_relations = True
pos_list = ["NOUN", "VERB"]
tfidf = ["nweight", "rel_weight", "pmi_weight", "cond_entropy_weight"]

filename = "".join(
    [output_path, "/OvrlCluster", str(focal_token), "_", "-".join(pos_list), "_max_degree",
     str(max_degree), "_algo", str(algo.__name__), "_depth", str(depth), "_conRel", str(contextual_relations),
     "_depth", str(depth), "_keeptopk", str(keep_top_k), "_cm", str(context_mode), "_keeponlyt_", str(depth == 0),
     str(cutoff), "_lev", str(level), "_tfidf", str(tfidf is not None), "_rs", str(rs)])
logging.info("Overall Profiles  ALL POS: {}".format(filename))
df, clusters = context_cluster_all_pos(semantic_network, focal_substitutes=focal_token, times=times,
                                       keep_top_k=keep_top_k, max_degree=max_degree, sym=sym, weight_cutoff=cutoff,
                                       level=level,
                                       depth=depth, context_mode=context_mode, algorithm=algo,
                                       contextual_relations=contextual_relations, pos_list=pos_list, tfidf=tfidf,
                                       filename=filename)

logging.info("Extracted Clusters, now getting relationships")

clusterdict, all_nodes = get_cluster_dict(clusters, level=level)
rlgraph = cluster_distances_from_clusterlist(clusters, level=level)
cl_df = nx.convert_matrix.to_pandas_edgelist(rlgraph)
cl_filename = filename + "_CLREL"
cl_df.to_excel(cl_filename + ".xlsx", merge_cells=False)
# Dump cluster dict
pickle.dump(clusterdict, open(filename + "_CLDICT.p", "wb"))

logging.info("Extracting cluster relationships across years")
df_list = []
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
