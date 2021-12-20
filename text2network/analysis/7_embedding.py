import logging
import pickle
from itertools import product

import networkx as nx
import pandas as pd
import numpy as np
from tqdm import tqdm

from text2network.classes.neo4jnw import neo4j_network
from text2network.functions.graph_clustering import consensus_louvain, cluster_distances, \
    cluster_distances_from_clusterlist, get_cluster_dict, infomap_cluster
from text2network.measures.context_profiles import context_cluster_all_pos, context_per_pos
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
setup_logger(config['Paths']['log'], config['General']['logging_level'], "7_embedding.py")

semantic_network = neo4j_network(config)



# %% Across different options


times = list(range(1980, 2021))
#times=[1988]
focal_token = "leader"
sym_list = [False]
rs_list = [100]
cutoff_list = [0.2,0.1,0.01]
post_cutoff_list=[0.01,None]
depth_list = [0, 1]
context_mode_list = ["bidirectional", "substitution", "occurring"]
rev_list = [False]
algo_list = [consensus_louvain, infomap_cluster]
ma_list = [(2, 2)]
pos_list = ["NOUN", "ADJ", "VERB"]
# TODO CHECK WITH X
tfidf_list = [["weight", "diffw", "pmi_weight" ]]
keep_top_k_list = [50,100,200,1000]
max_degree_list = [50,100]
level_list = [15,10,8,6,4,2]
keep_only_tokens_list = [True,False]
contextual_relations_list = [True,False]

paraml1_list = product(cutoff_list,context_mode_list,tfidf_list)
param_list = product(rs_list, depth_list,  max_degree_list, sym_list,
                     keep_top_k_list, ma_list, algo_list, contextual_relations_list, keep_only_tokens_list, post_cutoff_list)





# %% Sub or Occ
focal_substitutes = focal_token
focal_occurrences = None

if not (isinstance(focal_substitutes, list) or focal_substitutes is None):
    focal_substitutes=[focal_substitutes]
if not (isinstance(focal_occurrences, list) or focal_occurrences is None):
    focal_occurrences=[focal_occurrences]

logging.info("Getting level: {}".format(level))
output_path = check_create_folder(config['Paths']['csv_outputs'])
output_path = check_create_folder(config['Paths']['csv_outputs'] + "/profile_relationships/")
output_path = check_create_folder(
    "".join([output_path, "/", str(focal_token), "_cut", str(int(cutoff * 100)), "_tfidf",
             str(tfidf is not None), "_cm", str(context_mode), "/"]))
output_path = check_create_folder("".join(
    [output_path, "/", "conRel", str(contextual_relations), "_postcut", str(int(postcut * 100)), "/"]))
output_path = check_create_folder("".join(
    [output_path, "/", "keeptopk", str(keep_top_k), "_keeponlyt_", str(depth == 0), "/"]))
filename = "".join(
    [output_path, "/md", str(max_degree), "_algo", str(algo.__name__),
     "_rs", str(rs)])
output_path = check_create_folder("".join(
[output_path, "/",  "lev", str(level), "/"]))
logging.info("Getting tf-idf: {}".format(tf))
clusters=clusters_raw[tf].copy()
filename = check_create_folder("".join([output_path, "/"+ "tf_", str(tf) + "/"]))
filename =  "".join(
    [filename, "/md", str(max_degree), "_algo", str(algo.__name__),
     "_rs", str(rs)])
logging.info("Overall Profiles  Regression tables: {}".format(filename))

checkname=filename + "REGDF_allY_" + ".xlsx"
if os.path.exists(checkname) and os.stat(checkname).st_size > 0:
    logging.info("File {} exists - processing completed!".format(checkname))
else:
