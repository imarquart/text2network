
from itertools import product

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
years = list(range(1980, 2021))
focal_token = "leader"
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

focal_token="leader"
cluster=["boss", "superior"]
year=[2002,1980,2020]
# First, create an empty network
semantic_network = neo4j_network(config)

#test=get_pos_profile(snw=semantic_network,focal_token=focal_token,role_cluster=cluster,times=year, pos="VERB")
#print(test)
#print(len(np.unique(test.idx)))
max_degree=10
sym=True
rs=100
cutoff=0.002
df = average_cluster_proximities(focal_token=focal_token, nw=semantic_network, levels=6, times=years, do_reverse=False,
                                 depth=1, weight_cutoff=cutoff, year_by_year=False,
                                 add_focal_to_clusters=False,
                                 include_all_levels=False, add_individual_nodes=True,
                                 reverse_ties=False, symmetric=sym, seed=rs, export_network=False, max_degree=max_degree)

test2=create_YOY_role_profile(snw=semantic_network,focal_token=focal_token,role_cluster=cluster, times=year, keep_top_k=2 )
print(test2)