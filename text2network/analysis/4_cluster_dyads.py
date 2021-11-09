
from itertools import product

from text2network.utils.file_helpers import check_create_folder, check_folder
from text2network.measures.measures import average_cluster_proximities
from text2network.utils.logging_helpers import setup_logger
import logging
import numpy as np
from text2network.functions.graph_clustering import consensus_louvain
from text2network.classes.neo4jnw import neo4j_network
from text2network.measures.role_profiles import get_pos_profile





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

test=get_pos_profile(snw=semantic_network,focal_token=focal_token,role_cluster=cluster,times=year, pos="VERB")
print(test)
print(len(np.unique(test.idx)))