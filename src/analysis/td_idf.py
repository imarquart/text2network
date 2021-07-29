from itertools import product

from src.functions.file_helpers import check_create_folder
from src.measures.measures import average_cluster_proximities, extract_all_clusters
from src.utils.logging_helpers import setup_logger
import logging
import numpy as np
from src.functions.graph_clustering import consensus_louvain, louvain_cluster
from src.classes.neo4jnw import neo4j_network

# Set a configuration path
configuration_path = '/config/config.ini'
# Settings
years = None
# Load Configuration file
import configparser

config = configparser.ConfigParser()
print(check_create_folder(configuration_path))
config.read(check_create_folder(configuration_path))
# Setup logging
setup_logger(config['Paths']['log'], config['General']['logging_level'], "td_idf.py")


import os
os.environ['NUMEXPR_MAX_THREADS'] = '16'
# First, create an empty network
semantic_network = neo4j_network(config)


focal_token="bezos"
cutoff=0.1

semantic_network.context_condition(tokens=focal_token, times=None,  weight_cutoff=cutoff, occurrence=True, depth=1)