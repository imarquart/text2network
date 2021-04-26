from itertools import product

from src.functions.file_helpers import check_create_folder
from src.functions.measures import average_fixed_cluster_proximities, extract_all_clusters
from src.utils.logging_helpers import setup_logger
import logging
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from src.functions.node_measures import proximity, centrality
from src.functions.graph_clustering import consensus_louvain, louvain_cluster
from src.classes.neo4jnw import neo4j_network

# Set a configuration path
configuration_path = '/config/config.ini'
# Settings
years = 1986


# Load Configuration file
import configparser

config = configparser.ConfigParser()
print(check_create_folder(configuration_path))
config.read(check_create_folder(configuration_path))
# Setup logging
setup_logger(config['Paths']['log'], config['General']['logging_level'], "clusteringB")

# First, create an empty network
semantic_network = neo4j_network(config)

# Condition network
semantic_network.condition(years=years)




