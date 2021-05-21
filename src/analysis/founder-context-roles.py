import time
from itertools import product

from src.functions.file_helpers import check_create_folder
from src.functions.node_measures import proximity
from src.measures.ego_context_graph import create_ego_context_graph
from src.measures.measures import average_fixed_cluster_proximities, extract_all_clusters, return_measure_dict
from src.utils.logging_helpers import setup_logger
import logging
import numpy as np
from src.functions.graph_clustering import consensus_louvain
from src.classes.neo4jnw import neo4j_network
import pandas as pd
# Set a configuration path
configuration_path = '/config/config.ini'
# Settings
years = list(range(2010, 2020))

years=None
years=2010
focal_words = ["founder"]

# Load Configuration file
import configparser

config = configparser.ConfigParser()
print(check_create_folder(configuration_path))
config.read(check_create_folder(configuration_path))
# Setup logging
setup_logger(config['Paths']['log'], config['General']['logging_level'], "founder-analysis")


# Random seed
rs=1000
# Get a list of all clusters at level up to 6
rev=False
comp=False
depth=1
cutoff=0.1
level=8


# First, create an empty network
snw = neo4j_network(config, seed=rs)

ego_context_graph=create_ego_context_graph(snw, focal_words=focal_words, ego_years=years, context_years=years)


