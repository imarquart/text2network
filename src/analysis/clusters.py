from src.functions.file_helpers import check_create_folder
from src.functions.measures import average_fixed_cluster_proximities
from src.utils.logging_helpers import setup_logger
import logging
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from src.functions.node_measures import proximity, centrality
from src.functions.graph_clustering import consensus_louvain
from src.classes.neo4jnw import neo4j_network

# Set a configuration path
configuration_path = '/config/config.ini'
# Settings
years = range(1980, 2020)
focal_words = ["leader", "manager"]
focal_token = "leader"
alter_subset = ["boss", "coach", "consultant", "expert", "mentor", "superior"]
levels = [12, 12, 12, 12, 12, 12]
cutoffs = [0.1, 0.05, 0.01, 0, 0, 0]
depths = [1, 1, 1, 2, 3, 0]

##############################################################################

# Random Seed
np.random.seed(100)

# Load Configuration file
import configparser

config = configparser.ConfigParser()
print(check_create_folder(configuration_path))
config.read(check_create_folder(configuration_path))
# Setup logging
setup_logger(config['Paths']['log'], config['General']['logging_level'], "clustering")

# First, create an empty network
semantic_network = neo4j_network(config)

# years=np.array(semantic_network.get_times_list())
# years=-np.sort(-years)

logging.info("------------------------------------------------")

focal_token="leader"
interest_tokens=alter_subset
levels=8
cluster_cutoff=0.2
weight_cutoff=0.2
filename = "".join(
    [config['Paths']['csv_outputs'], "/", str(focal_token), "_con_yearfixed_lev", str(levels), "_clcut",
     str(cluster_cutoff),"_cut", str(weight_cutoff), "_depth", str(1),
     ".xlsx"])

df=average_fixed_cluster_proximities(focal_token, interest_tokens, semantic_network, levels, weight_cutoff=weight_cutoff,cluster_cutoff=cluster_cutoff, filename=filename)
logging.info(df)
