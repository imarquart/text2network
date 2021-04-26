from itertools import product

from src.functions.file_helpers import check_create_folder
from src.measures.measures import average_fixed_cluster_proximities, extract_all_clusters
from src.utils.logging_helpers import setup_logger
import logging
import numpy as np
from src.functions.graph_clustering import consensus_louvain
from src.classes.neo4jnw import neo4j_network

# Set a configuration path
configuration_path = '/config/config.ini'
# Settings
years = range(1980, 2020)
focal_words = ["founder"]

# Load Configuration file
import configparser

config = configparser.ConfigParser()
print(check_create_folder(configuration_path))
config.read(check_create_folder(configuration_path))
# Setup logging
setup_logger(config['Paths']['log'], config['General']['logging_level'], "clusteringleader")

# First, create an empty network
semantic_network = neo4j_network(config)

# Random seed
rs=1000

# Get a list of all clusters at level up to 6
rev=False
comp=False
depth=1
cutoff=0.01
level=6

filename = "".join(
    [config['Paths']['csv_outputs'], "/", str(focal_words), "_all_rev", str(rev), "_norm", str(comp), "_egocluster_lev",
     str(level), "_cut",
     str(cutoff), "_depth", str(depth), "_rs", str(rs)])
logging.info("Network clustering: {}".format(filename))

df = extract_all_clusters(level=level, cutoff=cutoff, focal_token=focal_words, semantic_network=semantic_network,
                          depth=depth, algorithm=consensus_louvain, filename=filename,
                          compositional=comp, reverse_ties=rev)

# Follow these across years

filename = "".join(
    [config['Paths']['csv_outputs'], "/", str(focal_words), "_rev", str(rev), "_norm", str(comp), "_yearfixed_lev",
     str(level), "_clcut",
     str(cutoff), "_cut", str(cutoff), "_depth", str(depth), "_ma", str(1,1)])

df = average_fixed_cluster_proximities(focal_words, interest_tokens, semantic_network, level, do_reverse=True,
                                       depth=depth, weight_cutoff=cutoff, cluster_cutoff=cutoff,
                                       moving_average=(1, 1), filename=filename, compositional=comp,
                                       reverse_ties=rev)


#### Cluster yearly proximities
# Random Seed
np.random.seed(100)

ma_list = [(1, 1)]
level_list = [5]
weight_list = [0.01, 0.1]
depth_list = [1]
rev_ties_list=[False]
comp_ties_list=[False]
param_list=product(depth_list, level_list, ma_list, weight_list, rev_ties_list,comp_ties_list)
logging.info("------------------------------------------------")
for depth, levels, moving_average, weight_cutoff, rev, comp in param_list:
    interest_tokens = alter_subset
    cluster_cutoff = 0.1
    # weight_cutoff=0

