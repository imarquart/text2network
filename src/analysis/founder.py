import time
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
years = list(range(2010, 2020))

years=None
#years=2010
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
level=10


# First, create an empty network
semantic_network = neo4j_network(config, seed=rs)



# Contextual analysis
del semantic_network
semantic_network = neo4j_network(config, seed=rs)
start_time=time.time()
semantic_network.context_condition(times=years, tokens=focal_words, depth=1, weight_cutoff=cutoff) # condition on context
logging.info("nodes in network %i" % (len(semantic_network)))
logging.info("ties in network %i" % (semantic_network.graph.number_of_edges()))
logging.info("finished in (across years) average {} seconds".format((time.time() - start_time)))
context_dyads= semantic_network.pd_format(semantic_network.proximities(focal_tokens=focal_words))[0].T
filename = "".join(
    [config['Paths']['csv_outputs'], "/prox-", semantic_network.filename,".xlsx"])
filename= check_create_folder(filename)
context_dyads.to_excel(filename)

del semantic_network
semantic_network = neo4j_network(config, seed=rs)
start_time=time.time()
semantic_network.context_condition(times=years, tokens=focal_words, depth=1, weight_cutoff=cutoff, occurrence=True) # condition on context
logging.info("nodes in network %i" % (len(semantic_network)))
logging.info("ties in network %i" % (semantic_network.graph.number_of_edges()))
logging.info("finished in (across years) average {} seconds".format((time.time() - start_time)))
context_dyads2= semantic_network.pd_format(semantic_network.proximities(focal_tokens=focal_words))[0].T

filename = "".join(
    [config['Paths']['csv_outputs'], "/occ-prox-", semantic_network.filename,".xlsx"])
filename= check_create_folder(filename)
context_dyads2.to_excel(filename)
# Contextual clustering
filename = "".join(
    [config['Paths']['csv_outputs'], "/cluster-", semantic_network.filename,""])
filename= check_create_folder(filename)
df = extract_all_clusters(level=level, cutoff=cutoff, focal_token=focal_words[0], snw=semantic_network,
                          depth=1, algorithm=consensus_louvain, times=years,filename=filename,add_focal_to_clusters=False, mode="context",seed=rs)

filename = "".join(
    [config['Paths']['csv_outputs'], "/cluster-focaladd-", semantic_network.filename,""])
filename= check_create_folder(filename)
df2 = extract_all_clusters(level=level, cutoff=cutoff, focal_token=focal_words[0], snw=semantic_network,
                          depth=1, algorithm=consensus_louvain, times=years,filename=filename,add_focal_to_clusters=True, mode="context",seed=rs)


# Contextual clustering
filename = "".join(
    [config['Paths']['csv_outputs'], "/occ-cluster-", semantic_network.filename,""])
filename= check_create_folder(filename)
df = extract_all_clusters(level=level, cutoff=cutoff, focal_token=focal_words[0], snw=semantic_network,
                          depth=1, algorithm=consensus_louvain, times=years,filename=filename,add_focal_to_clusters=False,occurrence=True, mode="context",seed=rs)

filename = "".join(
    [config['Paths']['csv_outputs'], "/occ-cluster-focaladd-", semantic_network.filename,""])
filename= check_create_folder(filename)
df2 = extract_all_clusters(level=level, cutoff=cutoff, focal_token=focal_words[0], snw=semantic_network,
                          depth=1, algorithm=consensus_louvain, times=years,filename=filename,add_focal_to_clusters=True, mode="context",occurrence=True,seed=rs)


