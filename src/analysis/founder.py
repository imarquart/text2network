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

years=[2010]
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
level=6


# First, create an empty network
semantic_network = neo4j_network(config, seed=rs)


filename = "".join(
    [config['Paths']['csv_outputs'], "/", str(focal_words), "_all_rev", str(rev), "_norm", str(comp), "_egocluster_lev",
     str(level), "_cut",
     str(cutoff), "_depth", str(depth), "_rs", str(rs)])
logging.info("Network clustering: {}".format(filename))

df = extract_all_clusters(level=level, cutoff=cutoff, focal_token=focal_words[0], snw=semantic_network,
                          depth=depth, algorithm=consensus_louvain, times=years,filename=filename,
                          compositional=comp, reverse_ties=rev, add_focal_to_clusters=False, seed=rs)

# Extract tokens of interest as those tokens that
# are in the same cluster as focal token
df_subset=df[df.Level==1]
focal_cluster = df_subset[df_subset.Token==focal_words[0]].Clustername
df_subset=df_subset[df_subset.Clustername==focal_cluster.iloc[0]]
df_subset=df_subset[df_subset.Delta_Proximity != -100]
df_subset=df_subset[df_subset.Proximity != 0]
interest_tokens=list(df_subset.Token)




# Follow these across years
filename = "".join(
    [config['Paths']['csv_outputs'], "/", str(focal_words), "_rev", str(rev), "_norm", str(comp), "_yearfixed_lev",
     str(level), "_clcut",
     str(cutoff), "_cut", str(cutoff), "_depth", str(depth), "_ma", str("1,1")])

df = average_fixed_cluster_proximities(focal_words[0], interest_tokens, semantic_network, level, times=years,do_reverse=True,
                                       depth=depth, weight_cutoff=cutoff, cluster_cutoff=cutoff,
                                       moving_average=(1, 1), filename=filename, compositional=comp,
                                       reverse_ties=rev, seed=rs)

# Contextual analysis
years=[2010]
semantic_network.decondition()
start_time=time.time()
semantic_network.context_condition(years, weight_cutoff=cutoff, batchsize=100) # condition on context
logging.info("nodes in network %i" % (len(semantic_network)))
logging.info("ties in network %i" % (semantic_network.graph.number_of_edges()))
logging.info("finished in (across years) average {} seconds".format((time.time() - start_time)))
context_dyads= semantic_network.proximities(focal_tokens=focal_words)
print(semantic_network.pd_format(context_dyads))

years=[2010,2011,2012,2015,2016]
semantic_network.decondition()
start_time=time.time()
semantic_network.context_condition(years, weight_cutoff=cutoff, batchsize=100) # condition on context
logging.info("nodes in network %i" % (len(semantic_network)))
logging.info("ties in network %i" % (semantic_network.graph.number_of_edges()))
logging.info("finished in (across years) average {} seconds".format((time.time() - start_time)))
context_dyads= semantic_network.proximities(focal_tokens=focal_words)
print(semantic_network.pd_format(context_dyads))

years = list(range(2010, 2020))
semantic_network.decondition()
start_time=time.time()
semantic_network.context_condition(years, weight_cutoff=cutoff, batchsize=100) # condition on context
logging.info("nodes in network %i" % (len(semantic_network)))
logging.info("ties in network %i" % (semantic_network.graph.number_of_edges()))
logging.info("finished in (across years) average {} seconds".format((time.time() - start_time)))
context_dyads= semantic_network.proximities(focal_tokens=focal_words)
print(semantic_network.pd_format(context_dyads))

# Contextual clustering

semantic_network.cluster(levels=level)


