import time
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
years = list(range(1980, 2021))
focal_token = ["founder"]

# Load Configuration file
import configparser

config = configparser.ConfigParser()
print(check_create_folder(configuration_path))
config.read(check_create_folder(configuration_path))
# Setup logging
setup_logger(config['Paths']['log'], config['General']['logging_level'], "founder-context")

# First, create an empty network
semantic_network = neo4j_network(config)

focal_token="founder"
alter_subset=None
level_list = [2,5]
weight_list = [0.3,0.1,0.2]
cl_clutoff_list = [80,90]
depth_list = [1]
rs_list = [100]
rev_ties_list = [False]
algolist=[louvain_cluster,consensus_louvain]
algolist=[consensus_louvain]
focaladdlist=[True,False]
comp_ties_list = [False]
occ_list=[False]
param_list = product(depth_list, level_list, rs_list, weight_list, rev_ties_list, comp_ties_list, cl_clutoff_list,algolist,focaladdlist,occ_list,occ_list)
logging.info("------------------------------------------------")
for depth, level, rs, cutoff, rev, comp, cluster_cutoff,algo,focaladd,occ,backout in param_list:
    del semantic_network
    np.random.seed(rs)
    semantic_network = neo4j_network(config)
    filename = "".join(
        [config['Paths']['csv_outputs'], "/ContextCluster_", str(focal_token), "_backout", str(backout),"_occ", str(occ),"_fadd", str(focaladd),"_rev", str(rev), "_norm", str(comp),
         "_lev", str(level), "_cut",
         str(cutoff), "_clcut", str(cluster_cutoff), "_algo", str(algo.__name__), "_depth", str(depth), "_rs", str(rs)])
    logging.info("Context Network clustering: {}".format(filename))
    # Random Seed
    #df = extract_all_clusters(level=level, cutoff=cutoff, times=years, cluster_cutoff=cluster_cutoff, focal_token=focal_token,
    #                          interest_list=alter_subset, snw=semantic_network,
    #                          add_focal_to_clusters=focaladd,
    #                          occurrence=occ, mode="context",to_back_out=backout,
    #                          depth=depth, algorithm=algo, filename=filename,
    #                          compositional=comp, reverse_ties=rev, seed=rs)

#### Cluster yearly proximities


ma_list = [(2, 2)]
level_list = [1]
weight_list = [0.1]
cl_clutoff_list = [0,100]
depth_list = [1,2]
rs_list = [200]
rev_ties_list = [False]
algolist=[consensus_louvain]
focaladdlist=[True]
comp_ties_list = [False]
param_list = product(ma_list,depth_list, level_list, rs_list, weight_list, rev_ties_list, comp_ties_list, cl_clutoff_list,algolist,focaladdlist,comp_ties_list,comp_ties_list)
logging.info("------------------------------------------------")
for ma,depth, level, rs, cutoff, rev, comp, cluster_cutoff,algo,focaladd,occ,backout in param_list:
    interest_tokens = None
    del semantic_network
    np.random.seed(rs)
    semantic_network = neo4j_network(config)
    # weight_cutoff=0
    filename = "".join(
        [config['Paths']['csv_outputs'], "/ContextClusterYOY_",str(focal_token),  "_backout", str(backout),"_occ", str(occ),"_fadd", str(focaladd), "_rev", str(rev), "_norm", str(comp), "_lev",
         str(level), "_clcut",
         str(cluster_cutoff), "_cut", str(cutoff), "_algo", str(algo.__name__), "_depth", str(depth), "_ma", str(ma), "_rs",
         str(rs)])
    logging.info("YOY Network clustering: {}".format(filename))
    df = average_cluster_proximities(focal_token=focal_token, nw=semantic_network, levels=level, interest_list=alter_subset, times=years,do_reverse=True,
                                     depth=depth, weight_cutoff=cutoff, cluster_cutoff=cluster_cutoff, year_by_year=True,
                                     add_focal_to_clusters=focaladd, occurrence=occ,to_back_out=backout,
                                     moving_average=ma, filename=filename, compositional=comp, mode="context",
                                     reverse_ties=rev, seed=rs)
