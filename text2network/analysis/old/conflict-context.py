from itertools import product

from text2network.utils.file_helpers import check_create_folder
from text2network.measures.measures import average_cluster_proximities
from text2network.utils.logging_helpers import setup_logger
import logging
import numpy as np
from text2network.functions.graph_clustering import consensus_louvain
from text2network.classes.neo4jnw import neo4j_network

# Set a configuration path
configuration_path = '/config/config.ini'
# Settings
years = list(range(1980, 2021))


# Load Configuration file
import configparser

config = configparser.ConfigParser()
print(check_create_folder(configuration_path))
config.read(check_create_folder(configuration_path))
# Setup logging
setup_logger(config['Paths']['log'], config['General']['logging_level'], "founder-context")

# First, create an empty network
semantic_network = neo4j_network(config)

focal_token="conflict"
alter_subset=None
level_list = [2,5]
weight_list = [0.3,0.1,0.2]
cl_clutoff_list = [0]
depth_list = [1]
rs_list = [100]
ma_list = [(2,2)]
sym_list = [True]
#algolist=[louvain_cluster,consensus_louvain]
algolist=[consensus_louvain]
focaladdlist=[True]
comp_ties_list = [False]
occ_list=[False, True]
backout_list=[False]
max_degree=50
param_list = product(depth_list, level_list, rs_list, weight_list, sym_list, comp_ties_list, cl_clutoff_list,algolist,focaladdlist,occ_list,backout_list)
logging.info("------------------------------------------------")
for depth, level, rs, cutoff, sym, comp, cluster_cutoff,algo,focaladd,occ,backout in param_list:
    del semantic_network
    np.random.seed(rs)
    semantic_network = neo4j_network(config)
    filename = "".join(
        [config['Paths']['csv_outputs'], "/ContextCluster_", str(focal_token), "_backout", str(backout),"_occ", str(occ),"_fadd", str(focaladd),"_sym", str(sym), "_norm", str(comp),
         "_lev", str(level), "_cut",
         str(cutoff), "_clcut", str(cluster_cutoff), "_algo", str(algo.__name__), "_depth", str(depth), "_rs", str(rs)])
    logging.info("Context Network clustering: {}".format(filename))
    #df = average_cluster_proximities(focal_token=focal_token, nw=semantic_network, levels=level, interest_list=alter_subset, times=years,do_reverse=True,
    #                                 depth=depth, weight_cutoff=cutoff, cluster_cutoff=cluster_cutoff, year_by_year=False, include_all_levels=True, add_individual_nodes=True,
    #                                 add_focal_to_clusters=focaladd, export_network=True,occurrence=occ,to_back_out=backout,filename=filename, compositional=comp, mode="context",
    #                                 symmetric=sym, max_degree=max_degree, seed=rs)#

#### Cluster yearly proximities

param_list = product(ma_list,depth_list, level_list, rs_list, weight_list, sym_list, comp_ties_list, cl_clutoff_list,algolist,focaladdlist,occ_list,backout_list)
logging.info("------------------------------------------------")
for ma,depth, level, rs, cutoff, sym, comp, cluster_cutoff,algo,focaladd,occ,backout in param_list:
    interest_tokens = None
    del semantic_network
    np.random.seed(rs)
    semantic_network = neo4j_network(config)
    # weight_cutoff=0
    filename = "".join(
        [config['Paths']['csv_outputs'], "/ContextClusterYOY_",str(focal_token),  "_backout", str(backout),"_occ", str(occ),"_fadd", str(focaladd), "_sym", str(sym), "_norm", str(comp), "_lev",
         str(level), "_clcut",
         str(cluster_cutoff), "_cut", str(cutoff), "_algo", str(algo.__name__), "_depth", str(depth), "_ma", str(ma), "_rs",
         str(rs)])
    logging.info("YOY Network clustering: {}".format(filename))
    df = average_cluster_proximities(focal_token=focal_token, nw=semantic_network, levels=level, interest_list=alter_subset, times=years,do_reverse=True,
                                     depth=depth, weight_cutoff=cutoff, cluster_cutoff=cluster_cutoff, year_by_year=True,
                                     add_focal_to_clusters=focaladd, occurrence=occ,to_back_out=backout,
                                     moving_average=ma, filename=filename, compositional=comp, mode="context",
                                     symmetric=sym, max_degree=max_degree, seed=rs)
