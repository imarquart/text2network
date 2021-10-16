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
focal_tokens = ["conflict"]
alter_subset=None
import os
os.environ['NUMEXPR_MAX_THREADS'] = '16'
alter_subset=None
# Load Configuration file
import configparser

config = configparser.ConfigParser()
print(check_create_folder(configuration_path))
config.read(check_create_folder(configuration_path))
# Setup logging
setup_logger(config['Paths']['log'], config['General']['logging_level'], "conflict_ego_clusters.py")
ma=(2,2)
# First, create an empty network
semantic_network = neo4j_network(config)
logging.info("first nw created")
level_list = [5]
weight_list = [0]
cl_clutoff_list = [0]
depth_list = [1,2]
rs_list = [100]
rev_ties_list = [False]
algolist=[consensus_louvain]
alter_set=[None]
focaladdlist=[True,False]
comp_ties_list = [False]
back_out_list= [False]
symmetry_list=[True]
max_degree=100
param_list = product(focal_tokens,depth_list, level_list, rs_list, weight_list, rev_ties_list, comp_ties_list, cl_clutoff_list,algolist,back_out_list,symmetry_list,focaladdlist,alter_set)
logging.info("------------------------------------------------")
for focal_token,depth, level, rs, cutoff, rev, comp, cluster_cutoff,algo,backout,sym,fadd,alters in param_list:
    del semantic_network
    np.random.seed(rs)
    semantic_network = neo4j_network(config)
    filename = "".join(
        [config['Paths']['csv_outputs'], "/EgoCluster_", str(focal_token), "_backout", str(backout),"_sym", str(sym),"_fadd", str(fadd),"_alters", str(str(isinstance(alters,list))),"_rev", str(rev), "_norm", str(comp),
         "_lev", str(level), "_cut","_max_deg", str(max_degree),
         str(cutoff), "_clcut", str(cluster_cutoff), "_algo", str(algo.__name__), "_depth", str(depth), "_rs", str(rs)])
    logging.info("Network clustering: {}".format(filename))
    #df = average_cluster_proximities(focal_token=focal_token, nw=semantic_network, levels=level, interest_list=alters, times=years,do_reverse=True,
    #                                 depth=depth, weight_cutoff=cutoff, cluster_cutoff=cluster_cutoff, year_by_year=False, add_focal_to_clusters=fadd,
    #                                 moving_average=None, filename=filename, compositional=comp, to_back_out=backout, include_all_levels=True, add_individual_nodes=True,
    #                                 reverse_ties=rev, symmetric=sym, seed=rs, export_network=True, max_degree=max_degree)


level_list = [5]
weight_list = [0]
cl_clutoff_list = [0]
depth_list = [1]
rs_list = [100]
rev_ties_list = [False]
algolist=[consensus_louvain]
alter_set=[None]
focaladdlist=[True]
comp_ties_list = [False]
back_out_list= [False]
symmetry_list=[True]
max_degree=100
param_list = product(focal_tokens,depth_list, level_list, rs_list, weight_list, rev_ties_list, comp_ties_list, cl_clutoff_list,algolist,back_out_list,symmetry_list,focaladdlist,alter_set)
logging.info("------------------------------------------------")
for focal_token,depth, level, rs, cutoff, rev, comp, cluster_cutoff,algo,backout,sym,fadd,alters in param_list:
    del semantic_network
    np.random.seed(rs)
    semantic_network = neo4j_network(config)
    filename = "".join(
        [config['Paths']['csv_outputs'], "/YOYCluster_", str(focal_token), "_backout", str(backout),"_sym", str(sym),"_fadd", str(fadd),"_alters", str(str(isinstance(alters,list))),"_rev", str(rev), "_norm", str(comp),
         "_lev", str(level), "_cut", "_ma", str(ma),"_max_deg", str(max_degree),
         str(cutoff), "_clcut", str(cluster_cutoff), "_algo", str(algo.__name__), "_depth", str(depth), "_rs", str(rs)])
    logging.info("Network clustering: {}".format(filename))
    df = average_cluster_proximities(focal_token=focal_token, nw=semantic_network, levels=level, interest_list=alters, times=years,do_reverse=True,
                                     depth=depth, weight_cutoff=cutoff, cluster_cutoff=cluster_cutoff, year_by_year=True, add_focal_to_clusters=fadd,
                                     moving_average=ma, filename=filename, compositional=comp, to_back_out=backout, include_all_levels=False, add_individual_nodes=False,
                                     reverse_ties=rev, symmetric=sym, seed=rs, export_network=False, max_degree=max_degree)