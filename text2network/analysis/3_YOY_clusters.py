
from itertools import product

from text2network.utils.file_helpers import check_create_folder, check_folder
from text2network.measures.measures import average_cluster_proximities
from text2network.utils.logging_helpers import setup_logger
import logging
import numpy as np
from text2network.functions.graph_clustering import consensus_louvain
from text2network.classes.neo4jnw import neo4j_network



# Set a configuration path
configuration_path = 'config/analyses/LeaderSenBert40.ini'
# Settings
years = list(range(1980, 2021))
focal_token = "leader"
import os
os.environ['NUMEXPR_MAX_THREADS'] = '16'
alter_subset=None
# Load Configuration file
import configparser

config = configparser.ConfigParser()
print(check_create_folder(configuration_path))
config.read(check_create_folder(configuration_path))
# Setup logging
setup_logger(config['Paths']['log'], config['General']['logging_level'], "3_YOY_clusters.py")

# First, create an empty network
semantic_network = neo4j_network(config)

#### Cluster yearly proximities
ma_list = [None, (2, 0),(1, 0),(1, 1)]
level_list = [10]
weight_list = [0.002]
max_degree_list = [200]
cl_clutoff_list = [0]
depth_list = [1]
rs_list = [100]
rev_ties_list = [False]
algolist=[consensus_louvain]
alter_set=[None]
focaladdlist=[False]
comp_ties_list = [False]
back_out_list= [False]
symmetry_list=[True, False]
param_list = product(depth_list, level_list, ma_list, weight_list, rev_ties_list,symmetry_list, comp_ties_list, rs_list,
                     cl_clutoff_list,back_out_list,focaladdlist,alter_set,max_degree_list,algolist)
logging.info("------------------------------------------------")
for depth, levels, moving_average, weight_cutoff, rev, sym, comp, rs, cluster_cutoff, backout,fadd,alters,max_degree,algo in param_list:
    interest_tokens = alter_subset
    del semantic_network
    np.random.seed(rs)
    semantic_network = neo4j_network(config)
    # weight_cutoff=0
    filename = "".join(
        [config['Paths']['csv_outputs'], "/EgoClusterYOY_",])
    filename = check_create_folder(filename)
    filename = "".join([filename, str(focal_token)])
    filename = "".join([filename, "_max_degree", str(max_degree)])
    filename = "".join([filename, "_rev", str(rev),"_sym", str(sym),])
    filename = "".join([filename, "_lev",  str(levels), ])
    filename = "".join([filename, "_fadd", str(fadd),"_alters", str(str(isinstance(alters,list))), "_norm", str(comp), "_clcut",
         str(cluster_cutoff), "_cut", str(weight_cutoff), "_algo", str(algo.__name__), "_depth", str(depth), "_ma", str(moving_average), "_rs",
         str(rs)])
    logging.info("YOY Network clustering: {}".format(filename))
    df = average_cluster_proximities(focal_token=focal_token, nw=semantic_network, levels=levels, interest_list=alters, times=years,do_reverse=True,
                                     depth=depth, weight_cutoff=weight_cutoff, cluster_cutoff=cluster_cutoff, year_by_year=True, add_focal_to_clusters=fadd,  include_all_levels=False, add_individual_nodes=False,
                                     moving_average=moving_average, filename=filename, compositional=comp, to_back_out=backout, symmetric=sym,
                                     reverse_ties=rev, seed=rs,export_network=True,max_degree=max_degree)
