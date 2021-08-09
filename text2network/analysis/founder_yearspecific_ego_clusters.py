from itertools import product
import pandas as pd
from text2network.functions.file_helpers import check_create_folder
from text2network.measures.measures import average_cluster_proximities, extract_all_clusters
from text2network.utils.logging_helpers import setup_logger
import logging
import numpy as np
from text2network.functions.graph_clustering import consensus_louvain, louvain_cluster
from text2network.classes.neo4jnw import neo4j_network

# Set a configuration path
configuration_path = '/config/config.ini'
# Settings
years = list(range(1980, 2021,5))
focal_token = "founder"

import os
os.environ['NUMEXPR_MAX_THREADS'] = '16'
alter_subset=None
# Load Configuration file
import configparser

config = configparser.ConfigParser()
print(check_create_folder(configuration_path))
config.read(check_create_folder(configuration_path))
# Setup logging
setup_logger(config['Paths']['log'], config['General']['logging_level'], "founder_ego_clusters.py")

# First, create an empty network
level_list = [5]
weight_list = [0]
cl_clutoff_list = [0]
depth_list = [1]
rs_list = [100]
rev_ties_list = [True]
algolist=[consensus_louvain]
alter_set=[None]
focaladdlist=[True]
comp_ties_list = [True]
back_out_list= [False]
symmetry_list=[False]
param_list = product(depth_list, level_list, rs_list, weight_list, rev_ties_list, comp_ties_list, cl_clutoff_list,algolist,back_out_list,symmetry_list,focaladdlist,alter_set)
logging.info("------------------------------------------------")
for depth, level, rs, cutoff, rev, comp, cluster_cutoff,algo,backout,sym,fadd,alters in param_list:
    df_list=[]
    for year in years:
        np.random.seed(rs)
        start_year = max(years[0], year - 5)
        end_year = min(years[-1], year)
        ma_years = list(np.arange(start_year, end_year + 1))
        semantic_network = neo4j_network(config)
        filename = "".join(
            [config['Paths']['csv_outputs'], "/5MA_YearSpecEgoCluster_", str(focal_token), "_y", "-".join([str(x) for x in ma_years]), "_sym", str(sym),"_rev", str(rev), "_norm", str(comp), str(level), "_cut",
             str(cutoff), "_clcut", str(cluster_cutoff), "_depth", str(depth), "_rs", str(rs)])
        logging.info("Network clustering: {}".format(filename))
        df = average_cluster_proximities(focal_token=focal_token, nw=semantic_network, levels=level, interest_list=alters, times=ma_years,do_reverse=True,
                                         depth=depth, weight_cutoff=cutoff, cluster_cutoff=cluster_cutoff, year_by_year=False, add_focal_to_clusters=fadd,
                                         moving_average=None, filename=filename, compositional=comp, to_back_out=backout, include_all_levels=True, add_individual_nodes=True,
                                         reverse_ties=rev, symmetric=sym, seed=rs, export_network=True)
        df["Year"]=year
        df_list.append(df.copy())
    filename = "".join(
        [config['Paths']['csv_outputs'], "/YearSpecEgoCluster_", str(focal_token), "_yALL", "_sym", str(sym),
         "_rev", str(rev), "_norm", str(comp), str(level), "_cut",
         str(cutoff), "_clcut", str(cluster_cutoff), "_depth", str(depth), "_rs", str(rs)])
    df = pd.concat(df_list)
    filename = check_create_folder(filename + ".xlsx")
    df.to_excel(filename)

