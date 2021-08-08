import time
from itertools import product

from text2network.functions.file_helpers import check_create_folder
from text2network.measures.measures import average_cluster_proximities, extract_all_clusters, proximities
from text2network.utils.logging_helpers import setup_logger
import logging
import numpy as np
from text2network.classes.neo4jnw import neo4j_network

# Set a configuration path
configuration_path = '/config/config.ini'
# Settings
years = list(range(1980, 2021))
alter_subset = None
# Load Configuration file
import configparser
config = configparser.ConfigParser()
print(check_create_folder(configuration_path))
config.read(check_create_folder(configuration_path))
# Setup logging
setup_logger(config['Paths']['log'], config['General']['logging_level'], "founder_proximities2.py")


import os

os.environ['NUMEXPR_MAX_THREADS'] = '16'
# First, create an empty network
semantic_network = neo4j_network(config)

weight_list = [0]
rs_list = [100]
batch_sizes=[1000]
rev_ties_list =[True,False]
comp_ties_list = [True]
back_out_list = [False]
focal_context_list = [("founder",None)]




param_list = product(rs_list, weight_list,focal_context_list,rev_ties_list,back_out_list,comp_ties_list)
logging.info("------------------------------------------------")
for rs, cutoff, fc_list, rev,backout,comp  in param_list:
    focal_token, context = fc_list
    ##
    if rev:
        depth=100000
    else:
        depth=0

    start_time = time.time()

    del semantic_network
    np.random.seed(rs)
    semantic_network = neo4j_network(config)
    filename = "".join(
        [config['Paths']['csv_outputs'], "/Prox_", str(focal_token), "_backout",
         str(backout), "_context", str(context is not None), "_rev", str(rev), "_norm",
         str(comp), "_cut",
         str(cutoff), "_rs", str(rs),".xlsx"])
    logging.info("Focal token:{}, {}".format(focal_token, filename))
    semantic_network.condition(tokens=focal_token, times=None, context=context, depth=depth, weight_cutoff=cutoff,
                 compositional=comp)
    if rev:
        semantic_network.to_reverse()

    df = proximities(semantic_network, focal_tokens=focal_token)
    df = semantic_network.pd_format(df)[0].T

    # This was the difference between founder and CEO, redo this elsewhere
    #df.iloc[:, 0]=df.iloc[:,0]/np.sum(df.iloc[:,0])
    #df.iloc[:, 1]=df.iloc[:,1]/np.sum(df.iloc[:,1])
    #df["diff"]=df.iloc[:,0]-df.iloc[:,1]
    #df["abs_diff"]=np.abs(df.iloc[:,0]-df.iloc[:,1])
    #df["sum"]=df.iloc[:,0]+df.iloc[:,1]
    #df["relative_diff"]=df["diff"]/df["sum"]
    #df["abs_rel_diff"]=df["abs_diff"]/df["sum"]
    #df["sum_same"]=df["sum"]/df["abs_diff"]
    #print(df)
    filename = check_create_folder(filename)
    logging.info("Time:{}".format(time.time() - start_time))
    df.to_excel(filename,merge_cells=False)



