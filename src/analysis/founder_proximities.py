import time
from itertools import product

from src.functions.file_helpers import check_create_folder
from src.measures.measures import average_cluster_proximities, extract_all_clusters, proximities
from src.utils.logging_helpers import setup_logger
import logging
import numpy as np
from src.functions.graph_clustering import consensus_louvain, louvain_cluster
from src.classes.neo4jnw import neo4j_network

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

weight_list = [0.1, 0.01,0]
depth_list = [1]
rs_list = [100]
batch_sizes=[1000]
rev_ties_list = [False]
focal_context_list = [("zuckerberg", ["facebook", "mark", "marc"]), ("jobs", ["steve", "apple", "next"]),
                      ("musk", ["elon", "tesla", "paypal"]), ("gates", ["bill", "microsoft"]),
                      ("page", ["larry", "google"]),("brinn", ["sergej", "google"]),("branson", ["richard", "virgin"]),("bezos", ["jeff", "amazon"]),]
focal_context_list = [("zuckerberg", ["facebook", "mark", "marc"]),("zuckerberg", None)]
focal_context_list = [("founder",None),("ceo", None)]
#focal_context_list = [("founder", None),("ceo", None)]

comp_ties_list = [False]
back_out_list = [False]
query_modes=["new", "old"]
param_list = product(rs_list, weight_list,focal_context_list,rev_ties_list,back_out_list,comp_ties_list,query_modes, batch_sizes)
logging.info("------------------------------------------------")
for rs, cutoff, fc_list, rev,backout,comp,mode,batch_size  in param_list:
    focal_token, context = fc_list
    ##
    start_time = time.time()
    logging.info("Focal token:{} Context: {}".format(focal_token,context))
    del semantic_network
    np.random.seed(rs)
    semantic_network = neo4j_network(config)
    filename = "".join(
        [config['Paths']['csv_outputs'], "/",str(mode),"cProx_", str(focal_token), "_backout",
         str(backout), "_context", str(context is not None), "_rev", str(rev), "_norm",
         str(comp), "_cut",
         str(cutoff), "_rs", str(rs),".xlsx"])
    logging.info("Proximities: {}, batch_size: {}".format(filename, batch_size))
    semantic_network.condition(tokens=focal_token, times=None, context=context, depth=1, weight_cutoff=cutoff,
                 compositional=comp, reverse_ties=rev, query_mode=mode, batchsize=batch_size)
    df = proximities(semantic_network, focal_tokens=focal_token)
    df = semantic_network.pd_format(df)[0].T
    filename = check_create_folder(filename)
    logging.info("Time:{}".format(time.time() - start_time))
    df.to_excel(filename,merge_cells=False)



