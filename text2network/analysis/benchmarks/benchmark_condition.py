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
focal_words = ["founder"]

# Load Configuration file
import configparser

config = configparser.ConfigParser()
print(check_create_folder(configuration_path))
config.read(check_create_folder(configuration_path))
# Setup logging
setup_logger(config['Paths']['log'], config['General']['logging_level'], "context-conditioning")


logging.info("-----------------Conditioning Benchmark---------------------")


weight_list = [0.1,0.25,0.5,0.01]
batch_size = [None,10,100,1000,10000]
year_list = [1981,1996,2010]
param_list=product(weight_list, batch_size)
rs=1000
rs_list=[1000,2000,3000]
logging.info("######################################### Benchmark 1-Year Ego Conditioning #########################################")

for cutoff,batch in param_list:

    logging.info("------------------------------------------------")
    test_name="1-Year Conditioning"
    param_string="Years: {},  Cutoff: {}, Batch Size: {}".format(year_list,cutoff,batch)
    time_list=[]
    for year in year_list:
        logging.disable(logging.ERROR)
        del semantic_network
        semantic_network = neo4j_network(config, seed=rs)
        start_time=time.time()
        semantic_network.context_condition(year, weight_cutoff=cutoff, batchsize=batch) # condition on context
        logging.disable(logging.NOTSET)
        time_list.append(time.time() - start_time)
    logging.info("------- {} -------".format(test_name))
    logging.info(param_string)
    logging.info("{} finished in (across years) average {} seconds".format(test_name,np.mean(time_list)))
    logging.info("Last Filename: {}".format(semantic_network.filename))
    logging.info("nodes in network %i" % (len(semantic_network)))
    logging.info("ties in network %i" % (semantic_network.graph.number_of_edges()))
    logging.info("------------------------------------------------")


logging.info("######################################### Benchmark #########################################")
for cutoff,batch in param_list:

    logging.info("------------------------------------------------")
    test_name="Multi-Year Conditioning"
    param_string="Years: {},  Cutoff: {}, Batch Size: {}".format(year_list,cutoff,batch)
    time_list=[]
    # Random Seed
    for rs in rs_list:
        logging.disable(logging.ERROR)
        del semantic_network
        semantic_network = neo4j_network(config, seed=rs)
        start_time=time.time()
        np.random.seed(rs)
        semantic_network.context_condition(year_list, weight_cutoff=cutoff, batchsize=batch) # condition on context
        logging.disable(logging.NOTSET)
        time_list.append(time.time() - start_time)
    logging.info("------- {} -------".format(test_name))
    logging.info(param_string)
    logging.info("{} finished in (across years) average {} seconds".format(test_name,np.mean(time_list)))
    logging.info("Last Filename: {}".format(semantic_network.filename))
    logging.info("nodes in network %i" % (len(semantic_network)))
    logging.info("ties in network %i" % (semantic_network.graph.number_of_edges()))
    logging.info("------------------------------------------------")


logging.info("######################################### Benchmark #########################################")
for cutoff,batch in param_list:

    logging.info("------------------------------------------------")
    test_name="All-Year Conditioning"
    param_string="Years: ALL,  Cutoff: {}, Batch Size: {}".format(cutoff,batch)
    time_list=[]
    # Random Seed
    for rs in rs_list:
        logging.disable(logging.ERROR)
        del semantic_network
        semantic_network = neo4j_network(config, seed=rs)
        start_time=time.time()
        np.random.seed(rs)
        semantic_network.context_condition(weight_cutoff=cutoff, batchsize=batch) # condition on context
        logging.disable(logging.NOTSET)
        time_list.append(time.time() - start_time)
    logging.info("------- {} -------".format(test_name))
    logging.info(param_string)
    logging.info("{} finished in (across years) average {} seconds".format(test_name,np.mean(time_list)))
    logging.info("Last Filename: {}".format(semantic_network.filename))
    logging.info("nodes in network %i" % (len(semantic_network)))
    logging.info("ties in network %i" % (semantic_network.graph.number_of_edges()))
    logging.info("------------------------------------------------")

