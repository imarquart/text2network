from text2network.utils.file_helpers import check_create_folder
from text2network.measures.centrality import yearly_centralities
from text2network.utils.logging_helpers import setup_logger
from text2network.classes.neo4jnw import neo4j_network
import logging


# Set a configuration path
configuration_path = 'config/analyses/LeaderSenBert40.ini'

# Settings
years = list(range(1980, 2021))
import os
os.environ['NUMEXPR_MAX_THREADS'] = '16'
# Load Configuration file
import configparser

config = configparser.ConfigParser()
print(check_create_folder(configuration_path))
config.read(check_create_folder(configuration_path))
# Setup logging
setup_logger(config['Paths']['log'], config['General']['logging_level'], "8_yearly_centralities.py")

# First, create an empty network
semantic_network = neo4j_network(config)

focal_tokens=["leader"]
types = ["frequency", "PageRank", "normedPageRank", "flow_betweenness", "rev_flow_betweenness", "local_clustering",
         "weighted_local_clustering"]

path = config['Paths']['csv_outputs']+"/yearly_centralities/gephi/"
cent=yearly_centralities(semantic_network, batch_size=100, depth=None, return_sentiment=False, weight_cutoff=None, year_list=years,focal_tokens=focal_tokens, max_degree=None, types=types, normalization="sequences", path=path)
cent=semantic_network.pd_format(cent)[0]

filename="/yearly_centralities/centrality_leader_YOY.xlsx"
path = config['Paths']['csv_outputs']+filename
path = check_create_folder(path)

cent.to_excel(path,merge_cells=False)