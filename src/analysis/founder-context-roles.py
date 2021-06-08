import time
from itertools import product

from src.functions.file_helpers import check_create_folder
from src.functions.node_measures import proximity
from src.measures.ego_context_graph import create_ego_context_graph
from src.measures.measures import average_cluster_proximities, extract_all_clusters, return_measure_dict
from src.utils.logging_helpers import setup_logger
import logging
import numpy as np
from src.functions.graph_clustering import consensus_louvain
from src.classes.neo4jnw import neo4j_network
import pandas as pd
import networkx as nx
# Set a configuration path
configuration_path = '/config/config.ini'
# Settings
years = list(range(1980, 2021))

#years=None
#years=2010
focal_words = ["founder"]
interest_list=None
# Load Configuration file
import configparser

config = configparser.ConfigParser()
print(check_create_folder(configuration_path))
config.read(check_create_folder(configuration_path))
# Setup logging
setup_logger(config['Paths']['log'], config['General']['logging_level'], "founder-context-roles.py")


# Random seed
rs=100
# Get a list of all clusters at level up to 6
rev=False
comp=False
depth=1
level=5
ego_cutoff=0.01
context_cutoff=0.1
ma=(2,2)
sym=False
# First, create an empty network



np.random.seed(rs)
snw = neo4j_network(config, seed=rs)



ego_context_graph=create_ego_context_graph(snw, focal_words=focal_words, ego_years=years, context_years=years, ego_cutoff=ego_cutoff, context_cutoff=context_cutoff, symmetric=sym, moving_average=ma)


filename = "".join(
    [config['Paths']['csv_outputs'], "/context-ego-", str(years[0]),"-", str(years[-1]), "_ecutoff", str(ego_cutoff),"_ccutoff",str(context_cutoff),"_sym",str(sym),"_ma", str(ma), "_rs",
         str(rs),".gexf"])
filename= check_create_folder(filename)
nx.write_gexf(ego_context_graph, filename)



# Random seed
rs=100
# Get a list of all clusters at level up to 6
rev=False
comp=False
depth=1
level=5
ego_cutoff=0.0
context_cutoff=0.1
ma=(2,2)
sym=False
# First, create an empty network



np.random.seed(rs)
snw = neo4j_network(config, seed=rs)



ego_context_graph=create_ego_context_graph(snw, focal_words=focal_words, ego_years=years, context_years=years, ego_cutoff=ego_cutoff, context_cutoff=context_cutoff, symmetric=sym, moving_average=ma)


filename = "".join(
    [config['Paths']['csv_outputs'], "/context-ego-", str(years[0]),"-", str(years[-1]), "_ecutoff", str(ego_cutoff),"_ccutoff",str(context_cutoff),"_sym",str(sym),"_ma", str(ma), "_rs",
         str(rs),".gexf"])
filename= check_create_folder(filename)
nx.write_gexf(ego_context_graph, filename)



# Random seed
rs=100
# Get a list of all clusters at level up to 6
rev=False
comp=False
depth=1
level=5
ego_cutoff=0.01
context_cutoff=0.1
ma=(3,2)
sym=False
# First, create an empty network



np.random.seed(rs)
snw = neo4j_network(config, seed=rs)



ego_context_graph=create_ego_context_graph(snw, focal_words=focal_words, ego_years=years, context_years=years, ego_cutoff=ego_cutoff, context_cutoff=context_cutoff, symmetric=sym, moving_average=ma)


filename = "".join(
    [config['Paths']['csv_outputs'], "/context-ego-", str(years[0]),"-", str(years[-1]), "_ecutoff", str(ego_cutoff),"_ccutoff",str(context_cutoff),"_sym",str(sym),"_ma", str(ma), "_rs",
         str(rs),".gexf"])
filename= check_create_folder(filename)
nx.write_gexf(ego_context_graph, filename)



# Random seed
rs=100
# Get a list of all clusters at level up to 6
rev=False
comp=False
depth=1
level=5
ego_cutoff=0.01
context_cutoff=0.1
ma=(1,1)
sym=False
# First, create an empty network



np.random.seed(rs)
snw = neo4j_network(config, seed=rs)



ego_context_graph=create_ego_context_graph(snw, focal_words=focal_words, ego_years=years, context_years=years, ego_cutoff=ego_cutoff, context_cutoff=context_cutoff, symmetric=sym, moving_average=ma)


filename = "".join(
    [config['Paths']['csv_outputs'], "/context-ego-", str(years[0]),"-", str(years[-1]), "_ecutoff", str(ego_cutoff),"_ccutoff",str(context_cutoff),"_sym",str(sym),"_ma", str(ma), "_rs",
         str(rs),".gexf"])
filename= check_create_folder(filename)
nx.write_gexf(ego_context_graph, filename)