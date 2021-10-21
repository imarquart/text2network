from text2network.utils.file_helpers import check_create_folder
from text2network.measures.extract_networks import extract_yearly_networks
from text2network.measures.centrality import yearly_centralities
from text2network.measures.proximity import yearly_proximities
from text2network.utils.logging_helpers import setup_logger
from text2network.classes.neo4jnw import neo4j_network

# Set a configuration path
configuration_path = 'config/analyses/replicationHBR40.ini'
# Settings
years = list(range(1980, 2021))
focal_token="manager"


import os
os.environ['NUMEXPR_MAX_THREADS'] = '16'
alter_subset=None
# Load Configuration file
import configparser

config = configparser.ConfigParser()
print(check_create_folder(configuration_path))
config.read(check_create_folder(configuration_path))
# Setup logging
setup_logger(config['Paths']['log'], config['General']['logging_level'], "replication.py")




# First, create an empty network
semantic_network = neo4j_network(config)

csv_folder=check_create_folder(config['Paths']['csv_outputs'])
network_folder=check_create_folder(csv_folder+"/yearly_networks")


# Extract yearly networks
ffolder=check_create_folder(network_folder+"/forward")
extract_yearly_networks(semantic_network, folder=ffolder, times=years)

ffolder=check_create_folder(network_folder+"/symmetric")
extract_yearly_networks(semantic_network, symmetric=True, folder=ffolder, times=years)

ffolder=check_create_folder(network_folder+"/compositional")
extract_yearly_networks(semantic_network, compositional=True, folder=ffolder, times=years)


# Extract yearly centralities
centrality_folder=check_create_folder(csv_folder+"/centralities")

cent=yearly_centralities(semantic_network, year_list=years,focal_tokens=focal_token, normalization=None, compositional=False, reverse=False, symmetric=False, types=["PageRank", "normedPageRank","local_clustering","weighted_local_clustering"])
cent=semantic_network.pd_format(cent)[0]
filename="/forward_centralities_"+str(focal_token)+".xlsx"
ffolder = check_create_folder(centrality_folder+filename)
cent.to_excel(ffolder,merge_cells=False)

cent=yearly_centralities(semantic_network, year_list=years,focal_tokens=focal_token, normalization="sequences", compositional=False, reverse=False, symmetric=False, types=["PageRank", "normedPageRank","local_clustering","weighted_local_clustering"])
cent=semantic_network.pd_format(cent)[0]
filename="/normed_forward_centralities_"+str(focal_token)+".xlsx"
ffolder = check_create_folder(centrality_folder+filename)
cent.to_excel(ffolder,merge_cells=False)

cent=yearly_centralities(semantic_network, year_list=years,focal_tokens=focal_token, normalization=None, compositional=False, reverse=False, symmetric=True, types=["PageRank", "normedPageRank","local_clustering","weighted_local_clustering"])
cent=semantic_network.pd_format(cent)[0]
filename="/sym_centralities_"+str(focal_token)+".xlsx"
ffolder = check_create_folder(centrality_folder+filename)
cent.to_excel(ffolder,merge_cells=False)

cent=yearly_centralities(semantic_network, year_list=years,focal_tokens=focal_token, normalization="sequences", compositional=False, reverse=False, symmetric=True, types=["PageRank", "normedPageRank","local_clustering","weighted_local_clustering"])
cent=semantic_network.pd_format(cent)[0]
filename="/normed_sym_centralities"+str(focal_token)+".xlsx"
ffolder = check_create_folder(centrality_folder+filename)
cent.to_excel(ffolder,merge_cells=False)


# Yearly proximities
proximitiy_folder=check_create_folder(csv_folder+"/proximities")

cent=yearly_proximities(semantic_network, year_list=years,focal_tokens=focal_token, max_degree=100, normalization=None, compositional=False, reverse=False, symmetric=False)
cent=semantic_network.pd_format(cent)[0]
filename="/forward_proximities_"+str(focal_token)+".xlsx"
ffolder = check_create_folder(proximitiy_folder+filename)
cent.to_excel(ffolder,merge_cells=False)


cent=yearly_proximities(semantic_network, year_list=years,focal_tokens=focal_token, max_degree=100, normalization="sequences", compositional=False, reverse=False, symmetric=False)
cent=semantic_network.pd_format(cent)[0]
filename="/normed_forward_proximities_"+str(focal_token)+".xlsx"
ffolder = check_create_folder(proximitiy_folder+filename)
cent.to_excel(ffolder,merge_cells=False)


cent=yearly_proximities(semantic_network, year_list=years,focal_tokens=focal_token, max_degree=100, normalization=None, compositional=False, reverse=False, symmetric=True)
cent=semantic_network.pd_format(cent)[0]
filename="/sym_proximities_"+str(focal_token)+".xlsx"
ffolder = check_create_folder(proximitiy_folder+filename)
cent.to_excel(ffolder,merge_cells=False)


cent=yearly_proximities(semantic_network, year_list=years,focal_tokens=focal_token, max_degree=100, normalization="sequences", compositional=False, reverse=False, symmetric=True)
cent=semantic_network.pd_format(cent)[0]
filename="/normed_sym_proximities_"+str(focal_token)+".xlsx"
ffolder = check_create_folder(proximitiy_folder+filename)
cent.to_excel(ffolder,merge_cells=False)