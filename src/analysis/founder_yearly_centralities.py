from src.functions.file_helpers import check_create_folder
from src.measures.measures import yearly_centralities
from src.utils.logging_helpers import setup_logger
from src.classes.neo4jnw import neo4j_network

# Set a configuration path
configuration_path = '/config/config.ini'
# Settings
years = None

tokens=["ceo", "cofounder", "owner", "leader", "insider", "director", "vice", "founding", "entrepreneur", "father", "head", "chair", "editor", "member", "pioneer", "man", "professor", "employee", "consultant", "boss", "visionary", "candidate", "inventor", "successor", "designer", "colleague", "son", "veteran", "builder", "creator", "donor", "champion", "incumbent", "coach", "husband", "salesman", "spokesperson", "predecessor", "governor", "victim", "star", "wizard", "writer", "speaker", "composer", "economist", "farmer", "brother", "cartoonist", "steward", "fellow", "alchemist", "poet", "devil", "facilitator", "historian", "deputy", "associate", "confidant", "actor", "driver", "chef", "ambassador", "daimyo", "superintendent", "songwriter", "advocate", "lawyer", "photographer", "commander", "millionaire", "undertaker", "comptroller", "teller", "treasurer", "blogger", "teammate", "pilgrim", "alpha", "citizen","founder"]
# Load Configuration file
import configparser

config = configparser.ConfigParser()
print(check_create_folder(configuration_path))
config.read(check_create_folder(configuration_path))
# Setup logging
setup_logger(config['Paths']['log'], config['General']['logging_level'], "yearly_cent")

# First, create an empty network
semantic_network = neo4j_network(config)

cent=yearly_centralities(semantic_network, semantic_network.get_times_list(),focal_tokens=tokens, reverse_ties=True, compositional=False, backout=False)
cent=semantic_network.pd_format(cent)[0]


filename="/rev_ycent_founder.xlsx"
path = config['Paths']['csv_outputs']+filename
path = check_create_folder(path)

cent.to_excel(path,merge_cells=False)