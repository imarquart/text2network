from text2network.functions.file_helpers import check_create_folder
from text2network.measures.measures import yearly_centralities
from text2network.utils.logging_helpers import setup_logger
from text2network.classes.neo4jnw import neo4j_network
import logging

# Set a configuration path
configuration_path = '/config/config.ini'
# Settings
years = None

tokens=["ceo", "cofounder", "owner", "leader", "insider", "director", "vice", "founding", "entrepreneur", "father", "head", "chair", "editor", "member", "pioneer", "man", "professor", "employee", "consultant", "boss", "visionary", "candidate", "inventor", "successor", "designer", "colleague", "son", "veteran", "builder", "creator", "donor", "champion", "incumbent", "coach", "husband", "salesman", "spokesperson", "predecessor", "governor", "victim", "star", "wizard", "writer", "speaker", "composer", "economist", "farmer", "brother", "cartoonist", "steward", "fellow", "alchemist", "poet", "devil", "facilitator", "historian", "deputy", "associate", "confidant", "actor", "driver", "chef", "ambassador", "daimyo", "superintendent", "songwriter", "advocate", "lawyer", "photographer", "commander", "millionaire", "undertaker", "comptroller", "teller", "treasurer", "blogger", "teammate", "pilgrim", "alpha", "citizen","founder"]

tokens=["conflicts", "problem", "war", "dispute", "tension", "competition", "change", "problems", "relationship", "crisis", "difference", "lack", "disagreement", "risk", "stress", "odds", "clash", "gap", "differences", "debate", "trouble", "interfere", "uncertainty", "disagreements", "friction", "issue", "issues", "confrontation", "ambiguity", "situation", "struggle", "confusion", "disputes", "dialogue", "battle", "failure", "resolution", "management", "among", "business", "cooperation", "work", "align", "interest", "fit", "resistance", "overlap", "question", "tensions", "line", "bias", "contact", "diversity", "process", "relationships", "agreement", "matter", "collaboration", "complexity", "rivalry", "compete", "politics", "action", "communication"]
# Load Configuration file
import configparser

config = configparser.ConfigParser()
print(check_create_folder(configuration_path))
config.read(check_create_folder(configuration_path))
# Setup logging config['General']['logging_level']
setup_logger(config['Paths']['log'],0, "yearly_cent")
logging.debug("Test")
# First, create an empty network
semantic_network = neo4j_network(config)
logging.getLogger().setLevel(0)
logging.debug("Test")
cent=yearly_centralities(semantic_network, semantic_network.get_times_list(),focal_tokens=tokens, reverse_ties=False, symmetric=True, backout=False, max_degree=100)
cent=semantic_network.pd_format(cent)[0]


filename="/rev_ycent_conflict.xlsx"
path = config['Paths']['csv_outputs']+filename
path = check_create_folder(path)

cent.to_excel(path,merge_cells=False)