from text2network.functions.file_helpers import check_create_folder
from text2network.measures.measures import yearly_proximities
from text2network.utils.logging_helpers import setup_logger
from text2network.classes.neo4jnw import neo4j_network

# Set a configuration path
configuration_path = '/config/config.ini'
# Load Configuration file
import configparser
config = configparser.ConfigParser()
print(check_create_folder(configuration_path))
config.read(check_create_folder(configuration_path))
# Setup logging
setup_logger(config['Paths']['log'], config['General']['logging_level'], "founder_yearly_proximities.py")
# First, create an empty network
semantic_network = neo4j_network(config, logging_level=10)

"""
This calculates YOY proximities for the focal token

It will generally extract all peers, which is useful in case of sparsity.
If checking outgoing connections (less sparse) or prominent tokens, you can use the option
alter_subset and it instead tracks a defined subset of tokens
"""
# Settings
years=list(range(1980,2021))
#years=list(range(1980,1985))
top_k_allyears=25
focal_tokens="conflict"
moving_average=None
#tokens=["ceo", "cofounder", "owner", "leader", "insider", "director", "vice", "founding", "entrepreneur", "father", "head", "chair", "editor", "member", "pioneer", "man", "professor", "employee", "consultant", "boss", "visionary", "candidate", "inventor", "successor", "designer", "colleague", "son", "veteran", "builder", "creator", "donor", "champion", "incumbent", "coach", "husband", "salesman", "spokesperson", "predecessor", "governor", "victim", "star", "wizard", "writer", "speaker", "composer", "economist", "farmer", "brother", "cartoonist", "steward", "fellow", "alchemist", "poet", "devil", "facilitator", "historian", "deputy", "associate", "confidant", "actor", "driver", "chef", "ambassador", "daimyo", "superintendent", "songwriter", "advocate", "lawyer", "photographer", "commander", "millionaire", "undertaker", "comptroller", "teller", "treasurer", "blogger", "teammate", "pilgrim", "alpha", "citizen","founder"]
#tokens=["ceo", "chairman", "president", "cofounder", "leader", "owner", "director", "insider", "vice", "entrepreneur", "founding", "executive", "manager", "father", "head", "chair", "managing", "member", "editor", "partner", "man", "consultant", "employee", "professor", "pioneer"]
tokens=["ambiguity","war","hostility","competition", "tension","dispute","change","diversity","struggle","cooperation"]

# rev Comp
prox=yearly_proximities(semantic_network, year_list=years,focal_tokens=focal_tokens, moving_average=moving_average,reverse_ties=False,symmetric=True, compositional=False, backout=False)
prox=semantic_network.pd_format(prox)[0]
prox.reset_index(inplace=True)
filename="/YOY_prox_sym_"+str(focal_tokens)+"_top"+str(top_k_allyears)+"_ma_"+str(moving_average)+".xlsx"
path = config['Paths']['csv_outputs']+filename
path = check_create_folder(path)
prox.to_excel(path,merge_cells=False)

# comp-rev alter subset
prox=yearly_proximities(semantic_network, year_list=years,focal_tokens=focal_tokens, alter_subset=tokens,moving_average=moving_average,symmetric=True,  reverse_ties=True, compositional=True, backout=False)
prox=semantic_network.pd_format(prox)[0]
prox.reset_index(inplace=True)
filename="/YOY_prox_sym_"+str(focal_tokens)+"_altertokens"+str(len(tokens))+"_ma_"+str(moving_average)+".xlsx"
path = config['Paths']['csv_outputs']+filename
path = check_create_folder(path)
prox.to_excel(path,merge_cells=False)





