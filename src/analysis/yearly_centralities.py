from src.functions.file_helpers import check_create_folder
from src.measures.measures import yearly_centralities
from src.utils.logging_helpers import setup_logger
from src.classes.neo4jnw import neo4j_network

# Set a configuration path
configuration_path = '/config/config.ini'
# Settings
years = None


# Load Configuration file
import configparser

config = configparser.ConfigParser()
print(check_create_folder(configuration_path))
config.read(check_create_folder(configuration_path))
# Setup logging
setup_logger(config['Paths']['log'], config['General']['logging_level'], "yearly_cent")

# First, create an empty network
semantic_network = neo4j_network(config)

cent=yearly_centralities(semantic_network, focal_tokens=["leader","manager"],years=semantic_network.get_times_list(), norm_ties=False)
cent=semantic_network.pd_format(cent)


filename="/ycent_manager-leader.xlsx"
path = config['Paths']['csv_outputs']+filename
path = check_create_folder(path)

cent.to_excel(path)