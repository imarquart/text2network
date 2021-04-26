from src.functions.file_helpers import check_create_folder
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
setup_logger(config['Paths']['log'], config['General']['logging_level'], "dyadic_context")

# First, create an empty network
semantic_network = neo4j_network(config)

# Condition network

# Condition network
ties=semantic_network.get_dyad_context(occurrence='manager', replacement='leader', years=years)
ties=semantic_network.pd_format(ties)[0]
ties.columns=["manager-leader"]
ties2=semantic_network.get_node_context("manager",years=years,occurrence=True)
ties2=semantic_network.pd_format(ties2)[0].T

frame=ties.merge(ties2,how='outer',left_index=True,right_index=True)
frame=frame.fillna(0)
frame['diff']=frame['manager']-frame['manager-leader']
frame=frame.sort_values('diff', ascending=False)


filename="/dyad_context_manager-leader.xlsx"
path = config['Paths']['csv_outputs']+filename
path = check_create_folder(path)

frame.to_excel(path)



